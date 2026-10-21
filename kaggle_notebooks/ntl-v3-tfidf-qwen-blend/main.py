# Name That LLM - v3: TF-IDF LR + frozen Qwen3-1.7B-Base layer heads (L4, L16, L19), log-prob blend.
# Nested prompt-grouped CV 0.444. GPU (T4) + input model qwen-lm/qwen-3 Transformers 1.7b-base. Internet off.
LAYERS = [4, 16, 19]
W = {'tfidf': 1.8, 4: 0.4, 16: 0.7, 19: 0.5}   # coordinate-search on OOF log-probs
LLM_C = 0.005

import glob, json, os, pathlib, time
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

T0 = time.time()
LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
DATA = next(p for p in [pathlib.Path('/kaggle/input/competitions/name-that-llm'),
                        pathlib.Path('/kaggle/input/name-that-llm'), pathlib.Path('data')]
            if (p / 'train.csv').exists())
train = pd.read_csv(DATA / 'train.csv')
test = pd.read_csv(DATA / 'test.csv')
test['text'] = test['text'].fillna('').astype(str)
y = train['model'].map({l: i for i, l in enumerate(LABELS)}).values
print('train', train.shape, 'test', test.shape, flush=True)


def log_proba(clf, X):
    p = np.full((X.shape[0], K), 1e-9)
    p[:, clf.classes_] = clf.predict_proba(X)
    return np.log(np.clip(p, 1e-9, 1))


feats = FeatureUnion([
    ('w', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                          token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2, max_features=30000)),
    ('c', TfidfVectorizer(analyzer='char', ngram_range=(1, 5), sublinear_tf=True, lowercase=False,
                          min_df=2, max_features=30000))])
logit = W['tfidf'] * log_proba(LogisticRegression(C=10.0, max_iter=3000).fit(feats.fit_transform(train['text']), y),
                               feats.transform(test['text']))
print(f'tfidf done {time.time() - T0:.0f}s', flush=True)


def find_model_dir():
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if c.get('model_type') == 'qwen3' and c.get('hidden_size') == 2048 and c.get('num_hidden_layers') == 28:
            return os.path.dirname(cfg)
    return 'Qwen/Qwen3-1.7B-Base'


@torch.no_grad()
def embed(texts, model, tok, bs=16):
    out = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    for b0 in range(0, len(texts), bs):
        idx = order[b0:b0 + bs]
        enc = tok([(tok.bos_token or '') + texts[i] for i in idx], return_tensors='pt', padding=True,
                  truncation=True, max_length=512, add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].unsqueeze(-1).float()
        for j, L in enumerate(LAYERS):
            out[idx, j] = ((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy()
    return np.nan_to_num(out)


mdir = find_model_dir()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('LLM', mdir, dev, flush=True)
tok = AutoTokenizer.from_pretrained(mdir)
tok.padding_side = 'right'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
try:
    llm = AutoModel.from_pretrained(mdir, dtype=torch.float16)
except TypeError:
    llm = AutoModel.from_pretrained(mdir, torch_dtype=torch.float16)
llm = llm.to(dev).eval()
E_tr = embed(list(train['text']), llm, tok)
E_te = embed(list(test['text']), llm, tok)
del llm
print(f'embeddings done {time.time() - T0:.0f}s', flush=True)

for j, L in enumerate(LAYERS):
    sc = StandardScaler().fit(E_tr[:, j])
    clf = LogisticRegression(C=LLM_C, max_iter=5000).fit(sc.transform(E_tr[:, j]), y)
    logit += W[L] * log_proba(clf, sc.transform(E_te[:, j]))

sub = pd.DataFrame({'id': test['id'], 'model': [LABELS[i] for i in logit.argmax(1)]})
assert sub['id'].is_unique and sub['model'].isin(LABELS).all() and len(sub) == len(test)
sub.to_parquet('submission.parquet', index=False)
print(sub['model'].value_counts().to_string())
print(f'total {time.time() - T0:.0f}s')
