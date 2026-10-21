# Name That LLM - v5: stylegram + Qwen3-8B L2+22 + Qwen3.5-9B L17/L18 heads + Qwen3-8B LoRA (nested CV 0.518)
# Log-prob blend of: stylegram n-grams (LinearSVC), TF-IDF LR, and frozen-LM layer heads.
# Train embeddings come precomputed from research kernels (kernel_sources: ntl-feat-*); test embeddings are
# computed here with the identical extraction code. Blend weights: L-BFGS on local OOF log-loss.
HEADS = [{'name': 'q8b', 'kernel': 'ntl-feat-qwen3-8b', 'hidden': 4096, 'nlayers': 36, 'groups': [[2, 22]], 'C': 0.0025, 'drop_bos': False}, {'name': 'q35', 'kernel': 'ntl-feat-qwen35-9b', 'hidden': 4096, 'nlayers': 32, 'groups': [[18], [17]], 'C': 0.0025, 'drop_bos': True}]   # [{name, kernel, hidden, nlayers, groups:[[L,..],..], C, drop_bos}]
W = {'stylegram': 2.1946, 'q8b_L2+22': 0.074, 'q35_L18': 0.172, 'q35_L17': 0.1578, 'lora_q8b': 0.4714}     # {component name: weight}
LORAS = [{'name': 'lora_q8b', 'kernel': 'ntl-lora-q8b'}]   # [{name, kernel}] LoRA classifiers trained by kaggle_research/lora-train (adapter/, head.pt, lora_cfg.json)
TSTD = True         # transductive standardisation of test features (no labels used)

import gc, glob, json, os, pathlib, re, subprocess, sys, time
if [{'name': 'lora_q8b', 'kernel': 'ntl-lora-q8b'}]:   # peft rejects Kaggle's old torchao even without quantisation; it is unused -> remove (offline)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], capture_output=True)
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
DATA = pathlib.Path(os.path.dirname(sorted(glob.glob('/kaggle/input/**/name-that-llm/test.csv', recursive=True) + glob.glob('/kaggle/input/**/test.csv', recursive=True))[0]))
train = pd.read_csv(DATA / 'train.csv')
test = pd.read_csv(DATA / 'test.csv')
test['text'] = test['text'].fillna('').astype(str)
y = train['model'].map({l: i for i, l in enumerate(LABELS)}).values
TR, TE = train['text'].tolist(), test['text'].tolist()
LOGIT = np.zeros((len(TE), K))
print('train', train.shape, 'test', test.shape, 'gpus', torch.cuda.device_count(), flush=True)


def add(name, p):
    if W.get(name, 0) > 0:
        LOGIT[:] += W[name] * np.log(np.clip(p, 1e-9, 1))
    print(f'{name} done (w={W.get(name, 0)}) {time.time() - T0:.0f}s', flush=True)


def full_proba(clf, X):
    p = np.full((X.shape[0], K), 1e-9)
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


# ---------------------------------------------------------------- stylegram n-grams + LinearSVC
FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


if W.get('stylegram', 0) > 0:
    A, B = [], []
    for an, lo, hi, maxf, w in SPEC:
        tf = shape_text if an == 'shape' else function_text if an == 'function' else str
        actual = 'char' if an == 'shape' else 'word' if an == 'function' else an
        opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf)
        if actual == 'word':
            opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**opts)
        A.append(v.fit_transform([tf(t) for t in TR]) * w)
        B.append(v.transform([tf(t) for t in TE]) * w)
    svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A, format='csr'), y)
    s = svc.decision_function(hstack(B, format='csr'))
    s = np.exp(s - s.max(1, keepdims=True))
    p = np.full((len(TE), K), 1e-9)
    p[:, svc.classes_] = s / s.sum(1, keepdims=True)
    add('stylegram', p)
    del A, B, svc

# ---------------------------------------------------------------- plain TF-IDF LR
if W.get('tfidf', 0) > 0:
    feats = FeatureUnion([
        ('w', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                              token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2, max_features=30000)),
        ('c', TfidfVectorizer(analyzer='char', ngram_range=(1, 5), sublinear_tf=True, lowercase=False,
                              min_df=2, max_features=30000))])
    add('tfidf', full_proba(LogisticRegression(C=10.0, max_iter=3000).fit(feats.fit_transform(TR), y),
                            feats.transform(TE)))


# ---------------------------------------------------------------- frozen-LM heads
def find_model_dir(hidden, nlayers):
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        tc = c.get('text_config', c)
        if tc.get('hidden_size') == hidden and tc.get('num_hidden_layers') == nlayers \
                and glob.glob(os.path.dirname(cfg) + '/*.safetensors'):
            return os.path.dirname(cfg)
    raise SystemExit(f'model {hidden}/{nlayers} not attached')


@torch.no_grad()
def embed(path, texts, layers, drop_bos):
    """Must match kaggle_research/feat_kernel.py exactly (same pooling, fp16, truncation, BOS handling)."""
    cfg = AutoConfig.from_pretrained(path)
    tcfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
    keep = max(layers)
    tcfg.num_hidden_layers = keep
    if getattr(tcfg, 'layer_types', None):
        tcfg.layer_types = tcfg.layer_types[:keep]
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map='auto').eval()
    if hasattr(model, 'language_model'):
        model = model.language_model
    dev = model.get_input_embeddings().weight.device
    E = np.zeros((len(texts), len(layers), tcfg.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    for b0 in range(0, len(texts), 8):
        idx = order[b0:b0 + 8]
        enc = tok([(tok.bos_token or '') + texts[i] for i in idx], return_tensors='pt', padding=True,
                  truncation=True, max_length=512, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if not all(torch.isfinite(hs[L]).all() for L in layers):
            with torch.autocast('cuda', enabled=False):
                model.float()
                hs = model(**enc, output_hidden_states=True).hidden_states
                model.half()
        m = enc['attention_mask'].unsqueeze(-1).float()
        if drop_bos:
            m[:, 0] = 0
        for j, L in enumerate(layers):
            h = hs[L].float()
            E[idx, j] = np.clip(((h * m.to(h.device)).sum(1) / m.sum(1).to(h.device)).cpu().numpy(), -6e4, 6e4)
        del hs, enc
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.nan_to_num(E)


for hd in HEADS:
    kdir = next(p for p in glob.glob('/kaggle/input/**/emb_train.npy', recursive=True) if hd['kernel'] in p)
    kdir = os.path.dirname(kdir)
    stored = json.load(open(os.path.join(kdir, 'layers.json')))
    Etr = np.load(os.path.join(kdir, 'emb_train.npy'), mmap_mode='r')
    need = sorted({L for g in hd['groups'] for L in g})
    Ete = embed(find_model_dir(hd['hidden'], hd['nlayers']), TE, need, hd['drop_bos'])
    for g in hd['groups']:
        Xtr = np.concatenate([np.asarray(Etr[:, stored.index(L)], dtype=np.float32) for L in g], 1)
        Xte = np.concatenate([Ete[:, need.index(L)] for L in g], 1)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=hd['C'] / len(g), max_iter=5000).fit(sc.transform(Xtr), y)
        Xte = StandardScaler().fit_transform(Xte) if (TSTD and len(TE) >= 2) else sc.transform(Xte)
        add(f"{hd['name']}_L{'+'.join(map(str, g))}", full_proba(clf, Xte))
    del Etr, Ete
    gc.collect()

# ---------------------------------------------------------------- LoRA classifiers
if LORAS:
    import torch.nn as nn
    from peft import PeftModel

    class Clf(nn.Module):   # identical to kaggle_research/lora-train worker
        def __init__(self, base, H, cfg):
            super().__init__()
            self.base, self.cfg = base, cfg
            self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in cfg['pool_layers']])
            self.drop = nn.Dropout(cfg['head_dropout'])
            self.out = nn.Linear(H * len(cfg['pool_layers']), K)

        def forward(self, ids, mask):
            hs = self.base(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
            m = mask.unsqueeze(-1).float().clone()
            m[:, 0] = 0
            pooled = [n((hs[L].float() * m).sum(1) / m.sum(1).clamp(min=1))
                      for n, L in zip(self.norms, self.cfg['pool_layers'])]
            return self.out(self.drop(torch.cat(pooled, -1)))

for lo in LORAS:
    kdir = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/head.pt', recursive=True) if lo['kernel'] in p))
    lcfg = json.load(open(os.path.join(kdir, 'lora_cfg.json')))
    path = find_model_dir(4096, 36)
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = lcfg['keep']
    if getattr(cfg, 'layer_types', None):
        cfg.layer_types = cfg.layer_types[:lcfg['keep']]
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16).cuda()
    adir = os.path.join(kdir, 'adapter') if os.path.exists(os.path.join(kdir, 'adapter', 'adapter_config.json')) else kdir
    base = PeftModel.from_pretrained(base, adir).eval()
    model = Clf(base, cfg.hidden_size, lcfg).cuda().eval()
    model.load_state_dict(torch.load(os.path.join(kdir, 'head.pt'), map_location='cuda'), strict=False)
    P = np.zeros((len(TE), K))
    order = np.argsort([len(t) for t in TE])
    with torch.no_grad():
        for s0 in range(0, len(order), 16):
            o = order[s0:s0 + 16]
            e = tok([(tok.bos_token or '') + TE[i] for i in o], return_tensors='pt', padding=True, truncation=True,
                    max_length=lcfg['max_len'], add_special_tokens=False)
            with torch.autocast('cuda', dtype=torch.float16):
                P[o] = model(e['input_ids'].cuda(), e['attention_mask'].cuda()).float().softmax(-1).cpu().numpy()
    add(lo['name'], P)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

sub = pd.DataFrame({'id': test['id'], 'model': [LABELS[i] for i in LOGIT.argmax(1)]})
assert sub['id'].is_unique and sub['model'].isin(LABELS).all() and len(sub) == len(test)
sub.to_parquet('submission.parquet', index=False)
print(sub['model'].value_counts().to_string())
print(f'total {time.time() - T0:.0f}s')
