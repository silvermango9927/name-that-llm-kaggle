# Name That LLM - research kernel: frozen-LM feature extraction + in-kernel prompt-grouped CV.
# Outputs (reused by submission kernels via kernel_sources):
#   emb_train.npy  (n_train, n_layers, H) float16 mean-pooled hidden states, layers.json
#   oof_L<k>.npy   OOF probabilities per layer head (tstd), cv.json
MODEL_TYPE = '__MODEL_TYPE__'   # config model_type to locate the attached model
HIDDEN = __HIDDEN__             # hidden size to disambiguate
NLAYERS = __NLAYERS__           # total layers in the checkpoint (disambiguates e.g. Qwen3-8B vs Qwen3.5-9B)
KEEP = __KEEP__                 # transformer blocks to load (later layers unused)
STRIDE = __STRIDE__
TAG = '__TAG__'

import gc, glob, json, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
DATA_DIR = os.path.dirname(sorted(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True) + glob.glob('/kaggle/input/**/train.csv', recursive=True))[0])
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
y = train['model'].map({l: i for i, l in enumerate(LABELS)}).values
texts = train['text'].tolist()


def find_model_dir():
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        tc = c.get('text_config', c)
        if (tc.get('hidden_size') == HIDDEN and tc.get('num_hidden_layers') == NLAYERS
                and glob.glob(os.path.dirname(cfg) + '/*.safetensors')):
            return os.path.dirname(cfg)
    raise SystemExit('model not found: ' + str(glob.glob('/kaggle/input/**/config.json', recursive=True)))


path = find_model_dir()
cfg = AutoConfig.from_pretrained(path)
tcfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
print('model', path, 'layers', tcfg.num_hidden_layers, '-> keep', KEEP, 'gpus', torch.cuda.device_count(), flush=True)
tcfg.num_hidden_layers = KEEP
if getattr(tcfg, 'layer_types', None):
    tcfg.layer_types = tcfg.layer_types[:KEEP]
tok = AutoTokenizer.from_pretrained(path)
tok.padding_side = 'right'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map='auto').eval()
if hasattr(model, 'language_model'):   # multimodal wrappers (Gemma 4): use the text tower only
    model = model.language_model
print(f'loaded {time.time() - T0:.0f}s', flush=True)

LAYERS = sorted(set(list(range(STRIDE, KEEP + 1, STRIDE)) + [KEEP]))
H = tcfg.hidden_size
E = np.zeros((len(texts), len(LAYERS), H), dtype=np.float16)
order = np.argsort([len(t) for t in texts])
dev = model.get_input_embeddings().weight.device
bad = 0
with torch.no_grad():
    for b0 in range(0, len(texts), 8):
        idx = order[b0:b0 + 8]
        enc = tok([(tok.bos_token or '') + texts[i] for i in idx], return_tensors='pt', padding=True,
                  truncation=True, max_length=512, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if not all(torch.isfinite(hs[L]).all() for L in LAYERS):   # fp16 overflow -> redo batch in fp32 path
            bad += 1
            with torch.autocast('cuda', enabled=False):
                model.float()
                hs = model(**enc, output_hidden_states=True).hidden_states
                model.half()
        m = enc['attention_mask'].unsqueeze(-1).float()
        m[:, 0] = 0   # drop BOS: attention-sink token with huge norm dominates the mean
        for j, L in enumerate(LAYERS):
            h = hs[L].float()
            E[idx, j] = np.clip(((h * m.to(h.device)).sum(1) / m.sum(1).to(h.device)).cpu().numpy(), -6e4, 6e4)
        del hs, enc
print(f'embedded {time.time() - T0:.0f}s, nonfinite batches {bad}', flush=True)
del model
gc.collect()
torch.cuda.empty_cache()
np.save('emb_train.npy', np.nan_to_num(E))
json.dump(LAYERS, open('layers.json', 'w'))

# ---- prompt-grouped CV per layer, transductive standardisation of the held-out fold
folds = list(GroupKFold(5).split(train, groups=train['prompt_id']))
C = 0.005 * 2048 / H
cv = {}
for j, L in enumerate(LAYERS):
    X = E[:, j].astype(np.float32)
    P = np.zeros((len(X), K))
    for a, b in folds:
        sc = StandardScaler().fit(X[a])
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[a]), y[a])
        P[b][:, :] = 0
        P[b] = clf.predict_proba(StandardScaler().fit_transform(X[b]))
    s = balanced_accuracy_score(y, P.argmax(1))
    cv[L] = round(float(s), 4)
    np.save(f'oof_L{L}.npy', P.astype(np.float32))
    print(f'{TAG} layer {L:3d}: OOF bal acc {s:.4f}  ({time.time() - T0:.0f}s)', flush=True)
json.dump(cv, open('cv.json', 'w'))
print('best', sorted(cv.items(), key=lambda kv: -kv[1])[:5])
