# Name That LLM - research kernel: do public extra generations help the frozen LM heads?
# Embeds the extra passages with the same extraction as feat_kernel.py (per model), reuses the train embeddings from the
# ntl-feat-* outputs, and CVs each head with extra rows added to training folds only (class-balanced weights).
# Outputs: emb_extra_<tag>.npy, extra_y.npy, oof_<tag>_L<..>_x<w>.npy, cv.json
MODELS = [  # tag, feat kernel, hidden, n_layers, drop_bos, head groups
    ('q35', 'ntl-feat-qwen35-9b', 4096, 32, True, [[18], [17]]),
    ('q8b', 'ntl-feat-qwen3-8b', 4096, 36, False, [[2, 22]]),
]
WS = [0.0, 0.3, 0.5]

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
L2I = {l: i for i, l in enumerate(LABELS)}
DATA_DIR = os.path.dirname(sorted(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True) +
                                  glob.glob('/kaggle/input/**/train.csv', recursive=True))[0])
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
y = train['model'].map(L2I).values
folds = list(GroupKFold(5).split(train, groups=train['prompt_id']))

parts = []
for fname, col in [('ntl_extra_passages_claude.csv', 'model'), ('ntl_generations.csv', 'label'),
                   ('accepted.csv', 'model'), ('generated_v1.csv', 'label')]:
    d = pd.read_csv(glob.glob('/kaggle/input/**/' + fname, recursive=True)[0])
    parts.append(pd.DataFrame({'text': d['text'].astype(str), 'model': d[col]}))
ex = pd.concat(parts).drop_duplicates('text')
ex = ex[ex['model'].isin(LABELS)].reset_index(drop=True)
ey = ex['model'].map(L2I).values
np.save('extra_y.npy', ey)
print('extra rows', len(ex), np.bincount(ey, minlength=K).tolist(), flush=True)


def find_model_dir(hidden, nlayers):
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        c = json.load(open(cfg))
        tc = c.get('text_config', c)
        if tc.get('hidden_size') == hidden and tc.get('num_hidden_layers') == nlayers \
                and glob.glob(os.path.dirname(cfg) + '/*.safetensors'):
            return os.path.dirname(cfg)


@torch.no_grad()
def embed(path, texts, layers, drop_bos):   # identical to feat_kernel.py / sub_template.py
    cfg = AutoConfig.from_pretrained(path)
    tcfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
    tcfg.num_hidden_layers = max(layers)
    if getattr(tcfg, 'layer_types', None):
        tcfg.layer_types = tcfg.layer_types[:max(layers)]
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map='auto').eval()
    if hasattr(model, 'language_model'):
        model = model.language_model
    dev = model.get_input_embeddings().weight.device
    E = np.zeros((len(texts), len(layers), tcfg.hidden_size), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])
    for b0 in range(0, len(texts), 8):
        idx = order[b0:b0 + 8]
        enc = tok([(tok.bos_token or '') + texts[i] for i in idx], return_tensors='pt', padding=True,
                  truncation=True, max_length=512, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
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


cv = {}
for tag, kname, hid, nl, drop_bos, groups in MODELS:
    kdir = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/emb_train.npy', recursive=True) if kname in p))
    stored = json.load(open(os.path.join(kdir, 'layers.json')))
    Etr = np.load(os.path.join(kdir, 'emb_train.npy'), mmap_mode='r')
    need = sorted({L for g in groups for L in g})
    Eex = embed(find_model_dir(hid, nl), ex['text'].tolist(), need, drop_bos)
    np.save(f'emb_extra_{tag}.npy', Eex)
    print(f'{tag} extra embedded {time.time() - T0:.0f}s', flush=True)
    for g in groups:
        Xtr = np.concatenate([np.asarray(Etr[:, stored.index(L)], dtype=np.float32) for L in g], 1)
        Xex = np.concatenate([Eex[:, need.index(L)].astype(np.float32) for L in g], 1)
        C = 0.0025 / len(g)
        for w in WS:
            P = np.zeros((len(y), K), dtype=np.float32)
            for a, b in folds:
                Xa, ya, sw = Xtr[a], y[a], np.ones(len(a))
                if w > 0:
                    ntr, nex = np.bincount(ya, minlength=K).astype(float), np.bincount(ey, minlength=K).astype(float)
                    swe = np.zeros(len(ey))
                    for c in range(K):
                        if nex[c] > 0:
                            sw[ya == c] = 1 - w
                            swe[ey == c] = w * ntr[c] / nex[c]
                    Xa, ya, sw = np.r_[Xa, Xex], np.r_[ya, ey], np.r_[sw, swe]
                sc = StandardScaler().fit(Xtr[a])
                clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xa), ya, sample_weight=sw)
                P[b] = clf.predict_proba(StandardScaler().fit_transform(Xtr[b]))
            name = f"{tag}_L{'+'.join(map(str, g))}_x{w}"
            cv[name] = round(float(balanced_accuracy_score(y, P.argmax(1))), 4)
            np.save(f'oof_{name}.npy', P)
            print(name, cv[name], f'({time.time() - T0:.0f}s)', flush=True)
json.dump(cv, open('cv.json', 'w'))
