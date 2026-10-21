# Name That LLM - v13: v12 with 6-adapter LoRA average (small-pool x4 + big-pool full/fold0)
# Log-prob blend of: stylegram n-grams (LinearSVC), TF-IDF LR, and frozen-LM layer heads.
# Train embeddings come precomputed from research kernels (kernel_sources: ntl-feat-*); test embeddings are
# computed here with the identical extraction code. Blend weights: L-BFGS on local OOF log-loss.
HEADS = [{'name': 'q8b', 'kernel': 'ntl-feat-qwen3-8b', 'hidden': 4096, 'nlayers': 36, 'groups': [[2, 22]], 'C': 0.0025, 'drop_bos': False, 'extra_w': 0.5, 'extra_kernel': 'ntl-extra-heads-pool', 'extra_layers': [2, 22]}, {'name': 'q35', 'kernel': 'ntl-feat-qwen35-9b', 'hidden': 4096, 'nlayers': 32, 'groups': [[18]], 'C': 0.0025, 'drop_bos': True, 'extra_w': 0.5, 'extra_kernel': 'ntl-extra-heads-pool', 'extra_layers': [17, 18]}]   # [{name, kernel, hidden, nlayers, groups:[[L,..],..], C, drop_bos}]
W = {'stylegram': 2.7839, 'q8b_L2+22': 0.1717, 'q35_L18': 0.8899, 'lora_q8b_x6': 0.2976}     # {component name: weight}
LORAS = [{'name': 'lora_q8b_x6', 'kernel': 'ntl-lora-q8b-extra2', 'members': [('ntl-lora-q8b-extra2', 'adapter'), ('ntl-lora-q8b-extra2', 'fold2'), ('ntl-lora-q8b-extra', 'fold0'), ('ntl-lora-q8b-extra', 'fold1'), ('ntl-lora-q8b-pool', 'adapter'), ('ntl-lora-q8b-pool', 'fold0')]}]   # [{name, kernel}] LoRA classifiers trained by kaggle_research/lora-train (adapter/, head.pt, lora_cfg.json)
TSTD = True         # transductive standardisation of test features (no labels used)
EXTRA_W = 0.5       # >0: stylegram also trains on public extra generations; extra rows share this fraction of each class's weight
CLAUDE_GATE = 1.0  # >0: after the blend predicts a Claude model, re-pick within Claude with a Claude-only stylegram SVC (lambda)
SINKHORN = 0.25     # >0: partial Sinkhorn balancing of test class mass (alpha)
TWIN_W = 0.5         # >0: add prompt-centered twin evidence for test rows in topic clusters of size 2..18 (x head weight)
TWIN_THRESH = 0.9           # average-linkage cosine cut on content-word TF-IDF (ARI 0.96 vs prompt_id on train)

import gc, glob, json, os, pathlib, re, subprocess, sys, time
if [{'name': 'lora_q8b_x6', 'kernel': 'ntl-lora-q8b-extra2', 'members': [('ntl-lora-q8b-extra2', 'adapter'), ('ntl-lora-q8b-extra2', 'fold2'), ('ntl-lora-q8b-extra', 'fold0'), ('ntl-lora-q8b-extra', 'fold1'), ('ntl-lora-q8b-pool', 'adapter'), ('ntl-lora-q8b-pool', 'fold0')]}]:   # peft rejects Kaggle's old torchao even without quantisation; it is unused -> remove (offline)
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


# --- curated host-format extra pool (mirrors src/extra.py; sources = public CC-BY/CC-BY-SA Kaggle datasets) ---
EXTRA_SOURCES = [  # (file name, label column)
    ('ntl_extra_passages_claude.csv', 'model'), ('ntl_generations.csv', 'label'), ('accepted.csv', 'model'),
    ('generated_v1.csv', 'label'), ('passages_clean.csv', 'model'), ('generations.csv', 'label'),
    ('ntl_alp_generations.csv', 'label'), ('ntl_proxy_generations.csv', 'label'), ('gen_passages.csv', 'model'),
    ('ntl_gpt_proxy_passages.csv', 'label'), ('passages.csv', 'label')]
EXTRA_DATASETS = ['seyitkaangunes/ntl-extra-api-passages', 'abdullah1afify/ntl-generations',
                  'keshavatearth/ntl-gpt-new-prompts-v1', 'prvsiyan/ntl-own-prompt-generations-v1',
                  'prvsiyan/prvsiyans-extended-name-that-llm-dataset', 'lxdlxdlxd/ntl-synth-generations-v1',
                  'alperen5252525/ntl-vertex-gemini-proxy-gpt-generations', 'dophuchung/ntl-proxy-generations',
                  'lucasboesen/ntl-free-proxy-generations', 'kaplinskiy/ntl-gpt-proxy-passages',
                  'haileycheng/name-that-llm-same-prompt-passages']


def load_extra_pool(labels):
    import glob, re
    import pandas as pd
    md = re.compile(r'\*\*|(^|\n)\s*#|(^|\n)\s*[-*•]\s|(^|\n)\s*\d+\.\s')
    parts = []
    for fname, col in EXTRA_SOURCES:
        for hit in glob.glob('/kaggle/input/**/' + fname, recursive=True):
            d = pd.read_csv(hit, low_memory=False)
            if 'text' not in d or col not in d or not d[col].isin(labels).mean() > 0.5:
                continue   # e.g. a same-named file in an unrelated input
            parts.append(pd.DataFrame({'text': d['text'].astype(str).str.strip(), 'model': d[col], 'src': fname}))
    ex = pd.concat(parts, ignore_index=True)
    ex = ex[ex['model'].isin(labels)]
    ok = ex['text'].map(lambda t: '\n' not in t and not md.search(t) and 30 <= len(t.split()) <= 300)
    ex = ex[ok].drop_duplicates('text').reset_index(drop=True)
    print('extra pool', len(ex), ex['model'].value_counts().to_dict(), flush=True)
    return ex
# --- end extra pool ---


POOL = True   # True: stylegram uses the full curated host-format pool (11 public datasets); False: original 4 files


def load_extra():
    if POOL:
        ex = load_extra_pool(LABELS)
        ex['y'] = ex['model'].map({l: i for i, l in enumerate(LABELS)})
        return ex
    specs = [('ntl_extra_passages_claude.csv', 'model'), ('ntl_generations.csv', 'label'),
             ('accepted.csv', 'model'), ('generated_v1.csv', 'label')]
    parts = []
    for fname, col in specs:
        hits = glob.glob('/kaggle/input/**/' + fname, recursive=True)
        assert hits, f'extra data file {fname} not attached'
        d = pd.read_csv(hits[0])
        parts.append(pd.DataFrame({'text': d['text'].astype(str), 'model': d[col], 'src': fname}))
    ex = pd.concat(parts).drop_duplicates('text')
    ex = ex[ex['model'].isin(LABELS)].reset_index(drop=True)
    ex['y'] = ex['model'].map({l: i for i, l in enumerate(LABELS)})
    return ex


EX = load_extra() if (EXTRA_W > 0 or CLAUDE_GATE > 0) else None


def stylegram_fit(fit_texts, fit_y, sw=None):
    A, B = [], []
    for an, lo, hi, maxf, w in SPEC:
        tf = shape_text if an == 'shape' else function_text if an == 'function' else str
        actual = 'char' if an == 'shape' else 'word' if an == 'function' else an
        opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf)
        if actual == 'word':
            opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**opts)
        A.append(v.fit_transform([tf(t) for t in fit_texts]) * w)
        B.append(v.transform([tf(t) for t in TE]) * w)
    svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A, format='csr'), fit_y, sample_weight=sw)
    return svc, svc.decision_function(hstack(B, format='csr'))


if W.get('stylegram', 0) > 0 and EXTRA_W > 0:
    fy = np.r_[y, EX['y'].values]
    sw = np.ones(len(fy))
    ntr, nex = np.bincount(y, minlength=K).astype(float), np.bincount(EX['y'].values, minlength=K).astype(float)
    for c in range(K):
        if nex[c] > 0:
            sw[:len(y)][y == c] = 1 - EXTRA_W
            sw[len(y):][EX['y'].values == c] = EXTRA_W * ntr[c] / nex[c]
    svc, s = stylegram_fit(TR + EX['text'].tolist(), fy, sw)
    s = np.exp(s - s.max(1, keepdims=True))
    p = np.full((len(TE), K), 1e-9)
    p[:, svc.classes_] = s / s.sum(1, keepdims=True)
    add('stylegram', p)
    del svc
elif W.get('stylegram', 0) > 0:
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


def topic_clusters(texts, thresh):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    Xt = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2 if len(texts) > 50 else 1).fit_transform(texts)
    Dm = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
    np.fill_diagonal(Dm, 0)
    return fcluster(linkage(squareform(Dm, checks=False), method='average'), thresh, criterion='distance')


def group_center(X, groups):
    X = np.array(X, dtype=np.float32, copy=True)
    for k in np.unique(groups):
        i = groups == k
        X[i] -= X[i].mean(0)
    return X


if TWIN_W > 0:
    G_TE = topic_clusters(TE, TWIN_THRESH)
    _sz = np.bincount(G_TE)[G_TE]
    TWIN_ROWS = (_sz >= 2) & (_sz <= 18)
    if len(TE) < 50:   # interactive 3-row run: force one cluster so the twin code path is exercised before scoring
        G_TE, TWIN_ROWS = np.ones(len(TE), dtype=int), np.ones(len(TE), dtype=bool)
    print('twin rows (clusters 2..18):', int(TWIN_ROWS.sum()), 'of', len(TE), '| cluster-size counts',
          {k: int(v) for k, v in zip(*np.unique(_sz, return_counts=True)) if k <= 20}, flush=True)
    G_TR = train['prompt_id'].values

for hd in HEADS:
    kdir = next(p for p in glob.glob('/kaggle/input/**/emb_train.npy', recursive=True) if f"/{hd['kernel']}/" in p)
    kdir = os.path.dirname(kdir)
    stored = json.load(open(os.path.join(kdir, 'layers.json')))
    Etr = np.load(os.path.join(kdir, 'emb_train.npy'), mmap_mode='r')
    need = sorted({L for g in hd['groups'] for L in g})
    Ete = embed(find_model_dir(hd['hidden'], hd['nlayers']), TE, need, hd['drop_bos'])
    for g in hd['groups']:
        Xtr = np.concatenate([np.asarray(Etr[:, stored.index(L)], dtype=np.float32) for L in g], 1)
        Xte = np.concatenate([Ete[:, need.index(L)] for L in g], 1)
        sc = StandardScaler().fit(Xtr)
        Xfit, yfit, swfit = Xtr, y, None
        if hd.get('extra_w', 0) > 0:   # add embedded public extra passages (ntl-extra-heads output), class totals preserved
            xdir = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/extra_y.npy', recursive=True)
                                        if f"/{hd['extra_kernel']}/" in p))   # exact segment (-pool variant exists)
            ey = np.load(os.path.join(xdir, 'extra_y.npy'))
            Eex = np.load(os.path.join(xdir, f"emb_extra_{hd['name']}.npy"))
            xl = hd['extra_layers']   # layer order as stored by ntl-extra-heads (may include layers this head doesn't use)
            Xex = np.concatenate([Eex[:, xl.index(L)].astype(np.float32) for L in g], 1)
            ntr, nex = np.bincount(y, minlength=K).astype(float), np.bincount(ey, minlength=K).astype(float)
            swt, swe = np.ones(len(y)), np.zeros(len(ey))
            for c in range(K):
                if nex[c] > 0:
                    swt[y == c] = 1 - hd['extra_w']
                    swe[ey == c] = hd['extra_w'] * ntr[c] / nex[c]
            Xfit, yfit, swfit = np.r_[Xtr, Xex], np.r_[y, ey], np.r_[swt, swe]
        clf = LogisticRegression(C=hd['C'] / len(g), max_iter=5000).fit(sc.transform(Xfit), yfit, sample_weight=swfit)
        Xte = StandardScaler().fit_transform(Xte) if (TSTD and len(TE) >= 2) else sc.transform(Xte)
        cname = f"{hd['name']}_L{'+'.join(map(str, g))}"
        add(cname, full_proba(clf, Xte))
        if TWIN_W > 0 and TWIN_ROWS.any() and W.get(cname, 0) > 0:
            Xc = group_center(Xtr, G_TR)
            scc = StandardScaler().fit(Xc)
            twin = LogisticRegression(C=hd['C'] / len(g), max_iter=5000).fit(scc.transform(Xc), y)
            Xte_raw = np.concatenate([Ete[:, need.index(L)] for L in g], 1)[TWIN_ROWS]
            pt = full_proba(twin, scc.transform(group_center(Xte_raw, G_TE[TWIN_ROWS])))
            LOGIT[TWIN_ROWS] += TWIN_W * W[cname] * np.log(np.clip(pt, 1e-9, 1))
            print(f'  {cname} twin added on {int(TWIN_ROWS.sum())} rows', flush=True)
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

def lora_dirs(lo):
    """[(adapter_dir, lora_cfg.json path)] per member; lo['members'] = [(kernel name, adapter subdir e.g. 'adapter'/'fold0')]."""
    out = []
    for kern, sub in lo.get('members', [(lo['kernel'], 'adapter')]):
        hits = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/adapter_config.json', recursive=True)
                if f'/{kern}/' in p and os.path.basename(os.path.dirname(p)) == sub]   # exact path segment
        assert len(hits) >= 1, f'LoRA member {kern}/{sub} not found'
        cfgp = next(p for p in glob.glob('/kaggle/input/**/lora_cfg.json', recursive=True) if f'/{kern}/' in p)
        out.append((hits[0], cfgp))
    return out


for lo in LORAS:
    members = lora_dirs(lo)
    print(lo['name'], 'members', [m[0] for m in members], flush=True)
    P = np.zeros((len(TE), K))
    for adir, cfgp in members:
        lcfg = json.load(open(cfgp))
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
        base = PeftModel.from_pretrained(base, adir).eval()
        model = Clf(base, cfg.hidden_size, lcfg).cuda().eval()
        hp = os.path.join(adir, 'head.pt') if os.path.exists(os.path.join(adir, 'head.pt')) else \
            os.path.join(os.path.dirname(cfgp), 'head.pt')
        missing = model.load_state_dict(torch.load(hp, map_location='cuda'), strict=False)
        assert not [k for k in missing.missing_keys if not k.startswith('base.')], missing
        order = np.argsort([len(t) for t in TE])
        with torch.no_grad():
            for s0 in range(0, len(order), 16):
                o = order[s0:s0 + 16]
                e = tok([(tok.bos_token or '') + TE[i] for i in o], return_tensors='pt', padding=True, truncation=True,
                        max_length=lcfg['max_len'], add_special_tokens=False)
                with torch.autocast('cuda', dtype=torch.float16):
                    P[o] += model(e['input_ids'].cuda(), e['attention_mask'].cuda()).float().softmax(-1).cpu().numpy() / len(members)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    add(lo['name'], P)

P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)
if SINKHORN > 0 and len(P) >= 2:
    target = (1 - SINKHORN) * P.sum(0) + SINKHORN * np.full(K, len(P) / K)
    for _ in range(50):
        P *= (target / (P.sum(0) + 1e-12))[None, :]
        P /= P.sum(1, keepdims=True) + 1e-12
    print('sinkhorn changed', int((P.argmax(1) != LOGIT.argmax(1)).sum()), 'of', len(P), flush=True)
pred = P.argmax(1)
if CLAUDE_GATE > 0:
    cidx = np.array([i for i, l in enumerate(LABELS) if l.startswith('claude')])
    capi = EX[EX['src'] == 'ntl_extra_passages_claude.csv']
    m = np.isin(y, cidx)
    svc, D = stylegram_fit([t for t, k in zip(TR, m) if k] + capi['text'].tolist(), np.r_[y[m], capi['y'].values])
    assert list(svc.classes_) == list(cidx)
    hit = np.isin(pred, cidx)
    new = cidx[(np.log(np.clip(P[:, cidx], 1e-12, 1)) + CLAUDE_GATE * D).argmax(1)]
    print('claude gate changed', int((new[hit] != pred[hit]).sum()), 'of', int(hit.sum()), flush=True)
    pred[hit] = new[hit]
sub = pd.DataFrame({'id': test['id'], 'model': [LABELS[i] for i in pred]})
assert sub['id'].is_unique and sub['model'].isin(LABELS).all() and len(sub) == len(test)
sub.to_parquet('submission.parquet', index=False)
print(sub['model'].value_counts().to_string())
print(f'total {time.time() - T0:.0f}s')
