# Name That LLM - LoRA fine-tune of Qwen3-8B-Base (first 24 blocks) as an 18-way classifier.
# Head: LayerNorm'd masked mean-pools of several hidden layers (BOS excluded) -> dropout -> linear.
# One job per T4 (no model splitting); jobs = prompt-grouped outer folds 0..4 and/or 'full' (all rows).
# Outputs per job: val_<job>_ep<k>.npy (OOF probs per epoch) and, for 'full', adapter/ + head.pt for inference.
JOBS = ['full', '0']
CFG = dict(keep=24, pool_layers=[4, 8, 12, 16, 20, 24], r=16, alpha=32, lora_dropout=0.05, head_dropout=0.1,
           lr_lora=1e-4, lr_head=1e-3, wd=0.01, epochs=2, micro=4, accum=4, max_len=448, seed=42,
           extra_cap=400, extra_w=0.5)   # extra rows per class (cap) and share of each class's weight given to them

WORKER = r'''
import json, os, sys, time, glob
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

CFG = json.loads(os.environ['NTL_CFG']); jobs = json.loads(os.environ['NTL_JOBS'])
LABELS = sorted(['gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','gemini-3.1-pro-preview',
  'gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','claude-fable-5','claude-opus-4-6','claude-opus-4-8',
  'claude-opus-5','claude-sonnet-5','qwen3.8-max','glm-5.3','glm-5.3-flash','kimi-k3'])
K = len(LABELS)
DATA_DIR = os.path.dirname(sorted(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True) + glob.glob('/kaggle/input/**/train.csv', recursive=True))[0])
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
y = train['model'].map({l: i for i, l in enumerate(LABELS)}).values
texts = train['text'].tolist()
folds = list(GroupKFold(5).split(train, groups=train['prompt_id']))

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


def load_extra():
    ex = load_extra_pool(LABELS)
    ex = ex.groupby('model', group_keys=False).apply(lambda g: g.sample(min(len(g), CFG['extra_cap']), random_state=0))
    return ex['text'].tolist(), ex['model'].map({l: i for i, l in enumerate(LABELS)}).values
EX_T, EX_Y = load_extra()
N_TR = len(texts)
texts = texts + EX_T                       # extra rows live at indices >= N_TR (never in any validation fold)
y = np.r_[y, EX_Y]
print('extra rows used', len(EX_T), np.bincount(EX_Y, minlength=K).tolist(), flush=True)

def sample_weights(a):
    # class totals preserved: train rows of a class with extra get (1-w), its extra rows share w * n_train_c
    w = np.ones(len(texts))
    ex_idx = np.arange(N_TR, len(texts))
    ntr = np.bincount(y[a], minlength=K).astype(float); nex = np.bincount(y[ex_idx], minlength=K).astype(float)
    for c in range(K):
        if nex[c] > 0:
            w[a[y[a] == c]] = 1 - CFG['extra_w']
            w[ex_idx[y[ex_idx] == c]] = CFG['extra_w'] * ntr[c] / nex[c]
    return w

def find_model_dir():
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        c = json.load(open(cfg))
        if c.get('model_type') == 'qwen3' and c.get('hidden_size') == 4096 and c.get('num_hidden_layers') == 36:
            return os.path.dirname(cfg)
PATH = find_model_dir()
tok = AutoTokenizer.from_pretrained(PATH); tok.padding_side = 'right'
if tok.pad_token is None: tok.pad_token = tok.eos_token

class Clf(nn.Module):
    def __init__(self, base, H):
        super().__init__()
        self.base = base
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in CFG['pool_layers']])
        self.drop = nn.Dropout(CFG['head_dropout'])
        self.out = nn.Linear(H * len(CFG['pool_layers']), K)
    def forward(self, ids, mask):
        hs = self.base(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        m = mask.unsqueeze(-1).float().clone(); m[:, 0] = 0
        pooled = [n((hs[L].float() * m).sum(1) / m.sum(1).clamp(min=1)) for n, L in zip(self.norms, CFG['pool_layers'])]
        return self.out(self.drop(torch.cat(pooled, -1)))

def build():
    cfg = AutoConfig.from_pretrained(PATH); cfg.num_hidden_layers = CFG['keep']
    if getattr(cfg, 'layer_types', None): cfg.layer_types = cfg.layer_types[:CFG['keep']]
    base = AutoModel.from_pretrained(PATH, config=cfg, dtype=torch.float16).cuda()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    base = get_peft_model(base, LoraConfig(r=CFG['r'], lora_alpha=CFG['alpha'], lora_dropout=CFG['lora_dropout'],
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    for p in base.parameters():
        if p.requires_grad: p.data = p.data.float()
    return Clf(base, cfg.hidden_size).cuda()

@torch.no_grad()
def predict(model, idx):
    model.eval(); idx = np.asarray(idx); P = np.zeros((len(idx), K), dtype=np.float32)
    order = np.argsort([len(texts[i]) for i in idx])
    for s in range(0, len(order), 16):
        o = order[s:s + 16]
        e = tok([(tok.bos_token or '') + texts[i] for i in idx[o]], return_tensors='pt', padding=True, truncation=True,
                max_length=CFG['max_len'], add_special_tokens=False)
        with torch.autocast('cuda', dtype=torch.float16):
            P[o] = model(e['input_ids'].cuda(), e['attention_mask'].cuda()).float().softmax(-1).cpu().numpy()
    model.train(); return P

for job in jobs:
    torch.manual_seed(CFG['seed']); rng = np.random.default_rng(CFG['seed'])
    a, b = (np.arange(N_TR), None) if job == 'full' else folds[int(job)]
    SW = sample_weights(a)
    a = np.r_[a, np.arange(N_TR, len(texts))]
    model = build()
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('base.')]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('base.')]
    opt = torch.optim.AdamW([{'params': lora, 'lr': CFG['lr_lora']}, {'params': head, 'lr': CFG['lr_head']}],
                            weight_decay=CFG['wd'])
    upd = CFG['epochs'] * (len(a) // (CFG['micro'] * CFG['accum']))
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * upd), upd)
    scaler = torch.amp.GradScaler(init_scale=256.0)
    t0 = time.time(); model.train()
    for ep in range(CFG['epochs']):
        perm = rng.permutation(a); losses = []
        for k, s in enumerate(range(0, len(perm) - CFG['micro'] + 1, CFG['micro'])):
            j = perm[s:s + CFG['micro']]
            e = tok([(tok.bos_token or '') + texts[i] for i in j], return_tensors='pt', padding=True, truncation=True,
                    max_length=CFG['max_len'], add_special_tokens=False)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(e['input_ids'].cuda(), e['attention_mask'].cuda()).float()
            sw = torch.as_tensor(SW[j], dtype=torch.float32).cuda()
            loss = (F.cross_entropy(logits, torch.as_tensor(y[j]).cuda(), reduction='none') * sw).sum() / sw.sum()
            scaler.scale(loss / CFG['accum']).backward(); losses.append(loss.item())
            if (k + 1) % CFG['accum'] == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
        msg = f'[{job}] ep {ep} loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s)'
        if b is not None:
            P = predict(model, b); np.save(f'val_{job}_ep{ep}.npy', P)
            msg += f' val bal_acc {balanced_accuracy_score(y[b], P.argmax(1)):.4f}'
        print(msg, flush=True)
    out = 'adapter' if job == 'full' else f'fold{job}'   # keep fold models too -> fold-ensemble inference
    model.base.save_pretrained(out)
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith('base.')}, f'{out}/head.pt')
    json.dump(CFG, open('lora_cfg.json', 'w'))
    del model, opt; torch.cuda.empty_cache()
print('worker done', flush=True)
'''

import json, os, subprocess, sys, time
# peft probes the optional torchao dispatcher even for plain fp16 Linear layers and rejects Kaggle's old
# torchao 0.10; we use no quantisation, so remove the unused package (offline uninstall, no installs).
print(subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], capture_output=True, text=True).stdout[-300:])
import torch
open('worker.py', 'w').write(WORKER)
n = max(torch.cuda.device_count(), 1)
procs = []
for w in range(n):
    my = JOBS[w::n]
    if not my:
        continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w), NTL_CFG=json.dumps(CFG), NTL_JOBS=json.dumps(my))
    procs.append(subprocess.Popen([sys.executable, 'worker.py'], env=env, stdout=open(f'log_w{w}.txt', 'w'),
                                  stderr=subprocess.STDOUT))
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(60)
for w in range(n):
    if os.path.exists(f'log_w{w}.txt'):
        lines = open(f'log_w{w}.txt').read().splitlines()
        print(f'--- worker {w}'); print('\n'.join(l for l in lines if l.startswith('[') or 'Error' in l or 'error' in l
                                                    or 'done' in l or 'Traceback' in l)[-6000:])
        print('\n'.join(lines[-15:]))
print(f'total {time.time() - t0:.0f}s')
