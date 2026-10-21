# Name That LLM - LoRA fine-tune of Qwen3-8B-Base (first 24 blocks) as an 18-way classifier.
# Head: LayerNorm'd masked mean-pools of several hidden layers (BOS excluded) -> dropout -> linear.
# One job per T4 (no model splitting); jobs = prompt-grouped outer folds 0..4 and/or 'full' (all rows).
# Outputs per job: val_<job>_ep<k>.npy (OOF probs per epoch) and, for 'full', adapter/ + head.pt for inference.
JOBS = ['0', '1']
CFG = dict(match=dict(hidden_size=3840, num_hidden_layers=48), keep=20, pool_layers=[4, 8, 12, 16, 20], r=16, alpha=32, lora_dropout=0.05, head_dropout=0.1,
           lr_lora=1e-4, lr_head=1e-3, wd=0.01, epochs=3, micro=4, accum=4, max_len=448, seed=42)

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

def find_model_dir():
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        c = json.load(open(cfg)); tc = c.get('text_config', c)
        if all(tc.get(k) == v for k, v in CFG['match'].items()) and glob.glob(os.path.dirname(cfg) + '/*.safetensors'):
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
    cfg = AutoConfig.from_pretrained(PATH)
    tcfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
    tcfg.num_hidden_layers = CFG['keep']
    if getattr(tcfg, 'layer_types', None): tcfg.layer_types = tcfg.layer_types[:CFG['keep']]
    base = AutoModel.from_pretrained(PATH, config=cfg, dtype=torch.float16, device_map={'': 0})
    if hasattr(base, 'language_model'):   # multimodal wrapper: train the text tower only, drop the vision tower
        base = base.language_model
    torch.cuda.empty_cache()
    print('gpu mem after load', torch.cuda.memory_allocated() / 2**30, flush=True)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    base = get_peft_model(base, LoraConfig(r=CFG['r'], lora_alpha=CFG['alpha'], lora_dropout=CFG['lora_dropout'],
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    for p in base.parameters():
        if p.requires_grad: p.data = p.data.float()
    return Clf(base, tcfg.hidden_size).cuda()

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
    a, b = (np.arange(len(y)), None) if job == 'full' else folds[int(job)]
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
            loss = F.cross_entropy(logits, torch.as_tensor(y[j]).cuda())
            scaler.scale(loss / CFG['accum']).backward(); losses.append(loss.item())
            if (k + 1) % CFG['accum'] == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
        msg = f'[{job}] ep {ep} loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s)'
        if b is not None:
            P = predict(model, b); np.save(f'val_{job}_ep{ep}.npy', P)
            msg += f' val bal_acc {balanced_accuracy_score(y[b], P.argmax(1)):.4f}'
        print(msg, flush=True)
    if job == 'full':
        model.base.save_pretrained('adapter')
        torch.save({k: v for k, v in model.state_dict().items() if not k.startswith('base.')}, 'head.pt')
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
