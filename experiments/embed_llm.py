"""Cache mean-pooled hidden states of a frozen causal LM for all train passages.

usage: python experiments/embed_llm.py <hf_id> [keep_layers] [stride]
  keep_layers: load only the first N transformer blocks (saves memory; later layers are unused anyway)
  stride:      store every `stride`-th hidden state (0..keep), always including the last kept one
writes results/emb_<slug>.npy  shape (n_train, n_stored, hidden) float16, and emb_<slug>_layers.json
"""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from common import RESULTS, load  # noqa: E402

model_id = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen3-1.7B-Base'
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
cfg = AutoConfig.from_pretrained(model_id)
tcfg = getattr(cfg, 'text_config', cfg)
keep = int(sys.argv[2]) if len(sys.argv) > 2 else tcfg.num_hidden_layers
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
tcfg.num_hidden_layers = keep
if getattr(tcfg, 'layer_types', None):
    tcfg.layer_types = tcfg.layer_types[:keep]
tok = AutoTokenizer.from_pretrained(model_id)
tok.padding_side = 'right'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(model_id, config=cfg, dtype=torch.float16).to(dev).eval()

texts = list(load().text)
order = np.argsort([len(t) for t in texts])
layers = sorted(set(list(range(0, keep + 1, stride)) + [keep]))
H = tcfg.hidden_size
out = np.zeros((len(texts), len(layers), H), dtype=np.float16)
t0 = time.time()
bad = 0
with torch.no_grad():
    for b0 in range(0, len(texts), 8):
        idx = order[b0:b0 + 8]
        batch = [(tok.bos_token or '') + texts[i] for i in idx]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=512,
                  add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].unsqueeze(-1).float()
        for j, L in enumerate(layers):
            h = hs[L].float()
            bad += int(not torch.isfinite(h).all())
            out[idx, j] = np.clip(((h * m).sum(1) / m.sum(1)).cpu().numpy(), -6e4, 6e4)
        if b0 % 400 == 0:
            print(b0, f'{time.time() - t0:.0f}s', flush=True)
slug = model_id.split('/')[-1]
np.save(RESULTS / f'emb_{slug}.npy', np.nan_to_num(out))
json.dump(layers, open(RESULTS / f'emb_{slug}_layers.json', 'w'))
print('saved', out.shape, 'layers', layers, 'nonfinite batches', bad, f'{time.time() - t0:.0f}s')
