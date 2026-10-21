"""E08: LoRA fine-tune of a decoder LM as an 18-way sequence classifier (last-token pooling head).

usage: python experiments/e08_lora.py <hf_id> <fold|all> [epochs] [lr] [r]
Saves OOF probabilities for the fold(s) to results/lora_<slug>_f<k>.npy
"""
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from common import K, RESULTS, folds, load  # noqa: E402

model_id = sys.argv[1]
which = sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
R = int(sys.argv[5]) if len(sys.argv) > 5 else 16
BS, ACC, MAXLEN = 4, 4, 384
dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

tr = load()
tok = AutoTokenizer.from_pretrained(model_id)
tok.padding_side = 'right'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
slug = model_id.split('/')[-1]


def run_fold(f, a, b):
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=K, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    cfg = LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=0.05, task_type='SEQ_CLS',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, cfg).to(dev)
    model.gradient_checkpointing_enable() if dev == 'cuda' else None
    texts, y = tr.text.values, tr.y.values
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(a) // (BS * ACC))
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng = np.random.default_rng(f)

    def enc(idx):
        return tok([texts[i] for i in idx], return_tensors='pt', padding=True, truncation=True, max_length=MAXLEN).to(dev)

    @torch.no_grad()
    def predict(idx):
        model.eval()
        out = np.zeros((len(idx), K))
        order = np.argsort([len(texts[i]) for i in idx])
        for s in range(0, len(idx), 16):
            j = order[s:s + 16]
            with torch.autocast(dev, dtype=torch.float16 if dev != 'cpu' else torch.bfloat16):
                out[j] = model(**enc(idx[j])).logits.float().softmax(-1).cpu().numpy()
        model.train()
        return out

    t0 = time.time()
    model.train()
    for ep in range(EPOCHS):
        perm = rng.permutation(a)
        losses = []
        for s in range(0, len(perm) - BS + 1, BS):
            idx = perm[s:s + BS]
            with torch.autocast(dev, dtype=torch.float16 if dev != 'cpu' else torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**enc(idx)).logits.float(),
                                                         torch.tensor(y[idx], device=dev)) / ACC
            loss.backward()
            losses.append(loss.item() * ACC)
            if (s // BS + 1) % ACC == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
        P = predict(b)
        print(f'fold {f} ep {ep} loss {np.mean(losses):.3f} val bal_acc {balanced_accuracy_score(y[b], P.argmax(1)):.4f} '
              f'({time.time() - t0:.0f}s)', flush=True)
    np.save(RESULTS / f'lora_{slug}_f{f}.npy', P)
    return P


fl = folds(tr)
for f, (a, b) in enumerate(fl):
    if which == 'all' or int(which) == f:
        run_fold(f, a, b)
