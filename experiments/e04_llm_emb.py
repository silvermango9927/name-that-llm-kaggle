"""E04: frozen-LLM embedding heads. StandardScaler + strongly-regularised LR on mean-pooled hidden states.

usage: python experiments/e04_llm_emb.py <slug> [spec ...] [--C 0.005] [--tstd]
  spec: a layer index ("16") or "+"-joined layers concatenated into one head ("7+22").
        default: sweep every stored layer.
  --tstd: transductive standardisation - scale the held-out fold with its *own* mean/std (mimics test-set
          standardisation, which needs no labels).
"""
import argparse
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from common import RESULTS, load, oof_proba, proba_full, report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('slug')
ap.add_argument('specs', nargs='*')
ap.add_argument('--C', type=float, default=0.005)
ap.add_argument('--tstd', action='store_true')
args = ap.parse_args()

E = np.load(RESULTS / f'emb_{args.slug}.npy', mmap_mode='r')
lj = RESULTS / f'emb_{args.slug}_layers.json'
stored = json.load(open(lj)) if lj.exists() else list(range(E.shape[1]))
col = {L: j for j, L in enumerate(stored)}
tr = load()
y = tr.y.values
specs = args.specs or [str(L) for L in stored if L > 0]

for spec in specs:
    Ls = [int(s) for s in spec.split('+')]
    X = np.concatenate([np.asarray(E[:, col[L]], dtype=np.float32) for L in Ls], axis=1)
    C = args.C / len(Ls)

    def fp(a, b, X=X, C=C):
        sc = StandardScaler().fit(X[a])
        Xb = StandardScaler().fit_transform(X[b]) if args.tstd else sc.transform(X[b])
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[a]), y[a])
        return proba_full(clf, Xb)

    P = oof_proba(tr, fp, verbose=False)
    tag = f'e04_{args.slug}_L{spec}' + ('_tstd' if args.tstd else '')
    report(tr, P, tag, notes=f'frozen {args.slug} layers {spec} mean-pool, LR C={C}' + (' tstd' if args.tstd else ''))
