"""E11: pseudo-labelling (self-training) on the held-out fold, which stands in for the unlabeled test set.

Per fold: fit components on train -> blend -> take held-out rows whose blended max-prob is in the top
`frac` -> add them with their argmax labels -> refit -> re-predict held-out. Labels of the held-out fold are
only used for scoring. Components available locally: stylegram LR and Qwen3-1.7B L4+19 head (tstd).

usage: python experiments/e11_pseudo.py [frac ...]
"""
import sys

import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from common import K, RESULTS, folds, load, proba_full  # noqa: E402
from e07_stylegram import featurize, svc_proba  # noqa: E402

tr = load()
y = tr.y.values
T = tr.text.values
E = np.load(RESULTS / 'emb_Qwen3-1.7B-Base.npy', mmap_mode='r')
X_emb = np.concatenate([np.asarray(E[:, 4], np.float32), np.asarray(E[:, 19], np.float32)], 1)
W = {'sty': 2.0, 'emb': 1.0}
fracs = [float(f) for f in sys.argv[1:]] or [0.3, 0.5, 0.7]


def fit_predict(Xs_a, ya, Xs_b, Xe_a, Xe_b):
    p_sty = svc_proba(Xs_a, ya, Xs_b)
    sc = StandardScaler().fit(Xe_a)
    clf = LogisticRegression(C=0.0025, max_iter=3000).fit(sc.transform(Xe_a), ya)
    p_emb = proba_full(clf, StandardScaler().fit_transform(Xe_b))
    z = W['sty'] * np.log(p_sty) + W['emb'] * np.log(np.clip(p_emb, 1e-9, 1))
    z -= z.max(1, keepdims=True)
    P = np.exp(z)
    return P / P.sum(1, keepdims=True)


res = {f: np.zeros((len(y), K)) for f in [0.0] + fracs}
for a, b in folds(tr):
    Xs_a, Xs_b = featurize(T[a], T[b])
    P0 = fit_predict(Xs_a, y[a], Xs_b, X_emb[a], X_emb[b])
    res[0.0][b] = P0
    conf = P0.max(1)
    for f in fracs:
        sel = np.argsort(-conf)[:int(f * len(b))]
        P1 = fit_predict(vstack([Xs_a, Xs_b[sel]]), np.r_[y[a], P0[sel].argmax(1)], Xs_b,
                         np.r_[X_emb[a], X_emb[b][sel]], X_emb[b])
        res[f][b] = P1
        print(f'  frac {f}: pseudo acc {np.mean(P0[sel].argmax(1) == y[b][sel]):.3f} '
              f'base {BA(y[b], P0.argmax(1)):.4f} -> {BA(y[b], P1.argmax(1)):.4f}', flush=True)
for f, P in res.items():
    print(f'frac {f}: OOF balanced acc {BA(y, P.argmax(1)):.4f}')
