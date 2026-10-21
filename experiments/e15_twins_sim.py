"""E15: additive prompt-centered "twin" evidence, simulated in the test regime (few passages per prompt).

Per fold: plain head (StandardScaler+LR) on Qwen3-1.7B L4+L19; twin head on prompt-centered train features.
Evaluation pool = held-out fold subsampled to m passages per prompt (m = 1,2,3,5). Within each prompt group of the
pool (oracle grouping = upper bound), twin log-probs on group-centered features are ADDED to the plain log-probs.
Repeated over several random subsamples.
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from common import RESULTS, folds, load, proba_full  # noqa: E402

tr = load()
y, g = tr.y.values, tr.prompt_id.values
E = np.load(RESULTS / 'emb_Qwen3-1.7B-Base.npy', mmap_mode='r')
X = np.concatenate([np.asarray(E[:, 4], np.float32), np.asarray(E[:, 19], np.float32)], 1)


def center(X, groups):
    X = X.copy()
    for k in np.unique(groups):
        i = groups == k
        X[i] -= X[i].mean(0)
    return X


rng = np.random.default_rng(0)
res = {m: {w: [] for w in [0, 0.5, 1.0]} for m in [1, 2, 3, 5]}
for a, b in folds(tr):
    sc = StandardScaler().fit(X[a])
    plain = LogisticRegression(C=0.00125, max_iter=3000).fit(sc.transform(X[a]), y[a])
    Xc = center(X[a], g[a])
    scc = StandardScaler().fit(Xc)
    twin = LogisticRegression(C=0.00125, max_iter=3000).fit(scc.transform(Xc), y[a])
    for m in res:
        for rep in range(4):
            pool = np.concatenate([rng.choice(np.where(g == k)[0], m, replace=False) for k in np.unique(g[b])])
            Lp = np.log(proba_full(plain, StandardScaler().fit_transform(X[pool])))
            Lt = np.zeros_like(Lp)
            if m >= 2:
                Lt = np.log(proba_full(twin, scc.transform(center(X[pool], g[pool]))))
            for w in res[m]:
                res[m][w].append(BA(y[pool], (Lp + w * Lt).argmax(1)))
for m, d in res.items():
    print(f'{m} passages/prompt: ' + '  '.join(f'twin w={w}: {np.mean(v):.4f}' for w, v in d.items()))
