"""E14: does public extra data help the 18-way stylegram model?

Extra rows are added to every training fold only (never validation). Per class, extra rows share fraction `w` of that
class's original total weight, so class balance is unchanged (classes without extra data are untouched).
LR is used (LinearSVC has no sample_weight-friendly calibration issue, but LR is equally strong here and cleaner).
usage: python experiments/e14_extra_stylegram.py [w ...]
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from common import K, load, oof_proba, report  # noqa: E402
from e07_stylegram import featurize  # noqa: E402
from extra import SOURCES, V1, load_extra  # noqa: E402

POOLS = {'v1': V1, 'all': list(SOURCES), 'all-ownx': [s for s in SOURCES if s != 'own_extra']}
pool = next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('--pool=')), 'v1')
tr = load()
ex = load_extra(POOLS[pool])
mincov = int(next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('--mincov=')), 0))
if mincov:   # keep only rows whose prompt is answered by >= mincov labels (avoids topic->label shortcuts)
    import pandas as pd
    cov = pd.read_csv(__import__('pathlib').Path(__file__).resolve().parents[1] / 'results/extra_coverage.csv')
    ex = ex[ex['text'].map(dict(zip(cov.text, cov['cov']))).fillna(0) >= mincov].reset_index(drop=True)
    pool += f'_cov{mincov}'
print('extra rows', len(ex), ex.groupby('src').size().to_dict())
y, T = tr.y.values, tr.text.values

for w in [float(x) for x in sys.argv[1:] if not x.startswith('--')] or [0.25, 0.5]:
    def fp(a, b, w=w):
        ft = np.r_[T[a], ex.text.values]
        fy = np.r_[y[a], ex.y.values]
        sw = np.ones(len(fy))
        n_tr = np.bincount(y[a], minlength=K).astype(float)
        n_ex = np.bincount(ex.y.values, minlength=K).astype(float)
        for c in range(K):
            if n_ex[c] > 0:   # class total stays n_tr[c]: train rows get (1-w), extra rows share w
                sw[:len(a)][y[a] == c] = 1 - w
                sw[len(a):][ex.y.values == c] = w * n_tr[c] / n_ex[c]
        Xa, Xb = featurize(ft, T[b])
        svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(Xa, fy, sample_weight=sw)
        s = svc.decision_function(Xb)
        s = np.exp(s - s.max(1, keepdims=True))
        return s / s.sum(1, keepdims=True)
    report(tr, oof_proba(tr, fp, verbose=False), f'e14_stylegram_extra_w{w}' + ('' if pool == 'v1' else f'_{pool}'),
           notes=f'stylegram SVC + extra pool {pool} w={w}')
