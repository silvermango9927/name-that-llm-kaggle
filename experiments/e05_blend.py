"""E05: log-prob blend of OOF members, weights by coordinate search; honest score via leave-fold-out weights.
Then apply topic-cluster assignment on top.

usage: python experiments/e05_blend.py name1 name2 ...   (names of results/oof_<name>.npy)
"""
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score as BA

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from assign import assign, topic_clusters  # noqa: E402
from common import RESULTS, folds, load, report  # noqa: E402


def fit_weights_nll(Ls, y, idx):
    """Non-negative weights minimising multinomial NLL of softmax(sum_j w_j * logp_j) (smooth, less overfit)."""
    from scipy.optimize import minimize
    from scipy.special import log_softmax
    S = np.stack([L[idx] for L in Ls])                      # (M, n, K)
    yy = y[idx]

    def f(w):
        z = np.tensordot(w, S, 1)
        lp = log_softmax(z, axis=1)
        p = np.exp(lp)
        g = np.einsum('mnk,nk->m', S, p) - S[:, np.arange(len(yy)), yy].sum(1)
        return -lp[np.arange(len(yy)), yy].sum() / len(yy), g / len(yy)
    r = minimize(f, np.full(len(Ls), 1.0 / len(Ls)), jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(Ls))
    return r.x


def fit_weights_grid(Ls, y, idx, grid=np.arange(0, 2.01, 0.1), rounds=3):
    w = np.ones(len(Ls))
    for _ in range(rounds):
        for j in range(len(Ls)):
            best = (-1, w[j])
            for v in grid:
                w[j] = v
                s = BA(y[idx], sum(wi * L[idx] for wi, L in zip(w, Ls)).argmax(1))
                best = max(best, (s, v))
            w[j] = best[1]
    return w


if __name__ == '__main__':
    names = [a for a in sys.argv[1:] if not a.startswith('--')]
    fit_weights = fit_weights_grid if '--grid' in sys.argv else fit_weights_nll
    tr = load()
    y = tr.y.values
    Ls = [np.log(np.clip(np.load(RESULTS / f'oof_{n}.npy'), 1e-9, 1)) for n in names]
    for n, L in zip(names, Ls):
        print(f'  {n}: {BA(y, L.argmax(1)):.4f}')
    w = fit_weights(Ls, y, np.arange(len(y)))
    blend = sum(wi * L for wi, L in zip(w, Ls))
    print('weights (in-sample)', dict(zip(names, np.round(w, 2))), f'in-sample {BA(y, blend.argmax(1)):.4f}')
    # nested: weights fit on 4 folds, applied to the 5th
    nested = np.zeros_like(blend)
    for a, b in folds(tr):
        wf = fit_weights(Ls, y, a)
        nested[b] = sum(wi * L[b] for wi, L in zip(wf, Ls))
    P = np.exp(nested - nested.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    tag = 'e05_blend_' + ('grid_' if '--grid' in sys.argv else 'nll_') + __import__('hashlib').md5('|'.join(names).encode()).hexdigest()[:6]
    print('tag', tag)
    report(tr, P, tag, notes=f'nested-CV log-prob blend of {names}; full weights {np.round(w, 2).tolist()}')
    g = topic_clusters(tr.text.values, 0.9)
    print(f'  + cluster assignment: {BA(y, assign(P, g)):.4f}')
