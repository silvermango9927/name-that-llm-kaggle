"""E13: post-blend family gates + Sinkhorn balancing on top of the v5 nested-OOF blend.

Gate: when the blend predicts a member of family F, rescore only F's members with
    log P_blend[F] + lam * decision_function(specialist_F)
where specialist_F is a stylegram LinearSVC trained on F's rows only (train folds, optionally + extra API passages).
Rows predicted outside F are never touched.

usage: python experiments/e13_gates.py <blend_oof_name>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.svm import LinearSVC

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from common import DATA, L2I, LABELS, RESULTS, folds, load  # noqa: E402
from e07_stylegram import featurize  # noqa: E402

FAMILIES = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ['claude', 'gpt', 'gemini', 'glm']}
LAMS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def sinkhorn(P, alpha, iters=50):
    P = P.copy()
    target = (1 - alpha) * P.sum(0) + alpha * np.full(P.shape[1], len(P) / P.shape[1])
    for _ in range(iters):
        P *= (target / (P.sum(0) + 1e-12))[None, :]
        P /= P.sum(1, keepdims=True) + 1e-12
    return P


def specialist_scores(tr, fam, extra=None):
    """OOF decision scores (n, |F|) of the within-family stylegram SVC for every train row."""
    idx = FAMILIES[fam]
    y, T = tr.y.values, tr.text.values
    D = np.zeros((len(tr), len(idx)))
    for a, b in folds(tr):
        a = a[np.isin(y[a], idx)]
        ft, fy = list(T[a]), list(y[a])
        if extra is not None:
            ft += list(extra.text)
            fy += list(extra.y)
        Xa, Xb = featurize(np.array(ft), T[b])
        svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(Xa, np.array(fy))
        assert list(svc.classes_) == list(idx)
        D[b] = svc.decision_function(Xb)
    return D


def apply_gate(P, D, idx, lam):
    pred = P.argmax(1).copy()
    hit = np.isin(pred, idx)
    s = np.log(np.clip(P[:, idx], 1e-12, 1)) + lam * D
    pred[hit] = idx[s.argmax(1)][hit]
    return pred


if __name__ == '__main__':
    tr = load()
    y = tr.y.values
    P = np.load(RESULTS / f'oof_{sys.argv[1]}.npy')
    print(f'base blend: {BA(y, P.argmax(1)):.4f}')
    for a in [0.25, 0.5, 0.75, 1.0]:
        print(f'  sinkhorn alpha {a}: {BA(y, sinkhorn(P, a).argmax(1)):.4f}')

    ext = pd.read_csv(DATA / 'extra/api_claude/ntl_extra_passages_claude.csv')
    ext['y'] = ext['model'].map(L2I)
    variants = [('claude', None), ('claude', ext), ('gpt', None), ('gemini', None), ('glm', None)]
    D = {}
    for fam, ex in variants:
        key = fam + ('+api' if ex is not None else '')
        D[key] = specialist_scores(tr, fam, ex)
        np.save(RESULTS / f'gate_{key}.npy', D[key])
        idx = FAMILIES[fam]
        m = np.isin(y, idx)
        within = BA(y[m], idx[D[key][m].argmax(1)])
        res = {lam: BA(y, apply_gate(P, D[key], idx, lam)) for lam in LAMS}
        print(f'{key:12s} specialist within-family acc {within:.3f} | gate: ' +
              ' '.join(f'{lam}:{s:.4f}' for lam, s in res.items()), flush=True)
