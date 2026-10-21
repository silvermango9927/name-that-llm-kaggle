"""E09: hand-crafted stylometric features (length/sentence/word stats, punctuation & function-word rates)
-> LightGBM and -> LR. Weak alone, but different inductive bias for the blend."""
import re
import sys

import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from common import load, oof_proba, proba_full, report  # noqa: E402

PUNCT = list(',.;:!?()[]"\'-') + ['—', '–', '‘', '’', '“', '”', '…', '*', '/', '%']
FUNC = sorted(ENGLISH_STOP_WORDS)
SENT = re.compile(r'(?<=[.!?])\s+')
WORD = re.compile(r"[A-Za-z’']+")


def stylo(t):
    words = WORD.findall(t)
    nw, nc = max(len(words), 1), max(len(t), 1)
    sents = [x for x in SENT.split(t.strip()) if x]
    sl = np.array([len(WORD.findall(x)) for x in sents]) if sents else np.array([0])
    wl = np.array([len(x) for x in words]) if words else np.array([0])
    lw = [x.lower() for x in words]
    f = [np.log1p(nw), len(sents), sl.mean(), sl.std(), sl.max(), sl.min(), wl.mean(), wl.std(),
         (wl >= 9).mean(), len(set(lw)) / nw, sum(c.isupper() for c in t) / nc, sum(c.isdigit() for c in t) / nc,
         sum(ord(c) > 127 for c in t) / nc, t.count('\n') / nw, t.count(', ') / nw, t.count(' and ') / nw,
         sum(1 for x in words if x.endswith('ly')) / nw, sum(1 for x in words if x.endswith('ing')) / nw,
         sum(1 for x in words if x.endswith('tion')) / nw, float(t.rstrip()[-1:] in '.!?'), len(t) - len(t.rstrip())]
    f += [t.count(q) / nw for q in PUNCT]
    cnt = {}
    for x in lw:
        cnt[x] = cnt.get(x, 0) + 1
    f += [cnt.get(x, 0) / nw for x in FUNC]
    return f


if __name__ == '__main__':
    tr = load()
    y = tr.y.values
    F = np.array([stylo(t) for t in tr.text], dtype=np.float32)

    def fp_lgb(a, b):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                               random_state=0, verbose=-1, n_jobs=6).fit(F[a], y[a])
        return proba_full(m, F[b])
    report(tr, oof_proba(tr, fp_lgb, verbose=False), 'e09_stylo_lgb', notes='stylometric feats + LightGBM')

    def fp_lr(a, b):
        sc = StandardScaler().fit(F[a])
        return proba_full(LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(F[a]), y[a]), sc.transform(F[b]))
    report(tr, oof_proba(tr, fp_lr, verbose=False), 'e09_stylo_lr', notes='stylometric feats + LR')
