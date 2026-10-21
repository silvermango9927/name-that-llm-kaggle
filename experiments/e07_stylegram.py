"""E07: "stylegram" n-grams (from public 0.499 notebook): char 3-5, word 1-3, word-shape char n-grams,
function-word/punctuation skeleton n-grams, each block weighted, LinearSVC C=0.5 -> softmax of margins.
Also: E07b same features with LR.
"""
import re
import sys

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from common import K, load, oof_proba, report  # noqa: E402

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def featurize(tr_texts, te_texts):
    A, B = [], []
    for an, lo, hi, maxf, w in SPEC:
        tf = shape_text if an == 'shape' else function_text if an == 'function' else str
        actual = 'char' if an == 'shape' else 'word' if an == 'function' else an
        opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf)
        if actual == 'word':
            opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**opts)
        A.append(v.fit_transform([tf(t) for t in tr_texts]) * w)
        B.append(v.transform([tf(t) for t in te_texts]) * w)
    return hstack(A, format='csr'), hstack(B, format='csr')


def svc_proba(Xa, ya, Xb, C=0.5, T=1.0):
    svc = LinearSVC(C=C, random_state=42, max_iter=4000).fit(Xa, ya)
    s = svc.decision_function(Xb) / T
    s = np.exp(s - s.max(1, keepdims=True))
    p = np.full((Xb.shape[0], K), 1e-9)
    p[:, svc.classes_] = s / s.sum(1, keepdims=True)
    return p


if __name__ == '__main__':
    tr = load()
    y = tr.y.values
    T = tr.text.values

    def fp_svc(a, b):
        Xa, Xb = featurize(T[a], T[b])
        return svc_proba(Xa, y[a], Xb)
    report(tr, oof_proba(tr, fp_svc), 'e07_stylegram_svc', notes='stylegram 4-block tfidf, LinearSVC C=0.5')

    def fp_lr(a, b):
        Xa, Xb = featurize(T[a], T[b])
        clf = LogisticRegression(C=20, max_iter=3000).fit(Xa, y[a])
        p = np.full((len(b), K), 1e-9)
        p[:, clf.classes_] = clf.predict_proba(Xb)
        return p
    report(tr, oof_proba(tr, fp_lr), 'e07b_stylegram_lr', notes='stylegram 4-block tfidf, LR C=20')
