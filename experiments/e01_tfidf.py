"""E01-E03: reproduce the public TF-IDF baselines under prompt-grouped 5-fold CV.

E01 starter:  word(1,2) 25k + char(3,5) 35k, lowercase, LR C=2           (official starter notebook)
E02 tuned:    case-sensitive, punctuation as tokens, char(1,5), min_df=2, LR C=10  (qwen-blend's TF-IDF arm)
E03 svm:      same features as E02, LinearSVC C=1 + sigmoid calibration  (public "CV 0.37" SVM notebooks)
"""
import sys

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from common import load, oof_proba, proba_full, report  # noqa: E402


def feats_starter():
    return FeatureUnion([
        ('w', TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)),
        ('c', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=35000, sublinear_tf=True))])


def feats_tuned(maxf=30000):
    return FeatureUnion([
        ('w', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                              token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2, max_features=maxf)),
        ('c', TfidfVectorizer(analyzer='char', ngram_range=(1, 5), sublinear_tf=True, lowercase=False,
                              min_df=2, max_features=maxf))])


CONFIGS = {
    'e01_starter': (feats_starter, lambda: LogisticRegression(C=2.0, max_iter=2000)),
    'e02_tfidf_lr': (feats_tuned, lambda: LogisticRegression(C=10.0, max_iter=3000)),
    'e03_tfidf_svm': (feats_tuned, lambda: CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), method='sigmoid', cv=3)),
}

if __name__ == '__main__':
    tr = load()
    for name in (sys.argv[1:] or CONFIGS):
        mk_feats, mk_clf = CONFIGS[name]

        def fp(a, b):
            f = mk_feats()
            Xa = f.fit_transform(tr.text.values[a])
            return proba_full(mk_clf().fit(Xa, tr.y.values[a]), f.transform(tr.text.values[b]))

        print(name)
        report(tr, oof_proba(tr, fp), name, notes=__doc__.split('\n')[2 + list(CONFIGS).index(name)].strip())
