# Name That LLM - v1 TF-IDF + LR (CV 0.349)
# Self-contained Kaggle submission script (internet off). Source of truth: github repo experiments/ + src/.
ASSIGN = False       # cluster test passages by topic, then one-model-per-passage assignment inside each cluster
THRESH = 0.9              # average-linkage cosine-distance cut (ARI 0.96 vs true prompt_id on train)

import pathlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
DATA = next(p for p in [pathlib.Path('/kaggle/input/competitions/name-that-llm'),
                        pathlib.Path('/kaggle/input/name-that-llm'), pathlib.Path('data')]
            if (p / 'train.csv').exists())
train = pd.read_csv(DATA / 'train.csv')
test = pd.read_csv(DATA / 'test.csv')
test['text'] = test['text'].fillna('').astype(str)
y = train['model'].map({l: i for i, l in enumerate(LABELS)}).values
print('train', train.shape, 'test', test.shape, flush=True)

feats = FeatureUnion([
    ('w', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                          token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2, max_features=30000)),
    ('c', TfidfVectorizer(analyzer='char', ngram_range=(1, 5), sublinear_tf=True, lowercase=False,
                          min_df=2, max_features=30000))])
clf = LogisticRegression(C=10.0, max_iter=3000).fit(feats.fit_transform(train['text']), y)
P = np.full((len(test), K), 1e-9)
P[:, clf.classes_] = clf.predict_proba(feats.transform(test['text']))
pred = P.argmax(1)


def topic_clusters(texts, thresh):
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2 if len(texts) > 50 else 1).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')


if ASSIGN:
    g = topic_clusters(list(test['text']), THRESH)
    L = np.log(np.clip(P, 1e-12, 1))
    pred = pred.copy()
    for c in np.unique(g):
        idx = np.where(g == c)[0]
        if len(idx) < 2:
            continue
        reps = -(-len(idx) // K)
        r, col = linear_sum_assignment(-np.tile(L[idx], (1, reps)))
        pred[idx[r]] = col % K
    sizes = np.bincount(g)[1:]
    print('clusters', len(sizes), 'size dist', pd.Series(sizes).describe().round(1).to_dict(), flush=True)

sub = pd.DataFrame({'id': test['id'], 'model': [LABELS[i] for i in pred]})
assert sub['id'].is_unique and sub['model'].isin(LABELS).all() and len(sub) == len(test)
sub.to_parquet('submission.parquet', index=False)
print(sub['model'].value_counts().to_string())
