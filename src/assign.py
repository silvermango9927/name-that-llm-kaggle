"""Prompt-group recovery + within-group assignment.

Train: every prompt was answered by all 18 models. If test is built the same way, passages that share a
prompt must come from *different* models, so we cluster passages by topic and solve a min-cost assignment
inside each cluster instead of taking a per-passage argmax.
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer


def topic_clusters(texts, thresh=0.9):
    """Average-linkage on topic TF-IDF cosine distance. ARI vs true prompt_id on train = 0.96 at 0.9."""
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2 if len(texts) > 50 else 1).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')


def assign(P, groups, min_size=2):
    """Each class used at most ceil(n/K) times per group; maximise total log-prob. Small groups keep argmax."""
    K = P.shape[1]
    L = np.log(np.clip(P, 1e-12, 1))
    pred = P.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < min_size:
            continue
        reps = -(-len(idx) // K)
        r, c = linear_sum_assignment(-np.tile(L[idx], (1, reps)))
        pred[idx[r]] = c % K
    return pred
