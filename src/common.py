"""Shared helpers: data loading, prompt-grouped CV, OOF bookkeeping, results logging."""
import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
L2I = {l: i for i, l in enumerate(LABELS)}
LAB = {l: l.split('-')[0].rstrip('0123456789.') for l in LABELS}  # gpt/gemini/claude/qwen/glm/kimi

N_FOLDS = 5


def load():
    tr = pd.read_csv(DATA / 'train.csv')
    tr['y'] = tr['model'].map(L2I)
    return tr


def folds(tr, n=N_FOLDS):
    """Prompt-grouped folds: every passage of a prompt lands in the same fold (test prompts are unseen)."""
    return list(GroupKFold(n_splits=n).split(tr, groups=tr['prompt_id']))


def oof_proba(tr, fit_predict, n=N_FOLDS, verbose=True):
    """fit_predict(train_idx, valid_idx) -> (len(valid_idx), K) probabilities. Returns OOF matrix."""
    P = np.zeros((len(tr), K), dtype=np.float64)
    for f, (a, b) in enumerate(folds(tr, n)):
        t0 = time.time()
        P[b] = fit_predict(a, b)
        if verbose:
            print(f'  fold {f}: bal_acc {balanced_accuracy_score(tr.y.values[b], P[b].argmax(1)):.4f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return P


def report(tr, P, name, notes=''):
    y = tr.y.values
    pred = P.argmax(1)
    ba = balanced_accuracy_score(y, pred)
    fold_scores = [balanced_accuracy_score(y[b], pred[b]) for _, b in folds(tr)]
    lab_acc = np.mean([LAB[LABELS[p]] == LAB[LABELS[t]] for p, t in zip(pred, y)])
    rec = pd.Series(recall_score(y, pred, average=None), index=LABELS).round(3)
    print(f'[{name}] OOF balanced acc {ba:.4f} (folds {np.round(fold_scores, 3)}, std {np.std(fold_scores):.3f}) '
          f'| lab acc {lab_acc:.3f}')
    RESULTS.mkdir(exist_ok=True)
    np.save(RESULTS / f'oof_{name}.npy', P)
    row = dict(name=name, oof_bal_acc=round(ba, 4), fold_std=round(float(np.std(fold_scores)), 4),
               lab_acc=round(float(lab_acc), 4), notes=notes, time=time.strftime('%Y-%m-%d %H:%M'),
               per_class_recall=rec.to_dict())
    with open(RESULTS / 'cv_log.jsonl', 'a') as fh:
        fh.write(json.dumps(row) + '\n')
    return ba, rec


def proba_full(clf, X):
    """predict_proba aligned to all K classes."""
    p = np.full((X.shape[0], K), 1e-9)
    p[:, clf.classes_] = clf.predict_proba(X)
    return p
