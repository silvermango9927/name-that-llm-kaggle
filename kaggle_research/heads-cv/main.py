# Name That LLM - CPU research kernel: CV of concatenated-layer heads on precomputed train embeddings.
# Reads emb_train.npy/layers.json from ntl-feat-* kernel outputs; writes oof_<model>_L<a+b>.npy + cv.json.
JOBS = {'qwen35-9b': [[3, 18], [2, 17], [3, 17], [2, 18], [3, 17, 18]]}   # {kernel_name_substring: [[La, Lb], ...]}

import glob, json, os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

T0 = time.time()
LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
DATA_DIR = os.path.dirname(sorted(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True) + glob.glob('/kaggle/input/**/train.csv', recursive=True))[0])
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
y = train['model'].map({l: i for i, l in enumerate(LABELS)}).values
folds = list(GroupKFold(5).split(train, groups=train['prompt_id']))
cv = {}
for kname, groups in JOBS.items():
    kdir = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/emb_train.npy', recursive=True) if kname in p))
    stored = json.load(open(os.path.join(kdir, 'layers.json')))
    E = np.load(os.path.join(kdir, 'emb_train.npy'), mmap_mode='r')
    H = E.shape[2]
    for g in groups:
        X = np.concatenate([np.asarray(E[:, stored.index(L)], dtype=np.float32) for L in g], 1)
        C = 0.005 * 2048 / H / len(g)
        P = np.zeros((len(X), len(LABELS)), dtype=np.float32)
        for a, b in folds:
            sc = StandardScaler().fit(X[a])
            clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[a]), y[a])
            P[b] = clf.predict_proba(StandardScaler().fit_transform(X[b]))
        tag = f"{kname}_L{'+'.join(map(str, g))}"
        cv[tag] = round(float(balanced_accuracy_score(y, P.argmax(1))), 4)
        np.save(f'oof_{tag}.npy', P)
        print(tag, cv[tag], f'({time.time() - T0:.0f}s)', flush=True)
json.dump(cv, open('cv.json', 'w'))
