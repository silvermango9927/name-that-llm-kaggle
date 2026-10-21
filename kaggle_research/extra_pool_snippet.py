# --- curated host-format extra pool (mirrors src/extra.py; sources = public CC-BY/CC-BY-SA Kaggle datasets) ---
EXTRA_SOURCES = [  # (file name, label column)
    ('ntl_extra_passages_claude.csv', 'model'), ('ntl_generations.csv', 'label'), ('accepted.csv', 'model'),
    ('generated_v1.csv', 'label'), ('passages_clean.csv', 'model'), ('generations.csv', 'label'),
    ('ntl_alp_generations.csv', 'label'), ('ntl_proxy_generations.csv', 'label'), ('gen_passages.csv', 'model'),
    ('ntl_gpt_proxy_passages.csv', 'label'), ('passages.csv', 'label')]
EXTRA_DATASETS = ['seyitkaangunes/ntl-extra-api-passages', 'abdullah1afify/ntl-generations',
                  'keshavatearth/ntl-gpt-new-prompts-v1', 'prvsiyan/ntl-own-prompt-generations-v1',
                  'prvsiyan/prvsiyans-extended-name-that-llm-dataset', 'lxdlxdlxd/ntl-synth-generations-v1',
                  'alperen5252525/ntl-vertex-gemini-proxy-gpt-generations', 'dophuchung/ntl-proxy-generations',
                  'lucasboesen/ntl-free-proxy-generations', 'kaplinskiy/ntl-gpt-proxy-passages',
                  'haileycheng/name-that-llm-same-prompt-passages']


def load_extra_pool(labels):
    import glob, re
    import pandas as pd
    md = re.compile(r'\*\*|(^|\n)\s*#|(^|\n)\s*[-*•]\s|(^|\n)\s*\d+\.\s')
    parts = []
    for fname, col in EXTRA_SOURCES:
        for hit in glob.glob('/kaggle/input/**/' + fname, recursive=True):
            d = pd.read_csv(hit, low_memory=False)
            if 'text' not in d or col not in d or not d[col].isin(labels).mean() > 0.5:
                continue   # e.g. a same-named file in an unrelated input
            parts.append(pd.DataFrame({'text': d['text'].astype(str).str.strip(), 'model': d[col], 'src': fname}))
    ex = pd.concat(parts, ignore_index=True)
    ex = ex[ex['model'].isin(labels)]
    ok = ex['text'].map(lambda t: '\n' not in t and not md.search(t) and 30 <= len(t.split()) <= 300)
    ex = ex[ok].drop_duplicates('text').reset_index(drop=True)
    print('extra pool', len(ex), ex['model'].value_counts().to_dict(), flush=True)
    return ex
# --- end extra pool ---
