"""Public extra generations (CC BY / CC BY-SA Kaggle datasets) unified to columns: text, model, y, src, prompt_id.

Only host-format rows are kept (single paragraph, no markdown, 30-300 words): the host's passages have 0% line breaks and
0% markdown, and mismatched external data was shown to *hurt* (forum thread 749447).
"""
import re

import pandas as pd

from common import DATA, L2I

SOURCES = {  # name: (path under data/extra, label column, prompt column or None)  -- Kaggle dataset in comment
    'api_claude': ('api_claude/ntl_extra_passages_claude.csv', 'model', 'prompt_id'),          # seyitkaangunes/ntl-extra-api-passages
    'gen_gpt_gem': ('ntl-generations/ntl_generations.csv', 'label', 'prompt_id'),              # abdullah1afify/ntl-generations
    'gpt_new': ('ntl-gpt-new-prompts-v1/accepted.csv', 'model', 'prompt_id'),                  # keshavatearth/ntl-gpt-new-prompts-v1
    'own_prompt': ('ntl-own-prompt-generations-v1/generated_v1.csv', 'label', 'prompt_id'),    # prvsiyan/ntl-own-prompt-generations-v1
    'prv_ext': ('prvsiyans-extended-name-that-llm-dataset/passages_clean.csv', 'model', 'prompt_id'),  # prvsiyan/prvsiyans-extended-name-that-llm-dataset
    'synth13': ('ntl-synth-generations-v1/generations.csv', 'label', 'pid'),                   # lxdlxdlxd/ntl-synth-generations-v1
    'own_extra': ('ntl-own-extra-v1/ntl_own_extra_v1.csv', 'model', 'group'),                  # michaelsemenoff/ntl-own-extra-v1
    'vertex': ('ntl-vertex-gemini-proxy-gpt-generations/ntl_alp_generations.csv', 'label', 'prompt_id'),  # alperen5252525/ntl-vertex-gemini-proxy-gpt-generations
    'proxy': ('ntl-proxy-generations/ntl_proxy_generations.csv', 'label', 'prompt_id'),        # dophuchung/ntl-proxy-generations
    'free_proxy': ('ntl-free-proxy-generations/gen_passages.csv', 'model', 'prompt_id'),       # lucasboesen/ntl-free-proxy-generations
    'gpt_proxy': ('ntl-gpt-proxy-passages/ntl_gpt_proxy_passages.csv', 'label', 'prompt_id'),  # kaplinskiy/ntl-gpt-proxy-passages
    'same_prompt': ('name-that-llm-same-prompt-passages/passages.csv', 'label', 'prompt_id'),  # haileycheng/name-that-llm-same-prompt-passages
}
V1 = ['api_claude', 'gen_gpt_gem', 'gpt_new', 'own_prompt']   # the set used up to v10
MD = re.compile(r'\*\*|(^|\n)\s*#|(^|\n)\s*[-*•]\s|(^|\n)\s*\d+\.\s')


def host_format(t):
    n = len(t.split())
    return '\n' not in t.strip() and not MD.search(t) and 30 <= n <= 300


def load_extra(sources=None, filter_format=True):
    out = []
    for name in (sources or SOURCES):
        path, col, pcol = SOURCES[name]
        df = pd.read_csv(DATA / 'extra' / path, low_memory=False)
        df = pd.DataFrame({'text': df['text'].astype(str).str.strip(), 'model': df[col],
                           'prompt_id': name + '_' + (df[pcol].astype(str) if pcol in df else df.index.astype(str)),
                           'src': name})
        out.append(df[df['model'].isin(L2I)])
    ex = pd.concat(out, ignore_index=True)
    if filter_format:
        ex = ex[ex['text'].map(host_format)]
    ex = ex.drop_duplicates('text')
    ex['y'] = ex['model'].map(L2I)
    return ex.reset_index(drop=True)
