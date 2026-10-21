"""Daily own-data generation through the Kaggle Benchmarks Model Proxy (free daily quota).

Each run: takes the next unused prompts from bank_main.json, pushes a new task version, runs it on the target models
with per-provider spend caps, downloads the outputs and appends host-format passages to data/ours/passages.csv.
Progress (which prompts each model has done) lives in datagen/progress.json, so runs never repeat a (prompt, model).

usage: python datagen/daily.py [--prompts 40] [--models gpt-5.6-sol,gemini-3.6-flash,...] [--dry]
Observed free-quota costs (pilot, USD/passage): gpt ~0.013-0.015 (terra 0.007, luna 0.001), gemini flash ~0.013,
gemini-3.1-pro 0.072; gpt-6-astra was refused (PermissionDenied) and OpenAI calls stop after ~$0.7/day.
"""
import argparse
import glob
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(ROOT / '.venv/bin/kaggle')
BANK = json.load(open(ROOT / 'datagen/bank_main.json'))
PROGRESS = ROOT / 'datagen/progress.json'
OUT = ROOT / 'data/ours/passages.csv'
SLUG2LABEL = {'gpt-5.5-2026-04-23': 'gpt-5.5', 'gpt-5.6-luna': 'gpt-5.6-luna', 'gpt-5.6-sol': 'gpt-5.6-sol',
              'gpt-5.6-terra': 'gpt-5.6-terra', 'gpt-6-astra': 'gpt-6-astra',
              'gemini-3.1-pro-preview': 'gemini-3.1-pro-preview', 'gemini-3.6-flash': 'gemini-3.6-flash',
              'gemini-3.8-flash': 'gemini-3.8-flash', 'claude-opus-4-6-default': 'claude-opus-4-6',
              'claude-opus-4-8-default': 'claude-opus-4-8', 'claude-opus-5-default': 'claude-opus-5',
              'claude-sonnet-5-default': 'claude-sonnet-5'}
# default rotation: weakest reachable classes first (recall 0.33-0.55 in the v11 blend), cheap models get more prompts
# OpenAI, Google and Anthropic appear to have separate daily free budgets, so all three run every day.
DEFAULT = ('gpt-5.6-sol,gpt-5.6-terra,gpt-5.5-2026-04-23,gpt-5.6-luna,gemini-3.6-flash,gemini-3.8-flash,'
           'gemini-3.1-pro-preview,claude-opus-4-8-default,claude-sonnet-5-default,claude-opus-4-6-default,'
           'claude-opus-5-default')
CAP_USD = {'gpt': 0.25, 'gemini-3.1-pro-preview': 0.4, 'gemini': 0.6, 'claude': 0.5}   # per model per run
MD = re.compile(r'\*\*|(^|\n)\s*#|(^|\n)\s*[-*•]\s|(^|\n)\s*\d+\.\s')


def sh(args, timeout=1800):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', type=int, default=40)
    ap.add_argument('--models', default=DEFAULT)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    prog = json.load(open(PROGRESS)) if PROGRESS.exists() else {}
    models = a.models.split(',')
    # next prompts: the first ones not yet done by ANY of today's models (keeps prompt blocks complete across models)
    todo = [r for r in BANK['records'] if not all(r['prompt_id'] in prog.get(m, []) for m in models)][:a.prompts]
    if not todo:
        print('prompt bank exhausted for these models')
        return
    day = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
    task = f'ntl-gen-{day}'
    caps = {m: next(v for k, v in CAP_USD.items() if m.startswith(k)) for m in models}
    tmpl = (ROOT / 'datagen/task_template.py').read_text()
    ids = {r['prompt_id'] for r in todo}
    # skip list for EVERY model (the push's automatic creation run uses the proxy's default model, not one of ours)
    done = {m: sorted(ids & set(prog.get(m, []))) for m in SLUG2LABEL}
    src = tmpl.replace('__BANK__', repr({'models': list(SLUG2LABEL), 'records': todo, 'done': done})) \
              .replace('__CAP__', repr(caps)).replace('__TASK__', task)
    f = ROOT / f'datagen/tasks/{task}.py'
    f.parent.mkdir(exist_ok=True)
    f.write_text(src)
    print(f'{task}: {len(todo)} prompts ({todo[0]["prompt_id"]}..{todo[-1]["prompt_id"]}) x {len(models)} models')
    if a.dry:
        return
    print(sh([KAGGLE, 'benchmarks', 'tasks', 'push', task, '-f', str(f), '--wait', '900'], 1000)[-400:])
    run = [KAGGLE, 'benchmarks', 'tasks', 'run', task, '--wait', '3000']
    for m in models:
        run += ['-m', m]
    print(sh(run, 3200)[-600:])
    dl = ROOT / f'datagen/out/{task}'
    print(sh([KAGGLE, 'benchmarks', 'tasks', 'download', task, '-o', str(dl), '-f'], 900)[-200:])
    rows = []
    for p in glob.glob(str(dl / '**/passages_*.jsonl'), recursive=True):
        rows += [json.loads(line) for line in open(p)]
    if not rows:
        print('no outputs downloaded')
        return
    d = pd.DataFrame(rows)
    d = d[(d.status == 'ok') & d.model_slug.isin(SLUG2LABEL)].copy()
    d['text'] = d['text'].astype(str).str.strip()
    d['model'] = d['model_slug'].map(SLUG2LABEL)
    d['n_words'] = d['text'].str.split().str.len()
    d['host_format'] = d['text'].map(lambda t: '\n' not in t and not MD.search(t) and 30 <= len(t.split()) <= 300)
    d['task'] = task
    keep = ['prompt_id', 'model', 'model_seen', 'target_words', 'n_words', 'host_format', 'text', 'cost_usd',
            'timestamp', 'task', 'user_message']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=keep)
    allr = pd.concat([old, d[keep]], ignore_index=True).drop_duplicates(['prompt_id', 'model'], keep='first')
    allr.to_csv(OUT, index=False)
    for (m, pid) in d[['model_slug', 'prompt_id']].itertuples(index=False):
        prog.setdefault(m, [])
        if pid not in prog[m]:
            prog[m].append(pid)
    json.dump(prog, open(PROGRESS, 'w'), indent=0)
    s = d.groupby('model').agg(n=('text', 'size'), cost=('cost_usd', 'sum'))
    print(s.round(3).to_string())
    print(f'total own passages: {len(allr)} | by model: {allr.model.value_counts().to_dict()}')


if __name__ == '__main__':
    sys.exit(main())
