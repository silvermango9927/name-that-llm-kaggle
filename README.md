# name-that-llm-kaggle
Kaggle **Name That LLM** — 18-way authorship attribution of short LLM passages (balanced accuracy).

- `docs/RESEARCH.md` — problem analysis, what the public notebooks do, strategy
- `docs/EXPERIMENTS.md` — every experiment: setup, CV, LB, why it worked / didn't
- `src/` — shared CV harness (`common.py`), topic-cluster assignment (`assign.py`)
- `experiments/` — local experiment scripts (write OOF preds + `results/cv_log.jsonl`)
- `kaggle_notebooks/` — exact scripts submitted to Kaggle (push with `kaggle kernels push -p <dir>`)

Setup: `python3 -m venv .venv && .venv/bin/pip install kaggle pandas pyarrow scikit-learn torch transformers`,
then `kaggle competitions download name-that-llm -p data && unzip data/*.zip -d data`.

| Best | CV | Public LB |
|---|---|---|
| v5 stylegram + Qwen3-8B/Qwen3.5-9B heads + Qwen3-8B LoRA | 0.519 | 0.503 |
