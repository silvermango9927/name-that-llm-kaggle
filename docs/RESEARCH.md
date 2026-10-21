# Name That LLM — Problem Analysis & Landscape (2026-10-15, competition day 2)

## The task in one paragraph
Given a single passage of 40–250 words, predict which of **18 frontier LLMs** wrote it (6 labs: OpenAI ×5,
Google ×4, Anthropic ×5, Qwen, ZAI ×2, Moonshot). Metric is **balanced accuracy** (mean per-class recall), so
every model counts equally regardless of how many test passages it has. Code competition: notebook ≤ 9h,
internet off, writes `submission.parquet`. ~4,000 hidden test passages; public/private splits have
different model mixes. 5 submissions/day.

## Is it multi-class classification or an ensemble-of-trees problem?
It is **single-label, 18-way multi-class classification** — that is the *problem formulation*.
"Ensemble of trees" is one possible *model family* you could use to solve it, not an alternative formulation.
In practice, gradient-boosted trees are a poor primary model here: the signal is in very high-dimensional,
sparse text features (n-grams) or dense neural embeddings, where linear models and transformers dominate.
Trees only make sense on a small set of hand-crafted stylometric features (sentence length, punctuation
rates…), which the public "stylometric fusion" notebook tried with modest results.

Ensembling **is** central, but it's ensembling of *diverse probabilistic classifiers* (log-prob blending /
stacking of TF-IDF-linear + LLM-embedding heads + fine-tuned encoders), not trees.

## Data facts that shape everything
| Fact | Consequence |
|---|---|
| train = 2,699 passages = 150 prompts × 18 models (one prompt is missing one model) | Tiny dataset: ~150 examples per class. Strong regularisation; pretrained representations matter. |
| Test prompts are **unseen** | CV **must** be `GroupKFold` on `prompt_id`, otherwise topic leaks and CV is inflated. |
| Train is perfectly balanced; test is not | Balanced accuracy protects us; no need to reweight classes. |
| Length differs only slightly per model (101–124 words mean) | Length is a weak feature by design (host randomised it). |
| Most errors are *within lab* (≈70% "right lab", ≈35% right model) | The hard part is separating siblings (e.g. gpt-5.6-luna/sol/terra, gemini flashes, claude opus versions). |

## What the public leaderboard / notebooks are doing (as of day 2)
Top public LB ≈ 0.47. Public approaches, best first:

| Approach | Notebook | Grouped CV | LB |
|---|---|---|---|
| TF-IDF LR + frozen **Qwen3-1.7B-Base** mean-pooled hidden states (layers 17, 20) → LR heads, log-prob blend | lucasboesen/qwen-blend | 0.431 | ~0.436 |
| Fine-tuned **UltraBERT-380M** encoder, 5-fold ensemble | pjmathematician/ntl-ultrabert-380m | 0.408 | — |
| LinearSVC on char(1-5)+word(1-2) TF-IDF | dariushafshar (+several clones) | 0.37 | — |
| TF-IDF + LR (case-sensitive, punctuation tokens) | qwen-blend control | ~0.35 | 0.347 |
| Official starter: TF-IDF word+char + LR | martynaplomecka/starter | ~0.31 | — |
| Stylometric features + CatBoost + SVM fusion | sharif485/stylometric-fusion | ~0.3x | — |

Key ideas circulating:
1. **Frozen LLM features beat n-grams.** A base LM's hidden states encode style; *middle/early* layers work best
   (final layers specialise in next-token prediction and lose stylistic info). Our sweep agrees: L4 > L16–19 > L28.
2. **Blending lexical + neural** is complementary (TF-IDF captures exact punctuation/typography quirks such as
   curly vs straight quotes and em-dashes; embeddings capture phrasing/semantics).
3. **Extra data** (generating more passages from the 18 models via APIs) is the most-discussed lever on the
   forum ("Will the winner be whoever gets the most extra training data?"). External data is allowed if public.
4. Someone published **reconstructed train prompts** for all 150 prompt_ids — useful for generating more data
   on matching prompt styles.
5. **Prompt-centering / topic-cluster tricks** (qwen-blend has an `ADAPT` flag, off by default). We tested the
   strongest version of this idea — see EXPERIMENTS.md E06 — and it does **not** transfer to the hidden test.

## Strategy going forward (ranked by expected value)
1. Stronger frozen-LM features: larger base models (Qwen3-4B/8B, Llama/Gemma/Mistral-class), more layers,
   better pooling, ensembling across LMs.
2. Fine-tuned classifiers (DeBERTa-v3 / ModernBERT / UltraBERT, LoRA on a 1–8B LM) — expensive on 2.7k rows
   but strongest single models in similar Kaggle authorship comps (e.g. LLM-Detect-AI-Generated-Text).
3. Perplexity-style features: per-token log-probs under several reference LMs (models' "surprise profiles").
4. Hierarchical (lab → model) heads and sibling-specific classifiers for the confusable groups.
5. External data generation (needs API spend; must be publicly shared to comply with rules).
