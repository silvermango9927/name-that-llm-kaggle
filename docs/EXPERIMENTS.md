# Experiment Log

All CV = prompt-grouped 5-fold (`GroupKFold` on `prompt_id`), metric = balanced accuracy, out-of-fold (OOF).
Raw numbers: `results/cv_log.jsonl`. Kaggle notebooks: `kaggle_notebooks/<name>/main.py` (frozen snapshot of what was submitted).

## Leaderboard summary
| Ver | Notebook | Idea | CV | Public LB | Verdict |
|---|---|---|---|---|---|
| v1 | ntl-v1-tfidf-lr | TF-IDF (word 1-2 + char 1-5, case-sensitive) + LR C=10 | 0.349 | 0.347 | baseline; **CV≈LB confirmed** |
| v2 | ntl-v2-tfidf-lr-assign | v1 + topic clustering + per-cluster Hungarian assignment | 0.452 | 0.346 | ❌ does not transfer |
| v3 | ntl-v3-tfidf-qwen-blend | TF-IDF LR + Qwen3-1.7B-Base L4/L16/L19 heads, log-prob blend | 0.444 (nested) | **0.430** | ✅ reproduces best public notebook (0.436) |
| v4 | ntl-v4-stylegram-q8b | stylegram SVC + Qwen3-8B L2/4/8/20/22 heads (tstd), NLL blend | 0.483 (nested) | **0.475** | ✅ +4.5 over v3 |
| v5 | ntl-v5-blend-lora | stylegram + Qwen3-8B L2+22 + Qwen3.5-9B L17/L18 heads + **Qwen3-8B LoRA** | 0.519 (nested) | **0.503** | ✅ +2.8 over v4 |
| v6 | ntl-v6-extra-gate | v5 + extra-data stylegram + Claude gate + Sinkhorn 0.25 | ~0.523 | **0.514** | ✅ +1.1 (CV said +0.5) |
| v7 | ntl-v7-twins | v6 + additive prompt-centered twins on frozen heads (clusters 2–18, w 0.5) | n/a (sim) | **0.530** | ✅ +1.6 |
| v8 | ntl-v8-twins1 | v7 with twin weight 1.0 | n/a | 0.525 | ❌ over-corrects; w 0.5 better |
| v10 | ntl-v10-lora-extra | v7 + frozen heads w/ extra data + extra-data LoRA (4-model avg: full+fold0/1/2) | 0.549 (folds 0–2) | **0.536** | ✅ best, but +0.6 vs ~+2 expected |
| v9 | ntl-v9-extra-heads | v7 + frozen heads w/ extra data (ablation for v10) | 0.535 | 0.531 | ≈ v7 → v10's gain is the LoRA |
| v11 | ntl-v11-pool | v10 + stylegram on the full curated pool | — | **0.538** | ✅ +0.2 |
| v12 | ntl-v12-pool-heads | v11 + frozen heads trained on the big pool | 0.556 (f0–2) | 0.537 | ≈ (CV +0.9 did not transfer) |
| v13 | ntl-v13-lora6 | v12 + 6-adapter LoRA average (small-pool ×4 + big-pool ×2) | — | 0.535 | ≈ (within noise; no gain from LoRA ensembling) |

---

## E01–E03 · Reproduce public TF-IDF baselines (`experiments/e01_tfidf.py`)
| Exp | Setup | CV | Lab acc |
|---|---|---|---|
| E01 | Official starter: lowercase word(1,2) 25k + char(3,5) 35k, LR C=2 | 0.314 | 0.664 |
| E02 | Case-sensitive, punctuation kept as word tokens, char(1,5), min_df=2, LR C=10 | **0.349** | 0.704 |
| E03 | E02 features + LinearSVC C=1 (sigmoid-calibrated) | 0.323 | 0.672 |

**Why E02 > E01:** model fingerprints live in *typography*: casing, curly vs straight quotes, em-dashes,
semicolons. Lowercasing and the default token pattern (which drops punctuation) throw that away; char 1-2 grams
keep single-punctuation-mark frequencies. Higher C (less regularisation) helps because the useful features are
many weak, rare n-grams.
**Why the SVM underperformed** the public "0.37" claim: we used a 3-fold internal calibration, which shrinks
each calibrator's training set; the public notebooks use raw `LinearSVC.predict` and different feature caps.
Not worth chasing: it's a linear model on the same features as E02.

## E06 · Topic-cluster + within-prompt assignment (`src/assign.py`) — ❌ negative result
**Hypothesis.** In train, every prompt was answered by all 18 models. If test is built the same way, passages
sharing a prompt must come from *different* models. Recover prompt groups by clustering on topic words, then
solve a min-cost assignment (Hungarian, each model ≤ ⌈n/18⌉ times per cluster) instead of per-passage argmax.

**Evidence in CV (very strong):**
- Oracle prompt groups: 0.349 → **0.461**.
- Blind average-linkage clustering of topic TF-IDF (cosine cut 0.9) recovers prompts with ARI **0.96**;
  cluster+assign: 0.349 → **0.452**; on the Qwen blend: 0.444 → **0.641**.
- Still +4–5 pts when 40–70% of passages are randomly dropped (simulating unequal per-model counts).

**Leaderboard:** v1 0.347 vs v2 **0.346** — zero gain.

**Why it failed.** The gain can only appear if test clusters contain several passages from *distinct* models on
one prompt. Zero change means the hidden test essentially lacks that structure: most likely ~one passage per
prompt (≈4,000 distinct prompts), so clusters are singletons and the assignment is a no-op. This also squares
with the host's statement that per-model counts differ and public/private mixes differ.
**Lesson:** a CV design that mirrors train's structure can validate a trick that the test set simply doesn't
support. We cannot see the scoring run's logs, so the LB is the only arbiter of test-structure assumptions.
Cost: 1 submission. Idea retired.

## E04 · Frozen Qwen3-1.7B-Base embeddings (`experiments/embed_llm.py`, `e04_llm_emb.py`)
Mean-pooled hidden state per layer (BOS prepended, max 512 tokens) → StandardScaler → LR C=0.005.
Extraction on M4 Pro MPS: 276 s for all 29 layers.

| Layer | 4 | 7 | 10 | 13 | 16 | 19 | 22 | 25 | 28 (last) |
|---|---|---|---|---|---|---|---|---|---|
| CV | **0.389** | 0.364 | 0.372 | 0.370 | 0.379 | 0.380 | 0.333 | 0.307 | 0.317 |

**Every layer from 4 to 19 beats the best TF-IDF model.** Early layers (≈ L4) encode surface form (token
identity, punctuation, local syntax) — exactly the stylistic signal we want; the final layers are specialised
for next-token prediction and become more abstract/semantic, so they lose fingerprint information.
Very strong L2 (C=0.005) is necessary: 2048 dims vs ~2,160 training rows per fold.

## E05 · Log-prob blend (`experiments/e05_blend.py`)
TF-IDF LR + Qwen L4 + L16 + L19, weights by coordinate search on OOF log-probs.
Full-data weights {tfidf 1.8, L4 0.4, L16 0.7, L19 0.5}; in-sample 0.454, **nested CV 0.444** (weights
re-fit on 4 folds, scored on the 5th — the honest number). Lab accuracy rises to 0.763.
**Why it works:** the members make different errors — TF-IDF sees exact characters, the LM sees
contextualised phrasing — so summing log-probs (a product of experts) cancels uncorrelated mistakes.
Note the blend's highest weight goes to TF-IDF despite it being the weakest member alone: it's the most
*different* from the three Qwen layers, which are correlated with each other.

**LB 0.430 vs nested CV 0.444.** The 1.4-pt gap is within expected noise: the public LB is a subset of ~4k
passages with a *different, unbalanced* model mix, and the blend weights were fit on all OOF data (the nested
number is the fair one). CV↔LB tracking remains good enough to iterate locally.

---
## Status after day 1 of work (2026-10-15)
- Best public LB: **0.430** (v3). Top of leaderboard: 0.473.
- Retired: topic-cluster assignment (E06), calibrated LinearSVC (E03).
- Next candidates (see RESEARCH.md "Strategy"): larger/multiple frozen LMs + more layers; fine-tuned encoder
  (DeBERTa/ModernBERT/UltraBERT) or LoRA LM classifier; per-token log-prob ("perplexity profile") features;
  hierarchical lab→model heads for the confusable sibling groups.

---
# Day 2 (2026-10-16): push toward the 0.54 frontier

## What changed on the leaderboard overnight
Top LB 0.544; best public notebooks 0.52–0.53. Public stacks combine: style n-grams, frozen hidden states of
8–14B LMs (Qwen3-8B/14B, Qwen3.5-9B, Gemma-4-12B), a **LoRA-fine-tuned Qwen3-8B** (largest blend weight), and
UltraBERT. Their "prompt-centering"/joint-decoding tricks report **+0.000 on LB**, matching our E06 finding. One
author simulated it explicitly: gain is +0.000 at 1–2 passages/prompt, which is evidently the test regime.

## Infra note — network-bound, so compute moved to Kaggle
Local bandwidth today ≈17 KB/s (HF and Cloudflare alike), so downloading 16–28 GB checkpoints is impossible.
New workflow: **research kernels on Kaggle GPUs** (`kaggle_research/`) extract embeddings and run the CV
in-kernel; we download only small OOF files (`--file-pattern`) and blend locally. Submission kernels reuse the
research kernels' train embeddings via `kernel_sources` and only embed the test set.
Gotchas hit: (1) max 2 concurrent GPU sessions; (2) once a kernel attaches other sources, the competition data
is no longer guaranteed at `/kaggle/input/competitions/name-that-llm/` → locate `train.csv` by glob.

## E07 · "stylegram" n-grams — ✅ 0.349 → 0.385
4 TF-IDF views, each block-weighted: char 3-5 (250k), word 1-3 incl. punctuation (180k), **word-shape** text
(letters→x/X, keeps punctuation/digits; char 2-6), **function-word skeleton** (content words→`CONTENT`; word 1-3).
LinearSVC C=0.5, softmax of margins. LR on the same features: 0.385 too.
**Why it works:** the shape and skeleton views remove topic entirely and leave pure syntax/punctuation rhythm,
which is what generalises across unseen prompts. Large vocabularies help because fingerprints are rare n-grams.

## E04b · Transductive standardisation (tstd) — ✅ +0.7 pts per head
Standardise the held-out/test features with *their own* mean/std instead of the training scaler (no labels used).
Qwen-1.7B L4: 0.389 → 0.396. **Why:** removes a global shift between train and test passage distributions
(different prompts/topics shift every embedding similarly); the classifier sees style deviations only.

## E04c · Concatenated layers — ✅ single head 0.396 → 0.422
One LR head on [early ‖ middle] layers (Qwen-1.7B L4+L19, C halved) beats either layer alone by ~3 pts and beats
averaging two separate heads. Early layers = surface form, middle = phrasing; the joint head can use interactions.

## E09 · Hand-crafted stylometry + LightGBM — ❌ for the blend
Alone 0.299 (LGB) / 0.275 (LR); blending it in lowered nested CV (0.4575 → 0.4538). Everything these ~200 features
encode (punctuation rates, sentence length, function-word frequencies) is already captured — at much higher
resolution — by the stylegram views. This is also the one place tree models were tried; they lose here.

## E05b · Blend weights by log-loss (L-BFGS) instead of grid search on balanced accuracy — ✅
Same members: grid 0.4575 → NLL **0.4612** nested; in-sample vs nested gap shrinks from 0.5 to 0.1 pts.
**Why:** balanced accuracy is a step function of the weights (noisy, overfits fold idiosyncrasies);
log-loss is smooth and uses the full probability information, so the weights generalise.

## E10 · Frozen big-LM layer sweeps on Kaggle T4×2 (`kaggle_research/feat_kernel.py`)
| Model | best layers (OOF, tstd) | notes |
|---|---|---|
| Qwen3-1.7B-Base | L4 0.396, L19 0.391 | local |
| **Qwen3-8B-Base** | **L8 0.425**, L22 0.420, L4 0.419, L6 0.418, L2 0.416 | U-shape: early & late-middle good, L12–18 dip |
| Gemma-4-12B | L26 0.425, L14 0.418, L16 0.415, L2 0.386 | best layers are *middle*, unlike Qwen |

Blends (nested, NLL weights):
- stylegram + TF-IDF + Qwen-1.7B heads: 0.461
- + Qwen3-8B L2/4/8/20/22: **0.482**; dropping Qwen-1.7B & TF-IDF: **0.483** (they're redundant given 8B)
- + Gemma-4-12B heads: 0.482 — no gain. Frozen-feature returns are saturating → next lever is fine-tuning.

## E08 · LoRA fine-tuning
Local MPS attempt with Qwen3-1.7B: too slow (no epoch in 30 min) → moved to Kaggle (`kaggle_research/lora-train`):
Qwen3-8B first 24 blocks, LoRA r16/α32 on all 7 projections, fp16 frozen base + fp32 trainables, head on
LayerNorm'd mean-pools of layers 4/8/12/16/20/24 (BOS excluded), LR 1e-4 adapters / 1e-3 head, eff. batch 16,
3 epochs, one fold per T4 (~20 min/epoch on a T4).

| fold | ep0 | ep1 | ep2 |
|---|---|---|---|
| 0 | 0.270 | 0.452 | **0.472** |
| 1 | 0.357 | 0.452 | **0.480** |

**Strongest single component so far** (best frozen head 0.467, stylegram 0.385), still improving at epoch 3.
Blend contribution measured 2-fold-nested on folds 0–1: frozen blend 0.467 → **0.494 with LoRA** (+2.8). LoRA's weight
(0.6) displaces most frozen heads — fine-tuning learns what the frozen probes saw, plus more.
**Why fine-tuning beats frozen probes:** the frozen hidden states are optimised for next-token prediction; the
discriminative signal is a small subspace. LoRA rotates the representation so that subspace becomes dominant,
across all 24 layers jointly, instead of a linear probe on one fixed layer.
Bugs hit: peft rejects Kaggle's torchao 0.10 even with no quantisation → uninstall it (offline) before import.

## E10b · More frozen heads
- Qwen3-8B concatenated heads (CPU kernel `heads-cv`, no GPU quota): **L2+22 0.467**, L2+8+22 0.465, L4+22 0.462 —
  best single heads yet, but the blend stays 0.483: same information as the separate layer heads.
- Qwen3.5-9B (hybrid linear-attention) layers: L18 0.423, L17 0.423, L3 0.422 (same early+mid pattern as Qwen3).
  Adding to the blend: 0.483 → **0.489**.
- Stylegram variants (C 0.25/1, char 1-5 / 2-6, function 1-4, shape 2-7): all 0.376–0.383 vs 0.385 → already tuned.

## E11 · Pseudo-labelling on the held-out fold — ❌ in CV (inconclusive for test)
Top-confidence 30/50/70% of held-out rows added with predicted labels, refit: 0.439 → 0.374 / 0.378 / 0.403, even
though pseudo labels were 72/60/53% correct. **Why:** the held-out fold contains all 18 passages of each prompt;
labelling one passage on prompt P as model A teaches the refit "topic P → A" and pulls the other 17 passages on P
toward A (topic leakage in reverse). The hidden test has ~1 passage/prompt, so CV can't fairly judge this; not
worth an LB probe yet. Parked.

## E08 (complete) · Qwen3-8B LoRA, all 5 folds + full-data model
| fold | 0 | 1 | 2 | 3 | 4 | OOF |
|---|---|---|---|---|---|---|
| ep2 bal acc | 0.472 | 0.480 | 0.520 | 0.485 | 0.497 | **0.491** |

~20 min/epoch/T4; 6 jobs on 2 T4s in two kernel runs (≈2 h each). Full-data model saved (adapter + head) for inference.

## E12 · Gemma-4-12B LoRA (20 blocks, folds 0–1) — ❌
0.357 / 0.411 after 3 epochs (Qwen LoRA: 0.472 / 0.480), loss still high (1.4 vs 0.6) → underfit. Likely causes: only
20 of 48 blocks kept (Gemma's useful frozen layers were 14–26, so we cut into them), and Gemma's much larger
vocabulary/embedding eats memory without helping. Not pursued; no full model trained.

## v5 · Final blend (nested CV 0.519)
NLL weights: stylegram 2.19, LoRA 0.47, Qwen3.5 L18 0.17, Qwen3.5 L17 0.16, Qwen3-8B L2+22 0.07. Zero-weight
heads (Qwen3-8B L4+22, L8, Qwen3.5 L2) dropped. Frozen-only 0.489 → +LoRA **0.519**.
Generated with `kaggle_notebooks/make_sub.py` (weights fit on OOF, kernel metadata assembled automatically).

---
# Day 3 (2026-10-17): extra data, gates, twins

## Survey: what the 0.55 public notebooks add (zbigniewkirchner/ntl-v6, lucasboesen/lora-x-adaptive-centering, …)
Same backbone as ours (stylegram + frozen LM heads + Qwen3-8B LoRA24) plus: a second LoRA seed, frozen DeBERTa/ELECTRA
encoders, a learned token-attention head, UltraBERT, **additive prompt-centered "twins"**, Sinkhorn balancing, and a
**Claude gate** (Claude-only SVC + LoRA Claude head trained with ~5k public Claude API passages).
New public **extra-data** datasets (all CC BY / CC BY-SA): Claude 4,998; GPT ~1,480; Gemini ~490 (no 3.7-flash);
**none for GLM, Qwen, Kimi**. Loader: `src/extra.py`.

## E13 · Post-blend family gates + Sinkhorn (on v5 OOF, 0.5186)
| | best CV |
|---|---|
| Sinkhorn α 0.25 | 0.5197 (+0.001) |
| Claude gate, train data only | ≤ 0.5186 (hurts at λ ≥ 1) |
| **Claude gate + 5k API passages** | **0.5227 (+0.004)** at λ 1.5 |
| GPT gate / Gemini gate (train only) | 0.5197 / 0.5189 |

**Lesson:** the gate *structure* adds nothing — a within-family specialist trained on the same rows only re-learns
what the 18-way models know. All of the gain is the **extra data**.

## E14 · Extra data in the 18-way stylegram — ✅
Extra rows only in training folds; per class, extra rows share fraction w of the class's original weight (balance kept).
Stylegram 0.385 → **0.398** (w 0.25 and 0.5 alike), lab acc 0.704 → 0.730. In the v5 blend: 0.5186 → **0.5223**;
Claude gate on top then only +0.001 (redundant with the extra-data stylegram).

## Error anatomy of the 0.522 blend
47.8% errors = 20.5% wrong lab + 27.3% right lab/wrong model. Worst recall: **glm-5.3 0.21, glm-5.3-flash 0.23,
claude-fable-5 0.29**, gpt-5.6-sol 0.33, gemini-3.7-flash 0.39. Top confusions: gemini-3.7↔3.8-flash (89),
gemini-3.6↔3.7 (59), **claude-fable-5↔glm-5.3-flash (53)**, glm-5.3↔glm-5.3-flash (49), gpt-5.6-sol↔terra (48),
**claude-fable-5↔glm-5.3 (45)**. The weakest classes are exactly those with no public extra data, and
claude-fable-5 is confused with the *GLMs* (cross-lab), not its siblings — consistent with GLM being tuned on
Claude-like outputs. → own-data generation for GLM/Qwen/Kimi/gemini-3.7 is the highest-ceiling lever.

## E15 · Additive prompt-centered twins, simulated in the test regime (oracle groups)
Pool = held-out fold subsampled to m passages/prompt; twin head trained on prompt-centered train features; its log-probs
are *added* for grouped rows. Qwen-1.7B L4+19 head:
| m | plain | + twin |
|---|---|---|
| 1 | 0.369 | 0.369 |
| 2 | 0.422 | **0.450** |
| 3 | 0.411 | **0.488** |
| 5 | 0.427 | **0.506** |
Big *if* the hidden test repeats prompts. v2 (hard assignment) said ~no, but additive evidence is far gentler → LB probe v7.

## LB results of day 3
- **v6 0.514 (+0.011):** extra data transfers to the LB *better* than CV predicts (+0.005). Plausible reason: CV folds
  share the train prompts' generation setup, so extra data's main benefit — more prompt/topic diversity per model — is
  undervalued by CV but matters for unseen test prompts.
- **v7 0.530 (+0.016): twins work on the hidden test.** This revises E06: the test *does* contain passages that share a
  prompt, just not in the "every model exactly once" form — forcing distinct labels (v2) gained nothing, while
  subtracting the cluster mean (removing topic, keeping style) and *adding* that evidence helps.

- **v8 0.525 (twin w 1.0) < v7 0.530 (w 0.5):** at full weight, cluster-centered evidence over-rides the plain heads on
  rows whose clusters are mixed (different prompts on similar topics). Keep w ≈ 0.5 (try 0.3 later).
  (A network hang made the CLI submit v8 twice — one slot wasted. All CLI calls now wrapped in `timeout`.)

## E16 · Extra data for the frozen LM heads (`kaggle_research/extra-heads`) — ✅
Embedded the 7,051 extra passages with each model (same extraction code); extra rows only in training folds with
class-balanced weights. Train scaler kept train-only.
| head | w 0 | w 0.3 | w 0.5 |
|---|---|---|---|
| Qwen3.5-9B L18 | 0.423 | **0.453** | 0.431 |
| Qwen3.5-9B L17 | 0.423 | **0.444** | 0.438 |
| Qwen3-8B L2+22 | 0.467 | **0.478** | 0.463 |
Blend (extra stylegram + extra heads + LoRA): 0.5223 → **0.5349**. w 0.5 is worse than 0.3: the extra data's
generation setup (API defaults, different prompts) differs from the host's, so it should inform, not dominate.

## E17 · LoRA trained with extra data (`kaggle_research/lora-train-extra`) — ✅ biggest single gain
Extra passages capped at 300/class (~2,900 rows), class-balanced loss weights (w 0.5), same recipe, 3 epochs
(~45 min/epoch/T4 with the larger set). Fold 0: 0.472 → **0.480**; fold 1: 0.480 → **0.506**.
Blend on folds 0–1 (2-fold nested): old LoRA 0.5176 → **extra-data LoRA 0.5472** (+3.0); both LoRAs 0.5481.
**Why so large:** the LoRA is the most data-hungry member (millions of trainable params, 120 rows/class/fold);
more prompts per model teach it what is *style* rather than *topic*. Full-data model + fold 2 now training
(`ntl-lora-q8b-extra2`); fold adapters are saved for fold-ensemble inference.

## E17b · Extra-data LoRA, folds 0–2 + full model (`ntl-lora-q8b-extra`, `ntl-lora-q8b-extra2`)
Fold 2: 0.520 → **0.526**. Folds 0–2 OOF: old LoRA 0.491 → **extra-data LoRA 0.504**.
3-fold nested blend on folds 0–2: v9 set (old LoRA) 0.530 → **v10 set (extra LoRA) 0.549** (+1.9).
v10 averages 4 LoRA models at inference (full + fold0/1/2 adapters) — a free fold-ensemble.
Blend weights for v10 are fit on folds 0–2 only (where the new LoRA has OOF).
Template fixes: explicit LoRA members `[(kernel, subdir)]`, exact path-segment matching (`ntl-lora-q8b` is a substring
of `ntl-lora-q8b-extra2`), head.pt looked up next to the adapter.

## Ops note
A network hang made one `kernels push` register silently, and the retry started a second identical LoRA run
(~5 extra GPU-hours). Combined with the LoRA runs this exhausted the weekly GPU quota → v9/v10 stuck in QUEUED.
All CLI calls are now wrapped in `timeout`; check `kernels status <ref>/<version>` before retrying a push.

## v10 · 0.536 — gain smaller than CV predicted (+0.006 LB vs ~+0.02 CV over the v7-equivalent set)
Likely contributors, to test next:
1. **Ensemble vs single-model mismatch:** blend weights were fit on single-fold-model OOF, but inference averages 4 LoRA
   models. Averaged probabilities are differently calibrated (usually softer), so the fitted LoRA weight no longer means
   the same thing.
2. **Weights fit on only 3 folds** (≈1,620 rows) — noisier than the 5-fold fits behind v6/v7.
3. **Twins interaction:** twins add evidence only from frozen heads; making those heads stronger (extra data) changes
   the plain/twin balance that w=0.5 was tuned for.
4. Two changes in one submission (extra heads + extra LoRA) — v9 (extra heads only) was never scored, so the split of
   the gain is unknown. Submit v9 as the ablation.

---
# Day 4 (2026-10-18): the external-data race

## Landscape
LB top 0.622 (single team, far ahead), pack 0.58–0.586, best *public* notebooks ≈0.552–0.554 (all variants of
stylegram + frozen heads + LoRA24 + twins + gates). The gap between public and private suggests private extra data.

Forum thread 749447 ("Will the winner be whoever gets the most extra data?"): one team added a public 18-model corpus
(~15k texts) and **lost** accuracy (0.326 → 0.279 with TF-IDF; same with Qwen3-8B embeddings). Cause: format mismatch —
host passages have **0% line breaks, 0% bold, median 101 words**; that corpus had 94% line breaks, 60% bold, median 174.
→ **Extra data must look like the host's data.**

## Curated host-format pool (`src/extra.py`, `kaggle_research/extra_pool_snippet.py`)
11 public CC-BY/CC-BY-SA datasets (Claude API passages, prvsiyan extended, synth-13, vertex-gemini, proxy generations, …).
Filters: no newline, no markdown, 30–300 words; exact-text dedup. Excluded: mega-corpus / extra-merged (25–46% markdown).
Result: **22,149 rows, every class ≥ 586** (GLM ×2, Qwen, Kimi finally covered), ≈8× the competition train set.

**Fidelity check** (stylegram trained on competition train only, accuracy on each source; in-domain CV ≈0.385/0.70 lab):
most sources 0.24–0.36 model / 0.60–0.78 lab → close to host distribution. Outlier `own_extra` (0.209 / 0.464; built by
chunking long answers into paragraphs) → excluded (CV-neutral: 0.4293 without vs 0.4282 with).

## E18 · Stylegram on the full pool — ✅ 0.398 → 0.429
| extra data | none | v1 pool (7k, 3 labs) | full pool (22k, 6 labs) |
|---|---|---|---|
| stylegram OOF | 0.385 | 0.398 | **0.429** (w 0.5); w 0.7 0.430; w 0.85 0.418 |

## Running
- `ntl-lora-q8b-pool`: LoRA on train + pool (cap 400/class ≈ 7.2k rows), 2 epochs, jobs full + fold 0.
- `ntl-extra-heads-pool`: frozen heads (Qwen3.5 L17/L18, Qwen3-8B L2+22) with pool (cap 600/class), w ∈ {0.3, 0.5, 0.7}.
- v11 = v10 + full-pool stylegram (LB pending); v9 ablation submitted.

## Where extra data pays off on the LB (decisive for GPU allocation)
| extra data added to | CV gain | LB gain |
|---|---|---|
| stylegram, v1 pool (v6) | +0.004 | +0.011 |
| frozen heads (v9) | +0.013 | +0.001 |
| stylegram, full pool (v11) | stylegram +3.1 alone | +0.002 |
| **LoRA (v10)** | ~+0.02 | **+0.006** |
Frozen-head and n-gram gains mostly saturate inside the blend; the fine-tuned LoRA is the member that converts extra
data into LB. → Spend GPU on LoRA(s) trained on the big pool (more data, more epochs, more seeds/bases).

## E19 · Frozen heads on the big pool (`ntl-extra-heads-pool`, cap 600/class ≈ 10k rows)
| head | none | small pool w0.3 | **big pool w0.5** | big pool w0.3 / w0.7 |
|---|---|---|---|---|
| Qwen3.5-9B L18 | 0.423 | 0.453 | **0.482** | 0.480 / 0.463 |
| Qwen3.5-9B L17 | 0.423 | 0.444 | **0.477** | 0.472 / 0.462 |
| Qwen3-8B L2+22 | 0.467 | 0.478 | **0.504** | 0.497 / 0.493 |
The Qwen3-8B frozen head (0.504) now beats the original LoRA (0.491). Blend (folds 0–2, 3-fold nested):
small-pool heads 0.547 → **big-pool heads 0.556**. → v12.

## E20 · LoRA on the big pool (`ntl-lora-q8b-pool`, cap 400/class ≈ 7.2k rows, 2 epochs)
Fold 0: 0.485 (no extra 0.472, small pool 0.480). Small single-model gain for ~2.5× more data — 2 epochs may be too
few for 9.4k rows/epoch, or the class-balanced weighting (w 0.5) caps how much the extra rows can teach.
Averaging the two extra LoRAs on fold 0: **0.500** single-model, but in the fold-0 blend 0.537 vs 0.544 (noise ±0.02
at 540 rows). → v13 tests the 6-adapter average on the LB.

## Topic-shortcut check (forum: "extra prompts must carry almost all labels")
Per-prompt label coverage after normalising the shared seyit prompt bank: 14.4k of 22.1k rows are in prompts with ≥13
labels; only 2.5k in ≤5-label prompts. Stylegram with coverage-filtered pools: cov≥8 0.428, cov≥12 0.430, all 0.429
→ no shortcut effect for n-gram models.

## How the leaders get data
Kaggle Benchmarks Model Proxy (free daily AI quota for every participant, used via `kaggle_benchmarks` tasks) reaches
12 labels (GPT ×5, Gemini, Claude); GLM/Qwen/Kimi came from paid official APIs / OpenRouter. Recipe used by every good
dataset: one user message, no system prompt, `<prompt> Write a single paragraph of about N words, with no title or
headings.`, N drawn per passage from the train length distribution, provider defaults.

## Reading the LB correctly: small deltas are noise
The public LB is a subset of ~4k passages with an unknown, unbalanced class mix. Two submissions that differ in one
component flip only a few % of predictions, so their LB difference has a noise floor of roughly ±0.005. v9 (+0.001),
v11 (+0.002), v12 (−0.001) are all inside it. **Use 5-fold CV for small changes; use the LB only to confirm large ones.**
The CV–LB gap has widened from ~0.01 to ~0.02 as more extra data went in: extra data helps on held-out *host* prompts
more than on the hidden test, i.e. the test's generation setup still differs somewhat from all the public pools.

---
# Day 5 (2026-10-19): generating our own data (free Kaggle Benchmarks Model Proxy)

## Setup (`datagen/`)
- `kaggle benchmarks init` → proxy credentials (kept outside the repo); tasks are pushed and run remotely with
  `kaggle b t push/run/download` (`datagen/task_template.py`).
- Reachable competition labels (12): GPT ×5, Gemini 3.1-pro / 3.6-flash / 3.8-flash, Claude opus-4-6 / 4-8 / 5,
  sonnet-5. Not reachable: claude-fable-5, gemini-3.7-flash, GLM ×2, Qwen3.8-max, Kimi-K3.
- Prompts: the CC-BY seyit 1,000-prompt bank; 399 ids (g0521–g0919) unused by any public proxy dataset. Each of these
  already has 5 Claude passages, so our GPT/Gemini passages complete 13-label prompt blocks.
- Request = host recipe: one user message, no system prompt, provider defaults,
  `<prompt> Write a single paragraph of about N words, with no title or headings.`, N drawn per (prompt, model) from
  the train word-count distribution (clipped 40–250).

## Pilot (20 prompts × 8 models)
| model | ok | USD/passage | notes |
|---|---|---|---|
| gemini-3.6-flash / 3.8-flash | 20 / 20 | 0.014 / 0.013 | 5–10% em-dashes (train Gemini has none; same in other proxy data) |
| gemini-3.1-pro-preview | 15 | 0.072 | hit the $1 cap |
| gpt-5.5 / 5.6-sol / 5.6-terra | 20 / 20 / 20 | 0.013 / 0.015 / 0.007 | |
| gpt-5.6-luna | 12 | 0.001 | then PermissionDenied for the rest |
| gpt-6-astra | 0 | — | PermissionDenied on every call |
Refusals start after ~$0.7 of OpenAI spend → the free quota is a **daily per-provider budget**. Expect a few hundred
passages/day in total → `datagen/daily.py` runs the next prompt slice per day, rotating weakest classes first,
appending to `data/ours/passages.csv` with full provenance (prompt id, model id seen, target length, cost, task).
Format: 0% line breaks / markdown, words/target = 1.00.
**Fidelity** (train-only stylegram): 0.315 model / **0.78 lab** — at the top of the public sources' range.

## Daily generation (automated)
- `datagen/run_daily.sh` via launchd (`~/Library/LaunchAgents/com.ntl.datagen.plist`), 09:00 local = 01:00 UTC, just
  after the presumed 00:00 UTC quota reset. Logs: `datagen/logs/`. Rotation: GPT ×4 (sol, terra, 5.5, luna), Gemini
  ×3, Claude ×4, per-model spend caps, stop after 3 PermissionDenied (= provider's daily budget used up).
- Refused from the first call for our account: **gpt-6-astra, claude-opus-4-6, claude-opus-4-8** (kept in rotation in
  case access opens; a refusal costs nothing).
- Costs per passage: claude-sonnet-5 0.003, claude-opus-5 0.014, gemini flash ~0.013, gpt ~0.007–0.015,
  gemini-3.1-pro 0.072.
- Bug fixed: the push's automatic creation run uses the proxy's default model (gemini-3.8-flash) → skip lists now
  cover every model so no (prompt, model) pair is generated twice.
- Total own passages after day 1: 224.

## E21 · 3-epoch big-pool LoRA (`ntl-lora-q8b-pool3`) — ❌ no gain
Rationale: every earlier LoRA gained +2…+5 points from epoch 2 → 3 (0.452→0.472, 0.452→0.480, 0.459→0.520,
0.478→0.526); the big-pool run stopped at 2 epochs (fold 0: 0.485) and was likely undertrained.
Result: fold 0 = 0.387 / 0.469 / **0.480** (ep 0/1/2) vs 0.485 for the 2-epoch run → undertraining was not the issue.
Every LoRA variant now lands at ~0.48 on fold 0 (small pool 3 ep 0.480, big pool 2 ep 0.485, big pool 3 ep 0.480):
the LoRA *as built* is saturated, and more of the same public data does not lift it.

## E22 · Twin-weight sweep (LB): v14 w 0.3, v15 w 0.7 (v7/v11 use 0.5; v8 w 1.0 was worse)
v15 (w 0.7): **0.526** vs v11 (w 0.5) 0.538 and v8 (w 1.0) 0.525 → heavier twins hurt. v14 (w 0.3): **0.539** (new best, ≈ v11 within noise) → twin weight optimum is ~0.3–0.5.

## E23 · LoRA with a prompt-centered head (`ntl-lora-q8b-twin`) — running
Own implementation of the idea behind the public 0.55 notebooks' "centered readout": micro-batches are triples of
passages answering the same prompt (train prompts; extra passages grouped by the shared seyit prompt bank → up to 13
labels per prompt), loss = CE(plain(z)) + CE(center(z − mean_triple(z))). At test time, rows in topic clusters of size
2–18 get the centered head's log-probs on cluster-centered LoRA features added (twin evidence from the strongest member,
instead of only the frozen heads). Config: small pool (cap 300/class), 3 epochs, micro 3 × accum 6; jobs full + fold 0.
Result: plain head fold 0 = 0.385 / 0.480 / **0.485** (ep 0/1/2) — same as other LoRAs (the extra centered objective does
not hurt). Its value is the twin evidence → v16 (LoRA twin w 0.5 + frozen-head twins w 0.5) on the LB.
