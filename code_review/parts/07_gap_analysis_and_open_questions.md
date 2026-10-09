# Part 7 — Why the numbers differ from the paper: ranked hypotheses, evidence, and open questions

> Part of the [deep code review](../../CODE_REVIEW.md). Previous: [Part 6](06_engineering_review.md).
> Per [CLAUDE.md](../../CLAUDE.md): *"Record guesses for why the discrepancy exists … that's analysis, not a to-do list. The code stays faithful."* This part is written in that spirit. It contains **no recommendation to change training or evaluation code to move a number**; the single diagnostic experiment (read-only re-scoring) is described as such.
> `REPRODUCTION_REPORT.md`, the file that CLAUDE.md designates for this content, does not exist in the repository; this part is written so it can be copied into it.

---

## 1. The gap, stated precisely

| set | ours (shipped rule) | paper (Table 1) | gap | ours, tag-span mean-p (D) | gap after D |
|---|---|---|---|---|---|
| All classes (87) | 0.5514 | 0.7301 | **−17.9** | 0.6776 | −5.3 |
| Bird classes (77) | 0.5803 | 0.7447 | **−16.4** | 0.7083 | −3.6 |
| Bird species (51) | 0.5841 | 0.7356 | **−15.2** | 0.6894 | −4.6 |

Other columns: top-3 0.56–0.59 vs 0.90 (−31 to −34); top-5 0.56–0.61 vs 0.93–0.94; ROC AUC 0.77–0.79 vs 0.75–0.77 (**ours higher**); AUC-mean-exp 0.98 vs 0.76–0.79 (**ours far higher**); parameters 2.49–2.59 M vs 2.5–2.6 M (**match**); training time 76 min vs 1.9–2.1 h (ours faster).

The mixed direction (some of ours higher, some lower) already says the discrepancy is not a single "model is worse" effect.

---

## 2. Hypotheses ranked by evidence

Legend — **Evidence**: ●●● measured here with the same weights; ●● measured indirectly; ● circumstantial. **Effect** = expected share of the gap.

| # | Hypothesis | Evidence | Effect | What I measured | What would settle it |
|---|---|---|---|---|---|
| H1 | **Decision is pooled over the whole file; the paper pools over each tagged call.** | ●●● | **+10.5 to +12.8 pts accuracy (~70 % of the gap)**, +30 pts top-3/5 together with H2 | rules A→D ([Part 5 §4](05_inference_and_evaluation.md)): D−A = +12.6 [8.9, 16.6], +12.8 [9.1, 16.6], +10.5 [7.1, 14.0] | the authors' evaluation code (not released); or their Fig. S1–S3 ROC data |
| H2 | **Saturated `softmax(Σ log p)` used for top-k / ROC AUC.** | ●●● | top-3/5: +28 pts with mean-p scores (0.84–0.86 / 0.92); ROC AUC meaningless | 100 / 100 / 99.85 % rows one-hot; AUC ≡ `(acc+1−FPR)/2` | same |
| H3 | **Residual 3.6–5.3 pts: model selection on the test list** ("best performing models" out of searches/runs). | ●● | up to the whole residual | no validation split exists in the repo ([Part 4 §6.3](04_training_loop_and_optimisation.md)); intervals ±4.5 pts | rerun many seeds/splits and report max vs mean (TODO §1) |
| H4 | **Different split / seed variance.** The tracked lists cannot be regenerated; each set has its own random test subset | ●● | ±2–5 pts | cluster-bootstrap 95 % CI ±4.5 pts; [Part 6 P-7](06_engineering_review.md) | repeat with several `split_seed`s |
| H5 | **Epoch 392 vs 399 / early-stopped weights.** | ● | < 1 pt (loss still falling slowly) | [Part 4 §5](04_training_loop_and_optimisation.md) | cannot be tested (epoch 399 not saved) |
| H6 | **"Tag span" in the paper is defined differently** (e.g. overlapping ≥ 50 %, or the cropped file of Part I) | ● | a few pts | my D uses frame centres in `[start, start+length]` | author code |
| H7 | **Data differences** (PCM_16 vs 32-bit as stated, clipping) | ● | ~0 | raw files are 16-bit; clipping is from the authors' own script | n/a |
| H8 | **Implementation differences in upstream SincNet** (version of `SincConv_fast`, BN `eps` bug) | ● | unknown | this clone is `d741648`; which upstream commit the authors used is not recorded anywhere | pin/compare commits |
| H9 | **Metric definition for Mean-Exp AUC** | ●● | explains the AUC-mean-exp mismatch (0.98 vs 0.76–0.79), not accuracy | ours uses flat OvR weighted AUC over files | author code |

H1 and H2 are not explanations "of the model": they are properties of how scores are produced from the same weights. H3–H6 are the plausible sources of the remaining 4–5 points; H7–H9 are provenance/definition questions.

---

## 3. Things that are the same as the authors' code (so *not* gap sources)

| Item | Why it cannot explain the gap |
|---|---|
| BatchNorm `eps = pooled length` ([dnn_models.py:415](../../SincNet_src/dnn_models.py#L415)) | the authors ran this line; hyper-parameters were tuned with it |
| Frozen Sinc cut-offs | same optimiser, same units |
| Normalisation clipping | same scripts ([nips4b_normalise_files.py:35](../../nips4b_normalise_files.py#L35)) |
| Tag-level split, leakage, ceiling | paper's own admission ("same data pool") |
| `abs()` missing in the enhanced path | same code path |
| 61 k dead parameters in the count | same count (2.5 M reproduced) |
| `fact_amp=0` | now configured exactly as Table S2 ([logs.md](../../logs.md)) |
| cfg values | match Table S2 (dropout 0.0, lr 0.001, batch 128, epochs 400, batches 80) |

The cfg/parameter match is reassuring: the *model and training* are as published; what is uncertain is how the published numbers were scored.

---

## 4. What the evidence says about the *model itself*

Independent of the score-pooling question:

* Single-frame accuracy inside a tag ≈ 0.54–0.55 for all three models ([Part 5 §2.4](05_inference_and_evaluation.md)) — the model's "per-16 ms" ability.
* Background frames receive a confident species label (top-1 ≈ 0.64–0.69) because there is no "none" class.
* Training accuracy 0.88–0.90 vs tag-span test accuracy 0.68–0.71 ⇒ overfitting by ≈ 20 pts despite tag-level leakage in favour of the test set.
* Per-class performance tracks support (ρ 0.43–0.65); macro-recall 0.40 vs accuracy 0.55–0.58.
* The front-end is a fixed mel-spaced band-pass bank with ≈ 390 Hz effective resolution ([Part 3 §2.5–2.6](03_model_theory_to_code.md)).

---

## 5. Open questions (answerable with the data on hand)

| Question | How to answer it (analysis only) | Cost |
|---|---|---|
| Does the choice of aggregation matter when tags are *known* vs *unknown*? | Report rule A–D plus a *detection-free* variant (e.g. top-N most confident frames) as labelled diagnostics | minutes (GPU) |
| How much of the 4–5-pt residual is seed/split variance? | Train 3–5 more seeds per set (76 min each) with `split_seed` ≠ 1234 and report mean ± sd; per [TODO.md §1](../../TODO.md) | hours |
| Is the paper's AUC-mean-exp computed per tag or per file? | Recompute from `rescored_*.npz` with different groupings and see which gives 0.76–0.79 | minutes |
| Does the "recording identity" shortcut inflate accuracy? | Evaluate on a **recording-level** split (new lists, labelled diagnostic) | hours |
| Do the 11–25 zero-recall classes have *anything* in common? | They are the classes with ≤ 62 training tags and short calls ([Part 5 §5](05_inference_and_evaluation.md)); a class-balanced re-score of the existing checkpoint (prior correction) is a pure post-hoc analysis | minutes |
| Would the checkpoint at epoch 399 differ? | cannot be answered (weights not saved) | — |
| What produced the 6.5× slower recorded runs? | record device, driver, torch version in the next run | — |

---

## 6. Limits of this review (what I did not do)

* No new training; no seed repetitions; no waveform+CNN or pre-trained baselines.
* The paper's Figures S1–S3 (ROC) and Fig. 2 (confusion matrix) were not re-created for *its* checkpoints (not available).
* The "tag-span" rule is my interpretation; I could not compare with the authors' evaluation code because it is not in the public repository.
* Statements about what the *authors* did (model selection, parameter counting, score computation) are inferences from the text and from reproducing their parameter count (2.5 M), not observations.
* Timing: the hardware of the committed runs is unknown ([Part 4 §2.1](04_training_loop_and_optimisation.md)).

---

## 7. Where to find things

| What | File |
|---|---|
| Figures (30) | [code_review/figures/](../figures/) (`architecture/ data/ evaluation/ filters/ training/`) |
| All measured numbers | [code_review/data/](../data/) (JSON/CSV/NPZ) |
| Scripts that produce them | [code_review/scripts/](../scripts/) (`run_all.sh` to regenerate) |
| Executable checks of the claims | [code_review/tests/test_review_claims.py](../tests/test_review_claims.py) |
| Parts | [00 problems/flow](00_problems_components_flow.md) · [01 repository](01_repository_and_scripts.md) · [02 data](02_data_pipeline.md) · [03 model](03_model_theory_to_code.md) · [04 training](04_training_loop_and_optimisation.md) · [05 evaluation](05_inference_and_evaluation.md) · [06 engineering](06_engineering_review.md) · 07 this file |
