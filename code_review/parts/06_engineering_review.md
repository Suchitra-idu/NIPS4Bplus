# Part 6 — Software-engineering review: structure, correctness, robustness, testing, reproducibility

> Part of the [deep code review](../../CODE_REVIEW.md). Previous: [Part 5](05_inference_and_evaluation.md). Next: [Part 7](07_gap_analysis_and_open_questions.md).
> Everything below is **review only**: proposed code is shown as text and was **not applied** ([CLAUDE.md](../../CLAUDE.md) forbids methodology changes aimed at the paper's numbers; several suggestions below *would* change numbers and are marked ⚠).
> Severity scale: **H** (changes results or blocks reproduction), **M** (misleads or breaks under plausible use), **L** (hygiene).

---

## 1. Design assessment at a glance

| Quality attribute | Rating | Why |
|---|---|---|
| Fidelity to the authors' released code | **high** | additions are additive and logged ([logs.md](../../logs.md)); training behaviour only changed through the documented `fact_amp=0` ([call_id.py:139-142](../../call_id.py#L139-L142)) |
| Modularity | **low** | module-level scripts, no `main()`, duplicated cfg parsing / model building ([Part 1 §2.1](01_repository_and_scripts.md)) |
| Testability | **low** | nothing importable without executing training or evaluation; no tests in the repo |
| Reproducibility | **medium-low** | seeds added, but model code untracked, canonical split non-regenerable, no env pin, absolute paths |
| Observability | **medium** | `res.res`, `time.res`, full stdout logs; no hardware/version/cfg stamp |
| Performance | **good enough** | preloading + vectorised evaluation; GPU ≈ 1/3 utilised ([Part 4 §6.4](04_training_loop_and_optimisation.md)) |
| Documentation | **medium** | extensive analysis docs but several stale statements ([§9](#9-documentation-accuracy-errata)) |
| Evaluation validity | **low** | whole-file rule, saturated scores, tag-level split ([Part 5](05_inference_and_evaluation.md)) |

---

## 2. [call_id.py](../../call_id.py) — finding by finding

### 2.1 Structure

* **M — import-time side effects.** [L125](../../call_id.py#L125) `options=read_conf()` parses `sys.argv`, then the module builds models, moves them to CUDA ([L233,252,267](../../call_id.py#L233)) and trains. `import call_id` in a test would start a training run (or fail without a GPU/`--cfg`). No `if __name__ == "__main__":`, no functions for train/validate/checkpoint.
* **M — duplicated parsing/model construction with [evaluate_metrics.py:22-89](../../evaluate_metrics.py#L22-L89).** ~70 lines are copied. The `fact_amp` change had to be implemented by a *second* `ConfigParser` pass ([L139-142](../../call_id.py#L139-L142)) because `read_conf` lives in the upstream file.
* **L — mid-file import** `import configparser` at [L140](../../call_id.py#L140), unused imports `F`, `flip`, `Variable` ([L30,32,36](../../call_id.py#L30)).
* **L — Mixed 1/2/3-space indentation** inside the epoch loop (`if epoch%N_eval_epoch==0:` at [L329](../../call_id.py#L329) is at 2 spaces, its body at 3, the nested `with` at 4). Python accepts it; reading the `if run_validation: … else:` pairing ([L331,403](../../call_id.py#L331)) requires counting spaces. A bug-prone layout inherited from upstream.

Suggested shape (not applied):

```python
def main(cfg_path):
    opt = load_options(cfg_path)                  # one shared parser for train and eval
    nets = build_models(opt)                      # shared with evaluate_metrics.py
    loader = BatchSampler(preload(opt.tr_lst), wlen, fact_amp, rng=np.random.RandomState(seed))
    for epoch in range(opt.N_epochs):
        train_one_epoch(nets, loader, optimisers)
        if epoch % opt.N_ckpt_epoch == 0 or epoch == opt.N_epochs - 1:
            save_checkpoint(...)
if __name__ == "__main__": main(parse_args().cfg)
```

### 2.2 Correctness and robustness

| ID | Line | Issue | Sev |
|---|---|---|---|
| C-1 | [329,414-418](../../call_id.py#L329) | checkpoint only at `epoch % N_eval_epoch == 0`: with 8 the final epoch (399) is lost; the saved model is epoch 392 (⟨measured⟩ `time.res`). Fix idea: also save on `epoch == N_epochs−1` ⚠ (changes the evaluated weights) | M |
| C-2 | [414](../../call_id.py#L414) | checkpoint has no epoch, cfg, RNG or optimiser state ⇒ cannot resume; `pt_file` loads only weights ([L270-274](../../call_id.py#L270-L274)) | L |
| C-3 | [L204-205](../../call_id.py#L204-L205) | seeds numpy + torch but not `cudnn.deterministic/benchmark`; `torch.cuda` ops (conv1d backward) may be non-deterministic | L |
| C-4 | [L63-120](../../call_id.py#L63-L120) | `np.random.randint(lo, hi)` raises `ValueError` if `lo > hi` (event extends beyond the file by more than `wlen`); ⟨measured⟩ it never happens on the current lists (0 cases) but the code has no guard | L |
| C-5 | [L107-110](../../call_id.py#L107-L110) | stereo check executes **after** slicing and prints once per sample; dead for this mono corpus | L |
| C-6 | [L196-200](../../call_id.py#L196-L200) | `try: os.stat(...) except: os.mkdir(...)` — bare `except`, non-recursive `mkdir` | L |
| C-7 | [L45,331](../../call_id.py#L45) | `run_validation` is a module constant; `N_eval_epoch` is really a checkpoint interval; test list still loaded and counted ([L190-193](../../call_id.py#L190-L193)) | L |
| C-8 | [L278-280](../../call_id.py#L278-L280) | three identical optimisers ([Part 4 §1.2](04_training_loop_and_optimisation.md)) | L |
| C-9 | [L117-118](../../call_id.py#L117-L118) | `Variable(...)` no-op; `.float().cuda().contiguous()` per batch (fine) | L |
| C-10 | [L318-319](../../call_id.py#L318-L319) | `loss_sum+loss.detach()` — good practice (no per-step sync); **keep** | — |
| C-11 | [L304-305](../../call_id.py#L304-L305) | `lab.long()` computed twice | L |
| C-12 | whole | no logging of: git commit, library versions, GPU, cfg copy, seed; `time.res` does not say which device produced the times ⇒ [Part 4 §2.1](04_training_loop_and_optimisation.md) discrepancy cannot be resolved | M |

### 2.3 The sampler in review ([L63-120](../../call_id.py#L63-L120))

Good: correct handling of both branches; per-sample loop vectorisation is unnecessary at 1 ms per batch. Concerns: (i) uniform over tags ⇒ imbalance ([Part 2 §4](02_data_pipeline.md)); (ii) no guarantee that *consecutive* draws are decorrelated across the same file (irrelevant here); (iii) the buffered branch's lower/upper bound semantics (window **contains** the event) differ subtly from the long branch (window **inside** the event) — the *label-to-window relationship* changes from "window ⊂ event" to "window ⊃ event" with no flag in the batch, so BatchNorm statistics mix the two populations (only 0.15–0.6 % of draws).

---

## 3. [evaluate_metrics.py](../../evaluate_metrics.py)

| ID | Line | Issue | Sev |
|---|---|---|---|
| V-1 | [123-148](../../evaluate_metrics.py#L123-L148) | decision over the **whole file**, not the tag; `start/length` never used ([Part 5](05_inference_and_evaluation.md)) ⚠ | **H** |
| V-2 | [143,147-152](../../evaluate_metrics.py#L143-L152) | `softmax(Σ log p)` is one-hot ⇒ top-k, ROC AUC degenerate ⚠ | **H** |
| V-3 | [91](../../evaluate_metrics.py#L91) | loads the checkpoint with `map_location='cuda'`: fails on CPU-only machines (evaluation is ≈ 100 s per model on the GPU, ≈ 18 k frames/s; a CPU run is slower but feasible) | L |
| V-4 | [22-89](../../evaluate_metrics.py#L22-L89) | duplicated cfg parsing / model building | M |
| V-5 | [120](../../evaluate_metrics.py#L120) | sampling rate returned by `sf.read` unused/unchecked | L |
| V-6 | [123,130](../../evaluate_metrics.py#L123-L130) | `N_fr+1` rows vs `n_frames` valid rows; harmless zero row | L |
| V-7 | [96-98](../../evaluate_metrics.py#L96-L98) | counts parameters that never receive gradients (61 k, [Part 3 §6.3](03_model_theory_to_code.md)) | L |
| V-8 | [104-107,158-164](../../evaluate_metrics.py#L104-L107) | stores `y_score` (post-softmax) but not the raw `Σ log p` ⇒ margins/saturation cannot be recovered from the artefact; storing the raw sum would have made the saturation analysis trivial | M |
| V-9 | [158-160](../../evaluate_metrics.py#L158-L160) | writes `key=value` text with 4 decimals (accuracy 0.5514); no cfg/commit/epoch stamp in `metrics.res` | L |
| V-10 | [114](../../evaluate_metrics.py#L114) | `wav_lst_te.loc[i, 'file']` per row (pandas scalar lookup) — fine at 1.3 k rows | — |
| V-11 | top | script-level (no function) — not importable | M |

---

## 4. [metrics_utils.py](../../metrics_utils.py) — the best-engineered file

* Pure functions, documented, torch-free, explicit `labels=` so absent classes do not shift columns.
* `weighted_fpr_fnr` uses `np.errstate` and `np.where` to guard zero denominators ([L23-25](../../metrics_utils.py#L23-L25)).
* **L** — `compute_metrics` requires `y_score` rows that sum to 1 ([L30-32](../../metrics_utils.py#L30-L32)) but does not validate it; `top_k_accuracy_score` silently accepts one-hot ties (the root of V-2). A guard "warn if > 95 % of rows have max-score == 1" would have exposed the issue on the first run.
* **L** — no tests (the file docstring says it was kept separate "so it can be tested on CPU"; no test was kept — [logs.md](../../logs.md)). [code_review/tests/test_review_claims.py](../tests/test_review_claims.py) now contains a metric identity test.

---

## 5. Upstream model code ([dnn_models.py](../../SincNet_src/dnn_models.py), [data_io.py](../../SincNet_src/data_io.py))

These are the authors' dependencies; findings are **documented, not repaired** (fidelity rule).

| ID | Location | Issue | Sev |
|---|---|---|---|
| U-1 | [dnn_models.py:415](../../SincNet_src/dnn_models.py#L415) | `BatchNorm1d(N_filt, <length>, momentum=…)` ⇒ `eps = 111/21/3` ([Part 3 §4.2](03_model_theory_to_code.md)) | **H** |
| U-2 | [:100-103,134-136](../../SincNet_src/dnn_models.py#L100-L103) | cut-off parameters in Hz with `lr=1e-3` ⇒ frozen filterbank ([Part 3 §2.6](03_model_theory_to_code.md)) | **H** |
| U-3 | [:446-456](../../SincNet_src/dnn_models.py#L446-L456) | `abs()` only on the LayerNorm path | M |
| U-4 | [:310-322](../../SincNet_src/dnn_models.py#L310-L322) | both norms always constructed; `bias=False` overwritten by a zero bias; tiny `U(±√(0.01/(in+out)))` init (0.04× Xavier) | L |
| U-5 | [:130-132](../../SincNet_src/dnn_models.py#L130-L132) | `self.n_ = self.n_.to(device)` re-assigns tensors every forward (not `register_buffer`): they are not in `state_dict` and `.cuda()` on the module does not move them — they are moved lazily inside `forward` | L |
| U-6 | [:107-108](../../SincNet_src/dnn_models.py#L107-L108) | window built with `K` instead of `K−1` and a non-integer `linspace` step (1.0068) — error ≤ 7.5×10⁻⁵ | L |
| U-7 | [:223-244](../../SincNet_src/dnn_models.py#L223-L244) | `act_fun` has no `else` ⇒ returns `None` for a typo; `"linear"` is `LeakyReLU(1)` "initialized like this, but not used in forward" | L |
| U-8 | [:247-258](../../SincNet_src/dnn_models.py#L247-L258) | custom LayerNorm: unbiased std, eps added to std | L |
| U-9 | [:163-220](../../SincNet_src/dnn_models.py#L163-L220), [:9-25](../../SincNet_src/dnn_models.py#L9-L25) | legacy `sinc_conv`, `sinc`, `flip` (hard-coded `.cuda()`, Python loop over filters) | L |
| U-10 | [data_io.py:17-78,121-181](../../SincNet_src/data_io.py#L17-L181) | 60-line copy-paste; strings only; `str_to_bool` bare `ValueError` | L |
| U-11 | [data_io.py:90-117](../../SincNet_src/data_io.py#L90-L117) | dead `create_batches_rnd` referencing un-imported `scipy` | L |

U-5 deserves a note: because `n_` and `window_` are plain attributes, `torch.save(model.state_dict())` does **not** contain them; they are re-derived from `kernel_size`/`sample_rate` at construction. That is fine for this code (deterministic) but means a checkpoint is only valid with the same constructor arguments — the checkpoint carries no architecture description ([§2.2 C-2](#22-correctness-and-robustness)).

---

## 6. Preprocessing and list scripts

| ID | File:line | Issue | Sev |
|---|---|---|---|
| P-1 | [nips4b_normalise_files.py:35](../../nips4b_normalise_files.py#L35), [cut_nips4bplus_files.py:62](../../cut_nips4bplus_files.py#L62) | `x/abs(max(x))` + PCM_16 write clips **187/687** files (≤ 23 % of samples) ⚠ | M |
| P-2 | [generate_mod_file_lists.py:98,116,135](../../generate_mod_file_lists.py#L98) | three independent tag-level splits; test files overlap training files (98.6–99.0 % of test rows) ⚠ | **H** (validity) |
| P-3 | [:77](../../generate_mod_file_lists.py#L77), [generate_file_lists.py:89](../../generate_file_lists.py#L89) | bare `except:` turns any lookup error into "drop row" | M |
| P-4 | [:52](../../generate_mod_file_lists.py#L52), [cut:43](../../cut_nips4bplus_files.py#L43) | file id from `str[-7:-4]` slicing of the CSV path | L |
| P-5 | [:36-43](../../generate_mod_file_lists.py#L36-L43) | positional `sys.argv`, no validation, no `--help` | L |
| P-6 | [:91-94](../../generate_mod_file_lists.py#L91-L94) | label integers by first-appearance order, mapping not saved | M |
| P-7 | tracked `mod_data_lists/` | **cannot be regenerated**: the lists predate the seed argument; running the script with the default seed 1234 gives a different split ⟨verified, [test](../tests/test_review_claims.py)⟩ (the repo says so: [logs.md](../../logs.md) "NOT regenerated and remain the canonical split") | M |
| P-8 | [README.md:31,55,80](../../README.md#L31) | `pyton`, `-–cfg` (en dash) | L |
| P-9 | [generate_file_lists.py:99-152](../../generate_file_lists.py#L99-L152) | three copy-pasted blocks building a label dictionary | L |

---

## 7. Configuration and repository hygiene

* **M — absolute paths** in tracked cfgs ([mod_cfg/mod_nips4bplus_bird_species.cfg:2-5](../../mod_cfg/mod_nips4bplus_bird_species.cfg#L2-L5): `/home/suchitra/MyStuff/NIPS4Bplus/...`); [logs.md](../../logs.md) mentions rewriting by `sed` for Colab.
* **M — model code ignored by git** ([.gitignore:7-9](../../.gitignore#L7-L9)) and replaced by symlinks.
* **L — bulky/duplicated artefacts**: `output/*/model_raw.pkl` (3 × 10 MB) tracked; `overleaf_package/` and `overleaf_v2/` plus two 7 MB zips duplicate 16 PDFs three times; `ArchitectureAnalysis.md` duplicates `DeepArchitectureAnalysis.md`; two PDFs of the SincNet paper (`sinceNet.pdf`, `SPEAKER RECOGNITION…pdf`) are identical (same size).
* **L — security**: loading `model_raw.pkl` uses `torch.load` (pickle). With PyTorch ≥ 2.6 the default `weights_only=True` makes it safe for state-dicts; with older versions loading untrusted checkpoints is code execution. Nothing in the repo pins the version.
* **L — no environment spec**: no `requirements.txt`/`pyproject`; the README defers to "the requirements of SincNet".
* **L — no CI and no pre-commit hooks.**

---

## 8. Testing strategy

The repository has **zero automated tests**. Because the code is script-style, tests either (a) import the authors' classes ([dnn_models.py](../../SincNet_src/dnn_models.py), [metrics_utils.py](../../metrics_utils.py)) or (b) run scripts via subprocess. [code_review/tests/test_review_claims.py](../tests/test_review_claims.py) does both and passes in ≈ 18 s on CPU (13 tests ⟨measured⟩):

| Test | What it pins down | Covers finding |
|---|---|---|
| `test_shapes_and_out_dim` | flatten sizes 240/180/180 and 300; output width = `class_lay` | architecture claims |
| `test_batchnorm_eps_is_the_pooled_length` | `[m.eps for m in cnn.bn] == [111, 21, 3]` | F3 |
| `test_sinc_filters_symmetric_centre_one_and_nyquist` | symmetry, centre tap 1, **matrix rank 76** | §2 Part 3 |
| `test_abs_only_on_layernorm_path` | cfg combinations that select the `abs` branch | §3 Part 3 |
| `test_parameter_count_matches_metrics_res` | constructed parameter count == `trainable_params` in `metrics.res` | Table 1 consistency |
| `test_lists_sizes_and_label_range` | 4110/1371, 3959/1320 sizes; labels contiguous | S4–S6 |
| `test_tag_level_leakage` | > 98 % of test rows share a file with training rows | F7 |
| `test_split_seed_determinism` | same seed ⇒ identical lists; different ⇒ different; tracked lists ≠ seed-1234 output | P-7 |
| `test_normalisation_clips_negative_dominant_files` | `x/abs(max(x))` > 1 and PCM_16 write clips | F8 |
| `test_recall_equals_accuracy_and_one_hot_auc_identity` | weighted recall ≡ accuracy; AUC ≈ `(acc+1−FPR)/2` for one-hot scores | F2 |
| `test_checkpoint_is_epoch_392` | `time.res` last line, cfg `N_eval_epoch=8` | F6 |
| `test_saved_scores_saturated` | > 99 % of saved rows have `max == 1.0` | F2 |
| `test_sinc_cutoffs_barely_moved` | `|Δ low_hz_| < 10 Hz` | F4 |

Recommended additions for the project itself (not implemented): a golden-output test for `create_batches_rnd` on a 3-row list (both branches, fixed seed); a regression test that `call_id.py` after refactoring still yields the same first-batch checksum; a test that `evaluate_metrics` frame count equals the original `while` loop on 5 synthetic lengths.

---

## 9. Documentation accuracy (errata)

| Document / location | Statement | Reality | Verdict |
|---|---|---|---|
| [DataPipelineAnalysis.md §5](../../DataPipelineAnalysis.md) | every draw re-reads the wav; `fact_amp` hard-coded; no `random_state`; evaluation not memoised | all four fixed in code | stale |
| [DataPipelineAnalysis.md §3](../../DataPipelineAnalysis.md) | normalised files "can exceed ±1.0" | PCM_16 write clips: 187/687 files, ≤ 3.77× overshoot, ≤ 23 % samples | incomplete |
| [DataPipelineAnalysis.md §0](../../DataPipelineAnalysis.md) | no runner for Part I | upstream [speaker_id.py](../../SincNet_src/speaker_id.py) (ignored clone) | wrong |
| [DataPipelineAnalysis.md §6](../../DataPipelineAnalysis.md) | scoring is whole-file | **correct**, and quantified here ([Part 5](05_inference_and_evaluation.md)) | ✔ |
| [DeepArchitectureAnalysis.md §14](../../DeepArchitectureAnalysis.md) | `abs()` after the Sinc layer | only on the LayerNorm path | incomplete |
| [DeepArchitectureAnalysis.md §21](../../DeepArchitectureAnalysis.md) | file prediction = argmax of **averaged probabilities** | shipped = argmax of **summed log-probabilities** | wrong for the code |
| [DeepArchitectureAnalysis.md §41, §64](../../DeepArchitectureAnalysis.md) | filters "move little" because of initialisation | they cannot move: Hz × `lr` ([Part 3 §2.6](03_model_theory_to_code.md)) | explained |
| [DeepArchitectureAnalysis.md §59](../../DeepArchitectureAnalysis.md) | 80 → 220 filters = more frequency channels | bank rank ≤ 76, effective bandwidth ≈ 390 Hz | overstated |
| [DeepArchitectureAnalysis.md:9-17](../../DeepArchitectureAnalysis.md#L9-L17) | five `images.openai.com` links | ephemeral URLs | broken |
| [logs.md](../../logs.md) | `N_eval_epoch` 8 → 57 | cfgs still 8; checkpoint is epoch 392 | not applied |
| [logs.md](../../logs.md) | "`output/` is empty and gitignored" | tracked, 34 MB | stale |
| [logs.md](../../logs.md) | ≈ 15 s/epoch, 2.4 h/run expected | recorded runs: 11.6 s/epoch, 76 min | updated |
| [TODO.md](../../TODO.md) | `evaluate_metrics.py` "needs a first GPU run" | has run | stale |
| [CLAUDE.md](../../CLAUDE.md), [DataPipelineAnalysis.md](../../DataPipelineAnalysis.md) | `REPRODUCTION_REPORT.md` | not in repo | missing |
| [DATASET_ANALYSIS.md §10](../../DATASET_ANALYSIS.md) | "`cw_len=10 ms` enhanced window" | enhanced is 16/18 ms; 10 ms is the default | wrong |
| `README.md` | "requirements are those of SincNet" | no version, no pin | vague |

---

## 10. Prioritised recommendations (none applied)

| Priority | Action | Changes numbers? |
|---|---|---|
| 1 | Write `REPRODUCTION_REPORT.md` with the gap hypotheses of [Part 7](07_gap_analysis_and_open_questions.md) (E1/E2/U-1/U-2) | no |
| 2 | Save the raw `Σ log p` (and per-file frame count) in `predictions.npz`; also report top-k/AUC from the mean-p scores next to the shipped ones | no (reporting) |
| 3 | Track `dnn_models.py`/`data_io.py` at a pinned SHA or as a submodule; add `requirements.txt` with exact versions | no |
| 4 | Make the checkpoint interval include the last epoch; record `N_eval_epoch` correctly in cfgs and `logs.md` | ⚠ new epoch-399 weights |
| 5 | Factor cfg parsing + model construction into one importable module used by both scripts | no |
| 6 | Add tests (§8) and a small CI job | no |
| 7 | Stamp `time.res`/`metrics.res` with device, torch version, commit, cfg hash | no |
| 8 | Evaluate additionally with a recording-level split and with tag-span scoring as **labelled diagnostics** | ⚠ new numbers (not a replacement) |

→ continue with [Part 7: gap analysis and open questions](07_gap_analysis_and_open_questions.md).
