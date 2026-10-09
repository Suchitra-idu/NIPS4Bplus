# Change log

Changes made to this repository, with the reason and the effect on
reproducing the results of Bravo Sanchez et al. (2021).

---

## 2026-10-08 — Speed up training / evaluation (Colab T4 run took ~6 h)

### Goal
Cut run time without changing the trained model or the reported metrics.

### Bottlenecks identified
1. **Per-epoch validation in `call_id.py`** (every `N_eval_epoch=8` epochs → 50 runs).
   Each run slides an 18 ms window with 1 ms shift over the *whole* file for every
   test row (~5.8M frames per run), copying frames one at a time in a Python loop,
   and re-processes the same file once per tag row (1371 rows, 442 unique files).
2. **Training data loading in `create_batches_rnd`**: every one of the
   400 × 80 × 128 ≈ 4.1M samples did a full `sf.read` of a wav file from disk
   (to keep 793 samples) plus 4 pandas `.loc` lookups, on the main thread while
   the GPU sat idle.
3. **`evaluate_metrics.py`** has the same per-frame Python loop as (1) (runs once).

### Changes

#### `call_id.py`
- **Validation off by default** — new flag `run_validation=False` at the top of the file.
  - Validation only monitors progress: no early stopping or LR schedule reads it,
    and it draws no random numbers, so skipping it does not change training.
  - Set `run_validation=True` to get the original `loss_te / err_te / err_te_snt` lines back.
  - With it off, epochs that would have been validated append
    `epoch N, loss_tr=… err_tr=…` to `res.res`, so the file still records progress.
- **Checkpoint schedule unchanged.** `model_raw.pkl` is still saved every
  `N_eval_epoch` epochs, whether or not validation runs. With 400 epochs and
  `N_eval_epoch=8`, the final checkpoint is still **epoch 392**, as in the
  original code. (Saving originally happened only inside the validation block.)
- **Training audio preloaded into RAM once** (new `preload_list`).
  - Each unique training wav is read once with `sf.read` (same float64 array,
    same `fs`), and the `file/start/length/label` columns become numpy arrays.
  - `create_batches_rnd` crops from memory instead of re-reading from disk.
  - The random number calls (`randint`, `uniform`, per-sample `randint`) and the
    crop/pad logic are unchanged, so batches are identical.
  - RAM cost ≈ 0.8 GB (≈550 files × ~4.2 s × 44.1 kHz × 8 bytes).
- Fixed the stereo warning message, which indexed the list as `wav_lst[...]`
  (would have raised an error with a DataFrame). Only printed for stereo
  files; the normalised NIPS4B files are mono.

#### `evaluate_metrics.py`
- Frames are built in one `signal.unfold(0, wlen, wshift)` call instead of a
  per-frame Python loop. The frame set is exactly the same as the original
  `while end_samp < len` loop: the strict `<` is kept, and `pout` keeps
  `N_fr+1` rows, with any uncovered row left at 0 as before.
- Whole-file scores are computed **once per unique file** and reused for each
  tag row of that file. Before, each row re-ran the identical forward passes.
- Eval batch size `Batch_dev` 128 → 1024. In eval mode, frames are scored
  independently (BatchNorm uses running stats), so this only changes speed.

### Not changed (deliberately, to keep reproduction faithful)
- No mixed precision (fp16 AMP), no `cudnn.benchmark`. Both could speed things
  up further but change floating-point numerics.
- Model, cfg files, hyper-parameters, RNG seeding, data lists, and the
  whole-file scoring rule in evaluation are unchanged.

### Verification
An equivalence test ran the original (git `HEAD`) and modified scripts side by side
(CPU, synthetic audio, a small stand-in for SincNet's `dnn_models`/`data_io`):
- `call_id.py`: training loss/error identical at every epoch; `model_raw.pkl`
  **bitwise identical**.
- `evaluate_metrics.py` on the same checkpoint: identical predictions, max
  per-row score difference 0.0. This included a file whose length triggers the
  leftover zero row in `pout`.
- Not yet run on the real data / GPU. On GPU, cuDNN may still give tiny
  run-to-run float differences, as it did with the original code.

---

## 2026-10-08 — Merge speed-up commit onto origin/master (752e6a6)

Rebasing the speed-up commit onto `origin/master` conflicted with remote commit
752e6a6, which added `fact_amp` from the cfg, `time.res` timing, and "Mean Exp"
metrics via `metrics_utils.py`.

### Resolution
- `call_id.py`: kept both sides. Training calls
  `create_batches_rnd(..., wav_data_tr, ..., fact_amp)` (preloaded data plus
  amplitude factor from the cfg). `time.res` logging and the checkpoint
  schedule comment are both kept.
- `evaluate_metrics.py`: the remote "Mean Exp" line used `count_fr_tot`, which
  the vectorised loop no longer defines. It would have raised `NameError`, and
  for repeated files it would have read a stale `pout`. Mean Exp is now cached
  per unique file next to the summed score, as
  `torch.exp(pout[:n_frames]).mean(dim=0)`. `n_frames` equals the original
  `count_fr_tot`.

### Verification
Same CPU equivalence test, now against 752e6a6 as the baseline: training
losses identical every epoch, `model_raw.pkl` bitwise identical, and
`y_true / y_pred / y_score / y_score_exp` in `predictions.npz` identical
(max diff 0.0).

Note: `CHANGES_LOG.md` matches `*.md` in `.gitignore`, so git does not track it
unless force-added.
