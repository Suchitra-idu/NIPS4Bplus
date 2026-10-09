# Change log

Record of code/config changes made to the authors' released code, with the
reason for each. Per CLAUDE.md, changes here must be authors' documented
settings or genuine blockers, never tweaks aimed at nudging numbers.

## 2026-10-07 — `fact_amp` set to 0 for the enhanced SincNet models

**Source:** Supplementary Information 1, Table S2 ("Training parameters for
enhanced SincNet models") lists `fact_amp = 0` for All Classes, Bird Classes
and Bird Species. The supplement also states `fact_amp` is hard-coded in the
SincNet code and needed a code modification to change.

**Before:** `call_id.py` passed a hard-coded `0.2` to `create_batches_rnd`,
i.e. random per-sample amplitude scaling in [0.8, 1.2] during training. The
earlier checkpoints were trained this way (since deleted, see below).

**Change:**
- `call_id.py`: `fact_amp` is now read from an optional `fact_amp` key in the
  cfg's `[windowing]` section, with a fallback of `0.2` (the released
  behaviour) when the key is absent. The call to `create_batches_rnd` uses it.
- `mod_cfg/mod_nips4bplus_{all_classes,bird_classes,bird_species}.cfg`: added
  `fact_amp=0` under `[windowing]`.
- `cfg/` (default/Part I settings) is untouched, so it keeps `0.2` via the
  fallback (Table S1 lists `0.2 (default hard coded)`).

**Effect on existing results:** the earlier checkpoints were trained with 0.2,
so they were not comparable to the paper's enhanced models, in addition to
being orphaned. Fresh training is required (and is what the Colab runs do).

**Other findings from the supplement (no code change):** all other `mod_cfg`
parameters already match Table S2 exactly (cw_len, filters, pooling, FC sizes,
dropout 0.0, lr, batch size, epochs, batches, seed). This rules out dropout
as a gap hypothesis. (`cfg/` vs Table S1 was checked afterwards, see below.)

## 2026-10-07 — Non-GPU prep (reproducibility, timing, extra metrics)

None of these change training behaviour or model results.

- **Seeded splits.** `generate_mod_file_lists.py` and `generate_file_lists.py`
  take an optional 4th argument `split_seed` (default 1234) passed as
  `random_state` to every `train_test_split`. Use a different seed per run for
  Part I's 5 different splits. Also wrapped the annotation `glob` in `sorted()`
  so label integers don't depend on filesystem order (matters on Colab).
  Verified: same seed gives byte-identical lists, a different seed differs;
  resulting train sizes (4110 / 3959 / 3959) match Supplementary S4–S6.
  The tracked `mod_data_lists/` are NOT regenerated and remain the canonical
  split for the three enhanced runs (they come from git, so Colab gets the same
  ones). Always keep the lists with the checkpoint they trained.
- **Training time.** `call_id.py` appends `epoch N, elapsed_s=...` to
  `<output_folder>/time.res` at every evaluation epoch (includes eval time).
- **Extra metrics.** New `metrics_utils.py` (no torch; checked ad hoc on
  synthetic data, no test file kept in the repo) computes Table S7's F1, weighted FPR/FNR and "ROC AUC Mean
  Exp" alongside the earlier metrics. `evaluate_metrics.py` uses it, and now
  also saves `predictions.npz` (y_true, y_pred, y_score, y_score_exp) for the
  confusion matrix and ROC plots. `evaluate_metrics.py` itself still needs a
  GPU run to be tested end to end.
- **Checked, no change:** `cfg/` matches Supplementary Table S1 (TIMIT column):
  cw_len 10, filters 80,60,60 / 251,5,5, pool 3,3,3, layer norm on, FC
  2048×3, N_batches 800, N_epochs 200. `fact_amp` absent, so it falls back to
  0.2 as S1 lists.

## 2026-10-07 — Old checkpoints deleted

Deleted `output/bird_classes`, `output/bird_classes_fixed (with norm)` and
`output/bird_species` (the last was only a smoke-test run, epoch 0). All three
used `fact_amp=0.2` and, for the first two, a train/test list that no longer
exists. No results from them are used anywhere. `output/` is now empty and is
gitignored.

## 2026-10-07 — `N_eval_epoch` 8 -> 57 for the enhanced runs (speed, not accuracy)

**Why:** the first Colab run (T4, `bird_species`) logged in `time.res`:
epoch 0 = 315.6 s, epoch 8 = 740.4 s, epoch 16 = 1163.0 s. Solving gives about
15 s per training epoch (similar to the paper's 1.9 h for 400 epochs) but about
300 s per full evaluation, i.e. evaluation was about 71% of the run time:
50 evaluations x 300 s + 400 x 15 s is about 5.9 h per run.

**Tried and rejected:** vectorising the evaluation windowing (replacing the
per-window Python copy loop with one batched `unfold`). It was bit-identical to
the original (max difference 0.0, including the edge case where the last
`pout` row stays zero) but only 1.4x faster, and batch sizes 128/512/2048 made no
difference. Evaluation is GPU-compute bound (about 5,000 frames per file x 1,321
files). The helper was deleted; `call_id.py` evaluation code is unchanged.

**Change:** `N_eval_epoch=57` in `mod_cfg/mod_nips4bplus_*.cfg`. 57 divides 399,
so evaluations (and checkpoint saves) happen at epochs 0, 57, ..., 399: the
final saved model is the real epoch-399 model (with 8 it would have been
epoch 392). Expected time about 2.4 h per run. Training is unaffected:
evaluation runs under `no_grad` in eval mode and does not use the RNG, so the
training trajectory is the same as with `N_eval_epoch=8`; only the density of
the `res.res` curve changes. `cfg/` (Part I, where the accuracy-over-epochs
curve for Fig. 1 matters) keeps `N_eval_epoch=8`.
