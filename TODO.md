# TODO — reproducing the SincNet / NIPS4Bplus paper

Scope: SincNet only. Waveform+CNN and pretrained-model baselines are out of
scope. Rule: no code changes aimed at matching numbers (see CLAUDE.md); log
any change in `logs.md`.

## 0. Prep
- [x] Check `cfg/` (default/Part I) against Supplementary Table S1 — matches the TIMIT column.
- [x] Make the split reproducible: `split_seed` argument added (default 1234). Tracked `mod_data_lists/` stay the canonical enhanced split; still archive the lists with each run's output.
- [x] Log wall-clock training time per run: `time.res` in each output folder.
- [x] Extend `evaluate_metrics.py` with F1, FPR, FNR and "ROC AUC Mean Exp" (Table S7); also saves `predictions.npz`. Needs a first GPU run to confirm end to end.
- [x] Old checkpoints deleted.

- [x] Colab setup: cfg paths rewritten with `sed`, outputs on Drive, norm files from a zip on Drive.

## 1. Table 1, SincNet rows (enhanced `mod_cfg`)
- [ ] Train `all_classes` (87), 400 epochs.
- [ ] Train `bird_classes` (77), 400 epochs.
- [ ] Train `bird_species` (51), 400 epochs.
- [ ] Evaluate each with `evaluate_metrics.py`: accuracy, ROC AUC, precision, recall, top-3, top-5, params, time.
- [ ] Repeat with several seeds/splits per class set; report mean, spread and best run.
- [ ] Compare with Table 1 / Table S7 and record the gap and hypotheses in `REPRODUCTION_REPORT.md`.

## 2. Part I, default settings (Fig. 1 and the text claims)
- [ ] Train default `cfg/` models: 5 runs per class set, 200 epochs, different random splits.
- [ ] Check: about 60% mean accuracy, 75.6% mean ROC AUC (30 runs), about 3.5 h per run.
- [ ] Check: accuracy above 65% within the first few epochs.
- [ ] Script to plot per-run and mean accuracy curves from `res.res` (Fig. 1).
- [ ] Optional: TIMIT/LibriSpeech default configs against Table S8.

## 3. Figures from trained models
- [ ] Fig. 2: confusion matrix over the 87 classes (save predictions, then plot).
- [ ] Fig. 4: learned sinc filters, time and frequency domain, from a checkpoint.
- [ ] Fig. S1–S3: ROC curves for the three class sets.
- [ ] Fig. 5 and dataset stats: check `dataset_analysis/` against 5,478 tags, 87 classes, 17m56s tagged audio (13m8s birds only), file lengths.

## 4. Discussion claims
- [ ] Overlap analysis: more than 20% of tags overlap another species; about 60:40 split of predictions between tagged and overlapping species; about +4% if either species counts.
- [ ] Confirm trainable parameters of about 2.5M.

## 5. Optional extras the paper describes
- [ ] Hyperparameter search over the ranges in Table S9.
- [ ] Transfer learning from TIMIT/LibriSpeech, fine-tuning only the last layer.
- [ ] AM-softmax (the paper reports no improvement).

## 6. Wrap-up
- [ ] Final results table (ours vs paper) in `REPRODUCTION_REPORT.md`.
- [ ] Note in the report that the baseline comparison is unchecked.
