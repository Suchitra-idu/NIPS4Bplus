"""Audit of the split lists and of call_id.create_batches_rnd's window sampler."""
import json
import numpy as np, pandas as pd, soundfile as sf
from _common import *

R = {}
norm = os.path.join(ROOT, "nips4b_norm_files")
files = sorted(os.listdir(norm))
info = {f: sf.info(os.path.join(norm, f)) for f in files}
nsamp = {f: info[f].frames for f in files}

# --- normalisation: overshoot over +-1 (max vs max|x|)
over = []
for f in files:
    x = sf.read(os.path.join(norm, f))[0]
    over.append((np.abs(x).max(), x.max(), x.min()))
over = np.array(over)
R["normalisation"] = dict(n_files=len(files), n_peak_gt_1=int((over[:, 0] > 1.0 + 1e-9).sum()), max_peak=float(over[:, 0].max()),
                          frac_peak_gt_1=float((over[:, 0] > 1.0 + 1e-9).mean()), n_pos_peak_le_0=int((over[:, 1] <= 0).sum()))
# per-file RMS after normalisation (what the network sees, since fact_amp=0 and no input norm)
rms = []
for f in files:
    x = sf.read(os.path.join(norm, f))[0]; rms.append(np.sqrt(np.mean(x ** 2)))
rms = np.array(rms); R["normalisation"].update(rms_p5=float(np.percentile(rms, 5)), rms_median=float(np.median(rms)), rms_p95=float(np.percentile(rms, 95)),
                                              rms_p95_over_p5=float(np.percentile(rms, 95) / np.percentile(rms, 5)))

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax[0].hist(over[:, 0], bins=40, color="#2a78d6"); ax[0].axvline(1, color=INK, ls="--", lw=1)
ax[0].set_xlabel("max |x| after 'normalisation'"); ax[0].set_title("Peak after x / |max(x)|  (expected ≤ 1)", fontsize=10, loc="left")
ax[1].hist(np.log10(rms), bins=40, color="#eb6834"); ax[1].set_xlabel("log10 RMS of normalised file"); ax[1].set_title("Residual loudness spread the net must absorb", fontsize=10, loc="left")
plt.close()  # superseded by 06_normalisation_clipping.py

# --- per-set audits
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
for j, s in enumerate(SETS):
    tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
    opt = load_options(CFG[s]); fs = 44100; wlen = int(fs * int(opt.cw_len) / 1000.0)
    allr = pd.concat([tr, te])
    out = dict(wlen=wlen, n_train=len(tr), n_test=len(te), n_classes=int(allr.label.nunique()), labels_contiguous=bool(sorted(allr.label.unique()) == list(range(allr.label.nunique()))))
    # split purity per class
    ct = tr.label.value_counts().reindex(range(out["n_classes"])).fillna(0); ce = te.label.value_counts().reindex(range(out["n_classes"])).fillna(0)
    out["test_share_min"] = float((ce / (ct + ce)).min()); out["test_share_max"] = float((ce / (ct + ce)).max())
    out["min_train_per_class"] = int(ct.min()); out["min_test_per_class"] = int(ce.min())
    # leakage + duplicated rows
    out["unique_test_files"] = int(te.file.nunique()); out["test_files_in_train"] = int(te.file.drop_duplicates().isin(set(tr.file)).sum())
    out["rows_dupe_across_split"] = int(te.merge(tr, on=["file", "start", "length"]).shape[0])
    # time-overlap between a test tag and train tags of the same file (window could contain train audio)
    ov = 0
    trg = tr.groupby("file")
    for r in te.itertuples():
        if r.file in trg.groups:
            g = trg.get_group(r.file)
            if ((g.start < r.start + r.length) & (g.start + g.length > r.start)).any(): ov += 1
    out["test_rows_time_overlapping_a_train_tag"] = ov
    # window geometry: branch and boundary behaviour for training rows
    ev = tr.length.values * fs
    out["frac_events_longer_than_wlen"] = float((np.floor(tr.length.values * fs).astype(int) > wlen).mean())
    crash = 0; short_window = 0; hi_excl = 0; lo_gt_hi = 0; edge_single = 0
    for r in tr.itertuples():
        n = nsamp[r.file]; t0 = int(r.start * fs); t1 = t0 + int(r.length * fs)
        if t1 - t0 > wlen:
            if t1 - wlen <= t0: crash += 1
            if t1 - wlen + wlen > n: short_window += 1
        else:
            lo, hi = max(0, t1 - wlen), min(t0, n - wlen)
            if lo > hi: lo_gt_hi += 1
            elif lo == hi: edge_single += 1
            else: hi_excl += 1
    out.update(sampler_crash_equal_bounds=crash, sampler_window_past_eof=short_window, sampler_lo_gt_hi_valueerror=lo_gt_hi,
               sampler_single_position=edge_single, sampler_random_branch=hi_excl)
    # event beyond EOF
    out["train_events_end_after_eof"] = int(sum((int(r.start * fs) + int(r.length * fs)) > nsamp[r.file] for r in tr.itertuples()))
    # sampling mass: tags are drawn uniformly, so class draw prob == class share
    out["max_over_min_class_share_train"] = float(ct.max() / ct.min())
    # how many frames / how much audio is "event" for the file-level decision
    fl = np.array([nsamp[f] / fs for f in te.file]); frac = te.length.values / fl
    out["test_event_fraction_of_file_median"] = float(np.median(frac)); out["test_event_fraction_p90"] = float(np.percentile(frac, 90))
    out["test_events_gt_50pct_of_file"] = float((frac > 0.5).mean())
    R[s] = out
    axes[j].hist(np.clip(frac, 0, 1), bins=40, color=COLORS[s]); axes[j].set_xlabel("event / file length")
    axes[j].set_title(f"{s.replace('_',' ')} (median {np.median(frac):.3f})", fontsize=10, loc="left")
axes[0].set_ylabel("test rows")
savefig("data", "event_fraction_of_scored_file.png")

# --- label ambiguity per test file
fig, ax = plt.subplots(figsize=(5.8, 3.4)); w = 0.26
for j, s in enumerate(SETS):
    te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
    g = te.groupby("file").label.nunique(); R[s]["test_files_with_ge2_distinct_labels"] = int((g >= 2).sum())
    R[s]["test_rows_in_multilabel_files"] = float(te.file.map(g).ge(2).mean())
    cnt = g.value_counts().reindex(range(1, 6)).fillna(0)
    ax.bar(np.arange(1, 6) + (j - 1) * w, cnt.values, w * 0.92, color=COLORS[s], label=s.replace("_", " "))
ax.set_yscale("log"); ax.set_xlabel("distinct labels in one file"); ax.set_ylabel("test files"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Labels per test file", fontsize=10, loc="left")
savefig("data", "multilabel_test_files.png")
json.dump(R, open(os.path.join(DATA, "data_audit.json"), "w"), indent=1); print(json.dumps(R, indent=1))
