"""Training-sampler statistics + a fully worked example on one real recording."""
import json
import numpy as np, pandas as pd, soundfile as sf
from _common import *

R = {}
fs = 44100
# ---- 1. simulated training draws (100k per set): class histogram vs class share; window composition; label noise
allann = {}
adir = os.path.join(ROOT, "raw", "nips4bplus_csv", "annotations", "temporal_annotations_nips4b")
for f in os.listdir(adir):
    try: a = pd.read_csv(os.path.join(adir, f), header=None)
    except pd.errors.EmptyDataError: continue
    allann["nips4b_birds_trainfile" + f[-7:-4] + ".wav"] = a.values
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for j, s in enumerate(SETS):
    opt = load_options(CFG[s]); wlen = int(fs * int(opt.cw_len) / 1000.0)
    tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); n = len(tr); C = tr.label.nunique()
    rng = np.random.RandomState(0); N = 100000; k = rng.randint(n, size=N)
    draw = np.bincount(tr.label.values[k], minlength=C) / N; share = tr.label.value_counts().reindex(range(C)).values / n
    # per-batch class richness (128 draws)
    cls_per_batch = [len(set(tr.label.values[rng.randint(n, size=128)])) for _ in range(500)]
    # label noise: fraction of sampled windows overlapped (in time) by another tag with a different class_name
    noisy = 0; noisy_any = 0; ovl_frac = []
    for i in rng.randint(n, size=5000):
        r = tr.iloc[i]; t0 = r.start; t1 = r.start + r.length
        if r.length * fs > wlen: s0 = rng.uniform(t0, t1 - wlen / fs)
        else: s0 = max(0, t1 - wlen / fs)   # deterministic buffered start (lower bound)
        s1 = s0 + wlen / fs
        ov = 0; anyov = 0
        for st, ln, lab in allann[r.file]:
            if abs(st - r.start) < 1e-9 and abs(ln - r.length) < 1e-9: continue
            if st < s1 and st + ln > s0:
                anyov += 1
                if lab != r.class_name: ov += 1
        noisy += ov > 0; noisy_any += anyov > 0
    R[s] = dict(wlen=wlen, window_ms=wlen / fs * 1000, max_draw_over_share_dev=float(np.abs(draw - share).max()),
                classes_per_batch_mean=float(np.mean(cls_per_batch)), classes_per_batch_min=int(np.min(cls_per_batch)),
                frac_windows_overlapped_by_other_class=noisy / 5000, frac_windows_overlapped_by_any_other_tag=noisy_any / 5000,
                expected_draws_rarest_class_per_epoch=float(share.min() * 128 * 80), expected_draws_commonest_class_per_epoch=float(share.max() * 128 * 80),
                draws_per_tag_per_run=float(128 * 80 * 400 / n))
    ax = axes[j]; ax.scatter(share, draw, s=12, color=COLORS[s]); ax.plot([0, share.max()], [0, share.max()], color="#9a9a9a", lw=0.8)
    ax.set_title(s.replace("_", " "), fontsize=10, loc="left"); ax.set_xlabel("class share of train tags")
axes[0].set_ylabel("share of drawn windows")
savefig("data", "sampler_class_share.png")

# ---- 2. worked example: trainfile007 (7 tags, 5 species, a Human tag)
fname = "nips4b_birds_trainfile007.wav"
x, _ = sf.read(os.path.join(ROOT, "nips4b_norm_files", fname)); T = len(x) / fs
ann = pd.read_csv(os.path.join(adir, "annotation_train007.csv"), header=None, names=["start", "length", "label"])
sp = pd.read_csv(os.path.join(ROOT, "raw", "nips4b_labels", "nips4b_birdchallenge_espece_list.csv")).set_index("class name")
ann["in_lists"] = ann.label.isin(sp.index)
ann.to_csv(os.path.join(DATA, "example_trainfile007_annotation.csv"), index=False)
sets = {}
for s in SETS:
    tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
    sets[s] = (tr[tr.file == fname][["class_name", "start", "length", "label"]].assign(split="train"), te[te.file == fname][["class_name", "start", "length", "label"]].assign(split="test"))
pd.concat([pd.concat(v).assign(set=s) for s, v in sets.items()]).to_csv(os.path.join(DATA, "example_trainfile007_rows.csv"), index=False)
opt = load_options(CFG["bird_species"]); wlen = 705; wshift = 44
rng = np.random.RandomState(3)
fig, ax = plt.subplots(2, 1, figsize=(10.5, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
t = np.arange(len(x)) / fs
ax[0].plot(t, x, color="#52514e", lw=0.4)
pal = {"Parate_song": "#2a78d6", "Gargla_call": "#eb6834", "Fricoe_song": "#1baf7a", "Turmer_song": "#e87ba4", "Human": "#9a9a9a"}
for r in ann.itertuples():
    c = pal.get(r.label, "#eda100"); ax[0].axvspan(r.start, r.start + r.length, color=c, alpha=0.30, lw=0)
    ax[1].barh(r.Index, r.length, left=r.start, color=c, height=0.6); ax[1].text(r.start + r.length + 0.03, r.Index, r.label, va="center", fontsize=7)
ax[1].set_yticks([]); ax[1].set_xlabel("time (s)"); ax[0].set_ylabel("amplitude")
tr = sets["bird_species"][0]
for _ in range(6):                      # sampled training windows
    r = tr.iloc[rng.randint(len(tr))] if len(tr) else None
    if r is None: break
    t0 = int(r.start * fs); t1 = t0 + int(r.length * fs)
    s0 = rng.randint(t0, t1 - wlen) if t1 - t0 > wlen else max(0, t1 - wlen)
    ax[0].plot([s0 / fs, (s0 + wlen) / fs], [1.12, 1.12], color=INK, lw=3, solid_capstyle="butt")
ax[0].set_ylim(-1.1, 1.25); ax[0].set_title("trainfile007: tags (colour), sampled 16 ms training windows (black bars)", fontsize=10, loc="left")
savefig("data", "worked_example_trainfile007.png")
R["example"] = dict(duration_s=T, n_samples=len(x), n_ann_rows=len(ann), n_rows_dropped=int((~ann.in_lists).sum()),
                    dense_frames=int(np.ceil((len(x) - wlen) / wshift)), peak=float(np.abs(x).max()), rms=float(np.sqrt((x ** 2).mean())))
json.dump(R, open(os.path.join(DATA, "sampler_stats.json"), "w"), indent=1); print(json.dumps(R, indent=1)); print(ann.to_string()); 
for s,(a,b) in sets.items(): print(s); print(pd.concat([a,b]).to_string())
