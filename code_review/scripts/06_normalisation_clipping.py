"""The x/|max(x)| normalisation + sf.write(PCM_16) => hard clipping when |min| > max."""
import json
import numpy as np, soundfile as sf
from _common import *
raw = os.path.join(ROOT, "raw", "nips4b_wav", "train"); norm = os.path.join(ROOT, "nips4b_norm_files")
rows = []
for f in sorted(os.listdir(norm)):
    x0 = sf.read(os.path.join(raw, f))[0]; x1 = sf.read(os.path.join(norm, f))[0]
    ratio = np.abs(x0).max() / x0.max()          # would-be peak after authors' division
    y = x0 / x0.max()
    n_clip = int((np.abs(y) > 1.0 + 1e-9).sum())   # samples the authors' division pushes past full scale
    rows.append((ratio, n_clip, x0.max(), x0.min(), n_clip / len(y)))
rows = np.array(rows)
sub = sf.info(os.path.join(norm, "nips4b_birds_trainfile001.wav")).subtype
R = dict(output_subtype=sub, n_files=len(rows), n_files_would_exceed_1=int((rows[:, 0] > 1 + 1e-9).sum()), max_overshoot=float(rows[:, 0].max()),
         n_files_with_clipped_samples=int((rows[:, 1] > 0).sum()), max_clipped_samples=int(rows[:, 1].max()), median_clipped_when_clipped=float(np.median(rows[rows[:, 1] > 0, 1])), max_clipped_fraction=float(rows[:, 4].max()), median_clipped_fraction_when_clipped=float(np.median(rows[rows[:, 1] > 0, 4])))
json.dump(R, open(os.path.join(DATA, "normalisation_clipping.json"), "w"), indent=1); print(R)
m = rows[:, 0] > 1 + 1e-9
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].hist(rows[m, 0], bins=30, color="#2a78d6"); ax[0].set_xlabel("peak before clipping (× full scale)"); ax[0].set_ylabel("files")
ax[0].set_title(f"{m.sum()} of 687 files overshoot", fontsize=10, loc="left")
ax[1].scatter(rows[m, 0], rows[m, 1], s=10, color="#eb6834"); ax[1].set_yscale("log"); ax[1].set_xlabel("peak before clipping (× full scale)"); ax[1].set_ylabel("clipped samples")
ax[1].set_title("Samples lost per file", fontsize=10, loc="left")
savefig("data", "normalisation_clipping.png")
