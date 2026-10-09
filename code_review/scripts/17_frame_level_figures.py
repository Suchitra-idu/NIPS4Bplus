import json
import numpy as np, pandas as pd
from _common import *
EX = np.load(os.path.join(DATA, "frame_examples.npy"), allow_pickle=True)[0]
R = json.load(open(os.path.join(DATA, "frame_level.json")))
ttl = {"fixed_by_span": "A wrong, tag-span right", "correct_both": "both right", "wrong_both": "both wrong"}
fig, axes = plt.subplots(3, 1, figsize=(10.5, 6.6), sharex=True)
meta = {}
for ax, key in zip(axes, ["fixed_by_span", "correct_both", "wrong_both"]):
    e = EX[("bird_species", key)]; meta[key] = dict(file=e["file"], label=e["label"], start=e["start"], length=e["length"])
    ax.plot(e["cen"], e["p_top"], color="#c9c9c9", lw=0.8, label="top-1 prob")
    ax.plot(e["cen"], e["p_true"], color="#1baf7a", lw=1.0, label="true class prob")
    ax.axvspan(e["start"], e["start"] + e["length"], color="#2a78d6", alpha=0.25, lw=0)
    for a, b, c in e["others"]:
        if abs(a - e["start"]) > 1e-6: ax.axvspan(a, a + b, color="#eb6834", alpha=0.10, lw=0)
    ax.set_ylim(0, 1.05); ax.set_ylabel("prob"); ax.set_title(f"{ttl[key]} — {e['file'][-11:-4]}", fontsize=9.5, loc="left")
axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right"); axes[-1].set_xlabel("frame centre (s)")
savefig("evaluation", "frame_posterior_timelines.png")
json.dump(meta, open(os.path.join(DATA, "frame_examples_meta.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for s in SETS:
    d = np.load(os.path.join(DATA, f"frame_level_{s}.npz"))
    ax[0].hist(np.log10(d["margin"]), bins=30, histtype="step", color=COLORS[s], lw=1.6, label=s.replace("_", " "))
    m = d["n_in"] > 0; ax[1].scatter(d["in_frame_acc"][m] + np.random.RandomState(0).uniform(-.01, .01, m.sum()), d["A_correct"][m] + np.random.RandomState(1).uniform(-.04, .04, m.sum()), s=4, color=COLORS[s], alpha=0.25)
    tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); n = len(d["bg_hist"]); prior = tr.label.value_counts().reindex(range(n)).fillna(0).values; prior = prior / prior.sum()
    ax[2].scatter(prior, d["bg_hist"] / d["bg_hist"].sum(), s=10, color=COLORS[s], alpha=0.8)
ax[0].set_xlabel("log10 margin (nats)"); ax[0].set_ylabel("test rows"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Top-1 − top-2 of Σlog p", fontsize=10, loc="left")
ax[1].set_xlabel("frame accuracy inside the tag"); ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["file wrong", "file right"]); ax[1].set_title("Frames vs file decision", fontsize=10, loc="left")
ax[2].plot([0, 0.12], [0, 0.12], color="#9a9a9a", lw=0.8); ax[2].set_xlabel("class share of train tags"); ax[2].set_ylabel("share of background frames"); ax[2].set_title("What background is called", fontsize=10, loc="left")
savefig("evaluation", "frame_level_summary.png")
print(meta)
