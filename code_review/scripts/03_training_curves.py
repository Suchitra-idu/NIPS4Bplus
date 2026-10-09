"""Training dynamics from output/*.log, res.res and time.res."""
import json, re
import numpy as np
from _common import *

def parse_log(p):
    ep, loss, err = [], [], []
    for l in open(p):
        m = re.match(r"epoch (\d+), loss_tr=([\d.]+) err_tr=([\d.]+)", l)
        if m: ep.append(int(m[1])); loss.append(float(m[2])); err.append(float(m[3]))
    return np.array(ep), np.array(loss), np.array(err)

def parse_time(p):
    t = [(int(a), float(b)) for a, b in re.findall(r"epoch (\d+), elapsed_s=([\d.]+)", open(p).read())]
    return np.array(t)

summary = {}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for s in SETS:
    ep, loss, err = parse_log(os.path.join(ROOT, "output", f"{s}.log"))
    t = parse_time(os.path.join(ROOT, "output", s, "time.res"))
    met = dict(l.strip().split("=") for l in open(os.path.join(ROOT, "output", s, "metrics.res")))
    k = 10; sm = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
    axes[0].plot(ep, loss, color=COLORS[s], alpha=0.25, lw=0.8); axes[0].plot(ep[k-1:], sm(loss), color=COLORS[s], lw=1.6, label=s.replace("_", " "))
    axes[1].plot(ep, 1 - err, color=COLORS[s], alpha=0.25, lw=0.8); axes[1].plot(ep[k-1:], 1 - sm(err), color=COLORS[s], lw=1.6)
    axes[1].scatter([392 + 8 * (SETS.index(s) - 1)], [float(met["accuracy"])], color=COLORS[s], marker="D", s=40, zorder=5, edgecolor="white")
    axes[2].plot(t[:, 0], t[:, 1] / 60, color=COLORS[s], lw=1.6, marker="o", ms=2.5)
    summary[s] = dict(final_train_loss=float(loss[-10:].mean()), final_train_acc=float(1 - err[-10:].mean()),
                      test_acc=float(met["accuracy"]), gap=float(1 - err[-10:].mean() - float(met["accuracy"])),
                      first_epoch_ge_50pct_train_acc=int(ep[np.argmax(1 - err >= 0.5)]),
                      epoch_last_improve_10ep_avg=int(ep[k - 1 + np.argmin(sm(loss))]),
                      wall_clock_min=float(t[-1, 1] / 60), sec_per_epoch=float(np.polyfit(t[:, 0], t[:, 1], 1)[0]),
                      last_saved_epoch=int(t[-1, 0]), epochs_logged=int(len(ep)), epochs_lost_after_last_ckpt=int(ep[-1] - t[-1, 0]))
axes[0].set_title("Train loss", fontsize=10, loc="left"); axes[0].set_xlabel("epoch"); axes[0].set_yscale("log"); axes[0].legend(frameon=False, fontsize=8)
axes[1].set_title("Accuracy (◆ test)", fontsize=10, loc="left"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1)
axes[2].set_title("Elapsed minutes", fontsize=10, loc="left"); axes[2].set_xlabel("epoch")
savefig("training", "training_curves.png")

# generalisation gap bars
fig, ax = plt.subplots(figsize=(5.8, 3.4))
x = np.arange(3); w = 0.36
tr = [summary[s]["final_train_acc"] for s in SETS]; te = [summary[s]["test_acc"] for s in SETS]
ax.bar(x - w / 2, tr, w * 0.92, color="#c9c9c9", label="train")
ax.bar(x + w / 2, te, w * 0.92, color=[COLORS[s] for s in SETS], label="test")
for i in range(3):
    ax.text(x[i] - w / 2, tr[i] + 0.01, f"{tr[i]:.2f}", ha="center", fontsize=8); ax.text(x[i] + w / 2, te[i] + 0.01, f"{te[i]:.2f}", ha="center", fontsize=8)
paper = [0.7301, 0.7447, 0.7356]
ax.scatter(x + w / 2, paper, marker="_", s=900, color=INK, lw=2, label="paper", zorder=4)
ax.set_xticks(x); ax.set_xticklabels([s.replace("_", " ") for s in SETS]); ax.set_ylim(0, 1.0); ax.set_ylabel("accuracy")
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), markerscale=0.35, handlelength=1.4)
ax.set_title("Train vs test accuracy", fontsize=10, loc="left", pad=22)
savefig("training", "generalisation_gap.png")
json.dump(summary, open(os.path.join(DATA, "training_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
