import json, glob
from _common import *
from sklearn.metrics import top_k_accuracy_score
R = {}
for s_ in SETS:   # rebuilt from the saved re-scored arrays written by 08_scoring_diagnostics.py
    z = np.load(os.path.join(DATA, f"rescored_{s_}.npz")); y = z["y"]; n = z["A"].shape[1]
    R[s_] = {k: dict(acc=float((z[k].argmax(1) == y).mean()), top3=float(top_k_accuracy_score(y, z[k], k=3, labels=list(range(n)))),
                     top5=float(top_k_accuracy_score(y, z[k], k=5, labels=list(range(n))))) for k in "ABCD"}
json.dump(R, open(os.path.join(DATA, "scoring_diagnostics.json"), "w"), indent=1)
paper = dict(all_classes=(.7301, .8993, .9329), bird_classes=(.7447, .8970, .9327), bird_species=(.7356, .9023, .9447))
rules = [("A", "whole file\nΣ log p\n(shipped)"), ("B", "whole file\nmean p"), ("C", "tag span\nΣ log p"), ("D", "tag span\nmean p")]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ai, (m, t) in enumerate((("acc", "accuracy"), ("top3", "top-3"), ("top5", "top-5"))):
    ax = axes[ai]; x = np.arange(4); w = 0.26
    for j, s in enumerate(SETS):
        v = [R[s][k][m] for k, _ in rules]
        ax.bar(x + (j - 1) * w, v, w * 0.92, color=COLORS[s], label=s.replace("_", " "))
        ax.hlines(paper[s][ai], 3 + (j - 1) * w - w / 2.2, 3 + (j - 1) * w + w / 2.2, color=INK, lw=2.2, label="paper" if (j == 0 and ai == 0) else None)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in rules], fontsize=7.5); ax.set_title(t, fontsize=10, loc="left"); ax.set_ylim(0.4, 1.05)
axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2); axes[0].set_ylabel("value")
savefig("evaluation", "scoring_rule_diagnostic.png")
