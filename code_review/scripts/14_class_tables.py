"""Per-class / per-group breakdowns of the saved predictions (CPU only)."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from _common import *

sp = pd.read_csv(os.path.join(ROOT, "raw", "nips4b_labels", "nips4b_birdchallenge_espece_list.csv")).set_index("class name")
rng = np.random.RandomState(0)
OUT = {}
for s in SETS:
    P = np.load(os.path.join(ROOT, "output", s, "predictions.npz")); yt, yp, ye = P["y_true"], P["y_pred"], P["y_score_exp"]
    tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
    n = int(yt.max()) + 1; names = class_names(s)
    cm = confusion_matrix(yt, yp, labels=list(range(n)))
    tp = np.diag(cm); rec = tp / np.maximum(cm.sum(1), 1); prec = tp / np.maximum(cm.sum(0), 1)
    rows = []
    for c in range(n):
        off = cm[c].copy(); off[c] = 0; j = int(off.argmax())
        rows.append(dict(label=c, name=names[c], train=int((tr.label == c).sum()), test=int(cm[c].sum()), recall=round(float(rec[c]), 3), precision=round(float(prec[c]), 3),
                         predicted=int(cm[:, c].sum()), top_confusion=names[j] if off[j] else "", top_confusion_n=int(off[j])))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(DATA, f"per_class_{s}.csv"), index=False)
    # group breakdowns (taxon, call type, duration bin, #labels in file)
    te = te.assign(correct=(yp == yt), correct_exp=(ye.argmax(1) == yt))
    te["call_type"] = te.class_name.str.split("_").str[-1]
    nl = te.groupby("file").label.nunique(); te["n_labels_file"] = te.file.map(nl).clip(upper=3)
    te["dur_bin"] = pd.cut(te.length, [0, 0.03, 0.06, 0.1, 0.2, 0.5, 6], labels=["<30ms", "30-60", "60-100", "100-200", "200-500", ">500ms"])
    g = {}
    for col in ["type", "call_type", "dur_bin", "n_labels_file"]:
        t = te.groupby(col, observed=True).agg(n=("correct", "size"), acc=("correct", "mean"), acc_mean_p=("correct_exp", "mean")).round(3)
        g[col] = {str(k): dict(v) for k, v in t.to_dict("index").items()}
    # cluster bootstrap (files) for accuracy
    files = te.file.unique(); idx = {f: np.where(te.file.values == f)[0] for f in files}
    accs = []
    for _ in range(2000):
        pick = rng.choice(files, len(files)); ii = np.concatenate([idx[f] for f in pick]); accs.append(te.correct.values[ii].mean())
    ci = np.percentile(accs, [2.5, 97.5])
    # top confusions overall
    off = cm.copy(); np.fill_diagonal(off, 0); flat = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:12], off.shape))[0]
    top = [dict(true=names[a], pred=names[b], n=int(off[a, b])) for a, b in flat if off[a, b]]
    OUT[s] = dict(groups=g, acc=float(te.correct.mean()), acc_ci95=[float(ci[0]), float(ci[1])], top_confusions=top,
                  n_classes_recall0=int((rec == 0).sum()), macro_recall=float(rec.mean()), macro_precision=float(prec[cm.sum(0) > 0].mean()),
                  classes_never_predicted=int((cm.sum(0) == 0).sum()))
    # figure: recall by group
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for a, col, ttl in zip(ax, ["call_type", "dur_bin", "n_labels_file"], ["Call type", "Event length", "Labels in file"]):
        t = te.groupby(col, observed=True).correct.agg(["mean", "size"]); a.bar(range(len(t)), t["mean"], color=COLORS[s], width=0.65)
        for i, (m, k) in enumerate(zip(t["mean"], t["size"])): a.text(i, m + 0.01, f"n={k}", ha="center", fontsize=7)
        a.set_xticks(range(len(t))); a.set_xticklabels([str(x) for x in t.index], fontsize=8); a.set_ylim(0, 1); a.set_title(ttl, fontsize=10, loc="left")
    ax[0].set_ylabel("accuracy")
    savefig("evaluation", f"accuracy_by_group_{s}.png")
    print(s, json.dumps({k: v for k, v in OUT[s].items() if k != "top_confusions"}, indent=0)[:900])
json.dump(OUT, open(os.path.join(DATA, "group_breakdowns.json"), "w"), indent=1)
