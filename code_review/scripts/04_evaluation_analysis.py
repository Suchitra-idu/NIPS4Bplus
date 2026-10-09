"""Post-hoc analysis of output/*/predictions.npz (no GPU needed)."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, top_k_accuracy_score
from _common import *
from metrics_utils import compute_metrics

R = {}
for s in SETS:
    P = np.load(os.path.join(ROOT, "output", s, "predictions.npz"))
    yt, yp, ys, ye = P["y_true"], P["y_pred"], P["y_score"], P["y_score_exp"]
    te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
    tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0]))
    n = ys.shape[1]; names = class_names(s)
    assert (te.label.values == yt).all()
    m = compute_metrics(yt, yp, ys, ye, n)
    rec = {k: float(v) for k, v in m.items()}
    # saturation of the decision score
    mx = ys.max(1)
    sat = dict(frac_max_prob_ge_0_999999=float((mx >= 0.999999).mean()), frac_max_prob_eq_1=float((mx == 1.0).mean()),
               median_max_prob=float(np.median(mx)),
               median_n_classes_with_prob_gt_1em6=float(np.median((ys > 1e-6).sum(1))),
               median_n_exact_zero=float(np.median((ys == 0).sum(1))))
    # top-k from the *mean-exp* score (already saved) -> what top-3/5 looks like when scores are not saturated
    exp_topk = {k: float(top_k_accuracy_score(yt, ye, k=k, labels=list(range(n)))) for k in (1, 3, 5)}
    exp_auc = float(roc_auc_score(yt, ye, labels=list(range(n)), multi_class="ovr", average="weighted"))
    exp_acc = float((ye.argmax(1) == yt).mean())
    # agreement between sum-log argmax and mean-prob argmax
    agree = float((ye.argmax(1) == yp).mean())
    # ceiling: one prediction per file, many labels per file
    g = te.groupby("file").label.agg(lambda x: x.value_counts().iloc[0]).sum() / len(te)
    uniq_files = te.file.nunique()
    # leakage: test files that also carry train rows
    trf = set(tr.file); leak_files = sum(f in trf for f in te.file.unique())
    leak_rows = float(te.file.isin(trf).mean())
    # per-class
    sup = tr.label.value_counts().reindex(range(n)).fillna(0).values
    cm = confusion_matrix(yt, yp, labels=list(range(n)))
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    from scipy.stats import spearmanr
    rho = spearmanr(sup, recall)
    # predicted-class histogram vs support: bias to frequent classes
    pred_cnt = np.bincount(yp, minlength=n); true_cnt = np.bincount(yt, minlength=n)
    R[s] = dict(metrics=rec, saturation=sat, exp_topk=exp_topk, exp_auc=exp_auc, exp_acc=exp_acc, argmax_agreement=agree,
                n_test_rows=len(te), n_test_files=int(uniq_files), file_level_accuracy_ceiling=float(g),
                test_files_also_in_train=int(leak_files), test_rows_with_train_file=leak_rows,
                classes_zero_recall=int((recall == 0).sum()), classes_recall_ge_80=int((recall >= 0.8).sum()),
                spearman_support_vs_recall=[float(rho.statistic), float(rho.pvalue)],
                top_class_pred_share=float(pred_cnt.max() / len(yt)), top_class_true_share=float(true_cnt.max() / len(yt)))
    np.save(os.path.join(DATA, f"confusion_{s}.npy"), cm)

    # ---------- confusion matrix figure (row-normalised)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, interpolation="nearest"); ax.grid(False)
    ax.set_xlabel("predicted class index"); ax.set_ylabel("true class index")
    ax.set_title(f"Confusion, {s.replace('_',' ')}", fontsize=10, loc="left")
    plt.colorbar(im, fraction=0.045, label="row share")
    savefig("evaluation", f"confusion_{s}.png")

# ---------- ROC curves (micro-average one-vs-rest), saturated vs mean-exp
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for j, s in enumerate(SETS):
    P = np.load(os.path.join(ROOT, "output", s, "predictions.npz")); yt = P["y_true"]; n = P["y_score"].shape[1]
    Y = np.eye(n)[yt]
    for key, sty, lab in (("y_score", "-", "Σlog p"), ("y_score_exp", "--", "mean p")):
        fpr, tpr, _ = roc_curve(Y.ravel(), P[key].ravel()); a = np.trapezoid(tpr, fpr)
        axes[j].plot(fpr, tpr, sty, color=COLORS[s], lw=1.6, label=f"{lab}  AUC {a:.2f}")
    axes[j].plot([0, 1], [0, 1], color="#9a9a9a", lw=0.8, ls=":"); axes[j].set_title(s.replace("_", " "), fontsize=10, loc="left"); axes[j].set_xlabel("FPR")
    axes[j].legend(frameon=False, fontsize=7, loc="lower right")
axes[0].set_ylabel("TPR")
savefig("evaluation", "roc_micro_saturated_vs_meanexp.png")

# ---------- score saturation + per-class recall vs support
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for s in SETS:
    P = np.load(os.path.join(ROOT, "output", s, "predictions.npz")); n = P["y_score"].shape[1]
    axes[0].bar(SETS.index(s), 100 * (P["y_score"].max(1) == 1.0).mean(), 0.6, color=COLORS[s])
    axes[1].hist(P["y_score_exp"].max(1), bins=30, histtype="step", color=COLORS[s], lw=1.6)
axes[0].set_xticks(range(3)); axes[0].set_xticklabels([s.replace("_", "\n") for s in SETS], fontsize=8); axes[0].set_ylim(0, 110)
axes[0].set_ylabel("% rows with max p = 1"); axes[0].set_title("Σlog p scores", fontsize=10, loc="left")
for i, s in enumerate(SETS):
    P = np.load(os.path.join(ROOT, "output", s, "predictions.npz")); axes[0].text(i, 100 * (P["y_score"].max(1) == 1.0).mean() + 2, f"{100*(P['y_score'].max(1)==1.0).mean():.1f}", ha="center", fontsize=8)
axes[1].set_xlabel("top mean-p"); axes[1].set_ylabel("test rows"); axes[1].set_title("mean p scores", fontsize=10, loc="left")
axes[1].legend(handles=[plt.Line2D([], [], color=COLORS[s], lw=1.6, label=s.replace("_", " ")) for s in SETS], frameon=False, fontsize=8, loc="upper right")
for s in SETS:
    P = np.load(os.path.join(ROOT, "output", s, "predictions.npz")); n = P["y_score"].shape[1]
    tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); sup = tr.label.value_counts().reindex(range(n)).fillna(0).values
    cm = np.load(os.path.join(DATA, f"confusion_{s}.npy")); rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    axes[2].scatter(sup, rec, s=14, color=COLORS[s], alpha=0.6, edgecolor="white", lw=0.4)
axes[2].set_xscale("log"); axes[2].set_xlabel("train tags per class"); axes[2].set_ylabel("test recall"); axes[2].set_title("Recall vs support", fontsize=10, loc="left")
savefig("evaluation", "score_saturation_and_support.png")

# ---------- ours vs paper
paper = {"all_classes": dict(accuracy=.7301, roc_auc=.7562, precision=.7489, recall=.7301, f1=.7231, top3_accuracy=.8993, top5_accuracy=.9329, roc_auc_mean_exp=.7591),
         "bird_classes": dict(accuracy=.7447, roc_auc=.7662, precision=.7625, recall=.7447, f1=.7408, top3_accuracy=.8970, top5_accuracy=.9327, roc_auc_mean_exp=.7865),
         "bird_species": dict(accuracy=.7356, roc_auc=.7485, precision=.7481, recall=.7356, f1=.7408, top3_accuracy=.9023, top5_accuracy=.9447, roc_auc_mean_exp=.7865)}
keys = ["accuracy", "roc_auc", "precision", "recall", "f1", "top3_accuracy", "top5_accuracy", "roc_auc_mean_exp"]
nice = ["Acc", "AUC", "Prec", "Recall", "F1", "Top-3", "Top-5", "AUC\nmean p"]
fig, ax = plt.subplots(figsize=(10.5, 3.9)); x = np.arange(len(keys)); w = 0.13
for j, s in enumerate(SETS):
    ax.bar(x + (j - 1) * 2 * w - w / 2 * 0, [R[s]["metrics"][k] for k in keys], w * 0.92, color=COLORS[s], label=s.replace("_", " "))
    ax.scatter(x + (j - 1) * 2 * w, [paper[s][k] for k in keys], marker="_", s=300, color=INK, lw=1.8, zorder=5, label="paper" if j == 0 else None)
ax.set_xticks(x); ax.set_xticklabels(nice, fontsize=8.5); ax.set_ylim(0, 1.05); ax.set_ylabel("value")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
savefig("evaluation", "ours_vs_paper.png")

json.dump(R, open(os.path.join(DATA, "evaluation_analysis.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
