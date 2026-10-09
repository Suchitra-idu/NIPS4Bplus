"""BatchNorm1d(N_filt, <int>, momentum=0.05) passes the int as *eps* (dnn_models.py:415).
Measure what that does to the activations of the trained enhanced models."""
import json
import numpy as np, torch
from _common import *

R = {}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for s in SETS:
    opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt); load_ckpt(s, cnn, d1, d2); cnn.eval()
    X, y, _ = sample_windows(LISTS[s][0], wlen, 1024, seed=1)
    x = torch.from_numpy(X).float()
    acts = {}
    hooks = []
    for i, bn in enumerate(cnn.bn):
        if not cnn.cnn_use_batchnorm[i]: continue
        hooks.append(bn.register_forward_hook(lambda m, inp, out, i=i: acts.__setitem__(i, (inp[0].detach(), out.detach(), m))))
    with torch.no_grad(): cnn(x)
    per = []
    for i, (a, b, m) in sorted(acts.items()):
        var_run = m.running_var; mean_run = m.running_mean
        # "what-if" with the conventional eps 1e-5 on the same running statistics
        b_std = ((a - mean_run[None, :, None]) / torch.sqrt(var_run[None, :, None] + 1e-5) * m.weight[None, :, None] + m.bias[None, :, None])
        per.append(dict(layer=i, eps=float(m.eps), median_running_var=float(var_run.median()), median_running_std=float(var_run.sqrt().median()),
                        eps_over_var_median=float(m.eps / var_run.median()), out_std_actual=float(b.std()), out_std_if_eps_1e5=float(b_std.std()),
                        gamma_mean=float(m.weight.mean()), gamma_min=float(m.weight.min()), gamma_max=float(m.weight.max()), in_absmax=float(a.abs().max())))
    R[s] = per
    ax = axes[["all_classes", "bird_classes", "bird_species"].index(s)]
    xi = np.arange(len(per)); w = 0.36
    ax.bar(xi - w / 2, [p["out_std_actual"] for p in per], w * 0.9, color=COLORS[s], label="as trained")
    ax.bar(xi + w / 2, [p["out_std_if_eps_1e5"] for p in per], w * 0.9, color="#c9c9c9", label="eps = 1e-5")
    ax.axhline(1, color=INK, lw=0.8, ls="--"); ax.set_ylim(0, 1.35); ax.set_xticks(xi); ax.set_xticklabels([f"BN{p['layer']+1}" for p in per])
    ax.set_title(s.replace("_", " "), fontsize=10, loc="left")
    if s == "all_classes": ax.set_ylabel("output std")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
savefig("architecture", "batchnorm_eps_effect.png")
json.dump(R, open(os.path.join(DATA, "batchnorm_eps.json"), "w"), indent=1)
print(json.dumps(R["bird_species"], indent=1))
