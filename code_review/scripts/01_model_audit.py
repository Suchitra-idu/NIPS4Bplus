"""Instantiate the enhanced + default models and audit shapes, parameters,
BatchNorm eps, dead modules and bias handling. Output: data/model_audit.json
and figures/architecture/*."""
import json
import numpy as np
import torch
from _common import *

out = {}
for cfgset, label in ((CFG, "enhanced"), (CFG_DEFAULT, "default")):
    for s in SETS:
        opt = load_options(cfgset[s])
        cnn, d1, d2, wlen = build_models(opt)
        # shape trace
        x = torch.randn(4, wlen)
        shapes = [("input", wlen)]
        h = x.view(4, 1, wlen)
        for i in range(cnn.N_cnn_lay):
            h = cnn.conv[i](h)
            shapes.append((f"conv{i}", tuple(h.shape[1:])))
            h = torch.nn.functional.max_pool1d(h, cnn.cnn_max_pool_len[i])
            shapes.append((f"pool{i}", tuple(h.shape[1:])))
        shapes.append(("flatten", cnn.out_dim))
        tot = lambda m: sum(p.numel() for p in m.parameters())
        # parameters that can never receive a gradient given cfg (unused norm modules)
        def unused(m, kind):
            n = 0
            for name, p in m.named_parameters():
                parts = name.split(".")
                if parts[0] in ("ln", "bn"):
                    i = int(parts[1])
                    use_ln = (m.cnn_use_laynorm[i] if kind == "cnn" else m.fc_use_laynorm[i])
                    use_bn = (m.cnn_use_batchnorm[i] if kind == "cnn" else m.fc_use_batchnorm[i])
                    if (parts[0] == "ln" and not use_ln) or (parts[0] == "bn" and not use_bn):
                        n += p.numel()
            return n
        # Linear biases that exist although cfg turned bias off (norm follows)
        fake_bias = sum(l.bias.numel() for i, l in enumerate(d1.wx) if (d1.fc_use_laynorm[i] or d1.fc_use_batchnorm[i]) and l.bias is not None)
        bn_eps = [float(m.eps) for m in cnn.bn] if label == "enhanced" else [float(m.eps) for m in cnn.bn]
        out[f"{label}/{s}"] = dict(
            wlen=wlen, shapes=[(a, list(b) if isinstance(b, tuple) else b) for a, b in shapes],
            params_cnn=tot(cnn), params_dnn1=tot(d1), params_dnn2=tot(d2),
            params_total=tot(cnn) + tot(d1) + tot(d2),
            params_unused_norm_cnn=unused(cnn, "cnn"), params_unused_norm_dnn1=unused(d1, "fc"),
            params_unused_norm_dnn2=unused(d2, "fc"),
            params_sinc=cnn.conv[0].low_hz_.numel() + cnn.conv[0].band_hz_.numel(),
            params_conv1=tot(cnn.conv[1]), params_conv2=tot(cnn.conv[2]),
            dead_bias_before_bn=int(fake_bias),
            cnn_bn_eps=bn_eps, cnn_use_bn=cnn.cnn_use_batchnorm, cnn_use_ln=cnn.cnn_use_laynorm,
        )
        o = out[f"{label}/{s}"]
        o["params_effective"] = o["params_total"] - o["params_unused_norm_cnn"] - o["params_unused_norm_dnn1"] - o["params_unused_norm_dnn2"] - o["dead_bias_before_bn"]
json.dump(out, open(os.path.join(DATA, "model_audit.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v["wlen"], v["shapes"][-1], "total", v["params_total"], "effective", v["params_effective"], "bn_eps", v["cnn_bn_eps"])

# ---- figure: parameter budget by component (enhanced)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
comps = [("Sinc", "params_sinc"), ("Conv2", "params_conv1"), ("Conv3", "params_conv2"),
         ("FC×3", "params_dnn1"), ("Output", "params_dnn2")]
x = np.arange(len(comps)); w = 0.26
for j, s in enumerate(SETS):
    vals = [out[f"enhanced/{s}"][k] for _, k in comps]
    ax.bar(x + (j - 1) * w, vals, w * 0.9, color=COLORS[s], label=s.replace("_", " "))
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([c for c, _ in comps], fontsize=8)
ax.set_ylabel("parameters"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
ax.set_ylim(top=3e7)
ax.set_title("Parameters per block", fontsize=10, loc="left", pad=14)
ax = axes[1]
for j, s in enumerate(SETS):
    o = out[f"enhanced/{s}"]; e = o["params_effective"]; t = o["params_total"]
    ax.bar(j, e / 1e6, 0.6, color=COLORS[s]); ax.bar(j, (t - e) / 1e6, 0.6, bottom=e / 1e6, color="#c9c9c9", hatch="//", edgecolor="white")
    ax.text(j, t / 1e6 + 0.04, f"{t/1e6:.2f} M", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels([s.replace("_", "\n") for s in SETS], fontsize=8); ax.set_ylim(0, 3.1)
ax.set_ylabel("millions"); ax.set_title("Effective (colour) + dead (grey)", fontsize=10, loc="left")
savefig("architecture", "param_budget.png")

# ---- figure: tensor flow (enhanced, 3 sets)
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off")
for r, s in enumerate(SETS):
    o = out[f"enhanced/{s}"]
    names = [str(o["wlen"]) + " smp"] + []
    labels = [f"input\n{o['wlen']}"]
    for a, b in o["shapes"][1:-1]:
        labels.append(f"{a}\n{b[0]}×{b[1]}")
    labels += [f"flatten\n{o['shapes'][-1][1]}", "FC×3\n1024", f"out\n{o['shapes'][-1][1] and {'all_classes':87,'bird_classes':77,'bird_species':51}[s]}"]
    n = len(labels)
    for i, t in enumerate(labels):
        ax.text(i / (n - 1), 1 - r * 0.38, t, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS[s], lw=1.4))
        if i < n - 1:
            ax.annotate("", xy=((i + 1) / (n - 1) - 0.045, 1 - r * 0.38), xytext=(i / (n - 1) + 0.045, 1 - r * 0.38),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.text(-0.07, 1 - r * 0.38, s.replace("_", " "), ha="right", va="center", fontsize=8.5, color=INK)
ax.set_xlim(-0.2, 1.05); ax.set_ylim(0.1, 1.2)
ax.set_title("Tensor shapes, enhanced models", fontsize=10, loc="left")
savefig("architecture", "tensor_shapes.png")
