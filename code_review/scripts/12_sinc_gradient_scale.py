"""Why the cut-offs do not move: gradient scale of low_hz_/band_hz_ at the trained
bird_species model (train mode, real windows), and the implied RMSprop drift."""
import json
import numpy as np, torch
from _common import *

s = "bird_species"; opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt, cuda=True); load_ckpt(s, cnn, d1, d2, "cuda")
for m in (cnn, d1, d2): m.train()
lr, alpha = 0.001, 0.95
G = {"low": [], "band": []}; Gc = []
nll = torch.nn.NLLLoss()
for b in range(120):
    X, y, _ = sample_windows(LISTS[s][0], wlen, 128, seed=100 + b)
    for m in (cnn, d1, d2): m.zero_grad()
    out = d2(d1(cnn(torch.from_numpy(X).float().cuda())))
    nll(out, torch.from_numpy(y).long().cuda()).backward()
    G["low"].append(cnn.conv[0].low_hz_.grad[:, 0].cpu().numpy().copy()); G["band"].append(cnn.conv[0].band_hz_.grad[:, 0].cpu().numpy().copy())
    Gc.append(cnn.conv[1].weight.grad.abs().mean().item())
R = {}
for k in G:
    g = np.array(G[k])                      # [120, 220]
    rms = np.sqrt((g ** 2).mean(0)); cons = np.abs(g.mean(0)) / np.maximum(rms, 1e-30)
    R[k] = dict(mean_abs_grad=float(np.abs(g).mean()), median_rms=float(np.median(rms)), median_sign_consistency=float(np.median(cons)),
                p90_sign_consistency=float(np.percentile(cons, 90)),
                frac_steps_same_sign_as_mean=float(np.mean(np.sign(g) == np.sign(g.mean(0)))))
    steps = 400 * 80
    R[k]["drift_upper_bound_hz"] = lr * steps                                 # every step moves lr (RMSprop step ~ lr*g/rms <= ~lr)
    R[k]["drift_systematic_hz"] = float(lr * steps * np.median(cons))        # consistent-direction part
    R[k]["drift_random_walk_hz"] = float(lr * np.sqrt(steps))                 # unbiased part
R["conv2_mean_abs_grad"] = float(np.mean(Gc))
R["first_step_hz"] = lr / np.sqrt(1 - alpha)                                 # RMSprop first update (v0=0): lr*g/sqrt((1-a)g^2)
R["steps_total"] = 400 * 80
R["fs_range_hz"] = 22050
R["relative_lr_old_sinc_conv"] = "params stored as f/fs: step 1e-3*fs = 44.1 Hz"
json.dump(R, open(os.path.join(DATA, "sinc_gradient_scale.json"), "w"), indent=1); print(json.dumps(R, indent=1))
fig, ax = plt.subplots(figsize=(5.8, 3.4))
names = ["max possible\n(every step)", "systematic\n(measured sign bias)", "random walk", "measured\nmedian"]
meas = json.load(open(os.path.join(DATA, "sinc_filters.json")))["bird_species"]["median_abs_dhigh_hz"]
vals = [R["low"]["drift_upper_bound_hz"], R["low"]["drift_systematic_hz"], R["low"]["drift_random_walk_hz"], meas]
ax.bar(range(4), vals, color=["#c9c9c9", "#2a78d6", "#2a78d6", "#1baf7a"], width=0.6); ax.set_yscale("log")
for i, v in enumerate(vals): ax.text(i, v * 1.15, f"{v:.2g}", ha="center", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("Hz after 32k steps"); ax.set_title("Cut-off drift budget", fontsize=10, loc="left")
savefig("filters", "drift_budget.png")
