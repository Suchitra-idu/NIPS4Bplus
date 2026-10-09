"""Receptive fields, MACs, activation statistics per stage, weight statistics vs init."""
import json
import numpy as np, torch
from _common import *

R = {}
for s in SETS:
    opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt); cnn0 = build_models(opt)[0]; load_ckpt(s, cnn, d1, d2); cnn.eval(); d1.eval(); d2.eval()
    K = cnn.cnn_len_filt; P = cnn.cnn_max_pool_len; N = cnn.cnn_N_filt
    # receptive field (samples) and jump
    rf, jump = 1, 1; rfs = []
    for k, p in zip(K, P):
        rf += (k - 1) * jump; rfs.append(("conv", rf, jump)); rf += (p - 1) * jump; jump *= p; rfs.append(("pool", rf, jump))
    # MACs per frame
    T = wlen; cin = 1; macs = []
    for k, n_, p in zip(K, N, P):
        Tout = T - k + 1; macs.append(n_ * cin * k * Tout); cin = n_; T = Tout // p
    fc = [cnn.out_dim] + [int(x) for x in opt.fc_lay.split(",")] + [int(opt.class_lay)]
    macs_fc = sum(a * b for a, b in zip(fc[:-1], fc[1:]))
    # activations per stage on 1024 real windows
    X, y, _ = sample_windows(LISTS[s][0], wlen, 1024, seed=7); x = torch.from_numpy(X).float()
    st = []
    with torch.no_grad():
        h = x.view(-1, 1, wlen); st.append(("input", float(h.mean()), float(h.std()), float((h == 0).float().mean())))
        for i in range(cnn.N_cnn_lay):
            c = cnn.conv[i](h); st.append((f"conv{i}", float(c.mean()), float(c.std()), float((c == 0).float().mean())))
            pl = torch.nn.functional.max_pool1d(c, P[i]); st.append((f"pool{i}", float(pl.mean()), float(pl.std()), 0.0))
            if cnn.cnn_use_batchnorm[i]: nb = cnn.bn[i](pl)
            elif cnn.cnn_use_laynorm[i]: nb = cnn.ln[i](pl)
            else: nb = pl
            st.append((f"norm{i}", float(nb.mean()), float(nb.std()), 0.0))
            h = cnn.act[i](nb); st.append((f"act{i}", float(h.mean()), float(h.std()), float((h <= 0).float().mean())))
        f = h.reshape(len(x), -1); st.append(("flatten", float(f.mean()), float(f.std()), float((f == 0).float().mean())))
        z = f
        for i in range(d1.N_fc_lay):
            z = d1.wx[i](z); z = d1.bn[i](z) if d1.fc_use_batchnorm[i] else z; z = d1.act[i](z)
            st.append((f"fc{i}", float(z.mean()), float(z.std()), float((z == 0).float().mean())))
        lp = d2(z); st.append(("logsoftmax", float(lp.mean()), float(lp.std()), 0.0))
        ent = float(-(lp.exp() * lp).sum(1).mean()); conf = float(lp.exp().max(1).values.mean())
    # weights vs init
    w = {}
    d1_0 = build_models(opt)[1]
    for i in range(d1.N_fc_lay):
        lim = float(np.sqrt(0.01 / (d1.wx[i].in_features + d1.wx[i].out_features))); xav = float(np.sqrt(6 / (d1.wx[i].in_features + d1.wx[i].out_features)))
        w[f"fc{i}"] = dict(init_limit=lim, xavier_limit=xav, init_std=float(d1_0.wx[i].weight.std()), trained_std=float(d1.wx[i].weight.std()), ratio_trained_over_init=float(d1.wx[i].weight.std() / d1_0.wx[i].weight.std()))
    for i in (1, 2):
        w[f"conv{i}"] = dict(init_std=float(cnn0.conv[i].weight.std()), trained_std=float(cnn.conv[i].weight.std()), ratio_trained_over_init=float(cnn.conv[i].weight.std() / cnn0.conv[i].weight.std()))
    R[s] = dict(rf=[(a, b, c) for a, b, c in rfs], rf_final_samples=rf, rf_final_ms=rf / 44.1, rf_fraction_of_window=rf / wlen,
                macs_conv=macs, macs_fc=macs_fc, macs_total=sum(macs) + macs_fc, stages=st, mean_entropy_nats=ent, mean_max_prob_per_frame_on_event_windows=conf,
                weights=w, window_label_chance=1.0 / int(opt.class_lay))
    print(s, "RF", rf, "MACs", sum(macs) + macs_fc, "conf", round(conf, 3), "entropy", round(ent, 3))
json.dump(R, open(os.path.join(DATA, "layer_stats.json"), "w"), indent=1)
# figure: MACs by layer (bird species), activation std per stage
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
r = R["bird_species"]; labs = ["Sinc", "Conv2", "Conv3", "FC×4"]; v = r["macs_conv"] + [r["macs_fc"]]
ax[0].bar(range(4), np.array(v) / 1e6, color="#1baf7a", width=0.6); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labs); ax[0].set_ylabel("M MACs / frame")
for i, x_ in enumerate(v): ax[0].text(i, x_ / 1e6 + 0.3, f"{x_/1e6:.1f}", ha="center", fontsize=8)
ax[0].set_title("Compute per 16 ms frame", fontsize=10, loc="left")
for s in SETS:
    stg = R[s]["stages"]; ax[1].plot(range(len(stg)), [a[2] for a in stg], marker="o", ms=3, color=COLORS[s], lw=1.3, label=s.replace("_", " "))
ax[1].set_xticks(range(len(R["bird_species"]["stages"]))); ax[1].set_xticklabels([a[0] for a in R["bird_species"]["stages"]], rotation=60, fontsize=7, ha="right")
ax[1].set_yscale("log"); ax[1].set_ylabel("std"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("Activation scale by stage", fontsize=10, loc="left")
savefig("architecture", "compute_and_activation_scale.png")
