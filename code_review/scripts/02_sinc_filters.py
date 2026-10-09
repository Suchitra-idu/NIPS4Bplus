"""Filter-bank analysis: init vs trained cutoffs, response curves, window quirk,
and a numerical check of SincConv_fast against the paper's Eq. 4."""
import json
import math
import numpy as np
import torch
from _common import *
from dnn_models import SincConv_fast

res = {}
FS = 44100


def cutoffs(layer):
    low = layer.min_low_hz + torch.abs(layer.low_hz_)
    high = torch.clamp(low + layer.min_band_hz + torch.abs(layer.band_hz_), layer.min_low_hz, layer.sample_rate / 2)
    return low[:, 0].detach().numpy(), high[:, 0].detach().numpy()


def freq_resp(layer, nfft=8192):
    layer(torch.zeros(1, 1, layer.kernel_size + 5))
    f = layer.filters.detach().numpy()[:, 0, :]
    return np.fft.rfftfreq(nfft, 1 / FS), np.abs(np.fft.rfft(f, nfft, axis=1)), f

fig, axes3 = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
axes = axes3[0]
fig2, axes2 = plt.subplots(2, 3, figsize=(12, 5.2))
for j, s in enumerate(SETS):
    opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt)
    init = cnn.conv[0]
    lo0, hi0 = cutoffs(init)
    load_ckpt(s, cnn, d1, d2)
    lo1, hi1 = cutoffs(cnn.conv[0])
    c0, c1 = (lo0 + hi0) / 2, (lo1 + hi1) / 2
    bw0, bw1 = hi0 - lo0, hi1 - lo1
    res[s] = dict(n_filters=len(lo0), kernel=cnn.conv[0].kernel_size,
                  median_abs_dlow_hz=float(np.median(np.abs(lo1 - lo0))), median_abs_dhigh_hz=float(np.median(np.abs(hi1 - hi0))),
                  max_abs_dlow_hz=float(np.max(np.abs(lo1 - lo0))), max_abs_dhigh_hz=float(np.max(np.abs(hi1 - hi0))),
                  median_rel_dcenter=float(np.median(np.abs(c1 - c0) / c0)),
                  median_bw_init=float(np.median(bw0)), median_bw_trained=float(np.median(bw1)),
                  frac_filters_moved_gt_5pct=float(np.mean(np.abs(c1 - c0) / c0 > 0.05)),
                  init_center_min=float(c0.min()), init_center_max=float(c0.max()),
                  trained_center_min=float(c1.min()), trained_center_max=float(c1.max()),
                  n_above_8k_init=int((c0 > 8000).sum()), n_above_8k_trained=int((c1 > 8000).sum()))
    ax = axes[j]
    k = np.arange(len(c0))
    ax.vlines(k, lo0 / 1e3, hi0 / 1e3, color="#b0b0b0", lw=2.2, label="initial (mel)")
    ax.vlines(k + 0.25, lo1 / 1e3, hi1 / 1e3, color=COLORS[s], lw=1.2, label="trained")
    ax.set_title(s.replace("_", " "), fontsize=10, loc="left")
    a2 = axes3[1][j]; a2.plot(k, lo1 - lo0, color=COLORS[s], lw=1.2, label="Δ low cutoff"); a2.plot(k, hi1 - hi0, color=INK2, lw=1.0, ls="--", label="Δ high cutoff")
    a2.set_xlabel("filter index"); a2.set_ylim(-60, 60)
    if j == 0: a2.set_ylabel("Δ cutoff (Hz)"); a2.legend(frameon=False, fontsize=8, loc="upper left")
    if j == 0:
        ax.set_ylabel("pass-band (kHz)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    # response overlays for 4 filters
    fr, R0, f0 = freq_resp(init); _, R1, f1 = freq_resp(cnn.conv[0])
    for r_, idx in enumerate([len(c0) // 8, len(c0) // 3, 2 * len(c0) // 3, len(c0) - 3]):
        axes2[0, j].plot(fr / 1e3, R0[idx], color="#b0b0b0", lw=1.3)
        axes2[0, j].plot(fr / 1e3, R1[idx], color=COLORS[s], lw=1.1)
    axes2[0, j].set_xlim(0, 22.05); axes2[0, j].set_title(s.replace("_", " ") + ": |H(f)|", fontsize=9, loc="left")
    axes2[0, j].set_xlabel("frequency (kHz)")
    axes2[1, j].plot(f0[len(c0) // 3], color="#b0b0b0", lw=1.4, label="initial")
    axes2[1, j].plot(f1[len(c0) // 3], color=COLORS[s], lw=1.1, label="trained")
    axes2[1, j].set_xlabel("tap"); axes2[1, j].set_title("filter %d, taps" % (len(c0) // 3), fontsize=9, loc="left")
    if j == 0: axes2[1, j].legend(frameon=False, fontsize=8)
    # coverage: sum of power responses
    if s == "bird_species":
        cov0, cov1 = (R0 ** 2).sum(0), (R1 ** 2).sum(0)
        res["coverage_fr"] = fr.tolist()[::16]; res["coverage_init"] = cov0.tolist()[::16]; res["coverage_trained"] = cov1.tolist()[::16]
fig.suptitle("Sinc pass-bands: initial vs trained", x=0.01, ha="left", fontsize=11)
plt.figure(fig.number); savefig("filters", "passbands_init_vs_trained.png")
plt.figure(fig2.number); savefig("filters", "filter_responses.png")

# ---- numerical check of SincConv_fast vs paper Eq.4 (windowed difference of sincs, centre-normalised)
layer = SincConv_fast(4, 151, FS)
layer(torch.zeros(1, 1, 200)); got = layer.filters.detach().numpy()[:, 0, :]
low, high = [x for x in cutoffs(layer)]
n = np.arange(-75, 76)
errs = []
for i in range(4):
    def lp(fc): return 2 * fc / FS * np.sinc(2 * fc / FS * n)  # np.sinc(x)=sin(pi x)/(pi x)
    g = lp(high[i]) - lp(low[i])
    w_ideal = 0.54 - 0.46 * np.cos(2 * np.pi * (n + 75) / 150)         # symmetric Hamming, N-1=150
    ref = g * w_ideal; ref = ref / ref[75]
    errs.append(float(np.max(np.abs(got[i] - ref))))
res["eq4_max_abs_err_vs_ideal_hamming"] = errs
# window quirk: linspace(0, K/2-1, steps=int(K/2)) for odd K
K = 151
nl = np.linspace(0, K / 2 - 1, int(K / 2)); wq = 0.54 - 0.46 * np.cos(2 * np.pi * nl / K)
wi = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(75) / 150)
res["window_quirk"] = dict(max_abs_dev=float(np.max(np.abs(wq - wi))), last_tap_code=float(wq[-1]), last_tap_ideal=float(wi[-1]), step=float(nl[1] - nl[0]))
# symmetric?
res["filters_symmetric"] = bool(np.allclose(got, got[:, ::-1], atol=1e-7))
json.dump(res, open(os.path.join(DATA, "sinc_filters.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("coverage")}, indent=1))

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(res["coverage_fr"] if False else np.array(res["coverage_fr"]) / 1e3, res["coverage_init"], color="#b0b0b0", lw=2, label="initial")
ax.plot(np.array(res["coverage_fr"]) / 1e3, res["coverage_trained"], color=COLORS["bird_species"], lw=1.3, label="trained")
ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("Σ |H|²"); ax.set_yscale("log"); ax.legend(frameon=False)
ax.set_title("Bank coverage (bird species)", fontsize=10, loc="left")
savefig("filters", "bank_coverage.png")
