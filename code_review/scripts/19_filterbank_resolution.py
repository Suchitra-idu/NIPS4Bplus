"""Effective frequency resolution and redundancy of the 220-filter Sinc bank (151 taps, Hamming)."""
import json
import numpy as np, torch
from _common import *
torch.set_grad_enabled(False)
s = "bird_species"; opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt); load_ckpt(s, cnn, d1, d2)
L = cnn.conv[0]; fs = 44100; L(torch.zeros(1, 1, 200)); W = L.filters[:, 0, :].numpy()          # 220 x 151
low = (L.min_low_hz + torch.abs(L.low_hz_))[:, 0].numpy(); high = np.minimum(low + L.min_band_hz + np.abs(L.band_hz_[:, 0].numpy()), fs / 2)
nfft = 16384; f = np.fft.rfftfreq(nfft, 1 / fs); H = np.abs(np.fft.rfft(W, nfft, axis=1))
bw3 = np.zeros(len(W)); peak = np.zeros(len(W))
for i in range(len(W)):
    h = H[i]; p = h.argmax(); peak[i] = f[p]; thr = h[p] / np.sqrt(2)
    lo = p
    while lo > 0 and h[lo] > thr: lo -= 1
    hi = p
    while hi < len(h) - 1 and h[hi] > thr: hi += 1
    bw3[i] = f[hi] - f[lo]
nom = high - low; centre = (low + high) / 2
# redundancy
sv = np.linalg.svd(W, compute_uv=False); en = np.cumsum(sv ** 2) / np.sum(sv ** 2)
rank99 = int(np.searchsorted(en, 0.99) + 1); rank999 = int(np.searchsorted(en, 0.999) + 1)
Hn = H / np.linalg.norm(H, axis=1, keepdims=True); cos_adj = np.sum(Hn[:-1] * Hn[1:], 1)
R = dict(n_filters=len(W), taps=W.shape[1], hamming_transition_hz_theory=3.3 * fs / W.shape[1], rect_resolution_hz=fs / W.shape[1],
         nominal_bw_median=float(np.median(nom)), eff_bw3db_median=float(np.median(bw3)),
         frac_filters_eff_bw_gt_2x_nominal=float(np.mean(bw3 > 2 * nom)), frac_filters_eff_bw_gt_nominal=float(np.mean(bw3 > nom)),
         eff_over_nom_at_low_f_below_2k_median=float(np.median(bw3[centre < 2000] / nom[centre < 2000])), n_below_2k=int((centre < 2000).sum()),
         n_2k_to_10k=int(((centre >= 2000) & (centre < 10000)).sum()), n_above_10k=int((centre >= 10000).sum()),
         rank_99pct_energy=rank99, rank_999pct_energy=rank999, max_possible_rank=min(W.shape), adjacent_cosine_median=float(np.median(cos_adj)),
         adjacent_cosine_below_2k_median=float(np.median(cos_adj[centre[:-1] < 2000])), adjacent_cosine_above_10k_median=float(np.median(cos_adj[centre[:-1] > 10000])),
         peak_gain_min=float(H.max(1).min()), peak_gain_max=float(H.max(1).max()), peak_gain_ratio=float(H.max(1).max() / H.max(1).min()),
         centre_vs_peak_max_dev_hz=float(np.max(np.abs(peak - centre))))
json.dump(R, open(os.path.join(DATA, "filterbank_resolution.json"), "w"), indent=1); print(json.dumps(R, indent=1))
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
ax[0].loglog(centre, nom, color="#9a9a9a", lw=1.6, label="nominal (high − low)"); ax[0].loglog(centre, bw3, color="#1baf7a", lw=1.6, label="measured −3 dB")
ax[0].set_xlabel("centre frequency (Hz)"); ax[0].set_ylabel("bandwidth (Hz)"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Bandwidth per filter", fontsize=10, loc="left")
ax[1].semilogy(sv / sv[0], color="#2a78d6", lw=1.4); ax[1].axvline(rank99, color="#eb6834", ls="--", lw=1); ax[1].axvline(151, color=INK2, ls=":", lw=1)
ax[1].text(rank99 + 3, 0.3, f"99 % energy: {rank99}", fontsize=8, color="#eb6834"); ax[1].set_xlabel("singular value index"); ax[1].set_ylabel("σ / σ₀"); ax[1].set_title("220 filters, 151 taps", fontsize=10, loc="left")
ax[2].plot(centre[:-1] / 1e3, cos_adj, color="#1baf7a", lw=1.2); ax[2].set_xlabel("centre frequency (kHz)"); ax[2].set_ylabel("cosine of |H| with next filter"); ax[2].set_title("Neighbour overlap", fontsize=10, loc="left"); ax[2].set_ylim(0, 1.02)
savefig("filters", "bank_resolution_and_redundancy.png")
