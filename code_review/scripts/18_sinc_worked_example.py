"""Step-by-step numbers for ONE Sinc filter of the trained bird_species model (mirrors SincConv_fast.forward)."""
import json, math
import numpy as np, torch
from _common import *
torch.set_grad_enabled(False)
s = "bird_species"; opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt); load_ckpt(s, cnn, d1, d2)
L = cnn.conv[0]; k = 73; K = L.kernel_size; fs = 44100
low = (L.min_low_hz + torch.abs(L.low_hz_)); high = torch.clamp(low + L.min_band_hz + torch.abs(L.band_hz_), L.min_low_hz, fs / 2); band = (high - low)[:, 0]
n_ = L.n_; win = L.window_
f_t_low = low @ n_; f_t_high = high @ n_
left = ((torch.sin(f_t_high) - torch.sin(f_t_low)) / (n_ / 2)) * win
centre = 2 * band.view(-1, 1); right = torch.flip(left, dims=[1]); bp = torch.cat([left, centre, right], 1) / (2 * band[:, None])
R = dict(filter=k, low_hz_param=float(L.low_hz_[k]), band_hz_param=float(L.band_hz_[k]), low=float(low[k]), high=float(high[k]), band=float(band[k]),
         n_first=[float(v) for v in n_[0, :3]], n_last=[float(v) for v in n_[0, -3:]], n_units="rad/sample * 1/fs: 2*pi*t, t=-75..-1 samples / fs",
         window_first=[float(v) for v in win[:3]], window_last=[float(v) for v in win[-3:]],
         left_first=[float(v) for v in left[k, :3]], left_last=[float(v) for v in left[k, -3:]], centre_raw=float(centre[k]), filt_centre=float(bp[k, 75]),
         filt_first=[float(v) for v in bp[k, :3]], l2=float(bp[k].norm()), sum_abs=float(bp[k].abs().sum()), K=K,
         dc_gain=float(bp[k].sum()), peak_response=float(np.abs(np.fft.rfft(bp[k].numpy(), 8192)).max()))
fr = np.fft.rfftfreq(8192, 1 / fs); H = np.abs(np.fft.rfft(bp[k].numpy(), 8192)); R["measured_peak_hz"] = float(fr[H.argmax()]); R["band_centre_hz"] = float((low[k] + high[k]) / 2)
R["bank_gain_peak_min_max"] = [float(np.abs(np.fft.rfft(bp.numpy(), 8192, axis=1)).max(1).min()), float(np.abs(np.fft.rfft(bp.numpy(), 8192, axis=1)).max(1).max())]
json.dump(R, open(os.path.join(DATA, "sinc_worked_example.json"), "w"), indent=1); print(json.dumps(R, indent=1))
fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))
t = np.arange(-75, 76); ax[0].plot(np.arange(-75, 0), left[k].numpy(), color="#2a78d6"); ax[0].set_title("① left half (sin diff / t · w)", fontsize=9.5, loc="left")
ax[1].plot(np.arange(-75, 0), win.numpy(), color="#eb6834"); ax[1].set_title("② half Hamming window", fontsize=9.5, loc="left")
ax[2].plot(t, bp[k].numpy(), color="#1baf7a"); ax[2].axhline(0, color="#9a9a9a", lw=0.6); ax[2].set_title("③ mirrored, ÷ 2·band (centre = 1)", fontsize=9.5, loc="left")
ax[3].plot(fr / 1e3, H, color="#1baf7a"); ax[3].axvspan(float(low[k]) / 1e3, float(high[k]) / 1e3, color="#2a78d6", alpha=0.25, lw=0); ax[3].set_xlim(0, 3); ax[3].set_title("④ |H(f)|, shaded = [low, high]", fontsize=9.5, loc="left"); ax[3].set_xlabel("kHz")
for a in ax[:3]: a.set_xlabel("tap offset")
savefig("filters", "sinc_forward_steps.png")
