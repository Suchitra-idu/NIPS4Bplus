"""Micro-benchmark of one training step split into phases (bird_species cfg, GTX 1650)."""
import json, time
import numpy as np, pandas as pd, soundfile as sf, torch
from _common import *

s = "bird_species"; opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt, cuda=True)
for m in (cnn, d1, d2): m.train()
params = list(cnn.parameters()) + list(d1.parameters()) + list(d2.parameters())
opts = [torch.optim.RMSprop(m.parameters(), lr=1e-3, alpha=0.95, eps=1e-8) for m in (cnn, d1, d2)]
df = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); audio = {f: sf.read(os.path.join(ROOT, "nips4b_norm_files", f)) for f in df.file.unique()}
sig = [audio[f][0] for f in df.file]; start = df.start.values; length = df.length.values; lab = df.label.values
nll = torch.nn.NLLLoss()
def batch(bs=128, fact=0.0):   # same logic as call_id.create_batches_rnd, fs=44100
    X = np.zeros([bs, wlen]); y = np.zeros(bs); ids = np.random.randint(len(df), size=bs); amp = np.random.uniform(1 - fact, 1 + fact, bs)
    for i in range(bs):
        sg = sig[ids[i]]; t0 = int(start[ids[i]] * 44100); t1 = t0 + int(length[ids[i]] * 44100)
        if t1 - t0 > wlen: st = np.random.randint(t0, t1 - wlen)
        else:
            lo, hi = max(0, t1 - wlen), min(t0, sg.shape[0] - wlen); st = lo if lo == hi else np.random.randint(lo, hi)
        X[i] = sg[st:st + wlen] * amp[i]; y[i] = lab[ids[i]]
    return X, y
T = dict(sample=[], h2d=[], fwd=[], bwd=[], step=[], total=[]); sync = torch.cuda.synchronize
for it in range(120):
    sync(); t0 = time.perf_counter(); X, y = batch(); t1 = time.perf_counter()
    inp = torch.from_numpy(X).float().cuda().contiguous(); lb = torch.from_numpy(y).float().cuda().contiguous(); sync(); t2 = time.perf_counter()
    out = d2(d1(cnn(inp))); loss = nll(out, lb.long()); sync(); t3 = time.perf_counter()
    for o in opts: o.zero_grad()
    loss.backward(); sync(); t4 = time.perf_counter()
    for o in opts: o.step()
    sync(); t5 = time.perf_counter()
    if it >= 20:
        for k, v in zip(T, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0]): T[k].append(v)
R = {k: float(np.mean(v)) * 1000 for k, v in T.items()}; R["ms_unit"] = "ms per batch of 128"; R["batches_per_epoch"] = 80
R["pred_epoch_s"] = R["total"] * 80 / 1000
# evaluation throughput
cnn.eval(); d1.eval(); d2.eval(); fr = torch.randn(1024, wlen).cuda()
with torch.no_grad():
    for _ in range(3): d2(d1(cnn(fr)))
    sync(); t = time.perf_counter()
    for _ in range(10): d2(d1(cnn(fr)))
    sync(); R["eval_frames_per_s"] = 10 * 1024 / (time.perf_counter() - t)
R["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
json.dump(R, open(os.path.join(DATA, "profile.json"), "w"), indent=1); print(json.dumps(R, indent=1))
fig, ax = plt.subplots(figsize=(6.2, 3.2)); ks = ["sample", "h2d", "fwd", "bwd", "step"]; labs = ["batch\nassembly", "H→D\ncopy", "forward", "backward", "3×RMSprop"]
v = [R[k] for k in ks]; ax.bar(range(5), v, color=["#eb6834", "#eda100", "#2a78d6", "#2a78d6", "#1baf7a"], width=0.65)
for i, x in enumerate(v): ax.text(i, x + 0.25, f"{x:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel("ms / batch"); ax.set_title("Training step breakdown", fontsize=10, loc="left")
savefig("training", "step_profile.png")
