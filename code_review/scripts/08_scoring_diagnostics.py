"""DIAGNOSTIC ONLY (does not touch evaluate_metrics.py / call_id.py and writes nothing into output/).

Re-scores the saved enhanced checkpoints on the unchanged test lists under four
file/tag-level aggregation rules, to quantify how much of the gap to the paper
is attributable to the *decision rule* rather than to the trained weights:
  A whole-file  sum(log p)         (what evaluate_metrics.py / call_id.py do)
  B whole-file  mean(p)            (the paper's 'posterior average', literally)
  C tag-span    sum(log p)         (frames whose centre lies inside the tagged event)
  D tag-span    mean(p)            (closest to the paper's Methods text)
"""
import json, sys, time
import numpy as np, pandas as pd, soundfile as sf, torch
from sklearn.metrics import top_k_accuracy_score
from _common import *

dev = "cuda"
which = sys.argv[1:] or SETS
R = {}
for s in which:
    opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt, cuda=True); load_ckpt(s, cnn, d1, d2, map_location="cuda")
    cnn.eval(); d1.eval(); d2.eval()
    fs = int(opt.fs); wshift = int(fs * int(opt.cw_shift) / 1000.0); n = int(opt.class_lay)
    te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
    scores = {k: np.zeros((len(te), n)) for k in "ABCD"}
    t0 = time.time()
    for fi, (fname, g) in enumerate(te.groupby("file", sort=False)):
        sig = torch.from_numpy(sf.read(os.path.join(ROOT, "nips4b_norm_files", fname))[0]).float().cuda()
        n_frames = max(0, -(-(sig.shape[0] - wlen) // wshift))
        frames = sig.unfold(0, wlen, wshift)[:n_frames]
        lp = torch.zeros(n_frames, n, device=dev)
        with torch.no_grad():
            for b in range(0, n_frames, 1024):
                lp[b:b + 1024] = d2(d1(cnn(frames[b:b + 1024].contiguous())))
        centres = (torch.arange(n_frames, device=dev) * wshift + wlen // 2).float() / fs
        p = lp.exp()
        A = lp.sum(0); B = p.mean(0)
        for r in g.itertuples():
            m = (centres >= r.start) & (centres <= r.start + r.length)
            if m.sum() == 0:
                m = torch.zeros_like(m); m[torch.argmin((centres - (r.start + r.length / 2)).abs())] = True
            i = te.index.get_loc(r.Index)
            scores["A"][i] = torch.softmax(A, 0).cpu().numpy(); scores["B"][i] = B.cpu().numpy()
            scores["C"][i] = torch.softmax(lp[m].sum(0), 0).cpu().numpy(); scores["D"][i] = p[m].mean(0).cpu().numpy()
        if fi % 100 == 0: print(s, fi, f"{time.time()-t0:.0f}s", flush=True)
    y = te.label.values
    R[s] = {k: dict(acc=float((v.argmax(1) == y).mean()),
                    top3=float(top_k_accuracy_score(y, v, k=3, labels=list(range(n)))),
                    top5=float(top_k_accuracy_score(y, v, k=5, labels=list(range(n))))) for k, v in scores.items()}
    R[s]["seconds"] = time.time() - t0
    print(s, json.dumps(R[s]), flush=True)
    np.savez_compressed(os.path.join(DATA, f"rescored_{s}.npz"), y=y, **scores)
    json.dump(R, open(os.path.join(DATA, "scoring_diagnostics_" + "_".join(which) + ".json"), "w"), indent=1)
