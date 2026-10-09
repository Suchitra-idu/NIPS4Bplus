"""DIAGNOSTIC (GPU). Frame-level behaviour of the trained models on the test files:
accuracy of single frames inside/outside tags, confidence, background predictions,
raw sum-log-prob margins, and example posterior timelines."""
import json, sys
import numpy as np, pandas as pd, soundfile as sf, torch
from _common import *

adir = os.path.join(ROOT, "raw", "nips4bplus_csv", "annotations", "temporal_annotations_nips4b")
def ann_of(f):
    try: return pd.read_csv(os.path.join(adir, "annotation_train" + f[-7:-4] + ".csv"), header=None).values
    except Exception: return np.zeros((0, 3))
R = {}; EX = {}
for s in SETS:
    opt = load_options(CFG[s]); cnn, d1, d2, wlen = build_models(opt, cuda=True); load_ckpt(s, cnn, d1, d2, "cuda"); [m.eval() for m in (cnn, d1, d2)]
    fs = 44100; wshift = 44; n = int(opt.class_lay)
    te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
    z = np.load(os.path.join(DATA, f"rescored_{s}.npz")); y = z["y"]; A = z["A"].argmax(1); D = z["D"].argmax(1)
    pick = {}
    if s == "bird_species":
        for name, cond in (("fixed_by_span", (A != y) & (D == y)), ("correct_both", (A == y) & (D == y)), ("wrong_both", (A != y) & (D != y))):
            idx = np.where(cond)[0]; pick[name] = int(idx[len(idx) // 3]) if len(idx) else None
    pickfiles = {te.file.iloc[i]: k for k, i in pick.items() if i is not None}
    rows = dict(margin=np.zeros(len(te)), nfr=np.zeros(len(te)), in_frame_acc=np.zeros(len(te)), in_votes_true=np.zeros(len(te)),
                in_conf=np.zeros(len(te)), n_in=np.zeros(len(te)), A_correct=(A == y))
    bg_hist = np.zeros(n); bg_conf = []; bg_ent = []; in_ent = []; bg_frames = 0
    for fi, (fname, g) in enumerate(te.groupby("file", sort=False)):
        sig = torch.from_numpy(sf.read(os.path.join(ROOT, "nips4b_norm_files", fname))[0]).float().cuda()
        F = max(0, -(-(sig.shape[0] - wlen) // wshift)); fr = sig.unfold(0, wlen, wshift)[:F]
        lp = torch.zeros(F, n, device="cuda")
        with torch.no_grad():
            for b in range(0, F, 1024): lp[b:b + 1024] = d2(d1(cnn(fr[b:b + 1024].contiguous())))
        p = lp.exp(); am = p.argmax(1); cen = (torch.arange(F, device="cuda") * wshift + wlen // 2).float() / fs
        conf = p.max(1).values; ent = -(p * lp).sum(1)
        anyspan = torch.zeros(F, dtype=torch.bool, device="cuda")
        for st, ln, _ in ann_of(fname): anyspan |= (cen >= st) & (cen <= st + ln)
        bg = ~anyspan
        if bg.sum() > 0:
            bg_hist += torch.bincount(am[bg], minlength=n).cpu().numpy(); bg_conf.append(conf[bg].mean().item()); bg_ent.append(ent[bg].mean().item()); bg_frames += int(bg.sum())
        tot = lp.sum(0); top2 = torch.topk(tot, 2).values
        for r in g.itertuples():
            i = te.index.get_loc(r.Index); m = (cen >= r.start) & (cen <= r.start + r.length)
            rows["margin"][i] = (top2[0] - top2[1]).item(); rows["nfr"][i] = F; rows["n_in"][i] = int(m.sum())
            if m.sum() > 0:
                rows["in_frame_acc"][i] = (am[m] == r.label).float().mean().item(); rows["in_conf"][i] = conf[m].mean().item(); in_ent.append(ent[m].mean().item())
        if fname in pickfiles:
            r0 = g.iloc[0] if pickfiles[fname] is None else te.iloc[pick[pickfiles[fname]]]
            EX[(s, pickfiles[fname])] = dict(file=fname, label=int(r0.label), start=float(r0.start), length=float(r0.length), cen=cen.cpu().numpy(),
                                             p_true=p[:, int(r0.label)].cpu().numpy(), p_top=conf.cpu().numpy(), pred=am.cpu().numpy(),
                                             others=[(float(a), float(b), str(c)) for a, b, c in ann_of(fname)])
    R[s] = dict(median_sumlogp_margin_nats=float(np.median(rows["margin"])), p10_margin=float(np.percentile(rows["margin"], 10)), frames_per_file_median=float(np.median(rows["nfr"])),
                in_span_frame_acc_mean=float(rows["in_frame_acc"][rows["n_in"] > 0].mean()), in_span_frame_acc_median=float(np.median(rows["in_frame_acc"][rows["n_in"] > 0])),
                in_span_conf=float(rows["in_conf"][rows["n_in"] > 0].mean()), bg_conf=float(np.mean(bg_conf)), bg_entropy=float(np.mean(bg_ent)), in_entropy=float(np.mean(in_ent)),
                bg_frames_total=bg_frames, bg_top_class_share=float(bg_hist.max() / bg_hist.sum()), bg_top3_share=float(np.sort(bg_hist)[-3:].sum() / bg_hist.sum()),
                rows_in_span_frames_lt_10=int((rows["n_in"] < 10).sum()),
                margin_gt_20_nats=float((rows["margin"] > 20).mean()), margin_gt_100_nats=float((rows["margin"] > 100).mean()),
                frame_acc_vs_file_acc_corr=float(np.corrcoef(rows["in_frame_acc"][rows["n_in"] > 0], rows["A_correct"][rows["n_in"] > 0])[0, 1]))
    np.savez_compressed(os.path.join(DATA, f"frame_level_{s}.npz"), bg_hist=bg_hist, **{k: v for k, v in rows.items()})
    print(s, json.dumps(R[s]), flush=True)
json.dump(R, open(os.path.join(DATA, "frame_level.json"), "w"), indent=1)
np.save(os.path.join(DATA, "frame_examples.npy"), np.array([EX], dtype=object), allow_pickle=True)
