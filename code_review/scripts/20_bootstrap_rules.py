import json
import numpy as np, pandas as pd
from _common import *
rng = np.random.RandomState(0); R = {}
for s in SETS:
    z = np.load(os.path.join(DATA, f"rescored_{s}.npz")); y = z["y"]; te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
    files = te.file.unique(); idx = {f: np.where(te.file.values == f)[0] for f in files}
    corr = {k: (z[k].argmax(1) == y).astype(float) for k in "ABCD"}
    boots = {k: [] for k in "ABCD"}; diff = []
    for _ in range(3000):
        ii = np.concatenate([idx[f] for f in rng.choice(files, len(files))])
        for k in "ABCD": boots[k].append(corr[k][ii].mean())
        diff.append(corr["D"][ii].mean() - corr["A"][ii].mean())
    R[s] = {k: dict(acc=float(corr[k].mean()), lo=float(np.percentile(boots[k], 2.5)), hi=float(np.percentile(boots[k], 97.5))) for k in "ABCD"}
    R[s]["D_minus_A"] = dict(mean=float(corr["D"].mean() - corr["A"].mean()), lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)))
    # McNemar-style discordant counts
    a, d = corr["A"].astype(bool), corr["D"].astype(bool); R[s]["discordant"] = dict(A_only=int((a & ~d).sum()), D_only=int((~a & d).sum()), both=int((a & d).sum()), neither=int((~a & ~d).sum()))
json.dump(R, open(os.path.join(DATA, "bootstrap_rules.json"), "w"), indent=1); print(json.dumps(R, indent=1))
