"""Shared helpers for the code-review analysis scripts.

Read-only with respect to the pipeline: nothing here edits or monkey-patches
call_id.py / evaluate_metrics.py / dnn_models.py. It only imports the authors'
model classes and re-reads their cfgs and checkpoints.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CR = os.path.join(ROOT, "code_review")
FIG = os.path.join(CR, "figures")
DATA = os.path.join(CR, "data")
sys.path.insert(0, ROOT)

SETS = ["all_classes", "bird_classes", "bird_species"]
CFG = {s: os.path.join(ROOT, "mod_cfg", f"mod_nips4bplus_{s}.cfg") for s in SETS}
CFG_DEFAULT = {s: os.path.join(ROOT, "cfg", f"nips4bplus_{s}.cfg") for s in SETS}
LISTS = {
    "all_classes": ("mod_all_classes_train_files.csv", "mod_all_classes_test_files.csv"),
    "bird_classes": ("mod_bird_classes_train_files.csv", "mod_bird_classes_test_files.csv"),
    "bird_species": ("mod_bird_sps_train_files.csv", "mod_bird_sps_test_files.csv"),
}
# categorical slots 1-3 of the reference palette (validated all-pairs)
COLORS = {"all_classes": "#2a78d6", "bird_classes": "#eb6834", "bird_species": "#1baf7a"}
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
})


def savefig(sub, name):
    d = os.path.join(FIG, sub)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    print("wrote", os.path.relpath(p, ROOT))
    return p


def load_options(cfg_path):
    """Use the authors' own data_io.read_conf (it parses sys.argv)."""
    from data_io import read_conf
    old = sys.argv
    sys.argv = ["x", f"--cfg={cfg_path}"]
    try:
        return read_conf()
    finally:
        sys.argv = old


def build_models(opt, cuda=False):
    """Mirror of the model construction block in call_id.py:219-267."""
    from data_io import str_to_bool
    from dnn_models import MLP
    from dnn_models import SincNet as CNN
    fs = int(opt.fs)
    wlen = int(fs * int(opt.cw_len) / 1000.0)
    L = lambda s, f: list(map(f, s.split(",")))
    cnn = CNN({"input_dim": wlen, "fs": fs, "cnn_N_filt": L(opt.cnn_N_filt, int),
               "cnn_len_filt": L(opt.cnn_len_filt, int), "cnn_max_pool_len": L(opt.cnn_max_pool_len, int),
               "cnn_use_laynorm_inp": str_to_bool(opt.cnn_use_laynorm_inp),
               "cnn_use_batchnorm_inp": str_to_bool(opt.cnn_use_batchnorm_inp),
               "cnn_use_laynorm": L(opt.cnn_use_laynorm, str_to_bool),
               "cnn_use_batchnorm": L(opt.cnn_use_batchnorm, str_to_bool),
               "cnn_act": L(opt.cnn_act, str), "cnn_drop": L(opt.cnn_drop, float)})
    fc_lay = L(opt.fc_lay, int)
    d1 = MLP({"input_dim": cnn.out_dim, "fc_lay": fc_lay, "fc_drop": L(opt.fc_drop, float),
              "fc_use_batchnorm": L(opt.fc_use_batchnorm, str_to_bool),
              "fc_use_laynorm": L(opt.fc_use_laynorm, str_to_bool),
              "fc_use_laynorm_inp": str_to_bool(opt.fc_use_laynorm_inp),
              "fc_use_batchnorm_inp": str_to_bool(opt.fc_use_batchnorm_inp), "fc_act": L(opt.fc_act, str)})
    d2 = MLP({"input_dim": fc_lay[-1], "fc_lay": L(opt.class_lay, int), "fc_drop": L(opt.class_drop, float),
              "fc_use_batchnorm": L(opt.class_use_batchnorm, str_to_bool),
              "fc_use_laynorm": L(opt.class_use_laynorm, str_to_bool),
              "fc_use_laynorm_inp": str_to_bool(opt.class_use_laynorm_inp),
              "fc_use_batchnorm_inp": str_to_bool(opt.class_use_batchnorm_inp), "fc_act": L(opt.class_act, str)})
    if cuda:
        cnn.cuda(); d1.cuda(); d2.cuda()
    return cnn, d1, d2, wlen


def load_ckpt(name, cnn, d1, d2, map_location="cpu"):
    import torch
    ck = torch.load(os.path.join(ROOT, "output", name, "model_raw.pkl"), map_location=map_location)
    cnn.load_state_dict(ck["CNN_model_par"]); d1.load_state_dict(ck["DNN1_model_par"]); d2.load_state_dict(ck["DNN2_model_par"])
    return ck


def class_names(name):
    import pandas as pd
    tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[name][0]))
    col = "species" if name == "bird_species" else "class_name"
    m = tr.drop_duplicates("label").set_index("label")[col].sort_index()
    return m.to_dict()


def sample_windows(csv_name, wlen, n, seed=0, fact_amp=0.0, folder=None):
    """Read-only re-implementation of call_id.create_batches_rnd (call_id.py:63-120)
    used only by diagnostics. Returns (batch[n,wlen], labels, rows)."""
    import pandas as pd, soundfile as sf
    folder = folder or os.path.join(ROOT, "nips4b_norm_files")
    df = pd.read_csv(os.path.join(ROOT, "mod_data_lists", csv_name))
    rng = np.random.RandomState(seed)
    cache = {}
    X = np.zeros((n, wlen)); y = np.zeros(n, dtype=int); rows = []
    for i, k in enumerate(rng.randint(len(df), size=n)):
        r = df.iloc[k]
        if r.file not in cache:
            cache[r.file] = sf.read(os.path.join(folder, r.file))
        sig, fs = cache[r.file]
        t0 = int(r.start * fs); t1 = t0 + int(r.length * fs)
        if t1 - t0 > wlen:
            s = rng.randint(t0, t1 - wlen)
        else:
            lo, hi = max(0, t1 - wlen), min(t0, sig.shape[0] - wlen)
            s = lo if lo == hi else rng.randint(lo, hi)
        X[i] = sig[s:s + wlen] * rng.uniform(1 - fact_amp, 1 + fact_amp); y[i] = r.label; rows.append(k)
    return X, y, rows
