"""Executable evidence for the claims in CODE_REVIEW. Pure unittest, CPU only, ~20 s.
Run:  .venv/bin/python -m unittest code_review/tests/test_review_claims.py -v
Nothing here modifies the pipeline; it imports the authors' classes and reads files."""
import os, sys, tempfile, unittest, subprocess
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code_review", "scripts")); sys.path.insert(0, ROOT)
from _common import load_options, build_models, CFG, CFG_DEFAULT, SETS, LISTS  # noqa: E402
from dnn_models import SincConv_fast  # noqa: E402
from metrics_utils import compute_metrics  # noqa: E402


class TestArchitecture(unittest.TestCase):
    def test_shapes_and_out_dim(self):
        exp = {"all_classes": 240, "bird_classes": 180, "bird_species": 180}
        for s in SETS:
            cnn, d1, d2, wlen = build_models(load_options(CFG[s])); self.assertEqual(cnn.out_dim, exp[s])
            self.assertEqual(d2(d1(cnn.eval()(torch.randn(3, wlen)))).shape[1], int(load_options(CFG[s]).class_lay))
        for s in SETS:
            cnn, *_ = build_models(load_options(CFG_DEFAULT[s])); self.assertEqual(cnn.out_dim, 300)

    def test_batchnorm_eps_is_the_pooled_length(self):          # finding F3
        cnn, *_ = build_models(load_options(CFG["bird_species"]))
        self.assertEqual([m.eps for m in cnn.bn], [111.0, 21.0, 3.0])
        self.assertNotAlmostEqual(cnn.bn[0].eps, 1e-5)

    def test_sinc_filters_symmetric_centre_one_and_nyquist(self):
        torch.set_grad_enabled(False); L = SincConv_fast(220, 151, 44100); L(torch.zeros(1, 1, 200)); W = L.filters[:, 0].numpy()
        np.testing.assert_allclose(W, W[:, ::-1], atol=1e-7); np.testing.assert_allclose(W[:, 75], 1.0, atol=1e-6)
        self.assertEqual(np.linalg.matrix_rank(W, tol=1e-6 * np.linalg.norm(W, 2)), 76)   # even filters => rank <= 76

    def test_abs_only_on_layernorm_path(self):                  # finding §3 Part 3
        d = load_options(CFG_DEFAULT["bird_species"]); e = load_options(CFG["bird_species"])
        self.assertIn("True", d.cnn_use_laynorm); self.assertNotIn("True", e.cnn_use_laynorm); self.assertIn("True", e.cnn_use_batchnorm)

    def test_parameter_count_matches_metrics_res(self):
        for s in SETS:
            cnn, d1, d2, _ = build_models(load_options(CFG[s])); n = sum(p.numel() for m in (cnn, d1, d2) for p in m.parameters())
            res = dict(l.strip().split("=") for l in open(os.path.join(ROOT, "output", s, "metrics.res")))
            self.assertEqual(n, int(res["trainable_params"]))


class TestData(unittest.TestCase):
    def test_lists_sizes_and_label_range(self):
        sizes = {"all_classes": (4110, 1371, 87), "bird_classes": (3959, 1320, 77), "bird_species": (3959, 1320, 51)}
        for s, (a, b, c) in sizes.items():
            tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
            self.assertEqual((len(tr), len(te)), (a, b)); self.assertEqual(sorted(pd.concat([tr, te]).label.unique()), list(range(c)))

    def test_tag_level_leakage(self):                           # finding F7
        for s in SETS:
            tr = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][0])); te = pd.read_csv(os.path.join(ROOT, "mod_data_lists", LISTS[s][1]))
            self.assertGreater(te.file.isin(set(tr.file)).mean(), 0.98)

    def test_split_seed_determinism(self):
        ann = os.path.join(ROOT, "raw", "nips4bplus_csv", "annotations", "temporal_annotations_nips4b"); sp = os.path.join(ROOT, "raw", "nips4b_labels", "nips4b_birdchallenge_espece_list.csv")
        if not os.path.isdir(ann): self.skipTest("raw data missing")
        out = []
        for seed in ("1234", "1234", "7"):
            d = tempfile.mkdtemp(); subprocess.run([sys.executable, os.path.join(ROOT, "generate_mod_file_lists.py"), ann, sp, d, seed], check=True, capture_output=True)
            out.append(open(os.path.join(d, "mod_bird_sps_test_files.csv")).read())
        self.assertEqual(out[0], out[1]); self.assertNotEqual(out[0], out[2])
        # the tracked canonical lists predate the seed argument: they CANNOT be regenerated (logs.md: "NOT regenerated")
        self.assertNotEqual(out[0], open(os.path.join(ROOT, "mod_data_lists", "mod_bird_sps_test_files.csv")).read())

    def test_normalisation_clips_negative_dominant_files(self):  # finding F8
        import soundfile as sf
        x = np.array([0.2, -0.8, 0.1], dtype=np.float64); y = x / np.abs(np.max(x)); self.assertGreater(np.abs(y).max(), 1.0)
        p = os.path.join(tempfile.mkdtemp(), "t.wav"); sf.write(p, y, 44100); self.assertLessEqual(np.abs(sf.read(p)[0]).max(), 1.0)


class TestMetrics(unittest.TestCase):
    def test_recall_equals_accuracy_and_one_hot_auc_identity(self):   # finding F2
        rng = np.random.RandomState(0); n, C = 600, 10; y = rng.randint(C, size=n); yp = np.where(rng.rand(n) < 0.6, y, rng.randint(C, size=n))
        one_hot = np.eye(C)[yp]; m = compute_metrics(y, yp, one_hot, one_hot, C)
        self.assertAlmostEqual(m["recall"], m["accuracy"]); self.assertAlmostEqual(m["false_negative_rate"], 1 - m["accuracy"])
        self.assertAlmostEqual(m["roc_auc"], (m["accuracy"] + 1 - m["false_positive_rate"]) / 2, places=2)
        self.assertLess(m["top3_accuracy"] - m["accuracy"], 0.15)      # only tie-breaking adds hits


class TestOutputs(unittest.TestCase):
    def test_checkpoint_is_epoch_392(self):                     # finding F6
        for s in SETS:
            last = open(os.path.join(ROOT, "output", s, "time.res")).read().strip().splitlines()[-1]; self.assertTrue(last.startswith("epoch 392,"))
        for s in SETS: self.assertIn("N_eval_epoch=8", open(CFG[s]).read())

    def test_saved_scores_saturated(self):
        for s in SETS:
            P = np.load(os.path.join(ROOT, "output", s, "predictions.npz")); self.assertGreater((P["y_score"].max(1) == 1.0).mean(), 0.99)

    def test_sinc_cutoffs_barely_moved(self):                   # finding F4
        import torch
        for s in SETS:
            cnn, d1, d2, _ = build_models(load_options(CFG[s])); init = cnn.conv[0].low_hz_.detach().clone()
            ck = torch.load(os.path.join(ROOT, "output", s, "model_raw.pkl"), map_location="cpu")
            self.assertLess((ck["CNN_model_par"]["conv.0.low_hz_"] - init).abs().max().item(), 10.0)


if __name__ == "__main__":
    unittest.main()
