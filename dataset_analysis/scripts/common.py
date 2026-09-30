"""Shared paths and small helpers for the dataset_analysis scripts."""
import os
import re

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "raw")
WAV_TRAIN_DIR = os.path.join(RAW_DIR, "nips4b_wav", "train")
WAV_TEST_DIR = os.path.join(RAW_DIR, "nips4b_wav", "test")
ANNOT_DIR = os.path.join(RAW_DIR, "nips4bplus_csv", "annotations", "temporal_annotations_nips4b")
ESPECE_LIST_CSV = os.path.join(RAW_DIR, "nips4b_labels", "nips4b_birdchallenge_espece_list.csv")
TPS_CSV = os.path.join(RAW_DIR, "nips4b_labels", "tps_canaux_sr_nbits_TRAIN.csv")

DATA_DIR = os.path.join(REPO_ROOT, "dataset_analysis", "data")
FIG_DIR = os.path.join(REPO_ROOT, "dataset_analysis", "figures")

TRAIN_FILE_RE = re.compile(r"nips4b_birds_trainfile(\d+)\.wav")
TEST_FILE_RE = re.compile(r"nips4b_birds_testfile(\d+)\.wav")
ANNOT_FILE_RE = re.compile(r"annotation_train(\d+)\.csv")


def train_file_number(filename):
    m = TRAIN_FILE_RE.match(filename)
    return int(m.group(1)) if m else None


def test_file_number(filename):
    m = TEST_FILE_RE.match(filename)
    return int(m.group(1)) if m else None


def load_species_table():
    """class_name (e.g. 'Aegcau_call') -> dict(english, scientific, taxon)."""
    import csv

    table = {}
    with open(ESPECE_LIST_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["class name"].strip()
            table[name] = {
                "english_name": row["English_name"].strip(),
                "scientific_name": row["Scientific_name"].strip(),
                "taxon": row["type"].strip(),
            }
    return table


def label_taxon(label, species_table):
    if label in ("Unknown", "Human"):
        return label
    info = species_table.get(label)
    return info["taxon"] if info else "unresolved"


def label_call_type(label):
    if label in ("Unknown", "Human"):
        return label
    for suffix in ("_call", "_song", "_drum"):
        if label.endswith(suffix):
            return suffix[1:]
    return "unknown_suffix"


def compute_waveform_features(data, sr):
    """Amplitude- and spectrum-domain features for an arbitrary 1-D signal
    (a whole file, or a short tagged-event slice). Shared by
    extract_audio_features.py (whole files) and
    extract_event_audio_features.py (per-tag slices) so both operate on
    an identical feature definition.
    """
    from scipy import stats

    n = len(data)
    if n < 2:
        return None

    mean = float(np.mean(data))
    peak = float(np.max(np.abs(data))) or 1e-12
    rms = float(np.sqrt(np.mean(data**2))) or 1e-12
    silence_ratio = float(np.mean(np.abs(data) < 0.02 * peak))

    signs = np.sign(data)
    signs[signs == 0] = 1
    zcr = float(np.mean(signs[1:] != signs[:-1]))

    window = np.hanning(n)
    spec = np.fft.rfft(data * window)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    power_sum = power.sum() or 1e-12
    p_norm = power / power_sum

    centroid = float(np.sum(freqs * p_norm))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p_norm)))
    cumulative = np.cumsum(p_norm)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    geo_mean = float(np.exp(np.mean(np.log(power + 1e-20))))
    arith_mean = float(np.mean(power)) or 1e-20
    flatness = geo_mean / arith_mean
    dominant_idx = int(np.argmax(power[1:]) + 1) if len(power) > 1 else 0
    dominant_freq = float(freqs[dominant_idx])
    spec_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-20)) / np.log(len(p_norm)))
    low_mask = freqs <= 2000
    low_energy_ratio = float(power[low_mask].sum() / power_sum)

    return {
        "n_samples": n,
        "mean": mean,
        "std": float(np.std(data)),
        "variance": float(np.var(data)),
        "skewness": float(stats.skew(data)),
        "kurtosis_excess": float(stats.kurtosis(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "abs_peak": peak,
        "rms": rms,
        "crest_factor": peak / rms,
        "dynamic_range_db": 20 * np.log10(peak / rms),
        "silence_ratio": silence_ratio,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff85_hz": rolloff,
        "spectral_flatness": flatness,
        "dominant_freq_hz": dominant_freq,
        "spectral_entropy": spec_entropy,
        "low_freq_energy_ratio_2khz": low_energy_ratio,
    }
