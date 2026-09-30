"""
Per-file waveform + spectral feature extraction for every raw NIPS4B wav
(687 train + 1000 test = 1687 files).

Writes dataset_analysis/data/audio_features.csv, one row per file.
Also cross-checks measured duration/sample rate against the authors'
declared values in tps_canaux_sr_nbits_TRAIN.csv (train set only).
"""
import csv
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    DATA_DIR,
    TPS_CSV,
    WAV_TEST_DIR,
    WAV_TRAIN_DIR,
    compute_waveform_features,
    test_file_number,
    train_file_number,
)

FIELDS = [
    "split",
    "file_number",
    "filename",
    "duration_s",
    "declared_duration_s",
    "duration_mismatch_s",
    "sample_rate",
    "channels",
    "subtype",
    "n_samples",
    "mean",
    "std",
    "variance",
    "skewness",
    "kurtosis_excess",
    "min",
    "max",
    "abs_peak",
    "rms",
    "crest_factor",
    "dynamic_range_db",
    "silence_ratio",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "spectral_flatness",
    "dominant_freq_hz",
    "spectral_entropy",
    "low_freq_energy_ratio_2khz",
]


def load_declared_durations():
    """tps_canaux_sr_nbits_TRAIN.csv is 687 rows, one per train file, in
    file-number order, columns: duration_s, channels, sample_rate, nbits."""
    rows = []
    with open(TPS_CSV, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rows.append(
                {
                    "duration_s": float(parts[0]),
                    "channels": int(parts[1]),
                    "sample_rate": int(parts[2]),
                    "nbits": int(parts[3]),
                }
            )
    return rows  # index 0 -> trainfile001


def analyze_file(path):
    info = sf.info(path)
    data, sr = sf.read(path, dtype="float64", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = len(data) / sr

    feats = compute_waveform_features(data, sr)
    feats["duration_s"] = duration
    feats["sample_rate"] = sr
    feats["channels"] = info.channels
    feats["subtype"] = info.subtype
    return feats


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    declared = load_declared_durations()

    out_path = os.path.join(DATA_DIR, "audio_features.csv")
    t0 = time.time()
    n_done = 0
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDS)
        writer.writeheader()

        for split, wav_dir, numbering_fn in (
            ("train", WAV_TRAIN_DIR, train_file_number),
            ("test", WAV_TEST_DIR, test_file_number),
        ):
            filenames = sorted(os.listdir(wav_dir))
            for filename in filenames:
                if not filename.endswith(".wav"):
                    continue
                num = numbering_fn(filename)
                path = os.path.join(wav_dir, filename)
                feats = analyze_file(path)

                declared_duration = None
                if split == "train" and num is not None and 1 <= num <= len(declared):
                    declared_duration = declared[num - 1]["duration_s"]

                row = {
                    "split": split,
                    "file_number": num,
                    "filename": filename,
                    "declared_duration_s": declared_duration,
                    "duration_mismatch_s": (
                        feats["duration_s"] - declared_duration
                        if declared_duration is not None
                        else None
                    ),
                }
                row.update(feats)
                writer.writerow(row)
                n_done += 1
                if n_done % 200 == 0:
                    elapsed = time.time() - t0
                    print(f"  processed {n_done} files ({elapsed:.1f}s elapsed)")

    print(f"Done. Wrote {n_done} rows to {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
