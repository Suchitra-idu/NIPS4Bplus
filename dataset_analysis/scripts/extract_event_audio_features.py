"""
For every tagged event in data/events.csv, slice the corresponding audio
segment out of its source wav file and compute the same waveform/spectral
feature set used for whole files. This is the feature table that actually
describes *what a call sounds like* (whole-file features are diluted by
silence and by other overlapping/unrelated calls in the same 5 s clip).

Writes data/event_audio_features.csv.
"""
import os
import sys
import time

import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, WAV_TRAIN_DIR, compute_waveform_features

MIN_SAMPLES = 32  # below this an FFT-based spectral estimate is not meaningful


def main():
    events = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    rows = []
    t0 = time.time()

    cache = {}
    for file_number, group in events.groupby("file_number"):
        wav_path = os.path.join(WAV_TRAIN_DIR, f"nips4b_birds_trainfile{file_number:03d}.wav")
        if wav_path not in cache:
            data, sr = sf.read(wav_path, dtype="float64", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            cache = {wav_path: (data, sr)}  # keep at most one file resident
        data, sr = cache[wav_path]
        n_total = len(data)

        for _, ev in group.iterrows():
            start_idx = max(int(round(ev["start_s"] * sr)), 0)
            end_idx = min(int(round(ev["end_s"] * sr)), n_total)
            segment = data[start_idx:end_idx]

            row = {
                "file_number": file_number,
                "label": ev["label"],
                "taxon": ev["taxon"],
                "call_type": ev["call_type"],
                "start_s": ev["start_s"],
                "duration_s": ev["duration_s"],
                "n_samples": len(segment),
                "too_short_for_spectrum": len(segment) < MIN_SAMPLES,
            }
            if len(segment) >= MIN_SAMPLES:
                feats = compute_waveform_features(segment, sr)
                feats.pop("n_samples", None)
                row.update(feats)
            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "event_audio_features.csv")
    out.to_csv(out_path, index=False)
    n_short = int(out["too_short_for_spectrum"].sum())
    print(
        f"Wrote {len(out)} event-level feature rows to {out_path} "
        f"({n_short} events shorter than {MIN_SAMPLES} samples skipped spectral calc) "
        f"in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
