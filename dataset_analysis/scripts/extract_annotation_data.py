"""
Parse the 674 strong temporal-annotation files (raw/nips4bplus_csv/annotations)
into an event-level table, plus a file-level summary table, plus a
sweep-line "simultaneous active classes" duration histogram (reproducing,
independently, the kind of statistic behind Morfi et al. Fig. 3 / Fig. 5).

Writes:
  data/events.csv           one row per tagged event (start, duration, label, ...)
  data/file_summary.csv     one row per of the 687 train files
  data/simultaneity.csv     duration-of-recording broken down by number of
                             simultaneously active classes
  data/cooccurrence_weak.csv    file-level (weak) label co-occurrence matrix
  data/cooccurrence_overlap.csv temporal-overlap label co-occurrence matrix
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    ANNOT_DIR,
    ANNOT_FILE_RE,
    DATA_DIR,
    TPS_CSV,
    WAV_TRAIN_DIR,
    label_call_type,
    label_taxon,
    load_species_table,
    train_file_number,
)


def load_declared_durations():
    rows = []
    with open(TPS_CSV, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rows.append(float(parts[0]))
    return rows  # index 0 -> trainfile001, length 687


def load_events(species_table):
    events = []
    for filename in sorted(os.listdir(ANNOT_DIR)):
        m = ANNOT_FILE_RE.match(filename)
        if not m:
            continue
        file_num = int(m.group(1))
        path = os.path.join(ANNOT_DIR, filename)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                start, dur, label = float(row[0]), float(row[1]), row[2].strip()
                events.append(
                    {
                        "file_number": file_num,
                        "start_s": start,
                        "duration_s": dur,
                        "end_s": start + dur,
                        "label": label,
                        "taxon": label_taxon(label, species_table),
                        "call_type": label_call_type(label),
                    }
                )
    return pd.DataFrame(events)


def sweep_line_simultaneity(file_events, file_duration):
    """Return list of (level, duration) covering the whole file timeline,
    where `level` = number of simultaneously-active labelled events."""
    if file_events.empty:
        return [(0, file_duration)]

    boundaries = set([0.0, file_duration])
    points = []
    for _, ev in file_events.iterrows():
        s, e = ev["start_s"], min(ev["end_s"], file_duration)
        if e <= s:
            continue
        points.append((s, 1))
        points.append((e, -1))
        boundaries.add(s)
        boundaries.add(e)
    points.sort()

    times = sorted(boundaries)
    level = 0
    p_idx = 0
    out = []
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        while p_idx < len(points) and points[p_idx][0] <= t0:
            level += points[p_idx][1]
            p_idx += 1
        seg = t1 - t0
        if seg > 1e-12:
            out.append((max(level, 0), seg))
    return out


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    species_table = load_species_table()
    declared_durations = load_declared_durations()

    events = load_events(species_table)
    events.to_csv(os.path.join(DATA_DIR, "events.csv"), index=False)
    print(f"Parsed {len(events)} events across {events['file_number'].nunique()} files")

    all_train_files = sorted(
        {train_file_number(f) for f in os.listdir(WAV_TRAIN_DIR) if f.endswith(".wav")}
    )
    # An annotation *file* existing on disk is a stronger and different
    # condition from "produced >=1 parsed event": 105 of the 674 annotation
    # files on disk contain no rows at all (just a bare CRLF) -- these are
    # recordings the annotator judged to have no taggable vocalisation
    # (background noise), not recordings that are simply undocumented.
    # We keep that distinction explicit rather than collapsing it.
    annotation_file_nums = {
        int(m.group(1))
        for m in (ANNOT_FILE_RE.match(f) for f in os.listdir(ANNOT_DIR))
        if m
    }
    annotated_files = set(events["file_number"].unique())
    empty_annotation_files = annotation_file_nums - annotated_files
    missing_annotation_files = set(all_train_files) - annotation_file_nums
    print(
        f"{len(annotation_file_nums)} annotation files on disk "
        f"({len(annotated_files)} with >=1 event, {len(empty_annotation_files)} empty); "
        f"{len(missing_annotation_files)} train files have no annotation file at all"
    )

    # --- file-level summary -------------------------------------------------
    summary_rows = []
    simultaneity_acc = {}  # level -> total seconds
    for num in all_train_files:
        duration = declared_durations[num - 1]
        has_annotation_file = num in annotation_file_nums
        has_events = num in annotated_files
        fev = events[events["file_number"] == num] if has_events else events.iloc[0:0]
        n_events = len(fev)
        distinct_labels = sorted(fev["label"].unique().tolist())
        distinct_labels_no_meta = [l for l in distinct_labels if l not in ("Unknown", "Human")]
        annotated_duration = float(fev["duration_s"].sum())

        summary_rows.append(
            {
                "file_number": num,
                "has_annotation_file": has_annotation_file,
                "annotation_file_empty": has_annotation_file and not has_events,
                "file_duration_s": duration,
                "n_events": n_events,
                "n_distinct_labels": len(distinct_labels),
                "n_distinct_species_labels": len(distinct_labels_no_meta),
                "has_unknown": "Unknown" in distinct_labels,
                "has_human": "Human" in distinct_labels,
                "annotated_duration_s": annotated_duration,
                "annotated_fraction": (
                    min(annotated_duration / duration, 1.0) if duration > 0 else np.nan
                ),
                "labels": ";".join(distinct_labels),
            }
        )

        if has_annotation_file:
            # Present on disk (possibly empty -> sweep-line degenerates to
            # a single level-0 segment spanning the whole file).
            for level, seg in sweep_line_simultaneity(fev, duration):
                simultaneity_acc[level] = simultaneity_acc.get(level, 0.0) + seg
        else:
            # No annotation file at all: one of the 13 files (6 ambiguous +
            # 7 insect-only) deliberately left undocumented. Genuinely
            # unknown activity level -- kept out of the level-0 bucket.
            simultaneity_acc["unannotated"] = simultaneity_acc.get("unannotated", 0.0) + duration

    file_summary = pd.DataFrame(summary_rows)
    file_summary.to_csv(os.path.join(DATA_DIR, "file_summary.csv"), index=False)

    simultaneity = pd.DataFrame(
        [{"level": k, "total_duration_s": v} for k, v in simultaneity_acc.items()]
    )
    simultaneity.to_csv(os.path.join(DATA_DIR, "simultaneity.csv"), index=False)

    # --- co-occurrence matrices ---------------------------------------------
    top_labels = events["label"].value_counts().index.tolist()

    # (a) weak / file-level co-occurrence: do labels A and B ever appear
    # together in the same file (regardless of timing)?
    weak_matrix = pd.DataFrame(0, index=top_labels, columns=top_labels, dtype=int)
    for num in annotated_files:
        labels_in_file = events.loc[events["file_number"] == num, "label"].unique()
        for i, a in enumerate(labels_in_file):
            for b in labels_in_file:
                weak_matrix.loc[a, b] += 1
    weak_matrix.to_csv(os.path.join(DATA_DIR, "cooccurrence_weak.csv"))

    # (b) strong / temporal-overlap co-occurrence: do labels A and B ever
    # have overlapping [start, end) intervals in the same file?
    overlap_matrix = pd.DataFrame(0, index=top_labels, columns=top_labels, dtype=int)
    for num, fev in events.groupby("file_number"):
        fev = fev.reset_index(drop=True)
        n = len(fev)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = fev.loc[i], fev.loc[j]
                if a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]:
                    overlap_matrix.loc[a["label"], b["label"]] += 1
    overlap_matrix.to_csv(os.path.join(DATA_DIR, "cooccurrence_overlap.csv"))

    print("Wrote file_summary.csv, simultaneity.csv, cooccurrence_weak.csv, cooccurrence_overlap.csv")


if __name__ == "__main__":
    main()
