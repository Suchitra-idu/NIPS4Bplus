"""
Core statistical analysis: descriptive stats, normality tests, distribution
fitting, class-imbalance metrics, correlations, and train/test distribution
shift, computed from the tables produced by extract_audio_features.py,
extract_annotation_data.py and extract_event_audio_features.py.

All numeric results referenced in DATASET_ANALYSIS.md come from the CSV/JSON
files this script writes into dataset_analysis/data/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR

RNG_SEED = 1234


def descriptive_stats(series):
    s = series.dropna().astype(float)
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "variance": float(s.var(ddof=1)),
        "skewness": float(stats.skew(s)),
        "kurtosis_excess": float(stats.kurtosis(s)),
        "min": float(s.min()),
        "p25": float(q1),
        "median": float(med),
        "p75": float(q3),
        "max": float(s.max()),
        "iqr": float(q3 - q1),
        "cv": float(s.std(ddof=1) / s.mean()) if s.mean() != 0 else np.nan,
    }


def normality_tests(series, label):
    s = series.dropna().astype(float).values
    n = len(s)
    rng = np.random.default_rng(RNG_SEED)
    s_for_shapiro = s if n <= 5000 else rng.choice(s, 5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(s_for_shapiro)
    dagostino_stat, dagostino_p = (
        stats.normaltest(s) if n >= 20 else (np.nan, np.nan)
    )
    mu, sigma = s.mean(), s.std(ddof=1)
    z = (s - mu) / sigma
    ks_stat, ks_p = stats.kstest(z, "norm")
    return {
        "feature": label,
        "n": n,
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "dagostino_k2_stat": float(dagostino_stat),
        "dagostino_k2_p": float(dagostino_p),
        "ks_vs_normal_stat": float(ks_stat),
        "ks_vs_normal_p": float(ks_p),
        "looks_normal_at_0.01": bool(shapiro_p > 0.01 and dagostino_p > 0.01),
    }


CANDIDATE_DISTS = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min,
    "exponential": stats.expon,
}


def fit_distributions(series, label, positive_only=True):
    s = series.dropna().astype(float).values
    if positive_only:
        s = s[s > 0]
    results = []
    for name, dist in CANDIDATE_DISTS.items():
        try:
            params = dist.fit(s)
            log_likelihood = np.sum(dist.logpdf(s, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            ks_stat, ks_p = stats.kstest(s, dist.cdf, args=params)
            results.append(
                {
                    "feature": label,
                    "distribution": name,
                    "params": [float(p) for p in params],
                    "log_likelihood": float(log_likelihood),
                    "aic": float(aic),
                    "ks_stat": float(ks_stat),
                    "ks_p": float(ks_p),
                }
            )
        except Exception as e:  # some dists can fail to fit degenerate data
            results.append({"feature": label, "distribution": name, "error": str(e)})
    results.sort(key=lambda r: r.get("aic", np.inf))
    return results


def gini_coefficient(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    cum = np.cumsum(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * cum[-1])) - (n + 1) / n)


def class_imbalance_metrics(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    n_classes = len(counts)
    total = counts.sum()
    p = counts / total
    shannon = float(-np.sum(p * np.log2(p)))
    max_entropy = np.log2(n_classes)
    ranks = np.arange(1, n_classes + 1)
    sorted_counts = np.sort(counts)[::-1]
    slope, intercept, r, p_val, se = stats.linregress(
        np.log(ranks), np.log(sorted_counts)
    )
    return {
        "n_classes": int(n_classes),
        "total_count": float(total),
        "min_count": float(counts.min()),
        "max_count": float(counts.max()),
        "median_count": float(np.median(counts)),
        "imbalance_ratio_max_min": float(counts.max() / counts.min()),
        "imbalance_ratio_max_median": float(counts.max() / np.median(counts)),
        "gini_coefficient": gini_coefficient(counts),
        "shannon_entropy_bits": shannon,
        "normalized_entropy": float(shannon / max_entropy),
        "effective_n_classes": float(2**shannon),
        "zipf_slope": float(slope),
        "zipf_r_squared": float(r**2),
    }


def main():
    audio = pd.read_csv(os.path.join(DATA_DIR, "audio_features.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    event_feats = pd.read_csv(os.path.join(DATA_DIR, "event_audio_features.csv"))
    file_summary = pd.read_csv(os.path.join(DATA_DIR, "file_summary.csv"))
    simultaneity = pd.read_csv(os.path.join(DATA_DIR, "simultaneity.csv"))

    # ---------------------------------------------------------------- #
    # 1. Whole-file descriptive stats + normality, by split
    # ---------------------------------------------------------------- #
    audio_features_to_test = [
        "duration_s",
        "rms",
        "crest_factor",
        "dynamic_range_db",
        "silence_ratio",
        "zero_crossing_rate",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_flatness",
        "skewness",
        "kurtosis_excess",
    ]

    desc_rows = []
    norm_rows = []
    for split in ["train", "test", "all"]:
        sub = audio if split == "all" else audio[audio["split"] == split]
        for feat in audio_features_to_test:
            d = descriptive_stats(sub[feat])
            d.update({"split": split, "feature": feat})
            desc_rows.append(d)
            n = normality_tests(sub[feat], feat)
            n["split"] = split
            norm_rows.append(n)
    pd.DataFrame(desc_rows).to_csv(os.path.join(DATA_DIR, "descriptive_stats_audio.csv"), index=False)
    pd.DataFrame(norm_rows).to_csv(os.path.join(DATA_DIR, "normality_tests_audio.csv"), index=False)

    # log-transformed versions of strictly-positive, right-skewed features
    log_norm_rows = []
    for feat in ["rms", "crest_factor", "spectral_bandwidth_hz"]:
        s = np.log(audio[feat].dropna() + 1e-12)
        n = normality_tests(s, f"log({feat})")
        n["split"] = "all"
        log_norm_rows.append(n)
    pd.DataFrame(log_norm_rows).to_csv(
        os.path.join(DATA_DIR, "normality_tests_audio_log.csv"), index=False
    )

    # ---------------------------------------------------------------- #
    # 2. Distribution fitting for whole-file features
    # ---------------------------------------------------------------- #
    fit_rows = []
    for feat in ["rms", "crest_factor", "spectral_centroid_hz", "spectral_bandwidth_hz"]:
        fit_rows.extend(fit_distributions(audio[feat], feat))
    pd.DataFrame(fit_rows).to_csv(os.path.join(DATA_DIR, "distribution_fits_audio.csv"), index=False)

    # ---------------------------------------------------------------- #
    # 3. Whole-file duration: mixture structure (full 5s segments vs. tail)
    # ---------------------------------------------------------------- #
    full_thresh = 5.0
    dur_mix = {
        split: {
            "n_total": int((audio["split"] == split).sum()),
            "n_full_length": int(((audio["split"] == split) & (audio["duration_s"] >= full_thresh)).sum()),
            "n_partial": int(((audio["split"] == split) & (audio["duration_s"] < full_thresh)).sum()),
        }
        for split in ["train", "test"]
    }
    with open(os.path.join(DATA_DIR, "duration_mixture.json"), "w") as f:
        json.dump(dur_mix, f, indent=2)

    # ---------------------------------------------------------------- #
    # 4. Train vs test distribution shift (whole-file features)
    # ---------------------------------------------------------------- #
    shift_rows = []
    train_df = audio[audio["split"] == "train"]
    test_df = audio[audio["split"] == "test"]
    for feat in audio_features_to_test:
        a, b = train_df[feat].dropna().values, test_df[feat].dropna().values
        ks_stat, ks_p = stats.ks_2samp(a, b)
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        shift_rows.append(
            {
                "feature": feat,
                "train_mean": float(a.mean()),
                "test_mean": float(b.mean()),
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
                "mannwhitney_p": float(u_p),
                "significant_shift_at_0.01": bool(ks_p < 0.01),
            }
        )
    pd.DataFrame(shift_rows).to_csv(os.path.join(DATA_DIR, "train_test_shift.csv"), index=False)

    # ---------------------------------------------------------------- #
    # 5. Correlation matrices (whole-file features)
    # ---------------------------------------------------------------- #
    corr_feats = [
        "duration_s", "rms", "crest_factor", "dynamic_range_db", "silence_ratio",
        "zero_crossing_rate", "spectral_centroid_hz", "spectral_bandwidth_hz",
        "spectral_rolloff85_hz", "spectral_flatness", "skewness", "kurtosis_excess",
    ]
    audio[corr_feats].corr(method="pearson").to_csv(os.path.join(DATA_DIR, "correlations_audio_pearson.csv"))
    audio[corr_feats].corr(method="spearman").to_csv(os.path.join(DATA_DIR, "correlations_audio_spearman.csv"))

    # ---------------------------------------------------------------- #
    # 6. Event (tag) level: descriptive stats + normality + distribution fit
    # ---------------------------------------------------------------- #
    event_desc_rows = [dict(descriptive_stats(events["duration_s"]), feature="event_duration_s", taxon="all")]
    event_norm_rows = [dict(normality_tests(events["duration_s"], "event_duration_s"), taxon="all")]
    log_dur = np.log(events["duration_s"])
    event_norm_rows.append(dict(normality_tests(log_dur, "log(event_duration_s)"), taxon="all"))

    for taxon, grp in events.groupby("taxon"):
        if len(grp) < 8:
            continue
        d = descriptive_stats(grp["duration_s"])
        d.update({"feature": "event_duration_s", "taxon": taxon})
        event_desc_rows.append(d)

    pd.DataFrame(event_desc_rows).to_csv(os.path.join(DATA_DIR, "descriptive_stats_events.csv"), index=False)
    pd.DataFrame(event_norm_rows).to_csv(os.path.join(DATA_DIR, "normality_tests_events.csv"), index=False)

    event_fit_rows = fit_distributions(events["duration_s"], "event_duration_s")
    for taxon in ["bird", "insect"]:
        sub = events[events["taxon"] == taxon]["duration_s"]
        event_fit_rows.extend(fit_distributions(sub, f"event_duration_s[{taxon}]"))
    pd.DataFrame(event_fit_rows).to_csv(os.path.join(DATA_DIR, "distribution_fits_events.csv"), index=False)

    # ---------------------------------------------------------------- #
    # 7. Per-class (label) statistics
    # ---------------------------------------------------------------- #
    label_counts = events["label"].value_counts()
    per_class = events.groupby("label").agg(
        n_events=("duration_s", "size"),
        total_duration_s=("duration_s", "sum"),
        mean_duration_s=("duration_s", "mean"),
        median_duration_s=("duration_s", "median"),
        std_duration_s=("duration_s", "std"),
        taxon=("taxon", "first"),
        call_type=("call_type", "first"),
    ).reset_index()

    spectral_by_label = event_feats[~event_feats["too_short_for_spectrum"]].groupby("label").agg(
        mean_spectral_centroid_hz=("spectral_centroid_hz", "mean"),
        mean_rms=("rms", "mean"),
        mean_zero_crossing_rate=("zero_crossing_rate", "mean"),
        mean_dominant_freq_hz=("dominant_freq_hz", "mean"),
    ).reset_index()
    per_class = per_class.merge(spectral_by_label, on="label", how="left")
    per_class = per_class.sort_values("n_events", ascending=False)
    per_class.to_csv(os.path.join(DATA_DIR, "per_class_stats.csv"), index=False)

    # ---------------------------------------------------------------- #
    # 8. Class imbalance metrics (three canonical tag selections)
    # ---------------------------------------------------------------- #
    species_events = events[~events["label"].isin(["Unknown", "Human"])]
    all_classes_counts = events["label"].value_counts().values  # 87+2 incl. Unknown/Human as tagged in raw data
    bird_insect_counts = species_events["label"].value_counts().values  # excludes Unknown/Human -> "All classes" (87)
    bird_only_counts = species_events[species_events["taxon"] == "bird"]["label"].value_counts().values

    imbalance = {
        "raw_labels_incl_unknown_human": class_imbalance_metrics(all_classes_counts),
        "species_labels_87_all_classes": class_imbalance_metrics(bird_insect_counts),
        "bird_only_labels": class_imbalance_metrics(bird_only_counts),
    }
    with open(os.path.join(DATA_DIR, "class_imbalance.json"), "w") as f:
        json.dump(imbalance, f, indent=2)

    # ---------------------------------------------------------------- #
    # 9. Simultaneity / overlap summary (validated against Morfi et al. Fig 5)
    # ---------------------------------------------------------------- #
    sim = simultaneity.set_index("level")["total_duration_s"]
    unannotated = float(sim.get("unannotated", 0.0))
    annotated_levels = sim.drop(index="unannotated", errors="ignore")
    annotated_levels.index = annotated_levels.index.astype(int)
    denom = annotated_levels.sum()
    sim_summary = {
        "total_train_duration_s": float(sim.sum()),
        "unannotated_duration_s": unannotated,
        "annotated_duration_s": float(denom),
        "pct_by_simultaneous_active_classes": {
            str(k): float(v / denom * 100) for k, v in annotated_levels.sort_index().items()
        },
    }
    with open(os.path.join(DATA_DIR, "simultaneity_summary.json"), "w") as f:
        json.dump(sim_summary, f, indent=2)

    # ---------------------------------------------------------------- #
    # 10. Overlap: fraction of events that overlap >=1 other event
    # ---------------------------------------------------------------- #
    overlap_matrix = pd.read_csv(os.path.join(DATA_DIR, "cooccurrence_overlap.csv"), index_col=0)
    label_overlap_any = (overlap_matrix.sum(axis=1) > 0)
    # An event overlaps another iff its label has >=1 recorded overlap
    # instance; approximate per-event flag via re-walking events (exact).
    n_overlapping_events = 0
    n_cross_label_overlap = 0
    for file_number, grp in events.groupby("file_number"):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        for i in range(n):
            a = grp.loc[i]
            any_overlap = False
            cross_label = False
            for j in range(n):
                if i == j:
                    continue
                b = grp.loc[j]
                if a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]:
                    any_overlap = True
                    if b["label"] != a["label"]:
                        cross_label = True
            if any_overlap:
                n_overlapping_events += 1
            if cross_label:
                n_cross_label_overlap += 1
    overlap_summary = {
        "n_events": int(len(events)),
        "n_events_with_any_overlap": int(n_overlapping_events),
        "pct_events_with_any_overlap": float(n_overlapping_events / len(events) * 100),
        "n_events_with_cross_label_overlap": int(n_cross_label_overlap),
        "pct_events_with_cross_label_overlap": float(n_cross_label_overlap / len(events) * 100),
    }
    with open(os.path.join(DATA_DIR, "overlap_summary.json"), "w") as f:
        json.dump(overlap_summary, f, indent=2)

    # ---------------------------------------------------------------- #
    # 11. Correlations among event-level acoustic features
    # ---------------------------------------------------------------- #
    ev_corr_feats = [
        "duration_s", "rms", "crest_factor", "zero_crossing_rate",
        "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_flatness",
        "dominant_freq_hz",
    ]
    ok = event_feats[~event_feats["too_short_for_spectrum"]]
    ok[ev_corr_feats].corr(method="spearman").to_csv(
        os.path.join(DATA_DIR, "correlations_events_spearman.csv")
    )

    print("Statistical analysis complete. Key numbers:")
    print(f"  Whole-file duration mixture: {dur_mix}")
    print(f"  Event overlap: {overlap_summary}")
    print(f"  Simultaneity %: {sim_summary['pct_by_simultaneous_active_classes']}")
    print(f"  Species-label (87-class) imbalance: {imbalance['species_labels_87_all_classes']}")


if __name__ == "__main__":
    main()
