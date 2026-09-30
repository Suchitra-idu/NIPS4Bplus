"""
Generates every figure referenced in DATASET_ANALYSIS.md from the CSV/JSON
tables produced by the other dataset_analysis scripts. Run this last.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, FIG_DIR, WAV_TRAIN_DIR

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
    }
)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  wrote {path}")


def fig_duration_hist(audio):
    fig, ax = plt.subplots(figsize=(6, 4))
    for split, color in [("train", "#3b6fa0"), ("test", "#c0703d")]:
        d = audio.loc[audio["split"] == split, "duration_s"]
        ax.hist(d, bins=60, alpha=0.55, label=f"{split} (n={len(d)})", color=color, density=True)
    ax.axvline(5.0039, color="k", ls="--", lw=1, label="5.0039 s protocol ceiling")
    ax.set_xlabel("Whole-file duration (s)")
    ax.set_ylabel("Density")
    ax.set_title("Recording duration")
    ax.legend(fontsize=8)
    savefig("01_file_duration_hist.png")


def fig_event_duration_dist(events):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    d = events["duration_s"].values
    shape, loc, scale = stats.lognorm.fit(d)
    x = np.linspace(d.min(), d.max(), 500)

    axes[0].hist(d, bins=80, density=True, alpha=0.6, color="#3b6fa0", label="observed")
    axes[0].plot(x, stats.lognorm.pdf(x, shape, loc, scale), "k-", lw=1.5, label="lognormal fit")
    axes[0].set_xlabel("Tagged-event duration (s)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Event duration (linear scale)")
    axes[0].legend(fontsize=8)

    logd = np.log(d)
    axes[1].hist(logd, bins=80, density=True, alpha=0.6, color="#3b6fa0")
    mu, sigma = logd.mean(), logd.std()
    xs = np.linspace(logd.min(), logd.max(), 500)
    axes[1].plot(xs, stats.norm.pdf(xs, mu, sigma), "k-", lw=1.5, label="normal fit to log(duration)")
    axes[1].set_xlabel("log(event duration / 1 s)")
    axes[1].set_title("Event duration (log-transformed)")
    axes[1].legend(fontsize=8)
    savefig("02_event_duration_distribution.png")


def fig_event_duration_by_taxon(events):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = ["bird", "insect", "amphibian", "Unknown", "Human"]
    data = [events.loc[events["taxon"] == t, "duration_s"].values for t in order]
    bp = ax.boxplot(data, tick_labels=order, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a7c7e7")
    ax.set_yscale("log")
    ax.set_ylabel("Event duration (s, log scale)")
    ax.set_title("Event duration by taxonomic group")
    savefig("03_event_duration_by_taxon.png")


def fig_qq_event_duration(events):
    d = events["duration_s"].values
    logd = np.log(d)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    stats.probplot(d, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q vs Normal (raw duration)")
    stats.probplot(logd, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q vs Normal (log duration)")
    savefig("04_qq_event_duration.png")


def fig_class_frequency(per_class):
    d = per_class[~per_class["label"].isin(["Unknown", "Human"])].sort_values("n_events", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = d["taxon"].map({"bird": "#3b6fa0", "insect": "#c0703d", "amphibian": "#4c9a5a"}).fillna("#999999")
    ax.bar(range(len(d)), d["n_events"], color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["label"], rotation=90, fontsize=5)
    ax.set_ylabel("Number of tagged events (log scale)")
    ax.set_title("Class frequency")
    savefig("05_class_frequency.png")


def fig_zipf(per_class):
    d = per_class[~per_class["label"].isin(["Unknown", "Human"])].sort_values("n_events", ascending=False)
    counts = d["n_events"].values
    ranks = np.arange(1, len(counts) + 1)
    slope, intercept, r, p, se = stats.linregress(np.log(ranks), np.log(counts))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(ranks, counts, "o", ms=4, color="#3b6fa0")
    ax.loglog(ranks, np.exp(intercept) * ranks**slope, "k--", lw=1.2, label=f"slope={slope:.2f}, R²={r**2:.2f}")
    ax.set_xlabel("Class rank (log)")
    ax.set_ylabel("Event count (log)")
    ax.set_title("Class frequency by rank")
    ax.legend(fontsize=8)
    savefig("06_zipf_plot.png")


def fig_lorenz(per_class):
    d = per_class[~per_class["label"].isin(["Unknown", "Human"])]
    counts = np.sort(d["n_events"].values)
    n = len(counts)
    cum_counts = np.cumsum(counts) / counts.sum()
    cum_classes = np.arange(1, n + 1) / n
    gini = (2 * np.sum(np.arange(1, n + 1) * counts) / (n * counts.sum())) - (n + 1) / n

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect equality")
    ax.plot(np.concatenate([[0], cum_classes]), np.concatenate([[0], cum_counts]), color="#3b6fa0", lw=2, label=f"Lorenz curve (Gini={gini:.3f})")
    ax.fill_between(np.concatenate([[0], cum_classes]), np.concatenate([[0], cum_classes]), np.concatenate([[0], cum_counts]), alpha=0.15, color="#3b6fa0")
    ax.set_xlabel("Cumulative fraction of classes (least -> most frequent)")
    ax.set_ylabel("Cumulative fraction of tagged events")
    ax.set_title("Lorenz curve")
    ax.legend(fontsize=8)
    savefig("07_lorenz_curve.png")


def fig_active_classes_hist(file_summary):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = file_summary["n_distinct_species_labels"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#3b6fa0")
    ax.set_xlabel("Number of distinct species/insect labels active in the file")
    ax.set_ylabel("Number of recordings")
    ax.set_title("Active classes per recording")
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y + 3, str(y), ha="center", fontsize=8)
    savefig("08_active_classes_per_file.png")


def fig_simultaneity(sim_summary):
    pct = sim_summary["pct_by_simultaneous_active_classes"]
    levels = sorted(pct.keys(), key=int)
    values = [pct[l] for l in levels]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(levels, values, color="#3b6fa0")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Number of simultaneously active tagged classes")
    ax.set_ylabel("% of annotated recording duration")
    ax.set_title("Temporal overlap of tags")
    savefig("09_simultaneity.png")


def fig_cooccurrence_heatmap():
    m = pd.read_csv(os.path.join(DATA_DIR, "cooccurrence_overlap.csv"), index_col=0)
    top_labels = m.sum(axis=1).sort_values(ascending=False).head(25).index
    sub = m.loc[top_labels, top_labels]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(sub.values, cmap="magma")
    ax.set_xticks(range(len(top_labels)))
    ax.set_xticklabels(top_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=6)
    ax.set_title("Label co-occurrence")
    fig.colorbar(im, ax=ax, shrink=0.8, label="# overlapping event pairs")
    savefig("10_cooccurrence_heatmap.png")


def fig_correlation_heatmap(audio):
    feats = [
        "duration_s", "rms", "crest_factor", "dynamic_range_db", "silence_ratio",
        "zero_crossing_rate", "spectral_centroid_hz", "spectral_bandwidth_hz",
        "spectral_rolloff85_hz", "spectral_flatness", "skewness", "kurtosis_excess",
    ]
    corr = audio[feats].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=7)
    for i in range(len(feats)):
        for j in range(len(feats)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=5.5)
    ax.set_title("Feature correlations")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig("11_correlation_heatmap.png")


def fig_spectral_by_taxon(event_feats):
    ok = event_feats[~event_feats["too_short_for_spectrum"]]
    order = ["bird", "insect", "amphibian", "Unknown", "Human"]
    data = [ok.loc[ok["taxon"] == t, "spectral_centroid_hz"].values for t in order]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c0703d")
    ax.set_ylabel("Spectral centroid (Hz)")
    ax.set_title("Spectral centroid by taxonomic group")
    savefig("12_spectral_centroid_by_taxon.png")


def fig_train_test_shift(audio):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, feat, xlabel in [
        (axes[0], "spectral_centroid_hz", "Spectral centroid (Hz)"),
        (axes[1], "zero_crossing_rate", "Zero-crossing rate"),
    ]:
        for split, color in [("train", "#3b6fa0"), ("test", "#c0703d")]:
            d = audio.loc[audio["split"] == split, feat].dropna()
            ax.hist(d, bins=50, density=True, alpha=0.5, label=split, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("Train vs. test spectral shift")
    savefig("13_train_test_shift.png")


def fig_amplitude_outlier():
    path = os.path.join(WAV_TRAIN_DIR, "nips4b_birds_trainfile439.wav")
    data, sr = sf.read(path, dtype="float64")
    t = np.arange(len(data)) / sr
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(t, data, lw=0.5, color="#3b6fa0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("trainfile439.wav")
    savefig("14_amplitude_outlier_trainfile439.png")


def fig_multi_tag_example():
    import csv

    events_path = os.path.join(
        os.path.dirname(DATA_DIR), "..", "raw", "nips4bplus_csv", "annotations",
        "temporal_annotations_nips4b", "annotation_train007.csv",
    )
    events_path = os.path.normpath(events_path)
    evs = []
    with open(events_path, newline="") as f:
        for row in csv.reader(f):
            if row:
                evs.append((float(row[0]), float(row[1]), row[2]))

    path = os.path.join(WAV_TRAIN_DIR, "nips4b_birds_trainfile007.wav")
    data, sr = sf.read(path, dtype="float64")
    t = np.arange(len(data)) / sr

    n_fft = 1024
    hop = 256
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    n_frames = 1 + (len(data) - n_fft) // hop
    spec = np.zeros((len(freqs), n_frames))
    window = np.hanning(n_fft)
    for i in range(n_frames):
        seg = data[i * hop : i * hop + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(seg))
    spec_db = 20 * np.log10(spec + 1e-6)
    times = (np.arange(n_frames) * hop + n_fft / 2) / sr

    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True, height_ratios=[2, 2, 1.3])
    axes[0].plot(t, data, lw=0.4, color="#3b6fa0")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("nips4b_birds_trainfile007.wav")

    im = axes[1].pcolormesh(times, freqs / 1000, spec_db, shading="auto", cmap="magma", vmin=-40, vmax=40)
    axes[1].set_ylabel("Frequency (kHz)")
    axes[1].set_ylim(0, 12)

    colors = plt.cm.tab10(np.linspace(0, 1, len(evs)))
    labels_seen = []
    for i, (start, dur, label) in enumerate(evs):
        axes[2].barh(i, dur, left=start, height=0.6, color=colors[i])
        axes[2].text(start, i, f" {label}", va="center", fontsize=7)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Tagged\nevents")
    axes[2].invert_yaxis()
    savefig("15_multi_tag_example_trainfile007.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    audio = pd.read_csv(os.path.join(DATA_DIR, "audio_features.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    event_feats = pd.read_csv(os.path.join(DATA_DIR, "event_audio_features.csv"))
    per_class = pd.read_csv(os.path.join(DATA_DIR, "per_class_stats.csv"))
    file_summary = pd.read_csv(os.path.join(DATA_DIR, "file_summary.csv"))
    with open(os.path.join(DATA_DIR, "simultaneity_summary.json")) as f:
        sim_summary = json.load(f)

    fig_duration_hist(audio)
    fig_event_duration_dist(events)
    fig_event_duration_by_taxon(events)
    fig_qq_event_duration(events)
    fig_class_frequency(per_class)
    fig_zipf(per_class)
    fig_lorenz(per_class)
    fig_active_classes_hist(file_summary)
    fig_simultaneity(sim_summary)
    fig_cooccurrence_heatmap()
    fig_correlation_heatmap(audio)
    fig_spectral_by_taxon(event_feats)
    fig_train_test_shift(audio)
    fig_amplitude_outlier()
    fig_multi_tag_example()
    print("All figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
