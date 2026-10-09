# NIPS4Bplus Dataset Analysis

An in-depth exploratory data analysis (EDA) of the raw NIPS4Bplus corpus
sitting in `raw/` — not of the reproduction pipeline's derived splits, and
not of the model. Every number, table and figure below is computed by the
scripts in [`dataset_analysis/`](dataset_analysis/) directly from the 1,687
wav files, the 674 temporal-annotation CSVs, and the species-list metadata
on disk. Where a claim from the two source papers could be checked against
the raw data, it was, and both matches and mismatches are reported —
several genuine discrepancies turned up and are documented rather than
smoothed over.

Sources cross-referenced throughout:
- Bravo Sanchez, F., Hossain, M. R., English, N. B. & Moore, S. T.
  *Bioacoustic classification of avian calls from raw sound waveforms with
  an open-source deep learning architecture.* Sci Rep 11, 15733 (2021).
- Morfi, V., Bas, Y., Pamuła, H., Glotin, H. & Stowell, D. *NIPS4Bplus: a
  richly annotated birdsong audio dataset.* arXiv:1811.02275 (2018) / PeerJ
  Comput. Sci. 5, e223 (2019).

---

## Table of contents

1. [What's actually on disk](#1-whats-actually-on-disk)
2. [Recording collection methodology (from the papers)](#2-recording-collection-methodology-from-the-papers)
3. [Whole-file waveform statistics](#3-whole-file-waveform-statistics)
4. [Tagged-event (annotation) statistics](#4-tagged-event-annotation-statistics)
5. [Class imbalance](#5-class-imbalance)
6. [Temporal overlap and co-occurrence](#6-temporal-overlap-and-co-occurrence)
7. [Acoustic content by taxonomic group](#7-acoustic-content-by-taxonomic-group)
8. [Train/test structure — a dataset-identity clarification](#8-traintest-structure--a-dataset-identity-clarification)
9. [Verified discrepancies between the papers and the raw data](#9-verified-discrepancies-between-the-papers-and-the-raw-data)
10. [Implications for the SincNet reproduction](#10-implications-for-the-sincnet-reproduction)
11. [Limitations of this analysis](#11-limitations-of-this-analysis)
12. [Reproducing this analysis](#12-reproducing-this-analysis)

---

## 1. What's actually on disk

```
raw/                                    610 MB
├── nips4b_wav/
│   ├── train/   687 wav files          nips4b_birds_trainfile001.wav ...687
│   └── test/   1000 wav files          nips4b_birds_testfile0001.wav ...1000
├── nips4b_labels/
│   ├── nips4b_birdchallenge_espece_list.csv     87-class species table
│   ├── nips4b_birdchallenge_train_labels.csv    687x89 weak-label matrix (original 2013 challenge)
│   └── tps_canaux_sr_nbits_TRAIN.csv            687 rows: duration, channels, sample_rate, nbits
└── nips4bplus_csv/
    └── annotations/temporal_annotations_nips4b/  674 per-file CSVs:
                                                    [start_s, duration_s, label]
```

1,687 wav files total, all confirmed (via `soundfile`, not assumed) to be:

| Property | Value | Paper claim | Match? |
|---|---|---|---|
| Sample rate | 44,100 Hz, all 1,687 files | "44.1 kHz" | ✅ |
| Channels | 1 (mono), all 1,687 files | "single channel" | ✅ |
| Bit depth | **16-bit PCM**, all 1,687 files | "32 bit depth" (Bravo Sanchez et al.) | ❌ see [§9](#9-verified-discrepancies-between-the-papers-and-the-raw-data) |
| Species/call-type classes | 87 (77 bird + 9 insect + 1 amphibian) | "87 classes" | ✅ |
| Distinct organisms | 61 (51 bird species + 9 insect + 1 amphibian) | "61... species," "7 insects" | ⚠️ insect count off, see §9 |
| Total tagged events | 5,775 raw rows across 569 files | "5,478" (after cleaning) | ✅ once cleaned identically, see §4 |
| Train total duration | 2,903.2 s measured (48.4 min) | "less than an hour" / "48 min" | ✅ |
| Test total duration | 4,172.5 s measured (69.5 min) | "nearly two hours" | ❌ see §9 |

The 87-class table (`nips4b_birdchallenge_espece_list.csv`) has 88 rows: 87
real classes plus an `Empty` row that the label README explicitly says is
*not* a class (it marks files with no tagged animal sound). Breakdown by
`type`:

| Taxon | Distinct species | Classes (call-type variants) |
|---|---|---|
| bird | 51 | 77 (song/call/drum kept separate per species) |
| insect | 9 | 9 (song only) |
| amphibian | 1 | 1 (song only) |
| **total** | **61** | **87** |

This 51/9/1 → 61 breakdown is what "All Classes" (87), "Bird Classes" (77)
and "Bird Species" (51, call types merged) refer to in the SincNet cfgs and
in `evaluate_metrics.py`/`REPRODUCTION_REPORT.md`.

---

## 2. Recording collection methodology (from the papers)

Not re-derivable from the raw files alone (no per-file GPS/location field
ships with the release), so quoted here for context rather than
independently verified:

- Recorders: SM2BAT + SMX-US microphones, originally deployed for bat
  echolocation sampling, opportunistically also recording birds for 3 hours
  starting 30 minutes after sunrise, with a 6 dB SNR trigger (2 s window).
- ~30 hours of field recordings collected across 39 sites in 7 regions:
  20% Haute-Loire (Central France), 65% Pyrénées-Orientales/Aude/Hérault
  (South France, Mediterranean), 15% Granada/Jaén/Almería (SE Spain).
- Any recording longer than 5 s was split into 5 s chunks; `SonoChiro`
  (a bat-call detector) flagged chunks likely to contain bird vocalisation;
  stratified sampling by location/feature-clustering picked ~5,000
  candidates; manual annotation + a ≥7-recordings-per-species cutoff
  produced the final 687-file training set (this cutoff is exactly why the
  class-count floor in [§5](#5-class-imbalance) is 9, not lower — 7 in the
  *challenge* labels plus a couple more once NIPS4Bplus's richer tagging
  found extra instances).
- Temporal annotations were produced by a single expert annotator (Hanna
  Pamuła) using Sonic Visualiser — i.e. every statistic in this report that
  touches timing/duration reflects one person's judgment calls about where
  a call starts and ends, not an objective ground truth.

Section 3 onward is all independently measured from the files.

---

## 3. Whole-file waveform statistics

### 3.1 Duration is not a continuous variable — it's a censored mixture

The collection protocol ("any recording longer than 5 s split into 5 s
files") means file duration is not drawn from a smooth distribution: most
files pile up at a hard ceiling (5.0039 s = 220,672 samples at 44.1 kHz),
with a second population of shorter files representing the *last* chunk of
whatever longer recording they were cut from.

| Split | n | at ceiling (≥5.00 s) | shorter (tail) |
|---|---|---|---|
| train | 687 | 425 (61.9%) | 262 (38.1%) |
| test | 1,000 | 583 (58.3%) | 417 (41.7%) |

![File duration histogram](dataset_analysis/figures/01_file_duration_hist.pdf)

Any statistical test that assumes file duration is unimodal/continuous
(e.g. fitting a single Gaussian or lognormal to it) is fitting the wrong
generative model — it's a mixture of "hit the ceiling" and "ran out of
recording." Descriptive stats are reported for completeness but should not
be over-interpreted:

| | train | test |
|---|---|---|
| mean | 4.226 s | 4.173 s |
| median | 5.004 s | 5.004 s |
| std | 1.150 | 1.125 |
| skewness | −1.09 | (similar) |

### 3.2 Amplitude- and spectrum-domain descriptive statistics (train, n=687)

| Feature | mean | std | skewness | excess kurtosis | min | median | max |
|---|---|---|---|---|---|---|---|
| RMS energy | 0.00352 | 0.02969 | 22.4 | 545 | 0.0003 | 0.00086 | 0.738 |
| Crest factor (peak/RMS) | 8.28 | 8.97 | 9.13 | 115 | 1.36 | 6.30 | 150.4 |
| Dynamic range (dB) | 16.86 | 4.23 | 1.76 | 5.95 | 2.64 | 15.99 | 43.5 |
| Silence ratio (\|x\|<2% peak) | 0.324 | 0.190 | 0.60 | 0.11 | 0.012 | 0.301 | 0.997 |
| Zero-crossing rate | 0.219 | 0.085 | −0.42 | 0.21 | 0.005 | 0.233 | 0.547 |
| Spectral centroid (Hz) | 4,770 | 2,516 | −0.06 | 1.40 | 32 | 5,350 | 19,017 |
| Spectral bandwidth (Hz) | 4,355 | 1,614 | −0.55 | −0.72 | 335 | 4,578 | 7,712 |
| Spectral flatness | 0.245 | 0.153 | −0.15 | −1.42 | 0.0007 | 0.254 | 0.483 |

RMS energy and crest factor are *extremely* right-skewed at the file level
(skew 22 and 9; excess kurtosis 545 and 115) — a handful of files are wildly
more impulsive than the rest. The single most extreme case is
`nips4b_birds_trainfile439.wav`: crest factor 150× (peak is 150 times the
RMS level) and excess kurtosis 5,453, an order of magnitude beyond the next
most extreme file. Its waveform makes the cause obvious — one sharp
transient (visually consistent with a close, loud call) in an otherwise
near-silent recording:

![Amplitude outlier example](dataset_analysis/figures/14_amplitude_outlier_trainfile439.pdf)

This matters methodologically: file-level amplitude statistics in this
dataset are dominated by a small number of impulsive outliers, so **mean ±
std summaries of amplitude features are misleading**; median/IQR or a
log transform are the honest summary here.

### 3.3 Formal normality tests reject everything — and that's expected, not informative

Shapiro-Wilk and D'Agostino K² both reject normality (p < 10⁻¹³) for every
feature in the table above, in both raw and log-transformed form. With
n=687–1,687, these tests have enough power to reject the null on
essentially any real-world measurement with non-zero higher moments — a
rejection here is not evidence of a *practically* large departure from
normality, just a *statistically detectable* one. The more useful signal is
the magnitude of skewness/kurtosis (above) and which parametric family
actually minimizes AIC / KS-distance (below) — i.e. this section treats "is
it normal" as the wrong question and "what shape is it, quantitatively"
as the right one.

### 3.4 Distribution fitting: RMS/crest factor are lognormal; spectral shape is closer to Weibull

Five candidate families (Normal, Lognormal, Gamma, Weibull, Exponential)
were fit by MLE to each feature and ranked by AIC, with a Kolmogorov-Smirnov
goodness-of-fit check:

$$
f_{\text{lognormal}}(x;\mu,\sigma)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right),\quad x>0
$$

| Feature | Best fit | ΔAIC to 2nd place | KS stat | KS p |
|---|---|---|---|---|
| RMS energy | **Lognormal** | 469 (vs. Weibull) | 0.054 | 1.1e-4 |
| Crest factor | **Lognormal** | 1,218 (vs. exponential) | 0.071 | 9.5e-8 |
| Spectral centroid | Weibull (barely) | 28 (vs. Normal) | 0.100 | 4.2e-15 |
| Spectral bandwidth | Weibull (barely) | 109 (vs. Normal) | 0.069 | 1.8e-7 |

RMS and crest factor are unambiguously best modeled as lognormal — a huge
AIC margin over every other candidate — which is exactly what's expected
for acoustic energy: it's the product of many roughly-independent
attenuating/amplifying factors (source loudness, distance, atmospheric
absorption, microphone gain), and products of positive random variables
converge to lognormal by the multiplicative central limit theorem. Spectral
centroid/bandwidth are close to symmetric (Weibull edges out Normal, but
only marginally) — consistent with them being *averages* over many
frequency bins, which pulls them back toward Gaussian-like behavior
regardless of the individual bins' distribution.

### 3.5 Feature correlations

![Correlation heatmap](dataset_analysis/figures/11_correlation_heatmap.pdf)

Three structural clusters:
- **Timbre/spectral-shape cluster** (r=0.6–0.92): zero-crossing rate,
  spectral centroid, bandwidth, rolloff and flatness all move together —
  they're different lenses on the same underlying "how broadband/noisy vs.
  tonal/narrowband is this recording" axis.
- **Impulsiveness cluster**: crest factor and dynamic range (r=0.83, both
  peak-vs-average measures), and skewness vs. kurtosis (r=−0.87 — files
  with an extreme kurtosis-driving spike tend to have the spike's sign
  dominate the skew too).
- **Silence ratio bridges both**: it correlates moderately with the
  spectral cluster (r=0.44–0.58) because the *background* in these
  recordings (insect stridulation, wind, distant traffic) is broadband
  noise, so a quieter, more silence-dominated file's frequency content
  looks flatter and noisier per unit energy than a file with more
  vocalisation filling the timeline.

### 3.6 A data-integrity check: file-level duration matches the authors' own metadata table

`tps_canaux_sr_nbits_TRAIN.csv` is the original NIPS4B-challenge metadata
table declaring each train file's duration/channels/sample-rate/bit-depth.
Cross-checking it against the actual wav headers/durations for all 687
files found exactly **one** mismatch:
`nips4b_birds_trainfile060.wav` is declared as 5.0039 s but is actually
only 2.103 s on disk (a 2.90 s discrepancy) — this is a genuine truncation
somewhere in the file's history, not a measurement artifact (`soundfile`'s
header-reported duration and the actual decoded sample count agree). Its
one tagged event (`Petpet_song`, 1.41–1.80 s) still falls safely inside the
truncated file, so it doesn't corrupt anything downstream, but it's worth
knowing about if anyone builds tooling that trusts the metadata table's
duration column instead of reading the wav header directly.

---

## 4. Tagged-event (annotation) statistics

### 4.1 What "rich annotation" means in numbers

`extract_annotation_data.py` parses all 674 temporal-annotation files. Of
the 687 train recordings:
- **569** have ≥1 parsed tagged event (5,775 events total).
- **105** have an annotation file that exists but is empty (just a bare
  line ending, zero events) — recordings judged to contain no taggable
  vocalisation.
- **13** have no annotation file on disk at all — genuinely undocumented
  (the paper's "6 ambiguous + 7 insect-only" files).

This distinction matters: a file with an *empty* annotation file is a
confirmed "nothing to tag here," while a file with *no* annotation file is
"we don't know" — collapsing the two (as a naive "does this file have
events" check would) silently misclassifies 105 recordings.

### 4.2 Event duration is strongly right-skewed and closely lognormal

![Event duration distribution](dataset_analysis/figures/02_event_duration_distribution.pdf)

| | all events (n=5,775) | bird (n=5,279) | insect (n=168) | amphibian (n=34) |
|---|---|---|---|---|
| mean | 0.195 s | 0.149 s | 1.670 s | 0.228 s |
| median | 0.090 s | 0.087 s | 1.033 s | 0.192 s |
| std | 0.455 | 0.220 | 1.775 | 0.203 |
| skewness | 7.29 | 5.14 | 0.78 | 3.33 |
| min / max | 23 µs / 5.00 s | 23 µs / 2.78 s | 25 ms / 5.00 s | 28 ms / 1.19 s |

Distribution fitting (same AIC/KS procedure as §3.4) picks **lognormal** as
the best fit for the overall and bird-only event-duration distributions by
a wide margin (ΔAIC > 3,200 over the next candidate), with fitted
parameters σ≈0.96, scale≈0.10 s (i.e. a fitted median call length of
~100 ms, matching the empirical 87–90 ms median closely). Log-transforming
turns a distribution that's unusable for Gaussian-based reasoning into one
that visually passes:

![Q-Q plots](dataset_analysis/figures/04_qq_event_duration.pdf)

Insect events are the exception: with only 168 samples and durations that
cluster into a bimodal-ish spread (brief stridulation bursts vs. long
sustained calls), Gamma edges out Lognormal, though the ranking is much
closer than for birds.

This lognormal-duration finding is a common pattern in bioacoustics (call
duration behaves like the product of many small multiplicative factors —
syllable count × syllable length × pause structure) and is worth keeping in
mind for anyone doing duration-based windowing or augmentation: **the
typical call is much shorter than the mean call**, because a shrinking tail
of very long songs pulls the mean up over 2x above the median.

![Event duration by taxon](dataset_analysis/figures/03_event_duration_by_taxon.pdf)

### 4.3 Independent validation against the papers' own reported numbers

Every number below was computed from the raw temporal-annotation files with
no reference to the papers' text during computation, then checked
afterward — and they match closely enough to give real confidence in both
this analysis pipeline and the papers' own reporting:

| Claim (Bravo Sanchez et al., Results) | Paper's number | This analysis |
|---|---|---|
| Total individual tagged events | 5,478 | 5,775 raw → 5,481 after dropping Unknown/Human (294) → **5,478** after dropping the 3 events under 10 ms the authors' own pipeline can't process |
| Shortest-mean-duration class | Sylcan_call, "~30 ms" | **31.1 ms** (n=124) |
| Longest-mean-duration class | Plasab_song, "more than 4.5 s" | **4.674 s** (n=11) |
| Most frequent class | Sylcan_song | **Sylcan_song, 282 events** |
| Least frequent class | Cicatr_song, 9 files | **Cicatr_song, 9 events** |

The three sub-10 ms events that close the 5,481→5,478 gap are identifiable
directly: `Poepal_call` (2.9 ms and 0.02 ms, both in file 409) and
`Sylmel_call` (8.7 ms, file 614) — exactly the "three very short files...
which could not be processed without code modifications" the paper
mentions in its methods section.

---

## 5. Class imbalance

Three canonical tag selections (matching the repo's own "All Classes" /
"Bird Classes" / "Bird Species" experiments) were each scored with a common
imbalance toolkit:

$$
\text{Gini} = \frac{2\sum_{i=1}^{n} i\,x_{(i)}}{n\sum_i x_i} - \frac{n+1}{n},
\qquad
H = -\sum_i p_i\log_2 p_i,
\qquad
N_{\text{eff}} = 2^{H}
$$

where $x_{(i)}$ are class counts sorted ascending and $p_i = x_i/\sum x_i$.
$N_{\text{eff}}$ ("effective number of classes") is the count of *equally
frequent* classes that would produce the same entropy — a single number
that answers "how many classes does this dataset really behave like it
has."

| Tag selection | classes | min / max count | imbalance ratio (max/min) | Gini | entropy (bits) | effective classes | Zipf slope |
|---|---|---|---|---|---|---|---|
| All Classes (87, incl. call-type splits) | 87 | 9 / 282 | 31.3× | 0.447 | 5.967 | 62.5 (72% of 87) | −0.84 (R²=0.88) |
| Bird Classes (77) | 77 | 12 / 282 | 23.5× | 0.426 | 5.839 | 57.2 (74% of 77) | −0.80 (R²=0.88) |

(Bird Species, which merges call/song/drum per species down to 51 classes,
was not separately re-tallied here since it's a strict aggregation of the
Bird Classes counts above; aggregating strictly *reduces* imbalance because
merging call-type variants pools the smaller call/drum counts into the
larger song counts for the same species.)

A Zipf slope near −1 with R²≈0.88 means class frequency decays roughly as
rank⁻¹ — closer to Zipfian than to either near-uniform or catastrophically
long-tailed. 31× between the best- and worst-represented class is
meaningful but not extreme by bioacoustic-dataset standards (many
citizen-science corpora exceed 100–1,000×): the paper's own "≥7 recordings
per class" floor put a hard ceiling on how thin the tail could get.

![Class frequency, log scale](dataset_analysis/figures/05_class_frequency.pdf)
![Zipf plot](dataset_analysis/figures/06_zipf_plot.pdf)
![Lorenz curve](dataset_analysis/figures/07_lorenz_curve.pdf)

The Lorenz curve gives Gini a visual: with perfect balance every class
would trace the diagonal. Here, the least-frequent half of the 87 classes
(by population share) account for only about 19% of all tagged events —
a moderate, not catastrophic, imbalance, but still enough that a
model with no class weighting will see roughly 6.6× more examples of the
median class than of the typical minority class (imbalance ratio to
*median*, not to the single rarest class), which plausibly interacts with
the zero-dropout, wide-FC-layer setup already flagged as a likely
overfitting driver in `REPRODUCTION_REPORT.md` — more evidence *for* that
hypothesis, not a new independent one.

### Active classes per recording

![Active classes histogram](dataset_analysis/figures/08_active_classes_per_file.pdf)

Independently counting distinct species/insect labels per file (excluding
`Unknown`/`Human`) gives 124/242/196/95/19/8/3 files with 0–6 active
classes respectively — the same right-skewed, mode-at-1 shape as Morfi et
al.'s Fig. 3 (which reports roughly 100/230/195/110/30/10/5). The "0
active classes" bucket differs (124 here vs. ~100 in the paper) mostly
because this count folds in both the 105 genuinely-empty annotation files
*and* the ~19 files whose only tags are `Unknown`/`Human` with no resolved
species — a slightly different definitional cut than whatever the original
figure used, not a data problem.

---

## 6. Temporal overlap and co-occurrence

### 6.1 Reproducing "over 20% of tags overlap" independently

A sweep-line algorithm was run over every file's tagged intervals to
compute, second-by-second, how many tags are simultaneously active. This
directly and independently reproduces Morfi et al.'s Fig. 5:

![Simultaneity](dataset_analysis/figures/09_simultaneity.pdf)

| Simultaneously active tags | % of annotated duration (this analysis) | Morfi et al., Fig. 5 |
|---|---|---|
| 0 | 66.32% | 66.3% |
| 1 | 28.09% | 28.1% |
| 2 | 5.24% | 5.2% |
| 3 | 0.36% | 0.4% |

An essentially exact match (denominator = the 2,846.6 s of recordings that
*have* an annotation file, excluding the 13 genuinely-undocumented files'
59.5 s, which have unknown — not zero — activity level).

At the individual-event level: **27.4%** of the 5,775 tagged events overlap
in time with at least one other tagged event of *any* label, and **24.0%**
overlap with a tag of a *different* label — matching the paper's own "over
20% of tagged sounds... overlap with sounds of other species" claim and
`REPRODUCTION_REPORT.md`'s independent note of the same phenomenon.

### 6.2 Who overlaps with whom

![Co-occurrence heatmap](dataset_analysis/figures/10_cooccurrence_heatmap.pdf)

The strongest individual overlap pairs are dominated by **insect songs**:
`Carcar_song`↔`Cicatr_song` (66 overlapping instances), plus `Plasab_song`,
`Cicorn_song`, `Tibtom_song` and `Lyrple_song` all appearing repeatedly in
the top-15 overlapping pairs. This has a straightforward ecological
explanation visible directly in the duration table above: insect
stridulation songs last seconds (up to the full 5 s file), so they act as
a near-continuous acoustic background that *any* bird call occurring in
the same recording will, almost by construction, overlap with. Overlap in
this dataset is less "two birds happened to call at once" and more
"a cicada was already going the whole time."

A concrete example, `nips4b_birds_trainfile007.wav` (7 tagged events, 5
species, plus an overlapping `Human` tag), makes the general picture
tangible: waveform, spectrogram, and the tag timeline all lined up on the
same time axis.

![Multi-tag example](dataset_analysis/figures/15_multi_tag_example_trainfile007.pdf)

---

## 7. Acoustic content by taxonomic group

Whole-file spectral features are diluted by silence and by unrelated
overlapping sounds in the same 5 s clip, so `extract_event_audio_features.py`
recomputes the full waveform/spectral feature set on the *exact tagged
segment* of each of the 5,775 events (down to individual samples), giving a
much cleaner taxon comparison:

![Spectral centroid by taxon](dataset_analysis/figures/12_spectral_centroid_by_taxon.pdf)

Insects register a visibly higher median spectral centroid than birds —
consistent with cicada/bush-cricket stridulation concentrating energy in a
narrow high-frequency band rather than birdsong's broader, lower-centred
range. At the individual-class extreme: `Plaaff_song` (Coastal
Bush-cricket) tops the whole 87-class table at 7,180 Hz mean centroid, with
`Cicatr_song` (Black Cicada) close behind at 6,613 Hz; the lowest-centroid
class is `Hirrus_call` (Barn Swallow) at 896 Hz.

Event-level (per-tagged-segment) Spearman correlations add a second layer:
RMS energy correlates *negatively* with spectral bandwidth (ρ=−0.70) and
flatness (ρ=−0.72) — louder-registering calls in this dataset tend to be
more tonal/narrowband (a strong, clean, nearby call) while quieter segments
skew toward broadband/noisy (either a distant, attenuated call blending
into background noise, or the recorder's own noise floor dominating a
weak signal). Duration correlates moderately with crest factor (ρ=0.40):
longer calls/songs tend to have more internal dynamic range, plausible for
syllable-repeated songs where amplitude rises and falls between syllables
rather than sitting at one constant level.

---

## 8. Train/test structure — a dataset-identity clarification

This is a structural finding worth stating plainly because it's easy to
get wrong when navigating the repo: **`raw/nips4b_wav/test/` is not the
test set that any of this repo's cfgs, `data_lists/`, or
`mod_data_lists/` actually train/evaluate against.**

- `raw/nips4b_wav/test/` (1,000 files) is the original 2013 NIPS4B
  challenge's held-out test set. It ships with **no labels** — the Bravo
  Sanchez paper states outright "[t]he 2013 Bird Challenge also includes a
  testing dataset with no labels that we did not use." There are no
  temporal-annotation files for it either.
- Every train/test split this repo actually trains and scores against
  (`data_lists/*_test_files.scp`, `mod_data_lists/*_test_files.csv`) is a
  **random subset carved out of the 687 labelled *training* files** by
  `generate_file_lists.py`/`generate_mod_file_lists.py`
  (`train_test_split`, 75:25) — a completely different, and much smaller,
  population than the 1,000-file folder on disk.

Because of that, statistically comparing the two on whole-file features is
still an interesting sanity check of the *raw corpus* (not of the training
pipeline), and it turns up a real, measurable distribution shift:

![Train/test shift](dataset_analysis/figures/13_train_test_shift.pdf)

Two-sample KS tests (train n=687 vs. test n=1,000) on whole-file features:

| Feature | train mean | test mean | KS stat | p | shift at α=0.01? |
|---|---|---|---|---|---|
| Duration | 4.226 s | 4.173 s | 0.050 | 0.25 | no |
| Crest factor | 8.28 | 7.73 | 0.054 | 0.18 | no |
| Silence ratio | 0.324 | 0.302 | 0.075 | 0.018 | no (borderline) |
| RMS energy | 0.00352 | 0.00337 | 0.085 | 0.0052 | **yes** |
| Zero-crossing rate | 0.219 | 0.207 | 0.103 | 3.1e-4 | **yes** |
| Spectral centroid | 4,770 Hz | 4,415 Hz | 0.101 | 4.4e-4 | **yes** |
| Spectral bandwidth | 4,355 Hz | 3,956 Hz | 0.115 | 3.5e-5 | **yes** |
| Spectral flatness | 0.245 | 0.213 | 0.103 | 3.2e-4 | **yes** |

Amplitude-shape statistics (duration, crest factor, silence ratio) are
statistically indistinguishable between the two pools, but every
*spectral* statistic shows a significant shift — the 1,000-file test pool
skews toward lower frequency content, less broadband energy, and slightly
lower overall RMS than the 687-file training pool. Since these are drawn
from the same overall NIPS4B 2013 recording campaign, the shift most
plausibly reflects the sampling/stratification step (the paper describes a
separate stratified-random-sampling pass per pool, "based on locations and
clustering of features") rather than a different recording protocol —
this is a description of what's measurably different, not a claim about
why, consistent with this report's policy of not asserting root causes it
can't verify from the data alone.

---

## 9. Verified discrepancies between the papers and the raw data

None of these are large enough to suggest anything is broken; they're
recorded because the user asked for a genuinely in-depth analysis, and
"checked the paper's stated numbers against the actual bytes on disk and
they mostly-but-not-perfectly agree" is a real, useful finding in its own
right.

| # | Claim | Paper says | Raw data says | Verdict |
|---|---|---|---|---|
| 1 | Bit depth | 32-bit (Bravo Sanchez et al., Methods) | 16-bit PCM, all 1,687 files, confirmed via `soundfile` header + `tps_canaux_sr_nbits_TRAIN.csv`'s own `nbits` column (=16 for all 687 rows) | Paper's stated bit depth doesn't match the released files |
| 2 | Insect species count | "7 insects" (Morfi et al.) | 9 distinct insect species in `nips4b_birdchallenge_espece_list.csv` (Cicatr, Cicorn, Lyrple, Phofem, Plaaff, Plasab, Ptehey, Tetpyg, Tibtom) | Off by 2 |
| 3 | "61 different bird species" | Stated as a bird-species count (Morfi et al., §3.1) | Only 51 distinct bird-species prefixes exist; 51+9 insects+1 amphibian=61 | The paper's "61" is the *total organism count* across all taxa, not a bird-only count as the sentence literally states |
| 4 | Empty-annotation-file count | "100 recordings contain only background noise" (Morfi et al., §3.2) | 105 annotation files on disk contain zero events | Off by 5 |
| 5 | Test-set total duration | "nearly two hours" for 1,000 test files (Bravo Sanchez et al., Methods) | 4,172.5 s = 69.5 min ≈ 1h10m measured directly from all 1,000 files | ~50 minutes short of the paper's claim; see below |
| 6 | File-level duration metadata | `tps_canaux_sr_nbits_TRAIN.csv` declares 5.0039 s for `trainfile060.wav` | Actual file is 2.103 s | One-file anomaly, not systematic |

For #5, the arithmetic doesn't work under any file-count/duration
assumption consistent with the rest of the dataset (1,000 files × ~5 s
would already only be ~83 min, and the measured mean is 4.17 s, not 5 s) —
so either the paper's "nearly two hours" describes a different/larger
pre-subsampling pool than the exact 1,000-file archive that ended up
released and used by this repo's own README instructions, or it's simply
an error in the paper. Both are plausible; nothing in the raw data
resolves which, so this is reported as an open, verified discrepancy
rather than resolved one way or the other. Per `CLAUDE.md`, no code in
this repo depends on that number anyway (the 1,000-file test folder isn't
used by the training pipeline at all — see [§8](#8-traintest-structure--a-dataset-identity-clarification)), so this is inert as far as
reproduction is concerned.

---

## 10. Implications for the SincNet reproduction

Purely observational connections between what's measured here and what's
already tracked in `REPRODUCTION_REPORT.md` — no code changes are implied
or intended by any of this, consistent with `CLAUDE.md`'s policy of
treating the training code as fixed.

- **Event durations are lognormal with a median near 90 ms, well under
  the enhanced pipeline's `cw_len=10 ms` window** — the paper's own
  rationale for shrinking the SincNet frame length from 200 ms to 10 ms.
  The distribution fit here (§4.2) quantifies *how* short: even the
  75th-percentile bird call is well under 200 ms, so the original
  TIMIT-scale window would have truncated or skipped the majority of
  tagged calls outright.
- **Class imbalance (31× max/min, 72% effective-class ratio) plus the
  cfgs' zero dropout on wide 3×1024/2048-unit FC layers** is a plausible
  compounding factor in the severe train/test gap already documented in
  `REPRODUCTION_REPORT.md` §4.2 — this analysis adds a quantified
  imbalance measurement to what was previously a qualitative "small/
  imbalanced dataset" observation, not a new hypothesis.
- **~24–27% of tagged events overlap another tag**, independently
  confirming `REPRODUCTION_REPORT.md` §4 item 4 and the paper's own
  "over 20%" claim from a completely different computation path (sweep-line
  over raw annotation timestamps vs. the paper's review of model
  predictions) — this is now a cross-validated number, not a single-source
  one.
- **The `raw/nips4b_wav/test/` folder is not part of the trained/evaluated
  pipeline at all** (§8) — worth keeping in mind if anyone is tempted to
  use it as an additional held-out set in future work; it has no temporal
  annotations and (per §9 #5) its exact provenance relative to the
  paper's stated test-set size is not fully reconciled.

---

## 11. Limitations of this analysis

- **No geographic metadata ships with the raw files** — the 7-region/
  France-Spain breakdown in Morfi et al. Fig. 1 could not be independently
  reproduced or checked; it's quoted in §2 from the paper, not verified.
- **Single annotator, single pass**: every temporal-annotation statistic
  in §4–§7 inherits whatever judgment calls Hanna Pamuła made about where
  one call ends and the next begins (explicitly flagged as inconsistent
  across files in the paper itself — "different syllables of a song were
  separated... in other occasions summarised into a larger event"), so
  measured event-duration variance conflates real acoustic variance with
  annotation-style variance. There's no second annotator's labels in this
  release to separate the two.
- **The 1,000-file test folder has no labels**, so none of the per-class,
  per-event, or overlap analyses in §4–§7 could be extended to it; only
  whole-file waveform statistics (§8) were available for that pool.
- **Spectral features are a single whole-signal FFT**, not a
  spectrogram/STFT average — appropriate for comparing files/events
  holistically (as done throughout), but not a substitute for a proper
  time-frequency analysis if someone wants to study within-call frequency
  modulation.
- **Formal normality tests are reported but explicitly discounted** (§3.3)
  given the sample sizes involved; this is a deliberate methodological
  choice, not an oversight, but it means this report leans on
  distribution-fitting/AIC comparisons and visual (Q-Q) diagnostics more
  than p-values for shape claims.

---

## 12. Reproducing this analysis

All code lives in [`dataset_analysis/`](dataset_analysis/), not in this
repo's training/eval pipeline — see
[`dataset_analysis/README.md`](dataset_analysis/README.md) for the exact
commands. In short:

```bash
cd dataset_analysis/scripts
python3 extract_audio_features.py
python3 extract_annotation_data.py
python3 extract_event_audio_features.py
python3 statistical_analysis.py
python3 make_figures.py
```

Each script reads only from `raw/` and `dataset_analysis/data/`, and writes
only into `dataset_analysis/data/` or `dataset_analysis/figures/`.
