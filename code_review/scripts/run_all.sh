#!/usr/bin/env bash
# Regenerates every figure/table under code_review/ (read-only w.r.t. the pipeline).
# Scripts 08, 10, 11, 12 need a CUDA GPU (~2-5 min each); everything else runs on CPU.
set -e
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
for s in 01_model_audit 02_sinc_filters 03_training_curves 04_evaluation_analysis 05_data_audit 06_normalisation_clipping 07_batchnorm_eps; do $PY $s.py; done
$PY 08_scoring_diagnostics.py bird_species
$PY 08_scoring_diagnostics.py all_classes bird_classes
$PY 09_scoring_figure.py
$PY 10_frame_level.py
for s in 11_profile 12_sinc_gradient_scale 13_layer_stats 14_class_tables 15_sampler_and_example 17_frame_level_figures 18_sinc_worked_example 19_filterbank_resolution 20_bootstrap_rules; do $PY $s.py; done
$PY ../tests/test_review_claims.py
