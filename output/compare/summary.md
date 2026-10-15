# Original vs fixed SincNet

Accuracy [file-cluster 95% CI]; paired diff = fixed - original [95% CI]. A = authors' whole-file rule; D = tag span, mean p; Dp = D with prior correction.

## all_classes (1371 rows, 442 files)

| rule | original acc | fixed acc | paired diff | orig macro-rec | fixed macro-rec | orig top-3 | fixed top-3 |
|---|---|---|---|---|---|---|---|
| A | 0.551 [0.507, 0.596] | 0.592 [0.547, 0.632] | +0.040 [+0.012, +0.069] | 0.395 | 0.415 | 0.796 | 0.815 |
| B | 0.570 [0.526, 0.614] | 0.595 [0.551, 0.638] | +0.025 [-0.001, +0.052] | 0.412 | 0.446 | 0.844 | 0.862 |
| C | 0.681 [0.647, 0.717] | 0.716 [0.681, 0.749] | +0.034 [+0.013, +0.057] | 0.561 | 0.594 | 0.883 | 0.901 |
| D | 0.678 [0.642, 0.712] | 0.705 [0.670, 0.738] | +0.027 [+0.006, +0.049] | 0.564 | 0.590 | 0.883 | 0.897 |
| Dp | 0.634 [0.595, 0.670] | 0.694 [0.660, 0.726] | +0.060 [+0.037, +0.084] | 0.590 | 0.644 | 0.867 | 0.887 |

- original: BN eps [128.0, 24.0, 4.0]; Sinc drift median 0.3 Hz (low) / 0.6 Hz (high), max 6 Hz

- fixed: BN eps [1e-05, 1e-05, 1e-05]; Sinc drift median 44.3 Hz (low) / 59.0 Hz (high), max 405 Hz
- check: original rule A 0.5514 vs metrics.res 0.5514

## bird_classes (1320 rows, 429 files)

| rule | original acc | fixed acc | paired diff | orig macro-rec | fixed macro-rec | orig top-3 | fixed top-3 |
|---|---|---|---|---|---|---|---|
| A | 0.580 [0.535, 0.625] | 0.616 [0.572, 0.659] | +0.036 [+0.010, +0.063] | 0.396 | 0.435 | 0.792 | 0.828 |
| B | 0.567 [0.519, 0.614] | 0.635 [0.589, 0.677] | +0.068 [+0.030, +0.111] | 0.426 | 0.471 | 0.858 | 0.873 |
| C | 0.703 [0.668, 0.737] | 0.747 [0.715, 0.774] | +0.044 [+0.022, +0.067] | 0.579 | 0.654 | 0.889 | 0.918 |
| D | 0.708 [0.674, 0.740] | 0.742 [0.708, 0.770] | +0.033 [+0.012, +0.055] | 0.606 | 0.661 | 0.898 | 0.918 |
| Dp | 0.695 [0.657, 0.727] | 0.710 [0.675, 0.742] | +0.015 [-0.007, +0.040] | 0.639 | 0.689 | 0.893 | 0.895 |

- original: BN eps [111.0, 21.0, 3.0]; Sinc drift median 0.2 Hz (low) / 0.4 Hz (high), max 5 Hz

- fixed: BN eps [1e-05, 1e-05, 1e-05]; Sinc drift median 49.2 Hz (low) / 58.6 Hz (high), max 427 Hz
- check: original rule A 0.5803 vs metrics.res 0.5803

## bird_species (1320 rows, 417 files)

| rule | original acc | fixed acc | paired diff | orig macro-rec | fixed macro-rec | orig top-3 | fixed top-3 |
|---|---|---|---|---|---|---|---|
| A | 0.584 [0.537, 0.630] | 0.605 [0.558, 0.651] | +0.021 [-0.004, +0.046] | 0.411 | 0.433 | 0.802 | 0.833 |
| B | 0.583 [0.535, 0.631] | 0.607 [0.562, 0.651] | +0.024 [-0.002, +0.051] | 0.437 | 0.450 | 0.856 | 0.889 |
| C | 0.683 [0.645, 0.719] | 0.730 [0.696, 0.764] | +0.048 [+0.025, +0.072] | 0.598 | 0.659 | 0.874 | 0.905 |
| D | 0.689 [0.651, 0.726] | 0.722 [0.688, 0.756] | +0.033 [+0.009, +0.056] | 0.620 | 0.662 | 0.877 | 0.901 |
| Dp | 0.667 [0.629, 0.705] | 0.702 [0.666, 0.737] | +0.036 [+0.012, +0.059] | 0.639 | 0.682 | 0.865 | 0.898 |

- original: BN eps [111.0, 21.0, 3.0]; Sinc drift median 0.3 Hz (low) / 0.5 Hz (high), max 6 Hz

- fixed: BN eps [1e-05, 1e-05, 1e-05]; Sinc drift median 46.2 Hz (low) / 66.1 Hz (high), max 528 Hz
- check: original rule A 0.5841 vs metrics.res 0.5841
