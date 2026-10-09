# Project notes

This repo reproduces *Bioacoustic classification of avian calls from raw
sound waveforms with an open source deep learning architecture* (Bravo
Sanchez et al., Sci Rep 2021) using the authors' own companion code.
Target numbers and full findings: `REPRODUCTION_REPORT.md`.

## Rule: do not change code to force numbers to match the paper

If our results differ from the paper's Table 1, do not edit training/eval
code to try to close the gap. Record *guesses* for why the discrepancy
exists in `REPRODUCTION_REPORT.md` — that's analysis, not a to-do list.
The code stays faithful to what the authors released; only fix things that
are genuine blockers (crashes, broken imports, missing files), not
methodology changes aimed at nudging accuracy toward a target number.

## Where things stand

See `REPRODUCTION_REPORT.md` for the full setup, the old (orphaned)
checkpoints, and the current ranked list of gap hypotheses.
