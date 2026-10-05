# Dynamic State Engine Foundation Receipt

**Decision:** ACCEPT as a research foundation; **NOT PROMOTED** into the accepted game engine  
**Branch:** `feature/dynamic-state-engine-v1`  
**Baseline parent:** `af92f0204ccf9590a4d78951d397acdc10938fb4`  
**Date:** 2026-10-04

## What changed

Created a downloadable source package and patch containing fail-closed, independently testable foundations for a dynamic state-conditioned simulator:

- event-keyed deterministic random streams;
- legal pitch-count Markov state and sequence simulator interfaces;
- continuous fatigue/workload feature contracts with no invented coefficients;
- empirical-Bayes shrinkage utilities;
- batted-ball EV/LA/spray transition kernel with explicit backoff provenance;
- fitted conditional-logit bullpen-selection interface;
- first-pitch cutoff and source-provenance contracts;
- randomized PIT, fractional discrete interval coverage, finite-path proper-score corrections, calibration fitting, and forward-month splits.

## Verification

```text
PYTHONPATH=. pytest -q
17 passed

python -m compileall -q research_lab tests
PASS
```

The unit tests cover deterministic and order-independent random streams, count legality, two-strike foul behavior, pitch-sequence termination, shrinkage, provenance cutoffs, batted-ball backoff, bullpen softmax normalization, randomized PIT bounds, finite-path scoring, and forward chronological splits.

## Claims boundary

This receipt does **not** claim that the full-game simulator is more accurate yet. No pitch-plan, pitch-event, fatigue-degradation, batted-ball, defense, runner, or reliever-selection model has been trained in this package. The accepted locked seven-outcome PA provider remains the benchmark. Each fitted component must earn promotion through a paired pregame-valid replay with proper scores and leakage controls.

## Next accepted task

After the private-data reproduction gate and clean 2,430-game benchmark are restored, integrate event-keyed streams and verify:

1. unchanged marginal distributions;
2. deterministic reproducibility;
3. lower paired-difference variance;
4. no change to proper-score point estimates beyond Monte Carlo tolerance.
