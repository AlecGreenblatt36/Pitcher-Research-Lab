# PA model v1 build notes

This branch replaces the missing, non-reproducible `ROLLING_PA_RESULT.json` summary with an executable benchmark.

## Decisions

- The old reported +0.905% PA log-loss gain is not imported as evidence.
- A strong empirical-Bayes matchup baseline is defined in code.
- Feature availability is enforced by date-blocked online counters.
- Rare completed-PA outcomes remain in the probability simplex instead of disappearing.
- The runner kernel is not trained here; its event-label leakage question remains separate.
- A push-triggered historical workflow downloads public Statcast data and emits auditable predictions, metrics and a serialized model without committing raw data.

## Promotion standard

The full model is promoted only when all are true on untouched 2025 PAs:

1. game-clustered 95% CI for full model minus empirical-Bayes log loss is below zero;
2. context adds signal beyond the talent-only model with the same criterion;
3. maximum absolute class calibration gap is below 0.015.

This is deliberately stronger than comparing against a weak structural table and avoids an arbitrary relative-improvement threshold.
