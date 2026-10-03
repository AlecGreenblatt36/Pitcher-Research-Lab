# Chronological PA model build notes

This branch replaces the missing, non-reproducible `ROLLING_PA_RESULT.json` summary with an executable benchmark and auditable predictions.

## Decisions

- The old reported +0.905% PA log-loss gain is not imported as evidence.
- A strong empirical-Bayes matchup baseline is defined in code.
- Feature availability is enforced by date-blocked online counters.
- Every PA on date D is scored before any result from D updates a feature.
- Rare completed-PA outcomes remain in the probability simplex instead of disappearing.
- The validation season is separated chronologically into model-tuning, probability-calibration, and ensemble-selection dates.
- The final candidate is a validation-selected blend of the calibrated multinomial model and empirical-Bayes baseline.
- The runner kernel is not trained here; its event-label leakage question remains separate.
- Historical workflows publish the result JSON, dataset receipt, predictions, tuning audit, and serialized model.

## Verified 2023–2025 development result

The latest real-data benchmark used 2023 for training, disjoint portions of 2024 for tuning/calibration/blending, and 182,926 PAs from 2,430 games in 2025 for evaluation.

| Model | Log loss | Multiclass Brier | Classwise ECE |
|---|---:|---:|---:|
| League | 1.481070 | 0.710593 | 0.001572 |
| Empirical Bayes matchup | 1.461277 | 0.702446 | 0.003021 |
| Talent only | 1.458833 | 0.701767 | 0.002947 |
| Talent + context | 1.457261 | 0.701214 | 0.002664 |
| Calibrated candidate | 1.456974 | 0.701125 | 0.001953 |

The candidate was selected as an 86% calibrated-model / 14% empirical-Bayes blend. It improved log loss by 1.627% relative to prior-date league rates and 0.294% relative to the fitted empirical-Bayes matchup baseline.

The candidate-minus-baseline log-loss difference was -0.004302 nats per PA, with a game-clustered 95% confidence interval of -0.004807 to -0.003803. Pre-PA game context also added signal beyond the talent-only model: -0.001572 nats per PA, with a 95% interval of -0.001806 to -0.001341. Maximum absolute class calibration error was 0.0027.

## Development gates

The candidate passed all three development gates:

1. game-clustered 95% CI for candidate minus empirical-Bayes log loss is below zero;
2. context adds signal beyond the talent-only model with the same criterion;
3. maximum absolute class calibration gap is below 0.015.

## Interpretation boundary

The 2025 season is now a **development holdout**, because its first-pass result informed the calibration and ensemble architecture. This result supports advancing the frozen PA architecture to a locked 2026 evaluation, but it is not a production promotion.

It does not validate runner transitions, pitcher removal, bullpen selection, team-run distributions, winner probabilities, or exact scores. Those claims require separate prospective tests after the PA layer survives the locked final holdout.
