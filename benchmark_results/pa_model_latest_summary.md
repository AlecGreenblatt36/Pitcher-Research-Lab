# Latest chronological PA benchmark

Generated: `2026-10-03T22:45:37.342909+00:00`
Claim status: `chronological_development_holdout_executed`
Data: 182,926 test PAs across 2,430 games.

| Model | Log loss | Multiclass Brier | Classwise ECE |
|---|---:|---:|---:|
| League | 1.481070 | 0.710593 | 0.001572 |
| Empirical Bayes matchup | 1.461277 | 0.702446 | 0.003021 |
| Talent only | 1.458833 | 0.701767 | 0.002947 |
| Talent + context | 1.457261 | 0.701214 | 0.002664 |
| Calibrated candidate | 1.456974 | 0.701125 | 0.001953 |

Relative log-loss gain vs league: **1.627%**
Relative log-loss gain vs empirical Bayes: **0.294%**
Candidate minus empirical-Bayes log loss: **-0.004302** (game-clustered 95% CI -0.004807 to -0.003803).
Context minus talent-only log loss: **-0.001572** (game-clustered 95% CI -0.001806 to -0.001341).
Maximum absolute class calibration gap: **0.0027**

Development gate: **PASS**
Production promoted: **NO**

Candidate blend: 86.0% calibrated model / 14.0% empirical Bayes.

The test seasons in this run are a development holdout because earlier results informed subsequent architecture work. This result can advance the model to a locked final evaluation, but it cannot validate runner transitions, bullpen logic, team-run distributions, winner probabilities, exact scores, or a production deployment.
