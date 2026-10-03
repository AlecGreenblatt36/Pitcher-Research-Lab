# Latest chronological PA benchmark

Generated: `2026-10-03T22:19:35.502827+00:00`
Data: 182,926 untouched test PAs across 2,430 games.

| Model | Log loss | Multiclass Brier | Classwise ECE |
|---|---:|---:|---:|
| League | 1.481070 | 0.710593 | 0.001572 |
| Empirical Bayes matchup | 1.461262 | 0.702423 | 0.002995 |
| Talent only | 1.459215 | 0.702111 | 0.005866 |
| Talent + context | 1.457458 | 0.701458 | 0.005533 |

Relative log-loss gain vs league: **1.594%**
Relative log-loss gain vs empirical Bayes: **0.260%**
Full minus empirical-Bayes log loss: **-0.003803** (game-clustered 95% CI -0.004368 to -0.003252).
Context minus talent-only log loss: **-0.001756** (game-clustered 95% CI -0.002022 to -0.001487).

Promotion: **FAIL**

This file is generated from the frozen chronological benchmark. It validates pre-PA probabilities only; it does not validate runner transitions, bullpen selection, team-run distributions, winner probabilities, or exact scores.
