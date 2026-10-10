# Locked 2026 chronological PA holdout

Generated: `2026-10-03T22:58:12.785176+00:00`
Source commit: `27e207b6e5ae4fdb788f42fae9d66f0021f7f6b6`
Candidate fingerprint: `649b59b91524c8d5f3c058db8ce585e5ac42618bc173a14dba151b0b429f188c`
Coverage: 183,849 2026 PAs across 2,429 games, 2026-03-25 00:00:00 through 2026-09-27 00:00:00.

| Model | Log loss | Multiclass Brier | Classwise ECE |
|---|---:|---:|---:|
| League | 1.484836 | 0.712472 | 0.001688 |
| Empirical Bayes matchup | 1.464168 | 0.703613 | 0.003653 |
| Talent only | 1.462042 | 0.703096 | 0.002688 |
| Talent + context | 1.460129 | 0.702425 | 0.002277 |
| Locked candidate | 1.459908 | 0.702371 | 0.002134 |

Candidate blend: **90.0% calibrated model / 10.0% empirical Bayes**.
Relative log-loss gain vs league: **1.679%**.
Relative log-loss gain vs empirical Bayes: **0.291%**.
Candidate minus empirical-Bayes log loss: **-0.004259** (game-clustered 95% CI -0.004730 to -0.003756).
Context minus talent-only log loss: **-0.001913** (game-clustered 95% CI -0.002215 to -0.001627).
Maximum absolute class calibration gap: **0.0040**.

Locked PA-layer gate: **PASS**
PA layer promoted: **YES**
Full game model promoted: **NO**

The test seasons in this run are the locked final PA holdout. This result may validate the pre-PA probability layer, but it does not validate runner transitions, bullpen logic, team-run distributions, winner probabilities, exact scores, or the full production application. The same holdout may not be used to retune the architecture and then be described as untouched.
