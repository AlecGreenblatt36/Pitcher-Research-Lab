# Baseball Research Lab — verified status

Verified on October 3, 2026.

## Verdict

The pre-plate-appearance probability layer is **verified on a locked, chronological, out-of-sample 2026 historical replay**. It provides a small but statistically precise improvement over a strong fitted empirical-Bayes matchup baseline and remains well calibrated.

This is stronger than an ordinary retrospective split because the candidate and gates were frozen, every feature was limited to information from dates before the target PA, and the entire 2026 block was scored without tuning to its results. It is **not the same as a live prospective forecast recorded before the games were played**. That distinction is preserved explicitly.

The full-game prediction system is **not** promoted by this result. Runner transitions, between-pitch events, pitcher removal, bullpen selection, team-run distributions, winner probabilities, and exact-score forecasts remain separate unverified layers.

## Locked 2026 PA evaluation

Frozen source commit: `27e207b6e5ae4fdb788f42fae9d66f0021f7f6b6`

Candidate fingerprint: `649b59b91524c8d5f3c058db8ce585e5ac42618bc173a14dba151b0b429f188c`

Protocol:

- training: 2023–2024
- tuning, classwise calibration, and ensemble selection: disjoint chronological blocks of 2025
- locked temporal test: 2026
- test coverage: 183,849 PAs across 2,429 games
- test dates: March 25 through September 27, 2026
- game-clustered bootstrap replicates: 1,000
- candidate: 90% calibrated multinomial model / 10% empirical-Bayes matchup baseline
- each 2026 PA may use completed PAs from earlier 2026 dates, but never the target date or a later date

| Model | Log loss | Multiclass Brier | Classwise ECE |
|---|---:|---:|---:|
| Prior-date league | 1.484836 | 0.712472 | 0.001688 |
| Empirical-Bayes matchup | 1.464168 | 0.703613 | 0.003653 |
| Talent only | 1.462042 | 0.703096 | 0.002688 |
| Talent + pre-PA context | 1.460129 | 0.702425 | 0.002277 |
| Locked candidate | **1.459908** | **0.702371** | **0.002134** |

Verified incremental value:

- log-loss improvement versus prior-date league rates: **1.679%**
- log-loss improvement versus empirical Bayes: **0.291%**
- candidate minus empirical-Bayes log loss: **-0.004259 nats/PA**
- game-clustered 95% CI: **-0.004730 to -0.003756**
- context minus talent-only log loss: **-0.001913 nats/PA**
- game-clustered 95% CI: **-0.002215 to -0.001627**
- maximum absolute class calibration gap: **0.003955**

All frozen PA-layer gates passed. The PA layer advances as the canonical Research Lab PA model; the game model does not.

## Evidence status

- **Reported:** the old summary-only `ROLLING_PA_RESULT.json` claim is not accepted as evidence.
- **Reproduced and locked temporal validation:** the 2025 development replay and frozen 2026 out-of-sample replay are complete, row-level predictions and metrics are preserved, and the 2026 result passed.
- **Live prospective validation:** not yet complete. This requires saving predictions before future PAs or games occur, then revealing and scoring outcomes afterward.

## Development replication

The architecture had previously passed the 2025 development holdout on 182,926 PAs across 2,430 games. That result was correctly relabeled as development evidence after it informed the final calibration and ensemble design. The separate frozen 2026 replay is the primary current validation claim.

The near-identical incremental results in the two seasons are important: the candidate improved log loss over empirical Bayes by 0.294% in the 2025 development replay and 0.291% in the frozen 2026 replay, with both game-clustered confidence intervals fully below zero. This is evidence of transportability, not proof of live prospective performance.

## Reproducibility and integrity

The locked run verified its schema, source SHA, evaluation mode, chronological year split, test coverage, test date range, prediction artifact, serialized model, and 64-character SHA-256 receipts before publishing.

Artifact hashes:

- serialized model: `3c87e4deedfb5253ac81252ad7fa2f117b16457f2c383c465670e2c3ee2fa095`
- locked predictions: `fd3708d64f68f804b95e4426817ced6536571f81084c1e82a587567d62474a58`
- tuning audit: `c2f8aa5b74e0d03e570791e5c62e64c3180ad6d942fa81bab22bb6a412d9186c`
- reproducibility-package ZIP: `a75b982f8e814cd427fc553e798a7b8533a46fcf7c9f880cd26d8a3dde722c95`

The 50,369,372-byte reproducibility package contains the model artifacts, all locked predictions, the complete PA dataset and receipt, model source, tests, requirements, documentation, and the exact locked workflow. It is retained by GitHub Actions for 90 days from the run.

## Test status

Final full repository CI:

- **53 tests passed**
- **48 subtests passed**
- Python compilation passed
- JavaScript syntax checks passed
- project validator passed all structure, dependency, secret, hygiene, and database checks
- all browser navigation tests passed and 14 browser screenshots were generated

Dedicated PA-model CI also passed all 11 PA tests before the locked run.

## Known warnings

The locked fit emitted scikit-learn ill-conditioned-Hessian and maximum-iteration warnings during multinomial fitting. The selected models used `C=0.001`; both held-out calibration optimizations reported successful convergence, the locked predictions passed artifact checks, and the candidate passed all proper-scoring and calibration gates. The warnings are preserved as an engineering caveat rather than hidden or used to retune against the locked 2026 outcomes.

Any solver change or convergence-focused refit must be evaluated on a new future holdout or explicitly labeled a post-holdout robustness analysis; the 2026 data must not be reused and described as untouched.

## What can be claimed now

> In a frozen chronological replay using only information available before each target PA, the model produced better seven-outcome PA probabilities than a strong empirical-Bayes batter/pitcher/platoon/park/recent-form baseline over 183,849 out-of-sample 2026 PAs. The gain was modest, statistically precise, well calibrated, and almost identical to the prior-season development result.

## What cannot be claimed now

- It has not yet been scored from predictions saved live before the games occurred.
- It does not predict the exact next PA outcome with certainty.
- It does not yet validate pitch-by-pitch sequencing.
- It does not validate the runner-state kernel or remove its outcome-conditioning concern.
- It does not validate bullpen or substitution logic.
- It does not establish superior team-run, winner, or final-score predictions.
- It does not justify an “extremely small margin of error” claim for individual games.
