# Edge research track

Status: five registered research families; no fitted edge model and no claimed gain. This track does not replace the conditional pitch-bridge sprint or silently change the live anchor. `experiments.json` records the proposed targets, clocks and metric cells. Fitted private artifacts must receive hashes before evaluation; every attempted grid/variant gets its own child experiment ID.

## Evaluation contract

Keep the existing PA/starter models for live forecasting. Build a separate replay lock whose model fitting, preprocessing, calibration and hyperparameter selection use only observations through 2024. Choose model complexity on chronological 2024 development folds. Actual pregame source-availability evidence is distinct from the date an event occurred. A new download of an old season is not an authenticated vintage snapshot.

Each experiment compares the candidate, its frozen current parent, and an independently frozen simple comparator on the exact same game/entity cases. Keep rejected cases, source gaps and zero-opportunity players in coverage accounting. Register each candidate before revealing its evaluation scores. Do not refit, choose a shrinkage factor, change a threshold or select a subgroup from its evaluation results. Repeatedly using 2025 to choose retained ideas makes that season an adaptive development benchmark even with through-2024 fitting; the public pregame record provides separate confirmation. Do not label adaptive 2025 results an untouched test.

Game-level retention requires lower corrected binary win Brier and lower declared log loss, with both paired 95% interval upper endpoints below zero, plus timing and integrity tests. A measured targeted skill improvement is required before describing the proposed mechanism as useful. Targeted effects and secondary harms are reported, not hidden behind an overall win score. No significance-based fishing across 7 metrics, many windows and player subsets: declare the primary skill target(s) in the child experiment before evaluation. Investigate secondary results on a NEW experiment, not by editing the old one.

Pair by game, forecast cutoff, side, player and metric. Average players within game before comparing aggregate pitcher performance; retain starter/reliever and missing-support breakdowns. Bootstrap paired calendar-date blocks for the historical replay, so both versions and all player lines share a resample. Report numerical simulation uncertainty separately; a sampling/bootstrap interval is not an estimate of Monte Carlo error. The current live score card is descriptive and does not calculate promotion intervals.

## Measurement contract

| Measurable | Proper score | Baseline | Interpretation |
|---|---|---|---|
| Pitcher strikeouts, walks, pitches | Fair CRPS of unconditional count distribution | Prior complete team-box resampling by starter/relief role | Never condition on the pitcher's realized innings |
| Pitcher innings | Fair CRPS of outs / 3 | Same prior-role baseline | Actual baseball `5.2` is 17 outs, not 5.2 decimal innings |
| Relievers who appear | Corrected Brier; binary log loss in registered research results | Random assignment of earlier relief appearances to the same pregame candidate pool | Nonappearance is scored; unprojected actual arms are reported as misses in roster coverage |
| Team total runs | Fair CRPS | Existing frozen team-strength negative-binomial distribution | Save complete team outcomes from each world; never add independent player marginals |
| High versus low total | Corrected Brier; binary log loss in registered research results | Convolution of the same team NB distributions | Fixed once: high = 9+ combined runs; low = 8 or fewer. Score once, not once per complement |
| Batter hits, HR, K | Existing fair CRPS; hit/HR occurrence Brier | Additional paired batter comparator still pending | Retain detail, not the headline |
| Winner | Existing corrected binary Brier and declared log loss | Existing fair team baseline | Do not confuse 10,000 worlds with 10,000 validation games |

The new simple pitcher baseline is intentionally transparent rather than a straw-man constant: sample an earlier complete team's starter line, and map that team's relief lines uniformly without replacement to the current pregame pool. Unassigned arms receive zero. If the earlier team used more arms than the current pool, choose a uniform subset of earlier relief lines. Each earlier team-game has equal weight. The saved distribution is analytic over these historical choices; it is a fitted empirical distribution, not a finite simulation sample, so it gets ordinary CRPS/Brier. Both comparators are saved with the same pregame box and independently source-audited.

This first implementation uses the accepted private daily cache, not a fabricated full-year pitcher baseline; the number of prior games and coverage dates travel with every box. Missing count fields are unavailable. Starter identity is the first pitcher in each complete prior official box; bullpen appearance does not establish physiological readiness. Before the clean replay, rebuild this exact comparator from its legitimate rolling pregame inputs. The team NB dispersion remains trained on 2023–2024; no baseline coefficients are chosen from target-game results.

## 1. Stuff change signal

Integrate Pitcher Research Lab's change-detection measurements, but not its retrospective dashboard windows as prediction inputs. Freeze recent and reference windows before evaluation; use nonoverlapping prior-date pitcher/pitch-type histories. Recent velocity, spin, horizontal/vertical movement, release coordinates and variability are deviations from that pitcher's own reference, shrunk toward no change with uncertainty in BOTH means. Control pitch-type reclassification, handedness and measurement regime. Learn shrinkage/thresholds from development data; do not hard-code an unearned velocity bonus.

Fit a new residual family with an explicit seven-outcome simplex and independent feature support. The old contact-only transport pilot intentionally holds K and pooled BB/HBP fixed and CANNOT implement this K/BB signal. Keep that pilot unchanged. Learn a two-stage tilt: terminal K / pooled BB-HBP / contact-or-other mass, then supported contact quality within contact. BB alone requires a separately fitted prior-date BB/HBP split; don't claim a pooled change is a walk-only effect. Zero/disabled residual must return the exact anchor vector. Missing signal support returns the anchor, not a mean-imputed intercept shift.

Primary skill targets: pitcher K and BB. Secondary: pitch counts, innings and team runs. Lead-time test: next start and next 2–3 starts, with overlapping windows clustered by pitcher/date; trigger membership is determined BEFORE observing those results. Compare against prior results-only signals on identical cases. Do not define 'before stabilization' by retrospectively selecting pitchers whose outcomes later changed.

## 2. Swing path versus pitch path

Use prior swings' attack angle, bat speed and swing-plane summaries with prior pitch arrival-angle/speed distributions. Respect handedness, coordinate conventions and the difference between release velocity and arrival velocity. Do not subtract differently signed angles as a ready-made physics advantage. Use strongly pooled nonlinear interaction terms learned in development, not assumed collision percentages.

Public bat tracking is not full 3D biomechanics, and contact-measured fields can exclude whiffs. Model measurement/coverage support explicitly; don't let contact-only sample selection become a whiff predictor evaluated on a different population. Squared-up contact is an official observed label, not a substitute name for every hard-hit ball. Same-swing values may be labels but never pregame features. First audit what was available through 2024; if the required feature has no sufficient legitimate development history, that subvariant cannot use 2025 to train and still claim a clean 2025 test. Defer it to a later forward lane.

Primary intermediate targets: whiff conditional on swing and squared-up contact with the declared denominator. Translate any gain into the PA/game engine before claiming improved games. Primary game-derived skill target: pitcher K; also report team runs and retained hitter detail. No fitted interaction is enabled in this sprint.

## 3. Arsenal versus weakness

Pool each hitter's pitch-type response toward handedness/league cohorts, then integrate against the opposing pitcher's prior-date usage by count and batter side. Keep league/cohort estimates inside development folds. Don't use tonight's realized counts, pitch mixture or terminal pitches. Terminal-pitch-labeled PA outcomes are selected exposures: they are not directly a hitter's per-pitch vulnerability. Fit at the appropriate pitch/swing/contact unit, then integrate/count-bridge or fit a residual whose exposure construction was fixed first. Residualize against existing anchor information to test incremental value, not rediscovered platoon/quality effects.

Primary skill targets: K and BB; secondary: pitches and runs. Compare league pitch mix, actual pitcher prior mix and the full matchup interaction as separate registered ablations. No random splits of pitches from the same game across time folds.

## 4. Bullpen availability and capacity

Use pregame active/postseason rosters, pitches and BF in the prior 1/2/3/7 days, consecutive-use days, time since last appearance and scheduled off days known at the cutoff. Training outcomes are future use and delivered workload, not presumed medical availability. Include zero-use opportunities from a legitimate historical eligible roster; do not infer exposure only from pitchers who appeared.

Proposed Bullpen Capacity Index (BCI): posterior expected deliverable bullpen outs, optionally with a separately learned quality adjustment. Model workload capacity and usage jointly with the starter hook and future score states, then average across worlds. Do not count an arm twice or add independent probabilities as if closer/setup usage were independent. Fit a postseason modifier with a chronological held-out postseason block, not on the same games scored. Primary skill targets: which relievers appear and innings; secondary: pitches, K, BB and runs. The BCI is a proposed index, not a proven public metric or health score.

## 5. Later in-game updating

Separate namespace `in_game`, not pregame. Freeze the pregame prior. At the end of each declared inning, filter only velocity/location observations already received; observed pitch location is not command error without a target. Update a strongly shrunk day state and replay only the remaining game from its real score, outs, bases, batting order, pitcher workload and remaining roster. Preserve each timestamped update. Never rewrite the pregame forecast or use later innings to infer today's early state.

Compare with a simple state-conditioned baseline and with 'same observed state, no day-state update'. Both must receive the same score/inning information. Score final wins, remaining runs and remaining pitcher contributions on a fixed update schedule; count each update horizon separately and cluster within game. More accurate later forecasts are not directly comparable to pregame forecasts with less information.

## Source definitions and old pilot boundary

Official Statcast CSV units/definitions (accessed October 6, 2026): https://baseballsavant.mlb.com/csv-docs

Official bat tracking: https://baseballsavant.mlb.com/leaderboard/bat-tracking

Proper scoring rule definitions: https://scoringrules.readthedocs.io/en/latest/theory.html

The previously supplied `transport_counterfactual.py` explicitly limits its redistribution to contact outcomes (lines 3–5); this track does not reinterpret it as a K/BB model. The old remediation/investor templates are historical drafts, not this track's frozen statistical protocol or measured evidence.
