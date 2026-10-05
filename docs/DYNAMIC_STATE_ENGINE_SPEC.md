# Baseball Research Lab — Dynamic State Engine Specification

**Status:** research architecture; not a production accuracy claim  
**Branch:** `feature/dynamic-state-engine-v1`

## Objective

Replace a plate-appearance-only simulation with a dynamic, state-conditioned system while preserving the validated locked seven-outcome PA model as a benchmark and fallback comparison. The goal is not to sound advanced. The goal is to improve out-of-sample proper scores without leakage.

No design element is accepted because it is physically plausible or popular. Every layer must beat the current accepted engine on the same games, with the same pregame cutoff and paired random streams.

## Why this branch exists

The current engine already corrected its largest hidden failure: simulated plate appearances now use the serialized locked model rather than a silent hand-weighted ratings provider. The remaining known limitations are structural: no pitch counts, no pitch-by-pitch sequence, unvalidated runner transitions, and heuristic bullpen behavior.

## Architecture

### Layer 0 — Provenance and cutoff firewall

Every input snapshot carries:

- source;
- effective timestamp;
- retrieval timestamp;
- checksum;
- first-pitch cutoff assertion.

Any input effective after first pitch fails closed.

### Layer 1 — Dynamic pitch state

Live count state is one of the 12 legal counts. Each pitch is simulated in two fitted steps:

1. pitch type and location region;
2. ball/called strike/swinging strike/foul/in-play/HBP.

Both distributions are conditioned on:

- count;
- batter and pitcher handedness;
- pitch repertoire;
- batter chase/contact/whiff profile;
- pitcher usage/zone/whiff profile;
- inning, outs, runners, score;
- pitch count and fatigue state;
- catcher and umpire context when available.

Fouls with two strikes preserve the count. The state machine terminates only on walk, strikeout, HBP, or ball in play.

### Layer 2 — Contact and batted-ball vector

A ball in play generates a joint distribution over:

- exit velocity;
- launch angle;
- spray angle;
- hang time or projected distance.

The first implementation is an empirical conditional density with hierarchical backoff. A later physics layer may adjust the sampled vector for air density, wind, altitude, park geometry, and roof status. A physics adjustment is accepted only if it improves held-out run CRPS/PIT and does not degrade winner scores.

### Layer 3 — Defense and runner resolution

The batted-ball vector is resolved through:

- fielder position and OAA/Fielding Run Value;
- outfielder arm strength;
- runner sprint speed;
- outs and base state;
- park geometry;
- catcher blocking/throwing for non-batted-ball runner events.

Terminal PA advancement and between-pitch hazards remain separate models.

### Layer 4 — Pitcher fatigue and manager policy

Fatigue is continuous, not a hard third-time-through switch. Raw features include:

- pitch count;
- pitches in inning;
- batters faced;
- inning;
- continuous TTO exposure;
- recent velocity/release/zone changes;
- prior 1–3 day workload;
- consecutive days used;
- role and rest.

A fitted degradation model may alter pitch choice, zone probability, whiff probability, EV/LA distribution, and command variance. No hand-selected degradation coefficients are promoted.

Bullpen behavior is decomposed into:

1. pregame active roster and availability;
2. current-pitcher exit hazard;
3. candidate-level reliever selection softmax.

### Layer 5 — Full game and product

The existing Game Room remains the product shell. The dynamic engine must expose distributions, not one deterministic script:

- winner probability;
- run PMFs;
- inning scoring;
- pitch counts;
- starter workload;
- reliever appearance/order;
- player lines;
- representative pitch/play sequence clearly labeled as one sampled path.

## Randomness contract

All stochastic draws are keyed by:

- master seed;
- game ID;
- Monte Carlo path;
- event index;
- draw type;
- sub-index.

This prevents a new runner or fatigue draw from shifting unrelated PA draws and increases power in paired model comparisons.

## Statistical acceptance

### Winner layer

- corrected Brier and corrected log loss versus current engine;
- fair negative-binomial team baseline;
- calibration slope/intercept with uncertainty;
- month-forward cross-fitted recalibration reported separately.

Winner accuracy is descriptive only.

### Run distribution

- CRPS;
- run MAE/RMSE/bias;
- randomized PIT histogram and uniformity diagnostics;
- fractional boundary coverage for discrete central intervals.

### Component gates

Each new layer receives one receipt:

- exact change;
- source/artifact hashes;
- pregame cutoff;
- paired proper-score differences and confidence intervals;
- calibration/PIT effect;
- `ACCEPT`, `REJECT`, or `INCONCLUSIVE`.

## Development order

1. Finish locked-PA reproduction and clean baseline replay from private data.
2. Implement event-keyed streams in the accepted engine and verify unchanged marginals.
3. Build pregame bullpen roster/availability pipeline.
4. Re-pull pitch-level Statcast data with terminal `pa_pitches` and train count/pitch sequence layers.
5. Fit continuous fatigue/degradation models.
6. Fit batted-ball, defense, and runner-resolution layers.
7. Add weather/park and catcher/umpire effects one at a time.
8. Recalibrate and blend only through chronological cross-fitting.
9. Start frozen prospective Scorebook forecasts.

## Claims boundary

This architecture can make the simulator more realistic and may improve forecast skill. It does not by itself prove superiority over commercial systems, betting markets, or academic models. That claim requires matched pregame data and proper-score evidence.
