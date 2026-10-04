# 2025 Bullpen Diagnostic Protocol V4

**Status:** controlling development protocol before reliever-model fitting  
**Date:** 2026-10-04

## What the actual-order oracle can establish

The same-game actual-reliever-order oracle is a postgame ceiling diagnostic. It
cannot be a forecast and cannot, by itself, authorize reliever modeling because
actual reliever identity/order contains realized leverage and game-path
information.

Two leakage-sensitivity checks have passed:

- the oracle gain remains in 1-2 run games rather than concentrating in
  blowouts;
- removing position-player-pitching games does not remove the gain.

These checks reduce the blowout/mop-up explanation but do not remove all
outcome leakage.

## Week-ahead placebo

Use each team's actual reliever sequence from a different game 5-9 calendar
days later, preferring exactly 7 days.

Rules:

- source date must be later than the target date;
- source gap must be 5-9 days;
- source date must contain exactly one game for the team, excluding
  doubleheaders;
- source game must contain at least one reliever;
- choose the source closest to 7 days, then date and game_pk deterministically;
- if no source qualifies, fall back to the normal pregame engine and record the
  fallback;
- future roster-only pitchers are allowed because this is a postgame negative
  control, but their rate must be reported.

The placebo measures the value of a more realistic bullpen identity/role list
without using the target game's realized usage.

## Revised decision logic

1. **Placebo minus current engine**
   - If clearly favorable, prioritize the forecast-valid active-roster,
     availability, role, and workload pipeline before complex reliever models.
   - This is the recoverable value of supplying a better bullpen candidate set.

2. **Oracle minus placebo**
   - Report as a descriptive upper-bound gap only.
   - It mixes game-day manager usage value with target-game outcome leakage and
     cannot authorize a reliever model by itself.

3. **Reliever exit and selection models**
   - May be developed after the candidate-set pipeline is forecast valid.
   - Promotion requires a pregame-only 2025 replay against the current engine
     with the same games, 1,000 paths, seeds, and proper-score metrics.
   - Oracle performance is not a promotion threshold.

## Recovery validity

Targeted recovery of games omitted by the pandas read-only-array error is valid
because chronological PA history is advanced from the immutable historical
PA table by cutoff date. It does not depend on whether an earlier simulated
game succeeded. At the next date, all actual prior-date PAs enter the counters;
same-date outcomes remain excluded by design.

Safeguards now required:

- exact Python/package lock for replay workflows;
- copy-on-first-write guard for read-only grouped arrays;
- shard jobs fail if any game fails;
- final merge requires exactly 2,430 unique games;
- receipts contain source, artifact, environment, and prediction hashes.

## Home/road scoring diagnostic

The earlier road-run concern is not a blocker.

From the 100-path development replay:

- candidate away-run bias: +0.0993 runs/game;
- prior-data league baseline away-run bias: +0.0840;
- candidate home-run bias: -0.0370;
- league baseline home-run bias: -0.0098.

The simulator-specific home-away differential deficit relative to the prior
league baseline is therefore about 0.0425 runs/game, not 0.10. Continue the
inning/PA/runner decomposition, but do not block the bullpen candidate-set
experiment on this small residual.

## Still blocked

- no claim that the winner layer is dependable;
- no production use of oracle/placebo inputs;
- no reliever-model promotion without forecast-valid replay;
- no further 2026 accept/reject decisions.
