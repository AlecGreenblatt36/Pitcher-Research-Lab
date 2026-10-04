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

## Week-before forecast-valid bullpen baseline

After the clean benchmark and week-ahead diagnostic finish, run a second
candidate-set baseline using each team's reliever sequence from a different
game 5-9 calendar days **before** the target game, preferring exactly 7 days.

This can be forecast-valid only if every target-game roster decision uses an
as-of-first-pitch cutoff:

- roster membership must come from an official roster snapshot or transactions
  effective strictly before the target game's first pitch;
- transactions announced later, retroactive knowledge, the target game's box
  score, target-game participants, and the set of pitchers who actually
  appeared are forbidden inputs;
- a source reliever who was optioned, designated, released, traded away, or
  placed on an inactive list before first pitch must be removed using only the
  pregame transaction record;
- same-day transactions are usable only when their effective timestamp is known
  to precede first pitch; otherwise fail closed or label the roster status
  unresolved;
- unresolved roster sides must fall back to the normal pregame candidate set and
  be counted explicitly;
- roster-source timestamp, transaction cutoff, overlap rate, removed-player
  count, supplemented-player count, and fallback rate must appear in receipts.

The target game's box score may be used later for scoring the forecast, but it
must never be used to construct or filter the pregame bullpen list.

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

## Locked PA pinned-environment gate

Before any clean benchmark result is interpreted, the serialized locked PA
provider must reproduce the saved locked predictions under the exact replay
environment lock, including pandas 3.0.6 and scikit-learn 1.9.1.

The compatibility receipt must include:

- Python and package versions;
- model, PA-history, and saved-prediction SHA-256 hashes;
- the three fixed verification dates and expected row counts;
- maximum absolute differences for the fitted model component, empirical-Bayes
  component, and 90/10 ensemble;
- fast NumPy path versus serialized sklearn pipeline;
- text-column dtypes after `read_csv` and the provider's normalization boundary;
- a fail-closed tolerance decision.

The clean full-game benchmark remains provisional until this receipt is PASS.
The reproduction receipt must be attached to the final benchmark certification;
it does not require rerunning completed game simulations if the simulation code,
artifact hashes, and environment lock are identical.

## Recovery validity and clean-rerun requirement

The immutable historical PA table means a clean replay does not learn from a
prior simulated outcome. However, the original read-only error could occur
midway through `SequentialHistoryState.advance_to`, after some counters were
mutated but before the history position was committed. The game runner also
updated its current-date marker before `advance_to` returned. Therefore, later
games in the same original shard may have inherited partially advanced or
partly double-applied history.

Consequently:

- the missing-game-only recovery is retained only as a debugging artifact;
- it is **not** accepted as final benchmark evidence;
- candidate, flat, and oracle must be rerun from scratch in clean shards under
  the fixed implementation and exact environment lock;
- each clean shard must fail if any game fails;
- the final merge must require exactly 2,430 unique games.

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
