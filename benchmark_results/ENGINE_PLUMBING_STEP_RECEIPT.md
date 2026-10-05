# Engine Plumbing Step Receipt

**Decision:** IMPLEMENTED, NOT YET PROMOTED  
**CI state at receipt creation:** queued  
**Accuracy decision:** INCONCLUSIVE — correctness plumbing only; no paired 2,430-game score has been run

## What changed

1. Added an official MLB schedule-backed game picker endpoint.
2. Added timestamped pregame snapshots for probable starters, posted lineups, venue/roof, weather fields, and plate umpire.
3. Pregame snapshots are marked forecast-valid only when captured before first pitch; optional archive writes are rejected after first pitch.
4. Disabled silent use of the development ratings provider unless `GAME_SIM_ALLOW_DEV_PROVIDER=True` is explicitly enabled.
5. Made the automatic-runner rule game-type aware inside the simulation engine. Regular-season games may use it; postseason/unknown game types do not.
6. Added a roof rule that neutralizes outdoor weather adjustments when the roof is reported closed.
7. Added official Baseball Savant Sprint Speed and OAA loaders with timestamped receipts, explicit missing-player imputation, and a prior-season replay policy to avoid final-season leakage.
8. Added unit tests for rule resolution, schedule/live-feed parsing, pregame cutoffs, provider fail-closed behavior, postseason extra innings, Sprint Speed, and OAA parsing.
9. Added a compact Savant smoke workflow that validates the public feeds without publishing bulk source tables.

## What did not change

- The hand-set runner transition kernel is still active.
- Sprint Speed and OAA are not yet wired into a promoted empirical transition model.
- Stolen bases, caught stealing, pickoffs, wild pitches, passed balls, and balks are still absent.
- Pregame bullpen availability is not yet fitted from five-day workload.
- No historical forecast-weather archive was reconstructed.
- No proper-score improvement is claimed.

## Required verification before promotion

- All new unit tests pass.
- Savant smoke workflow passes and reports plausible row counts/ranges.
- No existing locked-PA reproduction test regresses.
- Full-season replay is rerun only after the empirical runner and bullpen data layers are complete enough to compare fairly.
