# Sequential Game Simulator V1

## Scope

This package is the first end-to-end game-state layer around the validated seven-outcome PA model. It simulates a baseball game one plate appearance at a time and keeps batting order, bases, outs, inning, score, pitcher workload, starter removal, bullpen selection, the three-batter minimum, extra innings, and walk-offs synchronized.

The game engine and the PA probability provider are deliberately separate. `GameSimulator` accepts any provider implementing `probabilities(context)`. The included ratings provider exists so the engine and browser can run before the serialized PA artifact and live feature snapshot are connected. Its outputs must not be described as the validated PA model.

## Outputs

A Monte Carlo run returns:

- away, home, and tie simulation frequencies with Monte Carlo sampling intervals;
- expected, median, and 10th/90th percentile team and total runs;
- the most common exact scorelines and their frequencies;
- inning-by-inning run pressure and scoreless-inning probabilities;
- starter innings distributions;
- reliever appearance probabilities;
- a representative fully logged simulated game;
- provider and game-validation status metadata.

## Evidence boundary

The sequential engine is runnable and unit-tested, but it is not yet game-level validated. A displayed 61% means that 61% of simulations ended with that team winning under the current PA provider, runner kernel, and manager policy. It does not yet mean historically calibrated 61% real-world win probability.

Promotion requires a frozen chronological replay of untouched games that scores winner log loss/Brier/calibration, team-run error, total-run distribution coverage, starter workload, and reliever appearances. The runner-transition and manager-policy layers must also be trained or calibrated from time-valid historical data rather than repeatedly tuned on the final game holdout.

## Cloud entry point

`cloud_game_sim_app.py` registers the simulator blueprint, serves the simulator at `/game-simulator`, redirects the cloud root there, and preserves the existing pitcher dashboard at `/pitcher-lab`.

## Next data integrations

1. Load the verified serialized PA model and a timestamped live-feature snapshot.
2. Pull schedule, live game state, confirmed lineups, probable/current pitchers, venue, and weather from MLB feeds.
3. Estimate the empirical runner-transition kernel from prior play-by-play, conditional on PA category, base/out state, runner speed, batted-ball type when available, and defense/arm context.
4. Fit starter-removal and reliever-selection models from prior managerial decisions with workload and availability features.
5. Freeze and score a large chronological game replay before labeling win or score probabilities calibrated.
