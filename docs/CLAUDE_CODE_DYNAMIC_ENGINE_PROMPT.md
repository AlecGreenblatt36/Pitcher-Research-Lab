# Claude Code Execution Prompt — Dynamic State Engine

You are working in `AlecGreenblatt36/Pitcher-Research-Lab` on branch `feature/dynamic-state-engine-v1`.

Read these first:

1. `docs/DYNAMIC_STATE_ENGINE_SPEC.md`
2. `docs/DATA_SOURCE_REGISTRY.md`
3. `docs/MODEL_ACCURACY_TARGETS.md`
4. `docs/ENGINE_STEP_RECEIPT_TEMPLATE.md`
5. `schemas/dynamic_engine_manifest.json`

## Non-negotiable boundaries

- Do not alter or weaken `LockedPAModelProvider`.
- Do not silently fall back to ratings or league-average probabilities.
- Do not use target-game box scores, actual relievers, postgame weather, or future totals as pregame features.
- Do not claim that an architecture change improves accuracy until a frozen paired replay proves it.
- Do not work on UI before an engine component passes its gate.
- Keep bulk Statcast/MLB source rows out of the public repository.

## Foundation package

Use `Baseball_Research_Lab_Dynamic_Engine_Foundation.zip` or apply `dynamic_engine_foundation.patch`. It contains:

- `random_streams.py`: event-keyed deterministic random streams;
- `pitch_state.py`: legal count Markov state, pitch-plan/event interfaces, fail-closed table providers;
- `fatigue.py`: raw workload/state feature contract with no hand-tuned degradation;
- `shrinkage.py`: conjugate empirical-Bayes utilities;
- `forecast_diagnostics.py`: finite-path proper scores, randomized PIT, fractional discrete interval coverage, chronological month splits;
- `batted_ball.py`: empirical EV/LA/spray kernel with explicit backoff;
- `bullpen_policy.py`: fitted conditional-logit reliever selection interface;
- `data_contracts.py`: provenance and first-pitch cutoff enforcement;
- 17 passing unit tests.

## First coding task

Integrate event-keyed streams into `GameSimulator` without changing any marginal distribution:

1. retain backward-compatible `simulate(matchup, seed)` behavior;
2. derive a path ID explicitly in `simulate_many`;
3. use distinct draw keys for PA outcome, runner transition, pitcher removal, and reliever selection;
4. add regression tests showing:
   - same seed/path gives identical output;
   - reordering an unrelated draw does not change PA outcomes;
   - large-sample marginal outcome frequencies match the current generator within Monte Carlo tolerance;
   - paired-difference variance falls in an ablation fixture;
5. write one short receipt with `ACCEPT`, `REJECT`, or `INCONCLUSIVE`.

Do not start the pitch model until this task passes.

## Next task after random streams

Build the pregame bullpen roster/availability dataset using only roster/transaction/workload information available before first pitch. The week-before bullpen baseline may be used only with an as-of-first-pitch roster cutoff.

## Pitch-level task after private data is restored

Re-pull pitch-level Statcast data and preserve every pitch plus terminal `pa_pitches`. Fit chronologically:

1. pitch type/location policy;
2. pitch event transition model;
3. count-to-PA outcome aggregation check.

The pitch-level model must beat the locked PA benchmark or a defensible hybrid on proper scores. If it does not, reject it and keep the locked PA model.
