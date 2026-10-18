# Baseball Research Lab and Pitcher Research Lab

This repository holds two things.

**Baseball Research Lab (BRL)** is an MLB forecasting and scouting system that runs in GitHub Actions on sealed data
and publishes model outputs only: win chances, box-score distributions, in-game win chances, a public track record,
and a matchup report for every game (each hitter against the pitchers he will face, built from pitches thrown before
that day and graded after the game). Site: https://alecgreenblatt36.github.io/Pitcher-Research-Lab/. The evidence is
in LEDGER.md (every experiment with its registered prediction, result and decision, including the failed ones),
METHODS.md (what the system is and what the evidence does and does not show) and discovery/MATCHUP_PROGRAM.md (the
pitch-level research program). Operating state: HANDOFF.md. Code: brl_engine/ (engine and runtime), brl_live/ (the
live site), tools/ (research lanes and the report builder).

Reproducing the estimator tests without the sealed data: `tools/brl_synth.py` builds synthetic seasons with known truth
(eight world kinds) and runs the run-value estimator on them; `python tools/brl_synth.py` reads `tools/discovery_params.json`
(for example `{"kinds": ["effect", "null"], "reps": 2, "n_pa": 60000, "estimands": ["pitch"], "reprice_structural": true}`)
and prints each kind's summary: the coefficient's bias and coverage, the aiming figures against the true value of their own
choices, and the structural pricing's calibration. The lane runs the same code as experiment `value_synth`; the receipts
cited in LEDGER.md (VALUE-17, VALUE-18, VALUE-18I, VALUE-18J) are on the data branch under research/. The registration
record `research/registry.json` is generated from LEDGER.md by `python tools/registry.py`; the discovery lane refuses an
experiment run whose commit names an ID without a ledger row.

License: MIT (LICENSE). Citation: CITATION.cff.

**Pitcher Research Lab** is the original local application for studying how a pitcher's tracked profile moves over
time, kept in its own folder, [pitcher_app/](pitcher_app/README.md), with its own README, tests and launchers.
