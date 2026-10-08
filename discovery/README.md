# Discovery track

Research aimed at new baseball measurements, kept apart from the live product.

- `DECISION_HORIZON_PROTOCOL.md`: the question, the measurement and every prediction, each committed before the run
  it governs (addenda 1 to 12 carry the commit ids).
- `DECISION_HORIZON_RESULTS.md`: what each run found, with run ids, and the decision.
- Code: `tools/brl_discovery.py`. Experiments `horizon` (first profile), `horizon2` (decisive within-type test),
  `horizon3` (per-hitter incremental validity), `horizon4` (mechanism checks), `horizon5` (familiarity), `horizon6`
  (pitcher shape surprise), `horizon7` (steering limit from launch angle), `horizon8` (tunneling profile), `horizon9`
  (blind window against expectation pull), `horizon10` (arsenal flights and the paired 260 against 175 ms test),
  `horizon11` (within-pitcher change test), `horizon12` (per-hitter steering), `horizon13` (umpires as a negative control), `horizon14` (pitcher-level tunneling).

## Running an experiment

The sealed pitch-by-pitch seasons can only be opened inside Actions. Put the settings in
`tools/discovery_params.json` (`experiment` plus its parameters) on a branch and push it to `diag/discovery`; the
`brl-discovery` workflow runs it and writes `research/discovery-<experiment>-<run>.json` to the ledger branch.
Only model results, aggregates and per-player summaries leave the runner. Each experiment was first run on
synthetic pitches with a planted answer (planted commit times, planted null effects) before it touched real data.
