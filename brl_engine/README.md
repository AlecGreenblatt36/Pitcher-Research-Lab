# brl_engine: the project's own runtime

Public engine and pipeline code; only the data stays encrypted.

- `runtime/research_lab/` is the game engine and the locked plate-appearance provider, byte-identical
  to the copy that has been producing the live forecasts.
- `runtime/app/` and `runtime/cloud/` re-implement the pieces the inherited encrypted runtime kept
  private: data paths and matchup decoding, the locked provider and fitted starter policy, final-result
  parsing, the team model, forecast contracts (pregame inputs, the forecast record, audits, version
  scoring), the Git ledger store and daily runner, and AES-GCM sealing in the inherited wire format.
- `entrypoint.py` restores the inherited release asset only as the sealed container of the data files
  (seed history, locked model, starter hazard, names, team results), never imports its private code,
  and runs the same public `brl_live.box_runner` against these modules.
- `tests/` cover the contracts, the store and sealing; `brl_live_tests/` run unchanged on this runtime.

Shadow lane: `.github/workflows/brl-live-v2.yml` writes to the `brl-live-data-v2` branch and never
publishes the page. The live site switches to this runtime once the shadow lane agrees with it.
