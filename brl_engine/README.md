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

This runtime has been the live one since Oct 7, 2026 (the shadow lane that validated it is retired).
Receipts of live runs: `diagnostics/v2_receipt.json` on the `brl-live-data` branch.

## Models

`model.json` names the plate-appearance model the live runtime uses. `locked-pa-2026-v1` is the
locked model inside the data package. Any other name must have a manifest in `models/<name>.json`
(hashes, parameters, feature list, benchmark metrics) written by the fit job, with its sealed bundle
at `private/models/<name>.enc` on the ledger branch. The current live model is `pa-2026-v2-physics`:
the locked features plus pitch-physics features (`runtime/research_lab/pa_model/physics.py`).

## Lanes (all run inside GitHub Actions, where the key and the sealed data live; only metrics leave)

| lane | workflow / trigger | what it writes |
|---|---|---|
| live | `brl-live.yml`: trigger commits, pushes to `brl_live/**`, `brl_engine/**`, and each run dispatches the next | forecasts, boxes, the page; `diagnostics/v2_receipt.json` |
| backfill | `brl-bookkeeping-backfill.yml`: push to `diag/bookkeeping-backfill` (season in `tools/backfill_season.txt`), weekly cron | sealed `private/bookkeeping/season-<y>.enc`, `private/statcast/season-<y>.enc`, `private/statcast/physics-<y>.enc`; receipt `bookkeeping/season-<y>.json` |
| research | `brl-research.yml`: push to `diag/research` (set name in `tools/research_experiment.txt`) | `research/<experiment>-<run>.json` |
| fit | `brl-fit-model.yml`: push to `diag/fit-model` (`tools/fit_model_params.json`) | sealed `private/models/<name>.enc`, manifest `brl_engine/models/<name>.json` on main, receipt `research/fit-<name>-<run>.json` |
| replay | `brl-replay.yml`: push to `diag/replay` (`tools/replay_params.json`) | `research/replay-<tag>-<run>.jsonl.gz` and `.json` |
| public check | `brl-public-check.yml` after each live run | failing test or page-check output under `diagnostics/` |

Workflow logs cannot be read from the sandbox that develops this project, so every lane writes a
receipt (and failures write where they failed) to the ledger branch; `tools/brl_record_diagnostic.py`
does the same for test output.
