# BRL experiment and operations ledger

Missing metrics mean **not measured**. Operational improvements are not model-accuracy gains. Preserve failures and all model-selection attempts. The through-2024-model/full-2025 replay has not been run; previous 2025 inspection and any repeated selection must be disclosed.

| ID | Idea / action | Disposition and evidence | Delta corrected Brier, 2025 | Delta corrected log loss, 2025 |
|---|---|---|---|---|
| OPS-20261006-REFRESH | Encrypted prior-day history | Implemented. Earlier real-source refresh added 1,228 PAs in 17 games through Oct 5; healthy cloud runs reuse that accepted daily cache. Models/frozen seed unchanged. | Not run | Not run |
| OPS-20261006-FIRST | First two pregame forecasts | Locally simulated and publicly preserved at `26854d98d38d5864577351bb67a916d1d436c8e2`; awaiting completed-game eligibility/score audit. | Not run | Not run |
| OPS-20261006-MANUAL | User's configured cloud run | Run `37518902595`: setup/decryption and Pages worked; history read failed with `Unsupported encrypted package`; zero new forecasts. No missing user setting. | Not run | Not run |
| OPS-20261006-OBJECT-IO | Verify Git bytes before decrypting | Kept as reliability fix, not model upgrade. Diagnosis `37519489380` found a Contents payload declaring 687 bytes but decoding to 1,017 with wrong header/hash. Commit `f8c5b59e36b66ebc1314bdeb4e3f90ec3baf16c1` verifies size/hash, fetches once by immutable blob SHA when needed, and retains AES authentication. 21 new tests; 67 focused tests including existing refresh checks passed. | Not run | Not run |
| OPS-20261006-CLOUD | Cloud simulation plus official-lineup revision | Run `37520250698` completed history assembly through Oct 5, created Dodgers–Braves v2 with 10,000 simulations and official lineups, and deployed Pages. Publication `b8bd0836165252c6cd41d5ac1229b1b79618500a`; both v1 forecasts retained. Three versions across two games; no final scored forecast yet. | Not run | Not run |
| UI-20261006-DATE | Remove prior-day unforecast results from today's cards | Kept as display fix. Two regressions and real-document replay passed; all public JSON/scoring records preserved. Merged in PR #8 (`48fdd8c9b5dcd347db91fa991cc813e06f97ca67`) and deployed. Two current-game cards, no unnamed old-result placeholders. | Not run | Not run |
| OPS-20261006-REF-WRITE | Apply display/checker commit | Two direct ref updates failed: ReadTimeout, then GraphQL server error. That operation was stopped; PR #8 merged the tested patch while preserving newer documentation commits. No forced rollback or third direct-ref attempt. | Not run | Not run |
| OPS-20261006-REPEAT | Repeat healthy cloud iteration | Run `37522631086` completed, reused accepted history, preserved three versions, and correctly generated zero duplicate forecasts for unchanged inputs. Pages redeployed. | Not run | Not run |
| OPS-20261006-PHONE | Deployed public phone/desktop verification | Run `37522795380` passed normal HTTPS browser navigation at 390px and 1440px: cards match public JSON, version history/tabs work, no horizontal overflow or JavaScript errors. Five checker tests passed in cloud; 74 unique focused tests passed locally across the repair work. No private-runtime/key access in this check. | Not run | Not run |
| DOC-20261006 | Methods, ledger, cold-start handoff | Added and updated in the repo: METHODS.md, LEDGER.md, HANDOFF.md. | Not run | Not run |
| EVAL-2025-FREEZE | Through-2024 PA/starter copies and full 2025 replay | Planned; no fitted copies or full replay yet. Current live lock used 2025 calibration and cannot serve as the pre-2025 lock. | Not run | Not run |
| MODEL-BACKLOG | Fitted engine gaps and new contextual ideas | Not yet tried on the required frozen replay. Specify and register each separately before fitting/scoring. | Not run | Not run |

## Current live scorebook

At the deployed check on October 6, 2026 at 19:58 UTC: 2 games with forecasts, 3 preserved versions, 0 scored games. Model Brier, fair-baseline Brier and market Brier are unavailable, not zero. The two newly ingested prior-day finals had no saved forecasts and were not counted as prediction successes. No accuracy upgrade was adopted. Healthy forecast runs above were push-triggered; a cron-triggered forecast run has not yet been observed.

## Historical diagnostic, not the main scoreboard

September 27, 2026: 14 matched games, 140,000 completed paths. Prior receipt: model corrected Brier 0.20954985498549855; team baseline 0.2288995307206738; market unavailable. Both exploratory paired 95% difference intervals include zero. This is not a new test or adoption decision; never pool it into the live scorebook.

## Required entry for future model experiments

Record ID, timestamp before scoring, hypothesis, source vintage/coverage, training/tuning windows, current/candidate hashes, fixed game manifest, both paired deltas and 95% intervals, numerical MCSE separately, decision/reason, and later test-set reuse. A model is kept only after the declared gate; otherwise retain the current model and record the candidate as dropped or deferred. Repeatedly selecting on 2025 makes it development, even with through-2024 fitting.
