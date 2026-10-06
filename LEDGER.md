# BRL experiment and operations ledger

Missing metrics mean **not measured**. Operational improvements are not model-accuracy gains. Keep all failures and all model-selection attempts. The 2025 training-independent replay has not been run; prior 2025 inspection and subsequent selection must be disclosed.

| ID | Idea / action | Classification | Disposition | Delta corrected Brier, 2025 | Delta corrected log loss, 2025 | Evidence |
|---|---|---|---|---|---|---|
| OPS-20261006-REFRESH | Encrypted history through the prior calendar day | Data pipeline | Implemented; previous real-source exercise added 1,228 PAs in 17 games through Oct 5 | Not run | Not run | Previous History Refresh Checkpoint receipt; frozen seed/models unchanged |
| OPS-20261006-FIRST | Two locally simulated pregame forecasts committed publicly | Prospective capture | Preserved, awaiting completed-game eligibility/score audit | Not run | Not run | Data-branch commit `26854d98d38d5864577351bb67a916d1d436c8e2` |
| OPS-20261006-MANUAL | User's first configured cloud run | Operations | Runtime secret/decryption and Pages worked; history load failed; no new forecasts | Not run | Not run | Run `37518902595`, exact reason `Unsupported encrypted package` |
| OPS-20261006-OBJECT-IO | Verify Git object size/hash; recover inconsistent inline payload from immutable blob | Reliability, not model | 21 new tests and 67 combined focused tests passed locally; repaired cloud run completed, loaded history and saved an official-lineup forecast | Not run | Not run | Diagnosis `37519489380`: one object declared 687 bytes but decoded to 1,017; mismatched magic/hash; fix `f8c5b59e36b66ebc1314bdeb4e3f90ec3baf16c1` |
| OPS-20261006-CLOUD | First complete cloud forecast and lineup revision | Operations / prospective capture | Run `37520250698` saved Dodgers–Braves v2; retained both v1 forecasts; history through Oct 5; Pages deployed | Not run | Not run | New publication `b8bd0836165252c6cd41d5ac1229b1b79618500a`; 3 forecast versions, 0 scored games; push trigger, not observed cron |
| UI-20261006-DATE | Hide prior-day unforecast results from today's game cards | Display only | 2 regression tests and real-document replay passed; not deployed because ref-update failed twice; forecasts/scores/JSON unchanged | Not run | Not run | Prior page had 2 unnamed old-result cards; wrapper prepared to render only current-date cards |
| OPS-20261006-PHONE | Normal HTTPS phone/desktop page checker | Deployment checks | 5 local checker tests passed; prepared workflow not activated; separate normal desktop browser check passed; mobile viewport unavailable | Not run | Not run | Prepared `tools/check_brl_phone_page.py`; TinyFish run `d81c6249-270d-467d-89bb-e2c8f1d1890e`; no secret/private data access |
| OPS-20261006-REF-WRITE | Apply tested display/checker commit | Repository operation | Blocked after 2 attempts: ReadTimeout then GraphQL server error; main retained working cloud repair | Not run | Not run | Unapplied commit `e1c410c328c62d0de0ce839a36498d068a01e262`; no blind third retry |
| DOC-20261006 | Methods, ledger, cold-start handoff | Reproducibility | Added | Not run | Not run | This file, METHODS.md, HANDOFF.md |
| EVAL-2025-FREEZE | Through-2024 PA/starter lock and full 2025 replay | Evaluation prerequisite | Planned; no fitted copy or full replay yet | Not run | Not run | Current live lock used 2025 calibration; must not be reused as a pre-2025 lock |
| MODEL-BACKLOG | Advancement, defense, speed, SB/WP, bullpen/rest, postseason, talent uncertainty and new contextual ideas | Research backlog, not trials | Not yet tried on the required frozen replay | Not run | Not run | Specify and register each separately before fitting/scoring |

## Historical diagnostic, not the main scoreboard

September 27, 2026: 14 matched games, 140,000 completed simulated paths. Reported corrected Brier: model 0.20954985498549855; team baseline 0.2288995307206738; market unavailable. Both exploratory paired 95% difference intervals include zero. These are prior checkpoint results, not a new test or an adoption decision. Never pool them into the prospective scorebook.

## Required entry for every future model experiment

Record ID, timestamp before scoring, hypothesis, source vintage/coverage, training and tuning windows, current/candidate hashes, fixed game manifest, paired deltas and intervals for both metrics, numerical MCSE separately, decision and reason, and any subsequent test-set reuse. `Kept` requires the declared gate; otherwise retain the current version and mark the candidate dropped or deferred.
