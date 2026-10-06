# BRL experiment and operations ledger

Missing metrics mean **not measured**. Operational improvements are not model-accuracy gains. Preserve failures and all model-selection attempts. The through-2024-model/full-2025 replay has not been run; previous 2025 inspection and any repeated selection must be disclosed.

## Current checkpoint — October 6, 2026, 5:40 p.m. Eastern

Two live games, seven preserved forecast versions, two complete box projections, 10,000 completed worlds and five full examples per box. Zero final scored game/player forecasts. Model Brier: not yet. Fair baseline Brier: not yet. Market Brier: not yet. First schedule-triggered forecasting is now observed. Phone and desktop deployed box-page checks passed. No accuracy upgrade was adopted.

## Box-score sprint

| ID | Attempt | Disposition and evidence | Delta corrected Brier, 2025 | Delta corrected log loss, 2025 |
|---|---|---|---|---|
| BOX-01 | Observe full player and inning books without changing the engine | Kept as product instrumentation. One full 10,000-world frozen-input rerun reproduced every original away/home/seed vector. Team, inning, batting and pitching accounting reconciles. | Not run | Not run |
| BOX-02 | Use inherited PA pitch_number for counts | Dropped as a data source: uniformly normalized to 1. No hand-set replacement counts. | Not run | Not run |
| BOX-03 | Earlier official pitch events for empirical annotations | Kept as independently sampled bookkeeping, not a removal-model input. Prior pitcher/outcome/hand pools with declared fallbacks give pitch-count estimates and BB/HBP splits. 2,253 prior PAs supplied in the live run. No pitch sequences or accuracy claim. | Not run | Not run |
| BOX-04 | Final actual player-ID joins and proper player scores | Implemented. Seventeen genuine final feeds parsed; a separate 10,000-world historical comparison exercised actual batting/pitching rows. Fair CRPS for count distributions, corrected hit/HR occurrence Brier, independent pregame box-publication gate. First prospective final score still pending. | Not run | Not run |
| BOX-05 | Prototype-styled phone boxes, sample games and track record | Deployed via PR #9, merge `1fc11b2f2b2f78d1b6892c11756970125b714b9f`. Full line/batting/pitching boxes, five samples and half-inning PBP; actual comparisons and How tab. Exact-version matching prevents stale player boxes under a newer header. | Not run | Not run |
| BOX-06 | Cloud box forecasting and publication | Run `37532752518` succeeded. Dodgers–Braves box public at 5:25:25 p.m. Eastern before its 6 p.m. start; Brewers–Padres box public at 5:29:45 before 9:30 p.m. Each has 10,000 completed worlds. Seven total versions preserved. History through October 5. | Not run | Not run |
| BOX-07 | Normal HTTPS phone/desktop verification | Workflow `37534342077`, corrected second attempt, job `112514427447`, artifact `11445552293`, passed. 390px/1440px screenshots, both actual font families loaded, text >=16px, tap heights >=44px, at most one status note, no sideways overflow or JS errors. All five sample games opened. | Not run | Not run |
| BOX-08 | First isolated cloud regression execution | Initial failure: dashboard pytest config imported public app.py, causing ModuleNotFoundError for Flask. Fixed runtime isolation with `-c /dev/null --import-mode=importlib`; 47 focused tests passed, with no assertion weakened. | Not run | Not run |
| BOX-09 | Font verification | Initial browser check failed `Requested fonts unavailable` because it checked an unused default Barlow weight. Explicitly loaded/verifed body400 and heading700 FontFace objects, retained strict font requirement and added every-view geometry tests. One corrected cloud rerun passed. | Not run | Not run |
| BOX-10 | Final public-output audit | Both box hashes/publication times verified; all player histograms count 10,000; sample seeds distinct; team/inning/batting/pitching totals agree. No embedded fonts, private inputs, runtime or keys in the three-file public artifact. | Not run | Not run |
| OPS-20261006-CRON | Actual schedule-triggered execution | Run `37532008456` event=schedule generated two pregame win forecasts and published. This preceded the box rollout; it is not the box run. Future delivery is still subject to GitHub scheduling delays. | Not run | Not run |

47 focused regressions passed locally and in the isolated cloud runtime (42 box tests plus five public-check tests). A separate Node regression for exact-version matching and live status passed. These reruns overlap; do not sum them into a larger unique-test count. A local final-check command first referenced an absent test path; the actual repository test was retrieved and the successful combined run is recorded. No new model was fitted and the full 2025 benchmark was not run. Pitch-by-pitch remains the next separate sprint.

## Earlier operations — retained history

| ID | Idea / action | Disposition and evidence | Delta corrected Brier, 2025 | Delta corrected log loss, 2025 |
|---|---|---|---|---|
| OPS-20261006-REFRESH | Encrypted prior-day history | Implemented. Real-source refresh added 1,228 PAs in 17 games through Oct 5; healthy cloud runs reuse accepted daily cache. Models/frozen seed unchanged. | Not run | Not run |
| OPS-20261006-FIRST | First two pregame forecasts | Locally simulated and publicly preserved at `26854d98d38d5864577351bb67a916d1d436c8e2`; final first-pitch/score audit pending. | Not run | Not run |
| OPS-20261006-MANUAL | User's configured cloud run | `37518902595`: setup/decryption and Pages worked; history read failed with Unsupported encrypted package; zero new forecasts. No missing user setting. | Not run | Not run |
| OPS-20261006-OBJECT-IO | Verify Git bytes before decrypting | Kept as reliability fix. Diagnosis `37519489380` found a Contents payload declaring 687 bytes but decoding to 1,017 with wrong header/hash. `f8c5b59e36b66ebc1314bdeb4e3f90ec3baf16c1` verifies identity, fetches once by immutable SHA when needed, and retains AES authentication. 21 new tests; 67 overlapping focused checks passed. | Not run | Not run |
| OPS-20261006-CLOUD | First cloud official-lineup revision | `37520250698` assembled Oct 5 history, made Dodgers–Braves v2 with 10,000 simulations and official lineups, deployed Pages. Publication `b8bd0836165252c6cd41d5ac1229b1b79618500a`; v1 records retained. | Not run | Not run |
| UI-20261006-DATE | Remove prior-day unforecast results from today's cards | Kept. Two regressions and real-document replay passed; JSON and score records preserved. PR #8 (`48fdd8c9b5dcd347db91fa991cc813e06f97ca67`). | Not run | Not run |
| OPS-20261006-REF-WRITE | Apply display/checker commit | Two direct ref updates failed: ReadTimeout, then GraphQL error. Stopped that operation; merged PR #8 preserving newer docs, without forced rollback or third direct-ref attempt. | Not run | Not run |
| OPS-20261006-REPEAT | Repeat healthy iteration | `37522631086` completed, reused history, preserved versions and correctly made no duplicate for unchanged inputs. | Not run | Not run |
| OPS-20261006-PHONE | Earlier deployed phone/desktop verification | `37522795380` passed normal HTTPS at390/1440: JSON-matched cards, tabs/version history, no overflow/JS errors. Five cloud checker tests;74 overlapping unique focused local tests across earlier repair work. | Not run | Not run |
| DOC-20261006 | Methods/ledger/cold handoff | Added and updated in repo. | Not run | Not run |
| EVAL-2025-FREEZE | Through-2024 PA/starter copies and full2025 replay | Planned. Live lock used2025 calibration; cannot be the pre2025 lock. | Not run | Not run |
| MODEL-BACKLOG | Fitted engine/context ideas | Not yet tested on required frozen replay. Register each separately. | Not run | Not run |

At the earlier 3:58 p.m. Eastern deployed check there were three versions/two games/zero scored games and no witnessed cron run. The current checkpoint above supersedes those counts, without altering the historical receipt. Prior-day finals without forecasts never enter the prediction scoreboard.

## Historical diagnostic, not the main scoreboard

September 27, 2026: 14 matched games and140,000 completed paths. Previously recorded corrected Brier0.20954985498549855; fair team baseline0.2288995307206738; market unavailable. Both exploratory paired95% difference intervals include zero. Never pool this historical diagnostic into live scores.

## Required entry for model experiments

Record ID and timestamp before scoring, hypothesis, source vintage/coverage, training/tuning windows, current/candidate hashes, fixed game manifest, both paired deltas and95% intervals, numerical MCSE separately, disposition/reason and later reuse of evaluation results. A model is kept only after the declared gate; otherwise preserve the current model and mark the candidate dropped/deferred. Repeated selection on2025 makes it development despite through2024 fitting.


## Edge research track — registered October 6, 2026

No edge candidate has been fitted, evaluated on the frozen replay, or enabled in live forecasting. The effects below are **not run**, not zero. Each child variant must record its own artifact, source/manifest hashes, declaration time, target(s), coverage, paired intervals and separate numerical error. The machine ledger has all requested metric cells, including retained batter H/HR/K detail.

| ID | Idea | Status | Pitcher K | BB | Pitches | Innings | Relievers used | Team runs | High/low runs | Game Brier / log loss |
|---|---|---|---|---|---|---|---|---|---|---|
| EDGE-01 | Stuff change signal | Registered; not fitted | Not run | Not run | Not run | Not run | Not run | Not run | Not run | Not run / not run |
| EDGE-02 | Swing path vs pitch path | Registered; coverage audit required | Not run | Not run | Not run | Not run | Not run | Not run | Not run | Not run / not run |
| EDGE-03 | Arsenal vs weakness | Registered; not fitted | Not run | Not run | Not run | Not run | Not run | Not run | Not run | Not run / not run |
| EDGE-04 | Bullpen availability and BCI | Registered; not fitted | Not run | Not run | Not run | Not run | Not run | Not run | Not run | Not run / not run |
| EDGE-05 | In-game updating | Deferred to separate live-state lane | Not run | Not run | Not run | Not run | Not run | Not run | Not run | Not run / not run |

Measurement work: complete-world team/total distributions; pregame-frozen simple pitching/NB baselines; seven paired skill rows in How close and Track record; retained batting detail; per-metric date-block report utility. These are reporting/data-contract changes, not claimed accuracy gains. Source replay check used 31 earlier complete games (62 team boxes), produced 19 named pitcher comparator distributions, and changed no saved forecast. The initial live baseline's recent cache support is disclosed.

Local scoped suite: 84 tests passed including existing box and public-check tests. This is one combined count, not an addition of overlapping reruns. Full local 10,000-world rerun was not executed in this sprint: streaming process launch was unsupported (`StreamingExecNotEnabledContainerError`); the follow-on process found its harness file absent because that first command never ran. No further repeat of that full-run harness was attempted. The fix is to write/check the harness before launching a nonstreaming process. No claim of new whole-vector parity is made here; the previous box sprint's evidence remains separate. Cloud installation, new prospective execution and deployed screenshots must be read from their new receipts before being claimed.
