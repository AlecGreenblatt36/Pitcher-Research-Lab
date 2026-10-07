# BRL experiment and operations ledger

## Checkpoint — October 7, 2026 (Claude takes over the project)

Evidence base: a full out-of-sample replay of the 2026 regular season with the locked PA model and fitted starter hazard, every input prior-date (lineups and starters as they actually played, bullpens from the team's last 14 days, roles from the prior 365 days), 2,427 games, 200 worlds each, finite-path corrected. Scores are winner Brier (coin flip = 0.25000); "better than coin" = (0.25 - Brier) / 0.25. Fair team model: negative-binomial runs with decayed, shrunk offense/defense ratings and prior-data home field, tuned on 2025 only (tau = 180 days, k = 15).

| ID | Attempt | Disposition and evidence | Delta corrected Brier, 2026 |
|---|---|---|---|
| ACC-01 | Simulator alone | 0.24691 raw, 0.24570 corrected (1.7% better than coin). Calibration slope 0.64 over the season (0.18 in April, 0.44 in June, 0.98 in September). | reference |
| ACC-02 | Fair team model alone | 0.24576 (1.7%). | -0.00115 vs raw simulator |
| ACC-03 | Equal-weight log-odds average of simulator and team model (no fitted weights) | 0.24497 (2.0%); difference vs team model -0.00079 [-0.00203, +0.00037], May-Sept -0.00143 [-0.00281, -0.00003]. **Adopted as the headline win chance** (brl_live/record.py). | -0.00079 vs team |
| ACC-04 | Forward-fitted recalibration and home shift on the blend | 0.24461 (2.2%). Not adopted: weights fitted on the live season drift toward the simulator; headline stays unfitted. | -0.00036 vs blend |
| ACC-05 | Bullpen v2: rest from the last two days, shrunk K-BB-HR quality, save-based closer and setup roles, leverage from quality | Paired on 1,959 May-Sept games: +0.00017 [-0.00155, +0.00182]. Run levels unchanged. **Rejected**; regular-season heuristic bullpen stays. | +0.00017 |
| ACC-06 | PA-level residual diagnosis of the locked model (every 2025 and 2026 plate appearance, prior-date history state) | Visiting hitters in the top of the first: observed K 26.2% vs predicted 22.8%, run value 0.3085 vs 0.3298 per PA (2026); home hitters in the bottom of the first 0.3509 vs 0.3365. Same direction in 2025. Overall offense over-predicted by 0.17 (away) and 0.08 (home) runs per game. The model has no side x inning interaction. | diagnosis |
| ACC-07 | Context offsets: log(observed/predicted) per outcome class in six buckets (side x 1st / 2nd-8th / 9th+), 2025 prior plus expanding prior-date 2026, 200 pseudo-PA shrinkage, applied at simulation time | Paired on 490 May-Sept games (every 4th): -0.00122 [-0.00412, +0.00146]; log loss -0.00240; simulated home rate 0.5209 vs actual 0.5204 (base 0.5163); away runs 4.490 vs actual 4.414 (base 4.603); run CRPS 1.7301 vs 1.7325. **Adopted** (brl_live/provider_adjust.py, brl_live/context_offsets.json frozen through 2026-09-27). | -0.00122 |
| ACC-08 | Per-world talent noise c = 1 on top of the offsets (log-multipliers per player and class, sd c*sqrt((1-q)/(q(n+180))), Jensen-centred) | Same 490 games: +0.00126 [-0.00280, +0.00538] vs base, +0.0025 vs offsets alone; run CRPS no better. **Rejected**; code kept switched off (talent_noise_c = 0). | +0.00126 |
| UI-01 | Live page v2 (brl_live/page/template.html): slate cards, hero line score, Projected / High / Low / Upset versions chosen over all 10,000 worlds (brl_live/world_selection.py), Summary / Box score / Plays / Odds tabs, Record page, day switcher | Deployed to main (709eff1, acd220c). Deployed phone checks passed at 390 and 1440 px (run 37557075254). First scored live game: LAD at ATL, Oct 6, all methods missed (LAD 49%). | product |
| LIVE-01 | In-game updates (brl_live/live_sim.py, live_feed.py, live_update.py) | LiveSimulator runs the engine's own loop from an observed state; starting at the first pitch reproduces GameSimulator event for event on every tested seed. Each cycle every game in progress is continued in 2,000 worlds from the official linescore and boxscore (lineups with substitutions, pitchers used, line so far, runners, outs, score). First cloud run: MIL at SD, bottom 4th, SD 2-1, SD 74% live. Snapshots are replaced each cycle and never scored. Errors are recorded in the ledger, never raised. | product |
| MKT-01 | Betting-market reference (brl_live/market.py) | ESPN public scoreboard moneylines, vig removed, last capture while pregame, scored on the record only when captured before the observed first pitch. First evening: the Oct 6 games had already started, so no line was captured; shape diagnostics are in the ledger receipt for the first pregame cycle. A reference line, never an input. | pending |
| PITCH-01 | Pitch-by-pitch bookkeeping (brl_live/pitch_bridge.py) | Real count paths from prior official feeds by outcome (own pitcher with at least 12, else league by hand), only sequences consistent with the outcome; pitch types from the pitcher's mix by hand and count, speeds from his distribution by type, league fallbacks. Pitch counts now come from the paths; count pools remain the fallback. Outcomes and pitching changes untouched. Real-engine end-to-end build: 82 of 82 plays carried sequences. | product |
| BOX-11 | Bookkeeping v2: shrunk hit-by-pitch rate by batter hand, pitcher and batter from 70,550 prior walk/HBP events; pitcher pitch-count pools used only with at least 25 PA | Replaces raw per-pitcher pools that skewed HBP. End-to-end 10,000-world build on a real 2026 game: 0.50-0.67 HBP per team game, starters 82-89 pitches. | product |

Replay caveat: in the replay, early-season bullpens fall back to the previous season's last ten games because no roster is available; the live system takes the bullpen from the official pregame feed, so the replay's April number (0.25415) is pessimistic for live use and has not been separated from model error.

Missing metrics mean **not measured**. Operational improvements are not model-accuracy gains. Preserve failures and all model-selection attempts. The through-2024-model/full-2025 replay has not been run; previous 2025 inspection and any repeated selection must be disclosed.

## Current checkpoint — October 6, 2026, 6:59 p.m. Eastern

The edge measurement track is deployed at `f5de97fd7ddf3494ed70d75046ceef24b61b0e7a` (PR #10). Five hypotheses are registered, none fitted or adopted. How close and Track record now compare seven target families against simple pregame-frozen baselines. Cloud run `37543284982` completed one new 10,000-world Brewers–Padres forecast with those baselines, saved 6:57:21 p.m. Eastern and published 6:57:27 before its 9:30 p.m. scheduled start. Eight forecast versions across two games and three box versions remain preserved. Already-started Dodgers–Braves was not given a retroactive baseline. Zero final scored game/player/skill forecasts at the observed checkpoint; all three headline Briers remain not yet.

Normal deployed HTTPS checks passed at 390px and 1440px in run `37543886020`; 84 focused tests passed both locally and in isolated cloud run `37543029616` (overlapping counts, do not sum). Complete machine evidence: `research/edge_track/DEPLOYMENT_20261006.json`. A future score, first-pitch audit or accuracy improvement is not established by successful publication.

## Box-score sprint — earlier checkpoint retained

At 5:40 p.m. Eastern: two live games, seven preserved forecast versions, two complete box projections, 10,000 completed worlds and five full examples per box. Zero final scored game/player forecasts. First schedule-triggered forecasting was observed. Phone and desktop deployed box-page checks passed. No accuracy upgrade was adopted.

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
| BOX-09 | Font verification | Initial browser check failed Requested fonts unavailable because it checked an unused default Barlow weight. Explicitly loaded/verified body400 and heading700 FontFace objects, retained strict font requirement and added every-view geometry tests. One corrected cloud rerun passed. | Not run | Not run |
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

At the earlier 3:58 p.m. Eastern deployed check there were three versions/two games/zero scored games and no witnessed cron run. Later checkpoints supersede those counts without altering historical receipts. Prior-day finals without forecasts never enter the prediction scoreboard.

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

Measurement work: complete-world team/total distributions; pregame-frozen simple pitching/NB baselines; seven paired skill rows in How close and Track record; retained batting detail; per-metric date-block report utility. These are reporting/data-contract changes, not claimed accuracy gains. The local source check used 31 earlier complete games (62 team boxes), produced 19 named pitcher comparator distributions, and changed no saved forecast. The initial live baseline's recent cache support is disclosed.

| ID | Measurement work | Disposition and observed evidence | 2025 effects across all targets |
|---|---|---|---|
| EDGE-MEASURE-01 | Seven-target proper scoring and frozen simple baselines | Kept as measurement instrumentation, not a model upgrade. Model count distributions use fair finite-ensemble CRPS; fitted PMFs use exact-distribution scores. All eligible pregame relief candidates remain, including zero appearances. Game-balanced summaries and paired date-block report utility implemented. | Not run |
| EDGE-MEASURE-02 | Cloud tests and source verification | 84 combined tests passed locally and in cloud run37543029616;12 final source digests matched; Python and JS compilation passed. CLI synthetic arithmetic oracle passed but supplies no baseball-accuracy evidence. Pytest cache warning under /dev was nonfatal. | Not run |
| EDGE-MEASURE-03 | Live paired-target publication | Run37543284982 read/authenticated runtime, reused October5 history, assembled31 complete prior games for baselines, completed one new10000-world Brewers–Padres forecast. v4 public6:57:27p.m. Eastern before9:30p.m. Existing8forecast/3box versions retained. Publication32770ae6a5179f328019e026ea77d6d8cfd14f68. No post-start Dodgers–Braves backfill. | Not run |
| EDGE-MEASURE-04 | Phone-facing How close and Track record | Kept and deployed. Run37543886020 passed normal HTTPS at390/1440, fonts/16px text/44px taps/no overflow/no JS errors, seven-target views and five sample-game openings. How tab explains score scales, sparse baseline support and separate in-game lane. No scored finals yet. | Not run |
| EDGE-MEASURE-05 | Separate local full-repeat harness | Stopped after two setup failures: StreamingExecNotEnabledContainerError, then missing run_real_integration.py because the first command never ran. Fix: write and verify the file before a future nonstreaming launch. No new full-vector parity claim; the successful cloud forecast used a different input case. | Not run |

No predictive candidate was kept or dropped on accuracy this sprint. All five remain registered/unfitted. Frozen through-2024 models, full2025 replay, first final paired skill scores and conditional pitch sequences remain pending. All exact cloud timestamps, identifiers and pending boundaries are in `research/edge_track/DEPLOYMENT_20261006.json`.
