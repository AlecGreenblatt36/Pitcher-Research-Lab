# Baseball Research Lab handoff

## Current checkpoint, October 8, 2026, 5 p.m. Eastern (Claude in charge)

Read this section first, then LEDGER.md (every experiment with its registered prediction, result and decision). The
October 7 checkpoint below still describes the machinery (lanes, privacy, scheduling); this section says what changed.

### What the site runs now

- Player model: pa-2026-v2-physics (brl_engine/model.json). Refits that were better per plate appearance did not help
  at the game level and stay sealed, not served: RETRAIN-02 (aging), RETRAIN-03 (548-day physics window).
- Simulator settings (ADJUST in brl_live/boxscore.py): context offsets re-estimated on the simulator's own stack
  (CTX-02, brl_live/context_offsets.json, 2025 and 2026 through 2026-09-27); run environment env-v2 (ENV-03); team
  offsets centered on the league (TEAM-05, CENTER-01); steals and runner speed (RUN-01, RUN-02); postseason starter usage
  x0.91; base running after each outcome from every 2023-2026 play (TRANS-01, brl_live/transitions.json);
  running plays between plate appearances built but off (TRANS-02, replays running, see below).
- Headline win chance: taught-v4 from October 8, 14:00 UTC (HEADLINE-04): 60% market line, 40% our model, our model =
  0.039 + 0.543 x simulator log-odds + 0.545 x team-model log-odds. Versions live in brl_live/headline_params.json with
  effective times; never edit a version already in force.
- In-game win chance: the game's simulated table averaged with the league table on the log-odds scale (LIVE-04).
- Hitter chances on the page keep each lineup slot's hits with the starter's share of that slot's plate appearances
  (PLAYER-02, STARTER_SHARE in boxscore.py).
- Live runs chain every 12 to 15 minutes; the phone check passed at 19:23 UTC October 8 (390 and 1440 px, no errors).

### Evidence (full-season replays, 1,000 worlds per game, each season with tables from other seasons)

| Season | Simulator win Brier | Mean total vs actual | Notes |
|---|---|---|---|
| 2026, 2,454 games | 0.24378 before TRANS-01; kernel -0.00061, CTX-02 -0.00023 more | 8.98 vs 8.95 | headline 0.24317 vs closing line 0.24326 |
| 2025, 2,423 games | kernel -0.00010, CTX-02 +0.00045 (walk level carried from 2026) | 8.92 vs 8.90 | headline 0.24150 vs closing line 0.24235 |

Rejected with evidence on October 8: SKEW-02 (all physics seasons: right strikeout level, worse totals), ROLE-01
(superseded), RETRAIN-03 (window). Discovery (private page, not published): DISC-01 to DISC-07, Decision Horizon page
https://claude.ai/artifact/HFxTsd3ZXTQs7d7cTivr4h (Version 8 adds the ABS challenge test).

### In flight (updated 10:30 p.m. Eastern)

PROD-02 failed (ledger): the package with the running plays made win chances better (pooled -0.00033) but total runs
worse in both seasons (squared error +0.063 and +0.067), the plays' own miss. The plays stay off until the source of
the simulator's extra runs is found (RUNS-03). Who bats is ruled out: substitutes hit about as well as the starters
they replace (scratch trans/who_bats.py, effect -0.003 to -0.012 runs a game).

Replays queued or running (references are v2-prod-trans-ctx-1000-YYYY; one-shot package evaluation: scratch
trans/package_eval.sh PREFIX; totals detail trans/total_decomp.py NEW REF; headline refit harness/taught_small.py S1
TAG25 TAG26). The account's runner token cannot cancel runs, so everything queued runs:
- PROD-03 (bullpen choice, exits, hooks and the starter leash, no plays; do-no-harm rule in its row): tags
  v2-prod3-1000-2026 and v2-prod3-1000-2025. This is the test that decides what goes live next.
- LIVE-05 (in-game win chance with the PROD-03 package, 2026): v2-prod-trans-ctx-ws-1000-2026 (reference) and
  v2-prod3-ws-1000-2026; score with harness/ingame_eval2.py style cross-fit, paired by game.
- DISP-01 (day-form shock): 37864772513 (2026) and 37864803481 (2025), tags v2-prod-trans-ctx-df-1000-YYYY.
- RUNS-03 (simulated runs by inning against real): 37869327258, tag v2-prod-trans-ctx-inn-1000-2026; scratch
  trans/runs_by_inning.py.
- PROD-02 P2 (shape inside the package; cannot pass, recorded for the shape only): 37863398520, 37863432413.

If PROD-03 passes: ADJUST reliever_choice, relief_exit, relief_hooks, leash on (the postseason exit offset +0.4 then
acts in Wild Card, Division and League series games); refit the taught headline on the PROD-03 replays and add a new
version with a future effective time if the weights move; check the next live run's boxes and the phone check.

Recorded today: TRANS-02, BULLPEN-01, BULLPEN-02 (pooled bound missed by 0.000004), LEASH-01, RUNS-02, RELIEF-03
(passes, on with the package), PROD-02 (fails on totals), RUNS-01, RUN-03, DISC-08, POST-02 (postseason exit offset).

Private pages: Decision Horizon (discovery, version 10) and Going to the Pen (bullpen management measured,
https://claude.ai/artifact/KyCZ1uYHxQmSEribmxwuQA; source in the scratchpad pen_page/).

### What the bullpen work found (RELIEF-02)

The hand-set reliever exit rule pulled relievers mid-inning at about 57% of the decision points where a change is
allowed; real managers do it at about 9%. The engine used 4.6 relievers a team-game at 3.5 batters each (real 3.3 and
4.9). The fitted exits (brl_live/relief_exit.json, off) give 3.4 and 4.8. This is the largest structural miss found so
far and part of why simulated bullpens struck out fewer batters and allowed more hits than real ones.

### Known gaps, measured

- Run conversion: with real outcomes replayed through the kernel and steals, the simulator scores 0.04 to 0.07 runs per
  team per nine innings fewer than real half-innings; the running plays close it (+0.01 and +0.04).
- Bullpens: on real plate appearances the model is about 1.5% short of relievers' strikeouts (2026, starters right); in
  the simulated games the bullpens are another 1.5% short and allow about 0.7% more hits, because the simulator's
  manager picks relievers without regard to how much each one actually pitches (brl_replay/harness.bullpen takes every
  reliever used in the last 14 days; manager.select_reliever has no usage or quality term).
- Starters: the average (+0.2 batters too many in 2026) hides openers and short rest (+8.2), starts after a relief
  outing (+4.7), returns from layoffs (+1.5, then +0.7) and regular starts 0.3 to 0.4 short (LEASH-01 in flight).
- Extra innings are the in-game table's weakest spot.

### Working notes

Tests: `cd /tmp && PYTHONPATH="<repo>/brl_engine/runtime:<repo>" python3 -m pytest -c /dev/null --rootdir=<repo>
--import-mode=importlib <repo>/brl_engine/tests <repo>/brl_live_tests -q -p no:cacheprovider` (306 passed, 1 skipped).
Lanes start on a push to their diag branch with a params file: diag/replay (tools/replay_params.json), diag/research,
diag/fit-model, diag/discovery, diag/transitions, diag/challenges. Receipts land under research/ on brl-live-data.
Scratch analysis scripts live in the sandbox only (re-create from the ledger rows if lost).

## Checkpoint — October 7, 2026 (Claude in charge; ChatGPT no longer updates this project)

Site: https://alecgreenblatt36.github.io/Pitcher-Research-Lab/ — new page (brl_live/page/template.html, rendered by brl_live/box_page.py with window.BRL inlined). Headline win chance = our market-taught model combined with the market's pregame line from October 8 (brl_live/headline.py; the equal-weight average of the simulator and the team model before that); the simulator and the team model are shown on the Odds tab and scored separately on the Record page. Projected game = most central nine-inning world with the favorite winning, chosen over all 10,000 worlds (brl_live/world_selection.py); High / Low / Upset versions likewise. Full box scores stay on the page for the slate date and the day before (box_page.trim_boxes); everything stays in the private ledger.

Accuracy work and its evidence are in LEDGER.md (ACC-01 to ACC-08). Adopted: the blend and the context offsets (brl_live/provider_adjust.py, frozen table brl_live/context_offsets.json). Rejected with evidence: bullpen v2, per-world talent noise. The private runtime is unchanged; the public layer wraps the engine's provider inside brl_live/boxscore.run_box_worlds (ADJUST settings at the top of that function's section).

Local work that is not in the repository (sandbox scratch, re-creatable from the seed data): replay harness (games reconstruction, 2026 season replay, team baseline, evaluation, PA residuals, offsets estimation, paired adjustment replay). Re-create before extending; the methods are described in LEDGER.md.

Also live since Oct 7: in-game updates every cycle for games in progress (LIVE-01), a pregame betting-market reference on the record and Odds tab (MKT-01, first real capture expected on the first pregame cycle of Oct 7), and pitch-by-pitch lines under every simulated plate appearance (PITCH-01). The dashboard CI's browser regression no longer counts navigation-aborted requests as failures.

Own runtime (brl_engine/, live since Oct 7, 06:04 UTC): the live workflow runs `brl_engine/entrypoint.py`. Public engine and pipeline code; the inherited release asset is used only as the sealed container of the data files (seed history, locked model, starter hazard, names, team results), decrypted with the existing BRL_PA_PACKAGE_KEY; its private code is never imported. Every run's receipt is at `diagnostics/v2_receipt.json` on the `brl-live-data` branch, readable through git (`git fetch origin brl-live-data && git show FETCH_HEAD:diagnostics/v2_receipt.json`); per-game problems are recorded under `iteration.blocked` in ledger.json and never stop the other games. The shadow lane that validated the switch (branch `brl-live-data-v2`) reproduced the inherited runtime's forecasts within simulation noise (CLE at CWS 0.472 vs 0.464, LAD at ATL 0.416 vs 0.423) and its workflow is retired. 197 tests pass under the own runtime (`cd /tmp && PYTHONPATH=<repo>/brl_engine/runtime:<repo> python -m pytest -c /dev/null --rootdir=<repo> --import-mode=importlib <repo>/brl_engine/tests <repo>/brl_live_tests`), and the public-check workflow runs them. A new PA model or engine change now ships as public code; only a new data package would need a new key.

Scheduling: GitHub's cron has started only three runs in this repository's whole history; every other run came from a push. The workflow therefore (a) runs on any commit to `trigger/**` on main (a one-line timestamp commit starts a run with the page deploy) and (b) chains itself when a `BRL_DISPATCH_TOKEN` repository secret (fine-grained PAT, Actions: read and write) exists: each run dispatches the next with a 12-minute wait. Until the secret exists, runs must be started by trigger commits.

The lane now carries on with the last accepted history when yesterday cannot be accepted yet (a West Coast game past midnight Eastern, or Baseball Savant publishing late), never older than the day before yesterday; forecasts record the coverage they used in history_through.

Since the 08:00 UTC Oct 7 checkpoint: play-by-play wording with batted-ball shape and direction (CONTACT-01); a season backfill job that seals every game's pitch sequences and batted balls per season for the live bookkeeping and a pitch-physics dataset for modeling (BOOK-01; push to `diag/bookkeeping-backfill`, season chosen by `tools/backfill_season.txt` on that branch; receipts at `bookkeeping/season-<year>.json` on the data branch); a research lane that runs experiments inside Actions on the sealed data and reports metrics only (RESEARCH-01; push to `diag/research`, experiment name in `tools/research_experiment.txt`; receipts at `research/<experiment>-<run>.json`); failing test output of the public check is recorded at `diagnostics/tests_<run>.txt` on the data branch because workflow logs cannot be read from the sandbox (tools/brl_record_diagnostic.py).

Calibration is exhausted (ACC-12, ACC-13): the simulator is over-confident but the equal-weight blend already absorbs it, and decayed player rates do not help. Accuracy gains must come from new information; the research lane's first experiment is pitch quality and contact quality in the plate-appearance model.

Research results so far (receipts under research/ on the data branch): physics features cut the 2026 holdout log loss by 0.00123 nats per PA with a clean interval (RESEARCH-01); the best variant adds an expected run value by exit-velocity and launch-angle cell and 30-day form deviations (physics2 run 37597386690: -0.00154 [-0.00181, -0.00126] on 2026, -0.00150 on the 2025 blend dates). Shrinkage strength did not matter.

Production path (all built, tested, merged): research_lab/pa_model/physics.py (shared feature code; builder and live state proven equal), the provider appends physics features when the bundle declares them, tools/brl_fit_model.py (workflow brl-fit-model, push to diag/fit-model with tools/fit_model_params.json) seals the fitted bundle as private/models/<name>.enc and commits brl_engine/models/<name>.json to main, brl_engine/model.json selects the live model, brl_live/physics_inputs.py assembles the per-PA physics table through yesterday (sealed season tables private/statcast/physics-<year>.enc plus the day cache). Switching the site is a one-line change of brl_engine/model.json, to be made only after the replay gate below.

Replay lane (brl_replay/, workflow brl-replay, push to diag/replay with tools/replay_params.json): frozen 2026 replay of any model with production offsets, scores for the simulator, the team model and the blend, paired against a reference per-game file (research/replay-<tag>-<run>.jsonl.gz). The locked-v1 reference started 09:23 UTC Oct 7; the pa-2026-v2-physics fit started 09:31 UTC. Gate: the blend must not get worse and the simulator's Brier must improve with an interval excluding zero.

Live lane scheduling: each run dispatches the next with the repository token (tools/brl_chain_next.py; workflow_dispatch is allowed from GITHUB_TOKEN); a run only chains when no other instance is queued or active, so pushes never double the chain. The BRL_DISPATCH_TOKEN secret is no longer needed.

Game context (series, game number, series or season records, venue, weather) is captured per game in ledger context and shown on the slate cards and the game header. A game without a probable starter is forecast with an assumed opener (LIVE-02) instead of staying blocked.

Switched 11:35 UTC Oct 7: brl_engine/model.json selects pa-2026-v2-physics (REPLAY-02: simulator 0.24563 vs 0.24759, blend 0.24442 vs 0.24514 on the paired 2026 replay; the game-level interval touches zero, the PA-level gate is clean). The first live run with it (run 37614148781) created new versions for every pending game (receipt pa_model: 371,785 physics rows through 2026-10-06). A run that creates a wave of versions takes about 8 minutes a game, so the live job timeout is 90 minutes. Postseason starter usage is on (POST-01, factor 0.91 for F/D/L). Physics tables are schema v2 (per-pitch-family sums); a weekly cron tops up the current season's backfill.

To roll back: set brl_engine/model.json back to locked-pa-2026-v1 and merge; the next run uses the data-package model and the snapshot hash reverts to inputs only.

Afternoon of Oct 7 (UTC): the history refresh is robust to mid-PA substitutions and Statcast-only event names (HIST-02; Oct 6 accepted at 15:49 UTC after two rejections, which would have left Oct 8 without forecasts); GitHub API failures are waited out everywhere and the receipt save retries (API-01; the 14:59 run had failed silently and broken the chain); the page carries one box per game (PAGE-01, 5.6 MB to 1.9 MB); the real game is on the page in-game and after the final with a win chance chart from each forecast's own win table (UI-02, WP-01; first real use on the Oct 7 games, first box with a win table is each game's next version). The run receipt now reports history_day_notes (per-day capture notes) and storage_read_audit.requests and rate_limit_remaining.

Evening of Oct 7 (UTC): the team half of the headline blend reads the starting pitchers (TEAM-02, team-nb-decay-sp-v2: team model 0.24576 to 0.24497 on the 2026 replay with a clean interval, blend 0.24442 to 0.24425; production class reproduces the experiment exactly); reliever availability from the last three days did not help (RELIEF-01, off; replay switch `rest: true` kept) and neither did lineups in the team model (TEAM-03). Local experiment scripts for the team model live in the sandbox scratch (harness/team_sp.py, team_lineup.py) and use the seed history and the cloud replay's per-game file; re-create from the ledger rows if needed. The real game's plays move runners per event (the feed lists the batter first), and the page shows real plays even in a cycle whose continuation failed.

Real time (LIVE-03, from 19:30 UTC Oct 7): while a game is on, the page reads MLB's public feed in the browser (statsapi allows any origin for simple GETs; the live-client block in the template mirrors brl_live/real_game.py and a node test checks they read a feed the same way): the situation and the at-bat in progress every 15 s, the plays when a plate appearance ends, the lines on the Box tab. The model's numbers still come from saved runs; between them the win chance is the forecast's win table at the current situation. Local testing uses a mock feed (sandbox scratch render/mock_mlb.py) with `?api=http://127.0.0.1:PORT/api/` (honored only on localhost). The phone check records whether the deployed page followed a live game in diagnostics/phone_check_latest.json (browser_feed). A day feed that cannot be read for physics rows is now skipped and counted instead of blocking the run.

Research after the switch (all in LEDGER.md, none adopted): team defense (RESEARCH-03, +0.00003), gradient boosting on the same features (RESEARCH-04, worse), league run environment and day of season (RESEARCH-05, no change). The plate-appearance model has plateaued at about -0.00155 nats per PA against the locked one with these feature families; the production paths for defense (DefenseState, profile injection) and environment features exist behind the bundle's feature list if a later variant earns them.

Headline from October 8 (brl_live/headline.py, versions in brl_live/headline_params.json, each with an effective time; a game is scored with the version in force at its first pitch): our model = the simulator, the team model, the starters' log factors and the team ratings, weighted by least squares on the market's closing log-odds (2025 through Aug 16 and 2026, 4,140 games; DISTILL-02); headline = 70% market line, 30% our model on the log-odds scale when a pregame line was captured, our model alone otherwise (HEADLINE-02). The market never enters our model. The record publishes ours, blend (the headline) and recipe per forecast; the Record page has a When we disagree section; in-game numbers start from the headline and the gap fades with the square root of the game left (WP-02). Market history for fitting: research/market-2026-37684588887 (ESPN summaries) and research/market-sbr-37689046619 (2021 to 2025, mlb-odds-scraper dataset release; MKT-03). Refitting: re-run the scratch harness (harness/fit_headline.py, headline_grid.py, clv2.py; methods in the ledger) after new replays, then add a new version with a future effective time; never edit a version already in force.

Evening of Oct 7 (UTC), later: the run environment is on (ENV-03; brl_live/environment.py with environment.json env-v2: temperature, roof, day games, Wrigley wind and a shrunk venue residual per outcome class, from the official feed's weather at forecast time or the venue's usual weather). Past games: the page opens any date of 2025 and 2026 (brl_live/archive/season-<year>.json from tools/brl_season_archive.py, built locally from the replays, market files, conditions, team features and the Chadwick names; win-expectancy.json from 2023-2025 games); box scores and plays come from MLB's feed in the browser. Rebuild the season files after a new replay or recipe. Game conditions lane: brl-conditions (push to diag/conditions). Research set 'matchup' (pitch-family interaction features in physics.py) is running.

Next: watch the first physics-model finals on the Record page; new information rather than new features (health and roster news, catcher framing would need per-pitch catcher identity, weather forecasts); game-level aggregation (bullpen roles in October); a slower-learning gradient boosting run is the only model-class follow-up worth one job.

## Current checkpoint — October 6, 2026, 6:59 p.m. Eastern

Site: https://alecgreenblatt36.github.io/Pitcher-Research-Lab/

The box-score product and the edge-track measurement extension are deployed. Latest implementation: `f5de97fd7ddf3494ed70d75046ceef24b61b0e7a`, PR #10. Eight immutable forecast versions across two games, three full box versions, one carrying new simple skill baselines. All full boxes contain 10,000 completed worlds and five complete examples. Zero final scored game/player/skill forecasts at this observed checkpoint. Headline model/fair-team/market Briers are null, not zero. No edge model has been fitted or adopted.

Read METHODS.md, LEDGER.md, `research/edge_track/README.md`, `experiments.json`, and `DEPLOYMENT_20261006.json`. The latter contains exact successful run IDs, publication identity and observed clock boundaries. Previous detailed box-sprint handoff remains in Git history at `2a262c6c73aa839c2b9ba591b1c32dac3217fcab`; its older counts are superseded, not erased from the evidence.

## User priorities

The conditional pitch bridge remains the next separate engine sprint: produce pitches conditional on the PA outcome already chosen by the locked model. Use only prior-date pitch usage by count/batter hand and speed distributions; display count, type, speed and result, explicitly simulated rather than predicted. Only later test whether those counts improve starter removal. Attractive sequences do not authorize changes to live PA probabilities or the manager.

Parallel edge research now registers: EDGE-01 own-baseline stuff change; EDGE-02 swing-path versus pitch-path interactions; EDGE-03 arsenal versus hitter weakness; EDGE-04 bullpen workload/availability and BCI; EDGE-05 later in-game state updating, scored in a separate lane. These are hypotheses, not implemented model advantages. Each child experiment must freeze its features, clocks, training/tuning windows, settings, source/model/manifest hashes and score targets before evaluation. No coaching/injury causality is inferred from a predictive signal.

Then resume postseason bullpen/active-roster policy, daily digest, track-record completion, through-2024 model copies/full2025 replay and one fitted engine idea per sprint. Current live PA and starter locks are unchanged. The existing PA calibration used2025, so it is not eligible as the pre2025 model. Reconcile actual schedule coverage rather than manufacturing2430 rows. Repeated selection on2025 is development, not untouched confirmation. Both corrected winner Brier and log-loss paired95% intervals must pass the declared gate; target-specific effects are also mandatory. Keep numerical MC error separate.

## Working system and privacy

Repository `AlecGreenblatt36/Pitcher-Research-Lab`; default branch main; state branch `brl-live-data`. `.github/workflows/brl-live.yml` requests minute7,22,37,52 each hour plus manual/relevant-push triggers. Actual schedule-triggered forecasting was witnessed in run37532008456. Future timing is not guaranteed; late computations/publications never get backdated.

The correct secret, encrypted runtime release asset and Pages settings are installed. No user setup change is needed. Secret NAME only: `BRL_PA_PACKAGE_KEY`; never print, log, request in chat or commit its value. Release tag `brl-live-runtime-20261006`, asset `brl-live-runtime-20261006.zip.enc`, pins `brl_live/runtime.json`. Decrypted runtime lives under RUNNER_TEMP and is removed after jobs. Raw player history, model parameters, source snapshots and keys remain encrypted/private. No font files are shipped; pages load the requested public font provider.

`brl_live/entrypoint.py` restores pinned runtime and calls `box_runner.main`. `history_refresh.py` accepts reconciled prior-Eastern-day source data; `verified_store.py` checks returned byte count and Git object SHA, makes at most one immutable-blob recovery and still enforces AES authentication. `boxscore.py` is observation-only accounting; `edge_metrics.py` freezes and scores simple comparators; `box_page.py` renders Pages. Existing `live_extension.py` supplies refreshed inference. Private runtime supplies original app/engine/provider/manager/contracts/security. Public dashboard app.py can shadow private app package: use isolated runtime test paths, not additional dependencies to disguise the wrong import.

Public Pages artifact stays exactly index.html, predictions.json, .nojekyll. Ledger forecasts/publications/actuals/status coexist with box_scores, box_publications, actual_boxes, player_scores, skill_scores and view_scope. Immutable box outputs live in box_forecasts/<hash>.json. Public distributions and five generated examples are outputs, not raw source records. No source-level private input fields should enter public payloads.

## Latest observed cloud and phone evidence

- Cloud test run37543029616 passed84 focused tests and12-file source digest verification. Same84 passed locally; counts overlap. Python compilation and JS syntax passed. Pytest cache warning under /dev did not skip tests.
- Main live run37543284982 was push-triggered and completed. Secret read/authenticated runtime succeeded; accepted history throughOctober5 was reused, not freshly downloaded.31 complete earlier games supplied comparator history;2253 prior actual PAs supplied empirical bookkeeping counts. No engine rule/model coefficient changed.
- New Brewers–Padres(game849826) v4: forecast ID9e84d6ea50f6804e17f1886803d2897687f6d7a721d368bc31f7c959550aeb0a. Saved6:57:21p.m. Eastern, box public6:57:27 before9:30p.m. scheduled start. Box hash4f70c7b7a83dab63e9b72f9a830f9c4e63ab7d19d5361545de8ab1014c13afb0; commit32770ae6a5179f328019e026ea77d6d8cfd14f68. This was a new input case, not a repeat-run parity experiment.
- Dodgers–Braves was already in progress: earlier predictions were preserved, not resimulated or given a retroactive comparator. New metrics cannot inherit the timestamps of old win-only or baseline-less boxes.
- Deployed browser run37543886020 passed normal HTTPS navigation at390px and1440px. Artifact11448774261 includes screenshot/receipt evidence. Requested fonts loaded, text>=16px, taps>=44px, no horizontal overflow/JS errors; How close/Track record/How and all five sample-game openings checked. This is browser-viewport evidence, not testing every physical phone/browser.
- Independent original-dashboard CI37543285049 also passed. Do not claim its earlier intermittent selector issue was specifically repaired here.
- First actual-first-pitch eligibility and final game/player/skill scoring remain pending. Successful pregame publication does not establish postgame scoring or forecast accuracy.

## Metrics and exact-version publication

Every model count distribution includes all10000 worlds and zero opportunity for nonappearing pitchers. Team/combined runs are aggregated from coherent score pairs, never by summing unrelated marginal player draws. Five sample seeds replay exact paths: one modal-score example and four fixed world positions; a typical score is not the most probable detailed sequence.

New seven-target headline: pitcher K, BB, pitch counts, innings; reliever appearances; team total runs; high/low total. High is the declared9+ combined runs versus<=8, one binary event scored once. Batting H/HR/K remain secondary. Counts use fair finite-ensemble CRPS for model draws; binary targets use corrected model Brier. Exact fitted baseline PMFs use ordinary exact-distribution scores, not a correction pretending training observations were simulated forecasts. Metric scales differ; never average disparate statistics into an undocumented overall score.

Simple pitching comparator: empirical earlier complete-team starter lines and uniformly mapped relief lines/zero appearances for the same current pregame pitcher pool. It does not know which pitchers will appear and is not a fitted readiness model. Its current cache support is short(31 complete games), not a full-season comparator. Team/total comparator uses the unchanged team-strength NB recipe and dispersion. Raw source boxes remain encrypted; only derived forecast PMFs/coverage/hash are public.

Baselines freeze inside each new box before independent immutable publication. Latest eligible public box version per game is chosen without outcomes; older baseline-less boxes remain unpaired. All eligible pregame bullpen members, including nonappearances, are scored; unprojected actual participants remain visible and excluded coverage is reported. Missing fields are unavailable, not zeros. IP5.2 means17 outs, not5.2 decimal innings. Scoring checks player/game identity, unchanged content hash and publication before observed first pitch. Per-target player losses average within game, then equally across games. Player rows within games are not independent confirmation samples.

`research/edge_track/report.py` consumes matched already-scored rows for current/candidate/simple baseline; paired date-block confidence intervals are per metric. Its CLI arithmetic oracle passed with synthetic fixtures only. The report cannot establish provenance or authorize adoption just from a numerical gate; it explicitly leaves adoption false. Future experiment receipts need fitted artifacts, source vintages, declared splits and separate numerical MC evidence.

The attached legacy contact-only transport cannot move K or BB/HBP. EDGE-01 needs a separately declared probability adapter rather than claiming that contact restriction does the job. Preserve pooled anchor outcome definitions and fit any BB/HBP split separately. Sparse bat-tracking or pitch-path data must have coverage/measurement-regime audits; missing biomechanical fields are not fabricated. In-game observations are forbidden in pregame experiments and belong in separate origin-matched future-update comparisons.

## Failures and unfinished work

Separate local full-repeat setup failed twice: StreamingExecNotEnabledContainerError, then run_real_integration.py missing because the first command never ran. That operation stopped. Fix for a future repeat: write and verify the harness file before a nonstreaming launch. No new complete-vector parity is claimed this sprint. The previous box sprint did reproduce10000 frozen score/seed vectors; keep that older evidence distinct.

Earlier failures are retained in LEDGER.md: wrong pytest app namespace, font-check weight mismatch, inconsistent Contents bytes, two direct-ref write errors resolved through reviewed PR merge. Do not retry any same failing step more than twice. A green setup/publish job is not enough; read machine receipt stages and forecast counts.

Known engine limits remain regular-season bullpen policy in postseason, legacy speed/defense/rest, no pinch hitters/steals/WP/PB, no talent uncertainty, and initially unverified historical publication vintages. Pitch counts are still empirical prior-pitch-event annotations on an independent bookkeeping stream, not generated sequences or managerial inputs. Daily-history machinery works on accepted prior days; the next calendar-day refresh remains separately unobserved. No through-2024 refits,2025 replay,licensed-market comparison or fitted edge experiment has run.

For the next work session: read latest live run and public JSON first; verify final scoring if games have finished, retain all versions, then proceed to the conditional pitch bridge or a separately declared research sprint. Do not fit on scored games, overwrite original records, backdate forecasts, collapse metric scales, or describe null results as gains.
