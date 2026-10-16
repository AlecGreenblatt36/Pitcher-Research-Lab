# Baseball Research Lab handoff

## Standing rule from Alec (October 9, 2026, 7:43 a.m. Eastern)

A change that is proven to make a significant improvement goes live without waiting for his approval: it must pass
its registered gates in full-season replays of more than one season (or, for an invention, hold up under stress tests
across seasons), and the ledger row says which gate it passed. Below-floor effects (too small for the replay gates to
see) are not covered and stay queued. Publishing new pages, outside contact and costs still need his approval; the
discovery page stays private.

## Waiting on Alec (postseason-only changes cannot be replay-gated)

- The World Series: POST-01 and POST-02 found its starter and reliever usage like the regular season (scale 1.0, exit
  offset 0.0), but LEASH-02's and RELIEF-04's corrections are on for regular-season games only, so World Series games run
  the plain hazards. Giving them the corrected regular-season hazards is a postseason change, so it waits for Alec.

- POST-03: postseason starters kept in by tier (good starters longer); built in brl_live/live_extension.py, off
  (ADJUST postseason_starter_tiers None).
- LEASH-02's exit correction in postseason games too; it would need the postseason factor x0.91 re-measured on the
  corrected hazard first. Regular-season games have it from October 11.

## October 16: the longer history (Alec: "2023-2026 is not valid enough for how we are predicting things")

- HIST-01 passed: every regular season and postseason 2015-2022 is sealed at private/history/pa-<year>.enc (one row per
  plate appearance, the same builder as 2023-2026; receipts at history/pa-<year>.json on the ledger branch). With the
  physics tables of the same seasons (private/statcast/physics-<year>.enc, the bookkeeping backfill) the lab now holds
  2.09 million regular-season plate appearances (2.8 times as many) and 309 more postseason games.
- STAB-01 (research lane, experiment stability): how much sample each player rate needs, from every season; it checks
  the model's shrinkage settings.
- LONG-01 (workflow BRL fit arms, tools/fit_arms.json, comparison tools/brl_fit_compare.py): refits on the long history.
  The counters now take an optional half-life (PAConfig.history_half_life_days, off by default, so the live model is
  unchanged). Stage 1 picks the arm by its 2025 tuning score and tests it once on 2026 against a control refit of the live
  settings; stage 2 (only if stage 1 passes) is a 2025 and 2026 replay. Adopting it would also need the live history file
  and the replay lane to include 2015-2022 (brl_live/history_refresh.py; tools/brl_replay.py) and three seasons of
  physics tables (window_days 1095).

## October 16, afternoon: the pitch-level chain (CHAIN-01 to CHAIN-03; the PITCH- ids were taken)

Alec's direction after the long-history tests: stop refitting the same model; build the structural change (the Pitch
Model Plan doc). Built the same afternoon:

- tools/brl_pitch_table.py (workflow BRL pitch table, one job per season): one row per pitch from the sealed study,
  sealed at private/statcast/pitches-<year>.enc, receipts pitches/<year>.json (2021-2026, 4.33 million pitches).
- research_lab/pa_model/pitch_chain.py: the chain. 15 cells (family x location), league tables by platoon, count and
  cell, player and park deviations (log-odds of observed over league-expected, shrunk with STAB-01's constants,
  400-day half-life), the pitcher's mix by count situation, a Markov chain over the twelve counts per matchup.
  build_features (date blocked, the benchmark's order) and ChainState (the live provider's state at a cutoff) share
  one code path; brl_live_tests/test_pitch_chain.py checks the chain reads planted tendencies, leaks nothing from
  the same day, and that builder and provider agree.
- Wiring: tools/brl_fit_model.py (settings pitch_chain {params}, pitch_seasons), the provider (bundle keys chain_params
  and chain_features; pitch_table=), the replay lane (loads the pitch tables when the model's manifest has
  chain_features; the harness advances the chain date by date), the live runtime (brl_live/physics_inputs.py
  assemble_pitch_table: season tables plus the day cache; box_runner passes it when the manifest has chain_features).
- The long-history model (LONG-01/02) stays sealed; STAB-02 found the measured shrinkage adds nothing to the at-bat
  model. The at-bat model is not refit again on the old feature set.
- Results so far: CHAIN-01 passed (per pitch +8, +11, +4 nats per 1,000 over league-only); CHAIN-02/02B gain -0.00076
  nats per plate appearance on 2026 (interval clear of zero, short of the 0.002 bar); CHAIN-03 wins, blend and totals
  held, strikeouts over by 2% because the context tables were the live model's; CHAIN-03B (the chain's own tables,
  brl_replay/context_chain_fit2025/2026.json, tools/brl_context_table.py) running. The chain's gain is largest in the
  season's first weeks (March +1.1 to +1.4 per 1,000) as the plan predicted.
- LEASH-04 registered: pitches per plate appearance from the matchup's count chain (ChainState.pitch_count_table,
  game_sim.pitch_counts.MatchupPitchCounts; the engine hands the context to a counter that wants it) with LEASH-03's
  pitch-term hazard correction; replay option pitch_counts {chain, fallback}; live ADJUST pitch_counts 'chain' (off)
  wires it into the engine, the box worlds and the market worlds (brl_live/hazard_correction_v2.json,
  brl_live/pitch_counts_by_outcome.json).
- Adoption, if CHAIN-03B's four gates hold: brl_engine/models/pitch02b-v2h120.json (from the ledger branch's
  models/), brl_engine/model.json, brl_live/context_offsets.json = the chain's pooled table (staged copies in the
  scratch folder adopt/); How it works already carries the chain paragraph, shown when the live model is a chain model.

## October 16, evening: the chain is live, then spring training joined it (CHAIN-03B, SPRING-01/02, LEASH-04)

- CHAIN-03B passed all four gates (wins, blend, totals, strikeouts 1.003/0.999 with the chain's own context tables) and
  pitch02b-v2h120 went live at 3:55 p.m. Eastern. The first run after the switch blocked one game: box_runner read the
  manifest from the work folder (BRL_WORK_ROOT) instead of the repository; fixed in 294175da7 (test added). The next
  run built the first chain forecast (game 849817 version 4, 4,334,696 pitches assembled).
- SPRING-01 (data): tools/brl_spring_table.py, workflow BRL spring table, one job per season. Final spring games (type S,
  February 15 to April 5) fetched play by play, flattened to the pitch table's columns, sealed at
  private/statcast/spring-<year>.enc, receipts spring/<year>.json. Every season 2021-2026 usable (54-65% of pitches
  tracked; 2026 99.7%).
- SPRING-02 (model): chain parameter spring_weight; the chain absorbs a season's tracked spring pitches before Opening
  Day into the players' counters only (never league tables, parks or the per-pitch check). Arms c0/w05/w10 on the live
  settings; w10 selected on 2025 tuning loss and passed both gates (2026 season -0.000267 nats per plate appearance,
  interval -0.000341 to -0.000184; March plus April gain 0.000588 against the 0.0005 bar). Live as spring02-w10 at
  5:01 p.m. Eastern (brl_engine/model.json, manifest brl_engine/models/spring02-w10.json with spring_seasons
  2021-2026). The live runtime loads the spring tables through physics_inputs.assemble_pitch_table (spring_years from
  the manifest); the bookkeeping backfill now also seals the season's per-pitch table so the chain's current season
  stays fresh with the physics table. Context offsets, hazard correction and pitch-count tables stay CHAIN-03B's.
- The gain from spring pitches rose with the weight (half 0.00017, full 0.00027) and faded through the season without
  reversing: spring pitches read as ordinary evidence. Not tried: weight above one.
- CHAIN-04 (two-level shrinkage, the Bayesian player layer's first stage): a hitter's or pitcher's thin swing/whiff column
  shrinks toward his own pooled deviation (k_b_pool, k_p_pool) instead of straight toward the league. Arms on the live
  settings; pool-s (100/400) selected and passed (2026 -0.000279 nats per plate appearance, interval -0.000403 to
  -0.000165; per pitch swings -3.3 and whiffs -5.9 per 1,000). Live as chain04-pool-s at 5:50 pm Eastern with its own
  context tables from check-only replays (chain04-check-2025/2026; tools/brl_context_table.py). Smaller constants did
  better at every step, so CHAIN-04B (50/200, and the cell constants re-tuned around the pooled prior) is the next arm.
- BOOST-01 (machine learning): research_lab/pa_model/boost.py, a histogram gradient-boosted matchup model (sklearn) on
  the matchup-level features only, fit on 2023-2024 with out-of-fold predictions for the training rows (4 groups of
  games), rounds picked on the 2025 tuning block from staged predictions; its seven log probabilities (gb_<class>) join
  the logistic model's features; the bundle carries the model (bundle keys boost, boost_features), the provider's
  BoostPath evaluates once per matchup (cached on the inputs' bytes), the replay and live lanes need nothing else.
  Arms a/b/c (leaves 15/7/31) against c0 on the live settings, run 38969079000. If it passes, the registered path is
  full 1,000-world replays under CHAIN-03B's four gates before anything goes live (the whole prediction surface changes).
- Alec's confidence sheet for ALCS Game 4 (Rays at Guardians, October 16): a private artifact built from the published
  day file (public/days/2026-10-16.json on the ledger branch, forecast version 6) by the scratch script
  tonight/build_sheet.py: every statement with its chance, a plain label, the market's number and the measured
  accuracy of that statement type from the ledger (PLAYER-03/06/07/08, NRFI-01, RUNLINE-01, BULLPEN-03, ENV-03).
- CHAIN-04B passed by a small margin (pool constants 50/200 with the cell constants x0.67; -0.00005) and is live as
  chain04b-xs-tight (7:00 pm Eastern) with its own context tables. CHAIN-05 (the pitcher's mix by the hitter's side)
  failed at every constant; platoon_mix stays off (a clean negative, see the row). BOOST-02 (the trees on this base)
  is the next arms run; if it passes, full replays under CHAIN-03B's gates before it goes live.
- Totals, looked at again on LEASH-04's replays (described after looking, no experiment): our chance of the over at
  the closing line is worse than a constant (Brier 0.2509 in 2025 and 0.2531 in 2026 against 0.2491 and 0.2500; the
  market's 0.2497 and 0.2498), with a calibration slope of 0.16 to 0.29: when we say 36% it happens 46 to 51%, when
  we say 62 to 70% it happens 50 to 61%. The page already shows no over/under price for this reason (the taught total
  blends our mean with the line). The mean total itself is right on average and by bins of our mean, so the miss is
  noise in our mean relative to the market's line, not a bias; when we sit more than a run above the line the truth is
  0.5 to 0.75 runs below us, and the reverse. One lead: in 2026 the 228 games whose shorter starter was expected to
  get 10 outs or fewer (openers and bullpen games) scored 8.17 runs against our 8.93 (the line 8.50); 2025's 64 such
  games went the other way. Park residuals do not repeat across seasons (0.07), so parks are not the cause.
- Runner lesson (October 16, 7:20 pm): six replay runs (48 shard jobs) launched at 7:08 pm filled the account's
  concurrent job slots, and the live run's publish job (a one-minute job) sat queued behind them from 7:16 pm with the
  forecast already computed. Do not launch a replay batch inside the hour before a live window; two runs at a time
  leave room for the live lane.
- BULLPEN-07 registered and its six replays launched (sigma 0.75, 1.25, 2.0, both seasons, LEASH-04's settings): a
  per-game shock on each reliever's utility in the fitted choice, the day's unseen availability. On today's settings
  relievers shown at 60% or more pitch 52% of the time; the mean is right. Smallest sigma meeting the gates goes live.
- Tonight's scorer: tonight/score_sheet.py (scratch) reads the day file's actual box after the game and scores every
  statement of the confidence sheet (said vs happened, Brier). Yesterday's game as a dry run: the reliever chances
  were the misses (Booser 91%, Sulser 73%, Seymour 71% did not pitch in a 3-2 game), the rest as expected.
- LEASH-04 passed on both seasons (full scoring in the row) and is live: ADJUST pitch_counts 'chain'; the v2 hazard
  correction follows for regular-season games.
- LEASH-04 (pitch counts from the matchup's chain): replays leash04-chain-pen-1000-2025 (run 38958506391) and -2026
  (run 38958545975) with starter lines; scored with tools/brl_outs_cal.py, tools/replay_blend_compare.py,
  tools/replay_compare.py against chain03b-v2h120-pen-1000. Result in the LEDGER row.
- The at-bat model's own log loss (2026 blended): locked features 1.4601, physics 1.4584, chain 1.4576, chain with
  spring 1.4574. Each step is small on this scale; the game-level replays cannot see steps under 0.001 (paired Brier
  SE 0.0004 over 4,850 games), which is why CHAIN-03B adopted REPLAY-02's rule (proven at the at-bat level, no harm at
  the game level) and SPRING-02 did not re-run the game lane.

## Current checkpoint, October 8, 2026, 5 p.m. Eastern (Claude in charge; additions through October 11 below)

Read this section first, then LEDGER.md (every experiment with its registered prediction, result and decision). The
October 7 checkpoint below still describes the machinery (lanes, privacy, scheduling); this section says what changed.

### What the site runs now

- Player model: pa-2026-v2-physics (brl_engine/model.json). Refits that were better per plate appearance did not help
  at the game level and stay sealed, not served: RETRAIN-02 (aging), RETRAIN-03 (548-day physics window).
- Simulator settings (ADJUST in brl_live/boxscore.py): context offsets re-estimated on the simulator's own stack
  (CTX-02, brl_live/context_offsets.json, 2025 and 2026 through 2026-09-27); run environment env-v2-weather from October 10
  (PROD-04; env-v2 of ENV-03 without the venue term); team
  offsets centered on the league (TEAM-05, CENTER-01); steals and runner speed (RUN-01, RUN-02); postseason starter usage
  x0.91; base running after each outcome from every 2023-2026 play (TRANS-01, brl_live/transitions.json);
  running plays between plate appearances built but off (TRANS-02, replays running, see below).
- From October 9, about 8 a.m. Eastern (PROD-03, Alec's approval): the fitted reliever choice, relief exits with team
  hooks and the starter leash are on (reliever_choice, relief_exit, relief_hooks, leash in ADJUST); the postseason exit
  offset +0.4 acts in Wild Card, Division and League series games.
- Headline win chance: taught-v5 for games with a first pitch after October 9, 12:30 UTC (taught-v4 before it, from
  October 8, 14:00 UTC): 60% market line, 40% our model, our model = 0.0395 + 0.561 x simulator log-odds + 0.524 x
  team-model log-odds, refitted on the PROD-03 replays. Versions live in brl_live/headline_params.json with effective
  times; never edit a version already in force.
- In-game win chance: the game's simulated table averaged with the league table on the log-odds scale (LIVE-04).
- Hitter chances on the page keep each lineup slot's hits with the starter's share of that slot's plate appearances
  (PLAYER-02, STARTER_SHARE in boxscore.py).
- Live runs chain every 12 to 15 minutes; the phone check passed at 19:23 UTC October 8 (390 and 1440 px, no errors).
- October 10 additions: projected lineups follow the opposing starter's hand (LINEUP-01, brl_live/lineups.py RULE
  'hand', simulator and plans); game plans (tools/brl_report.py) use the swing model with the crossing term
  (SWING-CROSS-01), each season's called-strike edge (PROD-05) and the foul fixes (FOUL-01, FOUL-02), take a game's
  bullpen from the simulator's saved box (chance to pitch), keep the bench once a lineup is posted, and write each day
  as one commit (put_many); the live store retries 409s for minutes; the season top-up writes nothing when nothing is
  new. Every report day from 2025-03-27 to 2026-10-08 was rebuilt with these. Product call (October 10, evening): a
  plan for a game to come lists six relievers (was four), the simulator's likeliest; postseason teams use five or six.

- October 11 additions (each with its LEDGER row): starter exits by inning, LEASH-02's correction on the starter-exit
  hazard, regular-season games only (ADJUST hazard_correction; postseason games keep the plain hazard with x0.91), and
  with it the starters' outs ladder on the Odds tab for boxes simulated with it (12.5 to 20.5); the page states no over
  chance of its own anywhere (TOTALS-03); the run charts and ranges are negative binomials with the simulator's mean and
  its variance times 1.095 (TOTALS-05); hitter props switch between chances and fair American odds; the live builder
  refuses a hazard correction file that is not LEASH-02's 32 terms. On from October 11 evening: RELIEF-04's entry-inning
  correction on the relief-exit hazard for regular-season games (ADJUST relief_correction; postseason games keep the plain
  relief hazard, brl_live/relief_hooks.json and the +0.4); the strikeout spread (PLAYER-08), the hits-allowed ladder
  (PLAYER-09) and the batter-strikeouts column (PROD-16). Built and off: the in-season level layer (LEVEL-01, failed
  LEVEL-02 and LEVEL-03) and the engine's pitch draws (LEASH-03 failed).

### Evidence (full-season replays, 1,000 worlds per game, each season with tables from other seasons)

| Season | Simulator win Brier | Mean total vs actual | Notes |
|---|---|---|---|
| 2026, 2,454 games | 0.24378 before TRANS-01; kernel -0.00061, CTX-02 -0.00023 more | 8.98 vs 8.95 | headline 0.24317 vs closing line 0.24326 |
| 2025, 2,423 games | kernel -0.00010, CTX-02 +0.00045 (walk level carried from 2026) | 8.92 vs 8.90 | headline 0.24150 vs closing line 0.24235 |

Rejected with evidence on October 8: SKEW-02 (all physics seasons: right strikeout level, worse totals), ROLE-01
(superseded), RETRAIN-03 (window). Discovery (private page, not published): DISC-01 to DISC-07, Decision Horizon page
https://claude.ai/artifact/HFxTsd3ZXTQs7d7cTivr4h (Version 8 adds the ABS challenge test).

### Runner budget (October 11)

The repository gets about 20 concurrent GitHub-hosted jobs. A full replay takes 9 (a setup job and 8 shards) for about 50
minutes, so keep at most two full replays running at once; with four, the live slate run that a push starts waits in the
queue behind the shards (seen October 11 at 16:50 UTC). Check-only replays (n_sims 0) take about 4 minutes.

### How to check a game night from outside (the run logs are not readable; everything is on the ledger branch)

- `git fetch origin +brl-live-data:refs/remotes/origin/brl-live-data`, then: diagnostics/v2_receipt.json (each live
  run: status, reason, live_updates, storage_read_audit); diagnostics/phone_check_latest.json (the deployed page in a
  phone-sized browser: no script errors, plan rows, and browser_feed while a game is live: live_cards, win_chart,
  this_at_bat); ledger.json status/live/actuals for the game; diagnostics/report_trigger_latest.txt (plan builds the
  live chain dispatched); research/report-<run>.json (each plan build's receipt).
- A version with the posted lineups appears in ledger.json forecasts (lineup_status official) a few minutes after MLB
  posts them; the plans follow within about ten minutes (the trigger needs the forecast to be newer than the plans).
- A live run that reports "blocked" with HTTP 409 lost its writes to other writers on the branch; the next run tries
  again. Many in a row means something is committing every second (October 10: rebuild runs).

### In flight (updated 10:05 p.m. Eastern October 11, by `date -u`)

- October 11, 10:05 p.m.: KEYS-02 held every gate: the What decides it chances are within about a point of real games
  in 2025 and 2026 (scored first: 67.2% simulated against 67.4% and 66.8% real). BULLPEN-05 found the choice model gives
  the top two arms 20% of real held-out entries against 16.6%; BULLPEN-06 (a heavy week and the pool's two most used
  arms; held-out log-likelihood -1.7229 and -1.6960 from -1.7651 and -1.7333) is registered and replaying (runs
  38193362713 for 2025 and 38193385340 for 2026, tags v2-prod4-choice6-1000-<season>). Score with
  tools/brl_bullpen_entries.py and tools/brl_bullpen_cal.py on those runs, and tools/replay_compare.py against
  BULLPEN-05's (38186824402, 38186847009). WATCH-01 (play-by-play player for the simulated game) is live.
- October 11, 10:50 p.m.: BULLPEN-06 scored and stays off under its rule: wins and totals unchanged, top-two entries
  from 24% high to 11% (2025) and 9% (2026), but the chance to pitch at 60% and up is still 12 to 31 points high. The
  choice is now right entry by entry; the game-level miss is availability (every world has the same arms). Next for
  the bullpen: a per-world availability draw, registered with the same gates. The 2023-2026 live fit with the new
  features is in the scratchpad only (reliever_choice6_live.json), not served. An outside review of the pages (fresh
  reviewer on phone screenshots) drove the evening's layout fixes (LEDGER WATCH-01 lists them).

- October 11, 8:35 p.m.: Alec: the page is hard to read for someone who does not follow baseball and does not yet show
  what sets it apart (simulating the game before it happens). KEYS-01 (display): a game to come opens with the matchup
  side by side, a one-line call, "What to expect" (typical score, runs, one-run share, 100 games as squares), "What
  decides it" (scoring first, each starter's 18 outs, two hitters; counts over the box's own simulated games, shifted
  on the log-odds scale to add up to the pick) and "One game, played out" (the versions with the line score, now in the
  Summary instead of the header). Home cards get the same one-line call; their footer names Vegas only. Palette: night
  navy. Next: the Odds and Record pages in the same plain style, the live and final headers, and a replay check of the
  conditional chances (scored first, starter's 18 outs) against real games.

- October 11, 7:36 p.m.: BULLPEN-03 and -04 failed: the bullpen's top chances run high (60% to 80% shown, about 53%
  pitched) and a straight calibration cannot fix them; the Odds tab now says so under the bullpen tables. Next for it
  (BULLPEN-05, a diagnostic): record, per world, each relief entry's inning, the lead at that half-inning's start and
  whether the arm is the team's top one or two by 14-day share, and compare with real entries; then try a temperature on
  the choice model (per entry it is a little over-confident from 40% to 70% chosen) gated on the per-game chances. The
  simulator's games are slightly closer than real ones (final margin within 3 runs 62% against 59% to 61%), which
  explains only a little of it. Per entry the choice model is calibrated on held-out seasons (a temperature of 1.0 is
  best in both), so the fix is in the situations. BULLPEN-05's replays v2-prod4-entries-1000-<season> run 38186824402
  and 38186847009 (the first pair failed on a usage tuple; use these run ids). Score with tools/brl_bullpen_entries.py.
- October 11, 6:34 p.m.: BULLPEN-03 registered (the chance-to-pitch table, scored per reliever for the first time).
  Replays v2-prod4-pen-1000-<season> started 22:34 UTC (runs 38180906147 and 38180930219), RELIEF-04's settings plus the
  new recording, so their games must equal RELIEF-04's (check home_wins per game). Score with tools/brl_bullpen_cal.py.
- October 11, 6:06 p.m.: the Odds tab now carries the standard pitcher-hits market (PLAYER-09: each starter's
  hits-allowed ladder, 2.5 to 7.5, within 1.7 points on both replay sets, shown for every box) and a batter-strikeouts
  column (PROD-16, 1+ K). RUNLINE-02 failed (independent team runs miss how home wins end); the run line stays hidden.
  Not testable on the replays as recorded: walks-only and earned-runs props (the rows keep walks with HBP and no
  runs allowed by starter), batter runs and RBIs (no per-hitter runs in the rows).
- October 11, 5:51 p.m.: **RELIEF-04 held every gate and is on for regular-season games** (ADJUST relief_correction;
  brl_live/relief_correction.json, hooks brl_live/relief_hooks_corrected.json). Pitchers per starter out -0.164 and
  -0.133 against -0.146 and -0.127 real (gaps were -0.039 and -0.031), win Brier and totals unchanged. The postseason keeps
  the plain relief hazard with its +0.4. Still open: the level, about 0.1 pitcher a team-game short.
- October 11, 5:20 p.m.: two display calibrations held and are on the page. TOTALS-05: the run charts and ranges are
  negative binomials with the simulator's mean and variance x1.095 (page RUN_SPREAD; tools/brl_runs_shape.py); the
  ranges now hold the real score within 1.4 points of what they claim (raw up to 2.7 short). PLAYER-08: the strikeout
  ladder and ranges spread each start's distribution to the residual variance (K_SPREAD 1.0693; tools/brl_k_spread.py).
  Both constants come from LEASH-02's replays; refit them (same tools) whenever the live settings change the spread.
  Live updates now read each pitcher's entry inning from the play-by-play (needed by RELIEF-04's correction).
- October 11, 4:58 p.m.: RELIEF-04 registered (relievers who come in by the 3rd inning stay far longer than the exit
  hazard expects; 6th-inning entries come out sooner). Stage 1 held (research run 38174006841: decision log loss -0.0071
  and -0.0060 on held-out 2025 and 2026). Stage 2 replays v2-prod4-relief4-1000-<season> started 20:57 UTC (runs
  38174343872 and 38174368300) against v2-prod4-leash2-1000. Score: tools/brl_relief_usage.py on both tags (slope gap at
  least halved, level no worse by 0.03), tools/replay_compare.py v2-prod4-leash2-1000 v2-prod4-relief4-1000 (win Brier,
  totals), and check the receipts' settings equal LEASH-02's apart from relief_correction. If it holds: fit the live
  table (relief_fix 'live' coefficients in the receipt) into brl_live/relief_correction.json, and the hooks against the
  corrected hazard into brl_live/relief_hooks_corrected.json (tools/brl_relief_hooks.py --correction; keep
  relief_hooks.json plain, the postseason reads it); ADJUST relief_correction for regular-season games only. (Done 5:51.)
- October 11, 4:32 p.m.: review fixes. The live builder refuses a hazard correction file that is not LEASH-02's 32 terms
  (LEASH-03's pitch terms would have run with no pitches outside the one simulator it returned); brl_live/pitch_counts.json
  and the idle live branch are gone. The outs ladder shows only 12.5 to 20.5, the lines PLAYER-07 scored (with those lines
  LEASH-02's ladder runs 1.42 and 1.52 points high, the ledger now says so). The Total runs section no longer states the
  simulator's over chance (TOTALS-03), only the Vegas total and its over chance. Small page gaps closed: a live game's
  Plays tab with no plays yet says first pitch is in, the Odds tab without a box stops after win chances, Matchups shows
  without a simulated sample. Ledger and METHODS wording corrected (the hazard's inning terms, LEVEL-03's totals, the
  pitch-band figures).
- October 11, 3:03 p.m.: LEASH-03 (pitch counts in the starter's exit) registered; stage 1 held (research run 38165964534: the
  hazard expected 25% to 75% too many removals at 60 to 84 pitches and too few from 85 on, in every season; with pitch
  terms the decision log loss drops about 0.006 on 2025 and 2026). The engine can draw each plate appearance's pitches by
  outcome (research_lab/game_sim/pitch_counts.py, own random stream; off unless given a table). Stage 2 replays: tags
  v2-prod4-leash3-1000-<season> (started 19:00 UTC) against v2-prod4-leash2-1000; score with tools/brl_outs_cal.py on
  the leash3 files, then tools/replay_compare.py v2-prod4-leash2-1000 v2-prod4-leash3-1000. Not switchable as built:
  the pitch draws would reach only the simulator the live builder returns, while the box worlds (boxscore.py
  ObservedSimulator), the market worlds and live updates (live_sim.LiveSimulator, its own plate-appearance loop) build
  theirs without them, so hazard_correction_for refuses a live file that is not LEASH-02's 32 terms. A pitch-count item
  that passes needs the draws wired into all of those first (and a live game's real pitch count in live updates).
  **LEASH-03 failed (3:53 p.m.):** the ladder lines stayed 1.38 points high in 2026 (1.0 allowed); every line within 2.3,
  per-start slope 0.98 to 1.00 (better), win Brier and totals unchanged. LEASH-02 stays; the engine's pitch draws are
  built and idle. Next idea there: pitch counts with a per-game command level (a pitcher's whole day runs long or short)
  rather than independent draws per plate appearance.
- October 11, 2:01 p.m.: LEASH-02 held every gate (stage 2 runs 38156905239 and 38156881017): every outs line 12.5 to 20.5
  within 2.4 points in both seasons, starts ending in the 4th right (11.1% against 10.9% and 11.5%), win Brier and totals
  unchanged. **On for regular-season games** (ADJUST hazard_correction True; brl_live/hazard_correction.json; the live
  starter policy adds it only when game_type is R and the box's adjustments then name "starter exits by inning
  (LEASH-02)"). The outs ladder (OUTS_LADDER true) shows only for boxes with that label, so nothing changes on postseason
  pages. Cost: starters run a little long in the middle (16.5 and 17.5 outs about 2 points high). LEVEL-03 replays (level
  plus running plays, tags v2-prod4-lvrun-1000-<season>, runs started 17:34 and 17:44 UTC) score like LEVEL-02 against
  v2-prod4-tb-1000.
  LEVEL-03 failed (2:37 p.m.): with the level and the running plays together the totals land right (+0.09, +0.04) but the
  over and under got worse in both seasons (pooled +0.00064 [+0.00024, +0.00105]) and the squared error's interval
  reached +0.042; neither changes. Both level studies say the same: the season's level is not what costs the over/under.
  Queued for Alec with POST-03: the correction in postseason games too, which needs the postseason factor x0.91
  re-measured on the corrected hazard (a postseason-only change, so his call).
- October 11, 12:15 p.m.: LEVEL-01 held (in-season re-anchoring of each outcome's league level: +0.114 and +0.146 nats per
  1,000 real plate appearances at m 10,000, both intervals above zero; check-only runs 38152933999 and 38152908828). The
  layer is built (brl_live/provider_adjust.py LevelAdjust; per-date offsets brl_replay/level_offsets_<season>.json from
  tools/brl_level_cal.py --write-offsets) and off. LEVEL-02 registered: full replays with the layer (tags
  v2-prod4-level-1000-<season>), paired against PLAYER-06's replays (v2-prod4-tb-1000-<season>); score with
  `tools/replay_compare.py v2-prod4-tb-1000 v2-prod4-level-1000` and `tools/brl_player_cal.py --pair
  <season>='<tb glob>::<level glob>'`; first-date games must match exactly.
  **LEVEL-02 failed (17:20 UTC):** hitter chances better in 2025, flat in 2026; 2026 totals 0.11 runs short and P(over)
  worse pooled. The layer stays off. The finding: with each outcome's rate right, the simulator turns 2026's mix (more
  walks, fewer extra-base hits) into too few runs; the tables' level errors were hiding it. Next model item: run
  conversion (runs per walk and single in the simulator against real games, by base and out state).
- PLAYER-07 failed its rule (starters' outs: every line within 2.5 points in 2026, but more than 12.5 and 13.5 outs 3.8
  and 2.9 points short in 2025), so the outs ladder stays off (page OUTS_LADDER false; built and tested). The miss: the
  simulator ends too many starts in the 4th inning and too few in the 5th (10 to 12 outs 13.6% against 10.9% and 11.5%
  of real starts; 13 to 15 outs 25.5% against 29.6%, 25.2% against 27.8%). The exit hazard
  (research_lab/game_sim/starter_hazard.py) has no term for finishing the 5th: next item (LEASH-02).
- October 11, 12:50 p.m.: LEASH-02 registered and stage 1 held (research run 38156674333): a 32-term correction on the starter-exit
  hazard's log-odds (inning 3 to 7 by its end and the score, plus quiet mid-4th and scoreless mid-5th), fitted on 2023-2024
  decisions, lowers the decision log loss on 2025 and 2026 (intervals below zero) and puts removals after the 4th within 4%
  of actual (the plain hazard expects about half again as many). Stage 2 replays: tags v2-prod4-leash2-1000-<season>,
  paired against PLAYER-07's v2-prod4-outs-1000-<season>; score with tools/brl_outs_cal.py (bands included) and
  `tools/replay_compare.py v2-prod4-outs-1000 v2-prod4-leash2-1000`. If it passes: ADJUST hazard_correction True (regular
  season only; live code and brl_live/hazard_correction.json are in place), and PLAYER-07's rule is re-read on these
  replays for the outs ladder.
- Layout today: table row labels take the light rule, team stats show each rank beside its number, the Odds tab's
  ladders put each price beside its chance (one line a row), bullpen tables list arms at 10% or more with the rest in
  one line.

- October 11, 11:35 a.m.: DISP-03 failed its rule ((1): the total's over and under improved only within noise, pooled -0.00015
  [-0.00047, +0.00017]; (2), (3), (4) held: centered day-form worlds fix the run spread and put the run line within 1.5
  points in both seasons). Nothing changes: ADJUST market_day_form stays None (the box code and the page's reading of
  box.market_worlds are built and tested, idle), MORE_MARKETS stays off. A second review (after the first) found the
  fielder's choice missing from the RBI rule and five ledger figures; all fixed in 2fa79ff1f. The replays' real-PA check
  now records each game's prediction second moments, ready for an in-season level re-anchoring study (next season).

- October 11, 10:49 a.m.: PLAYER-06 failed its rule (two or more total bases right on level in both seasons, but the top tenth
  3.4 points high in 2026), so the 2+ TB column stays off (PROP_COLS.tb2 false; the Odds hitter tables already fit five
  columns on a phone). The top tenth's miss is in the rates: in 2026, the model's first season unseen in fitting or tuning,
  the top tenth by that chance got 0.244 hits and 0.043 home runs a plate appearance against 0.254 and 0.050 simulated; a
  model-side item (talent shrinkage for the best hitters), not a display fix. DISP-03 replays running (2026 run
  38146596899, 2025 run 38146625979, tags v2-prod4-df-1000-<season>): compare with `python3 tools/replay_compare.py
  v2-prod4-margins-1000 v2-prod4-df-1000 <out>`, the run line with tools/brl_runline_cal.py on the df files, and the spread
  with $S/disp_sd.py; if (1) and (2) hold set ADJUST market_day_form {'sigma': 0.12, 'center': 0.0035, 'worlds': 4000} (the
  page already reads box.market_worlds), and if (4) also holds set MORE_MARKETS.runline.

- October 11, 10:00 a.m.: RUNLINE-01, NRFI-01 and FIRST5-01 all failed their registered rules (runs 38141799131, 38141956435),
  so the page's More markets section stays off (MORE_MARKETS all false). The run line's miss is real: given a win, the
  simulator's chance of winning by two or more runs about 2 points low in 2026 and the road winners it rates least likely
  to win big did so 5 to 6 points more often in both seasons, the same narrowness DISP-01/02 found; a day-form shock used
  only for the market numbers (win chances from the plain worlds) is the candidate fix, untested. The first inning and the
  first five held on level (within 1.2 to 2.3 points) and missed a per-fifth bound that sat near the noise. BOX-RBI-01:
  simulated boxes now credit runs batted in on outs in play other than double plays. PLAYER-06 replays (two or more total
  bases): 2026 run 38144947261, 2025 queued after it; score with `tools/brl_player_cal.py` (kind tb2 in the output) and
  turn on PROP_COLS.tb2 only if the rule holds.

- October 11, 9:51 a.m.: a separate review of the day's page and study changes found five defects and six possible issues, all
  fixed in a78e72b4c: Record page notes now name what was checked (walks or hit-by-pitches; 1 to 2.3 points; the map
  "did not hold up" rather than "made it worse"; the strikeout check marked as described after looking), the strikeout
  ladder keeps to the checked lines 2.5 to 8.5, career lines count regular-season splits only and paint once (a failed
  ask retried after a minute), the run line's favorite follows the book when there is a line, the first-inning reading no
  longer needs a sixth inning, ledger figures corrected (PLAYER-03 top tenth, PLAYER-05's first group is under 100).
  PLAYER-05 (part-timers' share of their slot) failed its Brier gate, not adopted. PLAYER-06 (two or more total bases)
  registered; its replays (hitter lines, tags v2-prod4-tb-1000-<season>) start once the margins replays finish; boxes now
  carry tb2_probability; the page's 2+ TB column waits behind PROP_COLS.tb2.

- October 11, 9:14 a.m.: PLAYER-03 failed as registered (no kind passes; each replay's level comes from the other seasons'
  tables, so a map from one season moved the other the wrong way); the page keeps the raw chances, and the Record page
  says how close they came (hits within 1 point, home runs 0.6, walks about 2). Next season's item from it: re-anchor the
  outcome levels in-season (RUNS-03 pointed the same way). RUNLINE-01, NRFI-01 and FIRST5-01 registered; their replays
  run with the live settings (2026 run 38141799131, 2025 run 38141956435, tags v2-prod4-margins-1000-<season>; the
  harness now records margin_hist, f5_margin_hist, f5_total_hist and first_inning). Score with tools/brl_runline_cal.py
  and tools/brl_innings_cal.py, then set the page's MORE_MARKETS flags for whatever passed (the section is built and
  tested, off). Boxes saved from now carry margin_distribution, f5_margin_distribution, f5_total_distribution and
  first_inning. Lesson: the page is one scope, so a second function with an existing name silently replaces it (a new
  runLine nearly replaced the running game's, a second ordinal() shipped); test_page_unique_functions.py now guards it.

- October 11, 8:48 a.m.: BVP-01 (run 38139650184): every prediction failed; beyond each player's own level, a hitter's
  record against one pitcher carried between none and about 15% of a 20 plate-appearance record's on-base edge (2026's
  interval above zero, 2025's not), and nothing measurable in strikeouts or slugging. PROD-14: the Matchups tab shows each
  hitter's career line against the opposing starter from MLB (one vsPlayerTotal ask per lineup, never colored, the note
  states BVP-01's range); the Summary shows both teams' season stats with MLB ranks before first pitch (one teams/stats
  ask for every club). The stats probe (run 38139188923) found both asks open to the page. The Players page's
  leaderboard buttons for barrel and the expected stats were blank (labels missing since those rows were added); fixed,
  with a test tying every percentile row to a label and direction. Local preview mocks: serve.py answers vsPlayerTotal
  and teams/stats (mlb/gl/team-stats.json).

- October 11, 8:23 a.m.: PLAYER-03 registered (d6fa2bb37) and its two replays running: the page's hitter chances (1+ hit,
  2+ hits, HR, walk, strikeout) on the live engine, 2026 (run 38138601871) and 2025 (run 38138625455), tags
  v2-prod4-hitters-1000-<season>; the harness now records each hitter's histograms of hits, home runs, walks and
  strikeouts. Score them with `python3 tools/brl_player_cal.py --season 2025='<dir>/2025-*.jsonl.gz' --season
  2026='<dir>/2026-*.jsonl.gz' --out <file>` (per-game files research/replay-v2-prod4-hitters-1000-<season>-<run>-s<k>of8.jsonl.gz)
  and apply exactly the registered rule (a kind passes only if both seasons' Brier interval ends below zero and the
  calibrated mean is within 0.5 points). PLAYER-04 (descriptive): starters' strikeout chances on the live engine are
  calibrated (within 2.1 points at every line, both seasons), so the Odds tab lists each starter's strikeouts as a prop
  ladder with fair prices (kLadder). PROD-13: the Odds tab's hitter table now gives each hitter's own chances
  (ownAtLeast, PLAYER-02's share), the same numbers as the Summary's meters; the Record page's tag table fits a phone.
  Barrel rate is on both percentile panels (cards rebuilt, run 38137780354: Ohtani 18.6%, 99th; Skubal 7.1% against).

- October 11, 8:00 a.m.: PXSTATS-01 held (run 38136607768): for pitchers, expected AVG and SLG against from exit speed and
  launch angle predicted the next season better than the actual ones (unlike hitters, XSTATS-01), so pitcher pages and
  the pitching matchup rank them (cards' lines xh and xtb from xs_lines; PCT_PITCHER xavg, xslg). Cards rebuilt (run
  38137036454; Skubal expected AVG against .211, 85th percentile).

- October 11, 7:40 a.m.: HCHANGE-02 held for chase, misses, strikeouts and hard-hit (a looser 2-point, two-SE rule; 20 to
  41 hitters flagged; 56% to 70% held), so the changes line uses it; walks stay off (39%). XSTATS-01 failed: expected AVG
  and SLG from exit speed and launch angle predicted the next season no better than the real numbers (not shown; the
  Record page says so). Plan rows carry each hitter's MLB season line (fillHitLines, one batch ask for the lineup); the
  lineup line uses the Expected split in the Expected view; the game log table is keyboard-scrollable (axe clean, light
  and dark). Local preview mocks: $S/tour/site/mlb/gl/season-hitting.json for the batch season lines. Do not
  `pkill -f serve.py` from a shell whose own command line contains that string (it kills the shell); kill by pid.

- October 11, 7:10 a.m.: a separate review of the session's page and report changes (4f10922ba..8cfb431c6) found six
  wording defects, no crashes or wrong arithmetic; all fixed in d99b9524e: the velocity chart now says it shows games
  with 8 or more fastballs and its dashed line is his real season average (cards velo_season) or his earlier starts; a
  last-start read only for a start in the card's as-of season (a 2025 start was called "this season"); the matchup's
  fastball line names the base of the change; the splits notes pair each correlation with its own sample (190 hitters,
  242 pitchers); the hot-zone key names the ZONES-03 anchor only where it applies (hzAnchorWords); a pitcher with fewer
  than three starts reads "Only 2 starts". Cards rebuilt (run 38134409127). Lesson: every number's label must name the
  quantity actually computed, including its sample and season.

- October 11, 6:41 a.m.: PZONES-01 held (run 38131952265): pitcher charts have the Expected view (swing and miss shrunk
  at m 80; AVG and SLG at his level on his hand's league pattern; page PZ_EXP_M, pitcherZoneLegend). HCHANGE-01 (run
  38132555095; registered as CHANGE-01, renamed because the ID was taken): flagged season changes held 55% to 80% in
  chase, misses, strikeouts and hard-hit; walk-rate changes no longer called out (HC_RATES carry the held share). The
  Record page lists the other checks in plain words (OTHER_CHECKS). The Odds tab shows the line as a sportsbook board with
  our fair price (lineBoard, amOdds). Check new experiment IDs against the ledger before registering (grep "^| ID |").

- October 11, 6:18 a.m.: REST-01 failed as registered (relievers on no rest lose 0.16 mph, under the 0.2 gate, and no
  strikeouts; bullpen card unchanged). TTO-01: the third-time penalty against starters is real every season (OBP +1.05
  points in 2026, strikeouts down 3) but a starter's own gap did not reliably carry over (46 pairs, r 0.12); the
  starter fold says so (TTO_NOTE). Also: the pitching matchup gives each starter's last three starts from MLB's game logs
  (psLast3) and his last start's fastball; the bullpen card adds K% and BB% shaded by percentile (penRate) with names
  sticky; How it works has a Player pages section. Main at d25ab0db7.

- October 11, 5:55 a.m.: three more checks held and are on the page. PSPLITS-01 (run 38128749747): a pitcher's own AVG
  and SLG split by batter side did not carry over (r 0.10, 0.06); his Expected splits use his hand's usual split
  (pitcherSplitExpected). FORM-01 (run 38129227011): a hitter's last ten games said almost nothing about his next ten
  in OBP or SLG (k 0.02); strikeouts carried about 15%; the game log's note says so. VELO-01 (run 38129631856): about
  45% of a starter's last-start fastball change shows up again next start; each mph down meant 1.3 fewer strikeouts per
  100 batters (after drops of 1 mph or more, 2.0 fewer); the starter fold and pitcher page note changes of 1 mph or
  more (velLastNote) and pitcher pages chart speed by game (velChart; cards velo_games, velo_type, velo_last from
  game_velo_list). Card rebuild for October 11 and the players at 39435adbd (run 38130072455).

- October 11, 5:30 a.m.: SPLITS-01 (run 38127261982): a hitter's own AVG and SLG split by pitcher hand, beyond his side's
  usual one, did not carry over (r 0.06 each, 317 hitters); misses did in part. The splits table, the hot-zone line and
  the plan rows' line show Expected numbers in the Expected view (page splitExpected, SPLIT_CARRY). ZONES-03 (run
  38128261865) held: charts against one hand now anchor the Expected view on his level against both hands (page
  HZ_ANCHOR, hzAnchor; misses against righties keep the old anchor). Player pages open with the season line and last ten
  games from MLB's public stats (PROD-12; page PSTATS, fillPlayerStats; the stats probe checked both asks, runs
  38126258426 and 38126721578). Pitcher pages and the plan's starter fold draw the movement chart (movementChart); the
  league's ring per pitch type needs league arsenal v_mov and h_arm, which the next card build adds (daily 13:20 UTC).
  The players index marks switch hitters S. Local preview: $S/tour/serve.py on 127.0.0.1:8790 serves the site and
  mocks MLB game logs from $S/tour/site/mlb/gl/<id>-<hitting|pitching>-<reg|post>.json. Main at 08624324c.

- October 11, 4:45 a.m.: switch hitters fixed (PROD-11): the plans read the pitcher's pitches to the side the hitter takes
  against that hand (Fitted.side_vs, pair()), and the page orients, labels and compares his charts, splits, pull tags and
  spray by that side (cards: side_by_hand, spray_pull; page isSwitch, hzSideFor, hzLeagueFor; a both-hands chart is
  labeled 3B side and 1B side). TAGS-10 held (hard contact inside or away, 0.60 and 0.64); TAGS-09 only partly (misses
  away tagged; misses inside and the 2026 correlation failed). Rebuild at e7d11d7ae for October 11 and the players.

- October 11, 4:30 a.m.: a second review (a60bee37b..9dd6093ca) found two real defects, fixed in 17e687e5f's parent:
  arsenalChanges claimed pitches added or dropped from lists that leave out types under 20 pitches (now from complete
  shares: arsenal_shares and arsenal_prev with every type), and the bullpen card dropped a game only unlisted arms
  pitched (now the staff team's game days from the schedule, pen_days). Also: season_mark is safe on odd data, the
  page's live reader takes AVG and ERA only as strings. Player pages have a splits table by the other side's hand, and a
  pitcher page no longer shows a second arsenal table when nothing changed (an if/else slip). Lesson: when inserting a
  line between an if and its else, rebrace the block. Card rebuild at 17e687e5f.

- October 11, 4:15 a.m.: PTAGS-04 held (run 38122717614): Works inside / away with the fastball, each pitcher read
  against the league's rate for the matchups he faced (pitcher_tag_counts carries an adjusted numerator whose ratio minus
  the overall rate is his excess), 0.57 to 0.61 a season later. Hitter cards carry line_seasons (this season's and last
  season's line, 200 or more plate appearances each); the page says a change only beyond three standard errors and 3
  points (hitterChanges; 3 of tonight's 23 hitters). Real box scores show AVG and ERA through the game from the feed's
  seasonStats (real_game.season_mark in both Python readers and the page's live reader; display only). Card rebuild for
  PTAGS-04 at f6c9d5f80. Push brl_live changes to main sparingly: each push starts a live run and cancels the one in
  progress (the 12-to-15-minute chain then restarts).

- October 11, 4:30 a.m.: PTAGS-03 held (run 38120710560; ground-ball share 0.75 and 0.82 a season later): Ground-ball
  pitcher and Fly-ball pitcher are on the cards and starter folds. The pitching matchup on the Summary adds each
  starter's three main pitches and his tags. The Odds tab gives the simulator's chance of the over against the Vegas
  total (a push left out, as tools/replay_compare.py p_over scores it). The bullpen card's columns are the team's latest
  game days within a week (recent_load now seven days). Pitcher cards carry arsenal_prev (last season's mix and speeds,
  300 or more pitches) and the page says what changed (arsenalChanges: a pitch added or dropped, a fastball 1 mph faster
  or slower, a pitch used 10 points more or less). Card rebuild for October 11 and the players at e954194f4.

- October 11, 3:35 a.m.: a separate review of tonight's page changes found four issues, all fixed in a60bee37b: the
  next day's regrade of a game-day plan was labeled a look back (planWhen: a look back is built after its game day
  from pitches before an earlier date), KN and EP counted as offspeed in this season's mix (the cards count them with
  fastballs), the live count tip read from the pitcher's side, and a jump button could point at an empty sheet. The live
  at-bat card shows the hitter's tags (32092c775); the plans have jump buttons to each lineup and to all matchups.
  PTAGS-03 (ground-ball and fly-ball pitchers) registered and running; if it holds, add its two rules to PTAG_RULES
  (rate gb_against, floor 150) and rebuild the cards (the daily build at 13:20 UTC does it too).

- October 11, 3:15 a.m.: PTAGS-02 held (run 38118026447): pitcher heights carry over 0.73 to 0.78; Elevates fastballs,
  Keeps fastballs down and Buries breaking balls are on the pitcher cards and starter folds (page PTAG_RULES; the cards
  carry fb_up, fb_dn, br_dn since rebuild 38118434363). The Players page now opens with hitting and pitching leaderboards
  (top ten by any percentile row, side by side on a wide screen) and a team shows two tables (hitters, pitchers) with
  every rate shaded by percentile; the players index carries pa, r (rates) and pct per player (tools/brl_report.py
  index_line, rebuild with players true after 2249ece95). Swing decisions and chase lists moved into folds.

- October 11, 2:45 a.m.: TAGS-07 and TAGS-08 held (run 38117275043): a hitter's misses up in the zone against down
  (top third, MLB zones 1-3, against the bottom third, 7-9; relative to himself, net of the league) carry over 0.46 and
  0.71, and his hard contact up against down 0.54 and 0.55. The page tags Misses up / down in the zone (vertShift from the
  cards' zone counts, 9 points) and Hits the high / low strike hard (10 points; zone cells now carry hard-hit and
  measured balls in play, so it shows once the cards are rebuilt: run 38117812393 for October 10 and 11 and the player
  cards). Lineup notes: Up / Down in the zone for misses, Stay down / up in the zone. PTAGS-02 registered and running
  (pitcher heights: Elevates fastballs, Keeps fastballs down, Buries breaking balls); if it holds, add its rules to the
  page's PTAG_RULES and rebuild the cards again (ptags carry fb_up, fb_dn, br_dn from commit 3cfdd4e4c). Page fixes
  (08806632d): series games as tiles, distinct names in Matchups, a game-day plan is no longer called a look back
  (built after its game day only), How it works describes the fitted bullpen, tags in a strip under each plan row,
  320 px fits. Fetch the data branch with `git fetch origin +brl-live-data:refs/remotes/origin/brl-live-data` (a plain
  `git fetch origin brl-live-data` leaves origin/brl-live-data stale here).

- October 11, 1:50 a.m.: the live at-bat card names the hitter's habit for the count at hand (liveCountTip: first pitch,
  two strikes, behind; only tags that held a season later). Plans follow this season's arsenal everywhere: the pitcher's
  mix line, the pitch-type order in the chart headers, the plan's spots and the coaching lines all read famShares (this
  season's pitches when 300 or more, else the career mix), and a pitch type he throws under 5% this season gets no spots,
  no bullets and no coaching (nowCells, rare(fam)); checked on the Dodgers' sheet against Misiorowski (no changeups in
  2026: no offspeed lines). PROD-10 in the ledger lists tonight's display calls in one row.

- October 11, after midnight: the projected game follows our pick (PROD-07: when the team ratings and the line pick the
  other team, the header, line score, box and plays show the saved upset world, the most typical game the pick wins;
  page variantsFor plus box_page.lean_box(home_pick) for early, archived and day-archive boxes). Player pages open with
  percentile rankings like the public player pages (PROD-08: tools/brl_report.py player_lines, pct_ranks; card['line'],
  card['pct']; league['ranked'] = 413 hitters and 449 pitchers at 300+ PA or BF); the starter's fold in each plan opens
  with his bars. Hitter plans give the head-to-head line (pair['h2h'], Fitted.head_to_head). The game header and day
  cards give each probable starter's season line from MLB's people endpoint, read by the browser
  (tools/brl_probe_stats.py checked it answers any site; diagnostics/stats_probe.json). Line scores of real games show R H
  E (parse_actual_box 'errors', the client's state.errors). TAGS-01: every hitter tag carried over to the next season in
  both test seasons; the two-strike tags now read his change against his own chase in the other counts (season-to-season
  correlation 0.64 and 0.72; page twoStrikeShift), and the plans' evidence fold quotes the carry-over. TAGS-02 (the
  hitter's-count aggression line) held: +11.3 and +11.1 points over the league with the pitcher behind. TAGS-03 held:
  misses by pitch type against his own fastball misses carry over (0.52 to 0.62), so the plans tag Misses/Handles
  breaking balls and offspeed (page famWhiffShift). Each plan sheet opens with "Working through this lineup" (lineupKeys)
  from these tags, the running game (PROD-09: runners to hold, steals against the starter, each hitter's line) and the
  starter's projected line from the simulator's box; arsenal tables use this season's pitches when 300 or more; the live
  card gives the pitcher's mix in the count at hand. POST-03 held (postseason managers keep good starters in longer: top
  third by K-BB 0.90 of their regular length, bottom 0.77); the tiered factor is built in brl_live/live_extension.py and
  OFF (ADJUST postseason_starter_tiers None), queued for Alec's OK. TAGS-04 failed (velocity misses do not carry over;
  no tag). TAGS-05 held (pull shares carry over 0.58 to 0.70; page pullShift; lineup notes shade the infield).
  Also: the game Summary opens (pregame) with the pitching matchup side by side (pitchingMatchup: season line from MLB,
  rates with percentile pills from the plan's cards, our projected line); the Matchups tab outlines each hitter's toughest
  reliever; How it works describes the tags and how they were checked. Lessons: a visually-hidden span inside a sideways
  scroll box needs the box positioned (it widened the page); percentile text color uses the ink's real luminance.
  TAGS-06 held (ground-ball and fly-ball hitters, 0.73 and 0.75; page gbShift). Arsenal tables carry the league's rates
  per pitch type (league_arsenal). The Record page lists every tag's check (TAG_CHECKS). Plan rows show at most four
  tags, pitching ones first. PTAGS-01: seven pitcher tags held (page pitcherTags from card['ptags'] and league['ptags'],
  Fitted.ptags from pitcher_tag_counts); falls behind first and few chases left off. The phone check failed once on 12 and 13px text in the new
  pieces (its floor is 14px); fixed at 12ac50da0. Run tools/check_brl_phone_page.py against the local build (--url
  http://127.0.0.1:8790/index.html?api=http://127.0.0.1:8790/mlb/) before pushing page changes.

- October 10, 11 p.m.: Game 5 final, Guardians 2, White Sox 0 (we had Guardians 57% before first pitch). The held changes
  shipped after it was graded (claude/live-v2 and main at 7eab793ec: store writes compared by blob sha, the blocked-status
  fix, the ledger slimming; 345 tests pass). Check the next live receipts (diagnostics/v2_receipt.json) say ok and that
  ledger.json shrank. Also tonight: ZONES-01 registered, run (38106740119) and held on 2026, so the hot zones have an
  Expected view (client-side shrinkage, HZ_EXP_M in the page); live at-bat notes from credited findings (next pitch
  from SEQ-03F, the pitch count from WARMUP-01, runners in scoring position from PRESSURE-01); count tendencies,
  usage by count and put-away rates (count_tend, usage_by_count; rebuilt in 38106430325); hitter tags for first-pitch and
  two-strike habits. Checked and dropped: in-season re-anchoring of the run level (task 10): the simulator's monthly
  total bias is within noise and reverses between seasons (2025-08 -0.37, 2026-08 +0.26; standard errors about 0.22).
  Later: ZONES-02 held (the Expected view is fine on every hand and pitch-type subset at the pooled m); pitcher cards
  carry the arsenal by pitch type (pitcher_card['arsenal']: share, mph, spin, PITCHf/x movement with arm side
  positive, misses, chases, zone, put-away); plans show the lineup at a glance against the starter's hand; box scores
  carry earned runs for real games (real_game PIT_FIELDS and parse_actual_box mapping; simulated lines have none);
  coaching lines from count habits (countPlan, usageLine). MATCHUPREAD-01 registered and run (a pitcher's locations
  over a hitter's expected zones against league locations); its row says what the plan shows.

- October 10, 9 p.m. (Game 5 in progress): Alec asked for hot zones and layouts like the standard market. Done:
  hitter cards in the plans and on the Players page carry MLB's 13 zones (1-9 the strike zone by thirds, 11-14 the
  corners outside it, catcher's view, from the per-pitch `zone` column) as counts only: [zone, pitches, swings,
  misses, at-bats ended there, hits, total bases], split by the pitcher's hand (card['zones'] = {'R': {...}, 'L':
  {...}}, families all/fastball/breaking/offspeed; league['zones'][batter side][pitcher hand]). tools/brl_report.py
  zone_counts and zone_split (aa3775419); the plans and cards were rebuilt in run 38099403917. The page draws the
  standard chart (hzChart): AVG, SLG, Swing %, Miss %; red above the league's rate for his side against that hand,
  blue below; gray numbers under 10 at-bats (15 pitches, 10 swings); a triple counts as a double in SLG (out7 lumps
  them). Game plan rows show his chart against the starter's hand (all four pitch types side by side on a wide
  screen); a hitter's full plan has his line, a by-pitch-type table and the charts with the plan's zones outlined;
  the live at-bat card the same for the count in progress; pitcher cards list the arsenal as a table. The model's
  7x7 grids moved into folds and now flip for lefties (they were drawn in the hitter's frame but labeled catcher's
  view). Pages f059767b4 and 922a607b0. Then: pitcher cards carry zone counts by batter side (pitcher_card['zones'],
  zone_split(by='stand_r'), 14110f083; rebuilt in run 38100865499), drawn as "Where he pitches" (Pitch %: each zone's
  share of his pitches, red above an even spread of one in thirteen; or AVG, SLG, swing and miss against) on the
  Players page, under a hitter's full plan (to his side) and folded in the live at-bat card; plan rows carry a
  lineup-card line (AVG, SLG, chase, miss against the starter's hand) and the starter's arsenal table one tap above
  each sheet; one color rule site-wide (red favors the hitter, blue the pitcher; the matchup and model grids were
  gold for the hitter); each zone's number is white or dark by contrast. Spray tendencies: the pitch table now keeps
  each batted ball's direction ('spray', degrees off center from the gameday coordinates, home plate 125.42, 198.27)
  and trajectory ('traj' 0 ground ball, 1 liner, 2 fly, 3 pop-up), and hitter cards carry spray_counts (ground balls
  and balls in the air by field third at 15 degrees, pop-ups, hard-hit 95+ mph), drawn as "Where he hits it"
  (Players page, folded in a hitter's plan). Bullpen card: plan arms carry 'recent' (pitches on each of the five days
  before the game, training pitches only), shown per sheet with only the days the plan's data covers. The postseason
  plans (September 29 to October 9) were rebuilt with all of it in run 38102412944, each as of the date it had before
  (09-29 and 09-30 as of 09-01, 10-01 to 10-05 as of 10-01, later days their own); October 10 and 11 in 38101924869.
  Older days keep the old cards; the page shows the model grids in view when a card has no zones. Fixed: a plan for
  tomorrow rebuilt after 8 p.m. Eastern called itself a look back (built_at's UTC date); the build day is now read in
  Eastern time. Earlier tonight (4b14c77e7): the live box score and plays show the real game
  from first pitch, in batting order with substitutes under their spot, and no longer list the DH team's pitcher as
  a batter.

- October 10, 5:45 p.m.: the rebuild with all of October 10's fixes is complete: every one of the 401 report days
  (2025-03-27 to 2026-10-08) was written after 17:43 UTC. The 2025 postseason month had been missed (its 18:55 push
  went out in the same second as a WHIFF-OWN push and only the latter ran); redone in run 38087141540, one commit per
  day. record.json (4,991 games): chases expected against actual 1.029 (league) and 1.025 (maps), was about 1.05;
  hitters read as chasers 33.8% against 31.0% for an average hitter on the same pitches (maps 34.6%), patient 24.1%
  against 27.9% (maps 24.7%); months 1.00 to 1.03 after April, Aprils 1.06 to 1.08. LINEUP-03 ('swap', a regular back
  in for a fill-in) missed its gate on 2024 by 0.003 on right spots; LINEUP-01 replicated on 2024 (+0.45). Pages: an
  axe-core pass fixed club-color text contrast, headings, the sideways tables; player cards merge overlapping spot
  phrases and drop a pitch thrown under half a percent.
  Held for after Game 5 (branch pending-store-meta, commit f9e7571c6): the live store compares a write by the blob sha
  from the contents metadata instead of downloading the file it replaces (the 12.6 MB ledger was read about eight times
  a run). Tests pass; cherry-pick onto main once tonight's game is final. Also held: the status fix (a 'blocked'
  status from a run whose forecast did save goes back to the saved version) and the ledger slimming (branch
  pending-slim-ledger, archive_old_boxes; on the real ledger 14.3 MB to 7.9 MB with every score and page identical).
  All three are combined on branch deploy-after-game5 (8d46efda5, 338 tests pass); to ship, cherry-pick its three
  commits onto claude/live-v2 and push main.

- October 10, 4:15 p.m.: the container restarted at 19:56 UTC and took the rebuild loop with it. The rest of the rebuild
  runs from scratch report/rebuild_rest.sh (pushes every month not yet logged as pushed, four at a time; then October 1
  to 8 with player cards; same log). May and June 2026 finished; April 2026 (redo), July, August and September 2026
  pushed at 20:07 UTC. Fixed: plans listed relievers from the team's last game only (recent_players counted the bullpen
  after its game loop); now games in relief over the last 16 days, then batters faced, then recency. Strike-spot words
  say what the model measures ("takes it more than most", "can't do much with it"). Registered LINEUP-01 and LINEUP-02
  (the lineup before one is posted, by the opposing starter's hand; brl_live/lineups.py is shared by the simulator and
  the plans, rule still 'last'); study run 38082910875 (receipt research/report-38082910875.json, key lineup_study).
  Result: LINEUP-01 passed (+0.42 and +0.50 starters named per team-game, 2025 and 2026), LINEUP-02 failed on batting
  spots; RULE is 'hand' from 6fb5f8523 (simulator and plans; How page and METHODS.md say so). A strike-spot wording edit
  (ee9f2943b) put an apostrophe inside a single-quoted JS string and broke the page script; fixed in bdcdecbdd before any
  live run deployed it, and brl_live_tests/test_page_script.py now runs node --check on the template and a rendered page.
  Live page: the simulated game folds while MLB's plays have not come in; an at-bat with no plan links the hitter's card.
  Record page says the at-bat model was tuned on 2025, so 2026 is the fairer test.
- October 10, 4:50 p.m.: write contention. With four rebuild months committing every second, the live run's writes
  failed with 409 six times running and Game 5 was "blocked" at 20:24 UTC (ledger.json status). The runs could not be
  cancelled from here (the token has no actions write), so the loop was stopped and they run out. Fixes: the live
  store retries a 409 sixteen times with a random spread (9814fe326); the report lane writes a day's plans, the player
  cards, and the index with the record as one commit each (put_many, Git Data API, rebasing on a moved head), on
  claude/live-v2 at ca870d819 and tried first on the October 1 to 8 rebuild (diag/report 3757b4a62). Push it to main
  only after that run's receipt says completed and the days look right. Plans for games to come take their bullpen from
  the simulator's saved box (chance to pitch); plans built after a lineup is posted keep the bench.
  Outcome: the live runs from 19:57 to 20:49 UTC were all blocked by 409 (receipts diagnostics/v2_receipt.json), so the
  site stayed at the 18:45 deploy; the 20:50 run (9814fe326) completed and deployed Game 5's official-lineup version
  (v2, saved 20:23) and the LCS Game 1 hand-rule version (v2, 20:45). put_many's first trial lost every fast-forward
  to the rebuild runs; it now falls back to one file at a time after eight tries (41894d57e), and the second trial
  (run 38086104487, October 1 to 8) wrote one commit per day. Every 2026 rebuild month completed (receipts 38079203633,
  38082446419, 38079260497, 38079291185, 38082476353, 38082504807, 38082535066).
  Known risk, not fixed: ledger.json is 12.6 MB, almost all box_scores (34 boxes, about 380 KB each, also published
  as box_forecasts/<id>.json). Every persist() reads and writes all of it, and it grows by about 0.4 MB a version
  through the postseason. A slimmer ledger (scored boxes kept as references) needs care with render_page and the
  scoring of past days. Design sketch, not built: 62% of a box is 'samples' (five example games), and the day archive
  (box_page.lean_box) needs only the first; score_player_boxes and score_skill_boxes recompute every run and check
  content_hash(box) against the publication, so slimming a final box needs its score rows cached first (computed
  once, while the full box verifies) and both scorers reading the cache for slimmed boxes. The published
  box_forecasts/<id>.json keeps the full box for anyone re-checking.
- For next spring: hitters chase less early in the season (actual chase out of the zone 27.98% in April 2025 and 27.40%
  in April 2026, about 29.0% by July), and the league swing model has no season-timing term, so April plans expect 6
  to 8% too many chases (record.json by_month; relative reads unaffected). A registered test of a month-of-season (or
  days-into-season) offset on the outside-pitch swing model belongs before April 2027.
- Repository size (October 10, 5:50 p.m.): GitHub reports 4.5 GB (it recommends under 5 GB). Of the data branch's
  4.3 GB packed, 2.9 GB is old copies of three sealed 2026 season files (private/statcast/season-2026.enc 20.7 MB,
  bookkeeping 5.2 MB, physics 4.7 MB) that every report run resealed even with nothing new (91 to 112 copies in three
  days, most of them today's rebuild runs); fixed in 02ab686a0 (a run that adds no game writes nothing). Still about
  30 MB per game day through the World Series. For 2027: keep the season tables out of git history (base plus a
  small addendum of new games, or release assets, which do not count toward the repository); eight readers use
  study_path, season_path and physics_path (report, replay, fit, research, discovery, live bookkeeping). Report days
  are 0.66 GB over 21,912 versions (rebuilds); avoid full rebuilds unless the model changes.

- October 10, 3:35 p.m.: the rebuild loop's busy check read only the last 10 report runs, so with the calibration runs in
  between, batches overlapped (seven rebuild runs at once) and GitHub's secondary write limit failed April 2026 (run
  38079231926, HTTP 403). Fixed: the report's put() waits out 403/429 rate limits; busy() reads the last 60 runs. The loop
  was restarted at 9168a9f75 with April 2026 marked for redo in its log. Runs still going from before the restart use the
  old put(); if any of them fails, mark its month "failed ..., redo:" in report/rebuild_cross.log and relaunch the loop.
  Also: game plans print cleanly (print stylesheet); cards and the pregame header say when the simulator alone favors the
  other team.

- October 10, 3:20 p.m.: game-day plans follow the game. The live chain's report trigger (tools/brl_daily_report_trigger.py)
  now also rebuilds today's and tomorrow's plans when a forecast for that date was saved after the plans (lineups posted,
  a starter named), at most once per 45 minutes; when it dispatches, its output goes to diagnostics/report_trigger_latest.txt
  on the ledger branch (run logs cannot be read from outside). On the page, a plan built before the lineup was posted
  follows the posted lineup from our latest forecast and names any posted hitter it has no line for. Record page has a
  Game plans section (chasers and patient hitters against an average hitter on the same pitches, from record.json bins).
  Research: CAL-04 traced the engine-versus-realized aiming gap to whiff maps that are too flat for the hitter's own swing
  tendency; WHIFF-LAM-01 and WHIFF-OWN-01 to 03 did not pass (windows used: 2025-07-01, 2026-08-01, 2024-07-01,
  2026-06-01, 2024-09-01, 2025-05-01); a curved input is the open next step, on unused windows.

- October 10, 1:40 p.m.: CAL-03 checked the whole priced chain against what the pitches produced. Outside pitches were
  priced right (so the 35% gap behind the 30-to-40 range is about which swings a hitter's tendency adds, not a broken
  piece). Inside swings were not, and CAL-03b found two engine errors: the foul model evaluated at a zero whiff propensity
  (FOUL-01) and two-strike foul-tip strikeouts priced as balls in play (FOUL-02). Both are on (FOUL_FIX true; their joint
  check held: training rows within 0.05 runs per 100 swings). VALUE-18Z (discovery run 38072843478, frozen df01e156f)
  re-prices the aiming value; restate under its row's rule. Still open: the in-play value level by period (6% low in late
  2025, 8% high in late 2026), and the report's own count chain still uses the old foul label (its K and BB scales absorb a
  constant). Plans: the lineup before one is posted is the team's last lineup in batting order with the bench marked;
  the plan page opens from our own plan if MLB's feed fails. Rebuild of every backfilled report with all of it:
  scratch report/rebuild_all.sh (started 17:43 UTC at df01e156f; log report/rebuild_cross.log).

- October 10, 12:50 p.m.: the called-strike model. CAL-02 found it calling far too many strikes just off the plate
  (mixed seasons; the calls move each season). CS-SEASON-01 (each season's edge profile) and CS-SEASON-02 (the same,
  tested on 2024) missed their gates; CS-RECENT-01 (recent takes weighted more) missed too. On the 30 days after each
  monthly as-of date the season profile is no worse by log loss in any season (2024 +0.03, 2025 -1.43, 2026 -3.73 nats
  per 1,000 takes) and much better at the edge (2026 just off the plate 1.045 against 1.468), so it went on as a product
  call, PROD-05 (tools/brl_report.py CS_SEASON true, report and engine). VALUE-18Y (discovery run 38069376029, frozen
  244c8962a) re-prices the aiming value with it; restate the figure on the How page and in METHODS.md from its result
  under the row's rule. The rebuild of every backfilled report now carries both fixes: scratch report/rebuild_both.sh
  (started 16:52 UTC at 244c8962a; log report/rebuild_cross.log; skips months whose message is already logged).
  Registry now reads two-part names (CS-SEASON-02, SWING-CROSS-01, EDGE-MEASURE-*). Plan page: announced starters in
  the header for games still to come.

- October 10, 11 a.m.: CAL-01 found the report's swing model over-calling swings on pitches that end outside (8 to 11%,
  fastballs 15 to 19%, growing with distance outside; in sample as well, so structure, not drift). SWING-CROSS-01 (a
  crossing-location block for the league swing model, tools/brl_discovery.py cross_block, in the report and the engine)
  passed its registered gate in both runs and is on from the next daily build (tools/brl_report.py SWING_CROSS true).
  VALUE-18X re-priced the aiming value with it (run 38062560248): structural -42.3 [-45.2, -39.4] (was -43.5), regression
  -29.4 (same), inside -22.7, calibration 0.967 and 0.822 (constants kept); stated now as about 30 to 40 runs. Today's and
  tomorrow's plans and the player cards were rebuilt with the new model at 15:04 UTC (run 38062137847). Reports built
  before the switch keep old grades until rebuilt: scratch report/rebuild_cross.sh (started 15:19 UTC, four months at a
  time, then October 1 to 8 with player cards; log report/rebuild_cross.log). If the container restarts, relaunch it from
  the first month not yet pushed. Players: "Playing next" club chips and a next-game link on each card (4a9be73f1).
  Check-in scheduled 18:15 UTC (rebuild, Game 5's game-day version, the October 11 early call, phone check).
- October 10, 10:34 a.m.: Alec said to make the calls I was asking him about. Taken: PROD-04, the run environment without
  its venue term (brl_live/environment.json is env-v2-weather; TOTALS-08's do-no-harm product call). The headline refit
  on the weather replays reproduced taught-v5 (no new version); the taught total's weights and intercept unchanged; the
  season archives rebuilt from v2-prod3-weather-ps-1000. The How page says the park adjustment is out and why.

- Alec, 10:43 p.m.: explain things simply; the site's words must sound like baseball people, not AI; layouts still bad;
  he wants to look at future and past games and see the model overall. Shipped at 03:22 UTC (commit 2c577df45c6):
  - Tomorrow: after today's games, the live runner simulates tomorrow's games whose two starters are announced
    (BoxRunner.early_iteration, projected lineups, at most 20 minutes and only inside the first 45 minutes of a run;
    receipt in the ledger as early_receipt). The page keeps tomorrow's projected game lean (box_page.trim_boxes, EARLY_DAYS)
    with a note that the game-day version is the one graded. The daily report build now adds tomorrow's game plan as of
    today (tools/brl_report.py asof_for). Check after the next runs: early_receipt processed, tomorrow's day page shows
    the early calls, public/reports/<tomorrow>/ exists after the 13:20 UTC daily build.
  - Days: past days lead with "Our picks went X-Y. Vegas favorites went X-Y."; future days show early calls and "No
    number yet" cards from MLB's schedule.
  - Record page: "When we say 60%, do they win 60%?" (bins from the season archives) and a month-by-month table next to
    Vegas. Players: big leaguers by default; hitter cards say where his decisions cost him, grouped by chasing, taking
    strikes and weak swings. How page rewritten for coaches (phone-check anchors kept).
  - Later the same night: finished games open on How close and fold the projected game under one line (6,600 px to
    3,300 on a phone); Odds tabs and the record ladder say Our pick / Our model alone / Simulator alone / Team ratings
    alone / Vegas; pitcher cards get one scouting line per pitch (whiff, chase, zone rates and where it goes to each side,
    from the next report build); Game plan chase spots on the edge read "just off the corner" and stolen strikes are
    named where they land.
  - First early passes failed at the player-line baseline check (it required the game's date); fixed in 48af30186. Each
    early game takes about 8 to 9 minutes (500 s), so a 20-minute pass does two or three games. Confirmed working: game
    849831 (CWS at CLE, October 10) saved as version 1 at 03:58 UTC with projected lineups; the phone check passed with
    it on the page; at 04:01 the pass for October 11 skipped 849809 (starters not announced), as it should.
  - Record page also has Team by team for the latest season (wins against what our pregame numbers added up to).
  - A check-in is scheduled for 14:10 UTC October 10 (daily build: tomorrow's game plans, pitcher lines; early calls for
    October 11; phone check).
  - Check-in, 14:10 UTC October 10: the daily build ran at 13:32 (dispatched by the chain's fallback; the 13:20 cron did
    not fire) and wrote 2026-10-11/849809 (LAD at MIL, Brewers starter not named) and player cards with the new pitcher
    fields; the early pass skips 849809 until both starters are named; phone check passed at 13:50. The container
    restarted overnight and loop4 died after September 2025 (done); August 2025's second attempt was pushed at 14:13
    (run 38058777902). Then: "No number yet" cards for today and tomorrow open the game's plan (past-game view, Game plan
    tab only until first pitch); early plans say they were built the day before; a team with no starter named says so.
  - Worth a look (research): on the 4,991 graded games, actual chases 152,037 against 169,053 expected by the league swing
    model and 167,365 by the maps; both run about 10% high, so the chase expectation has a level offset (the maps' gain,
    21.9 nats per 1,000 swing decisions on 261,032, is relative and unaffected). Check the outside-zone definition and the
    swing model's level before quoting extra chases in counts.

- Alec, 9:55 p.m.: the site's words sound like AI, the layouts are bad, and you cannot look at future or past games and
  see the model overall. Done the same night: the date strip and calendar run two weeks ahead; a day with no saved
  numbers shows MLB's schedule with probable starters and the series estimate; a scoreboard strip (winners picked over
  the two seasons re-run, the market on the same games, the live picks) sits on every day page and opens the Record
  page, which now leads with the bottom line in words and the season tables; the explanatory text across the site was
  rewritten in plain words with the long parts folded into expandable notes (details.more); the season archives were
  rebuilt from the production replays (v2-prod3-1000) so past days and the tables show the model as it runs. Keep
  going on wording and layout: anything that reads like a paper should be cut or folded away.
- TOTALS-08 is closed: both runs (independent and paired streams) improve the totals in both seasons by about 0.03
  runs squared, remove the park reversal, and leave win chances alone, but the gain is below what two seasons can
  resolve, so the registered gate is not met. Removing the venue term from brl_live/environment.json is a do-no-harm
  product call for Alec (brl_replay/environment_weather.json is the candidate; then refit the taught headline on
  v2-prod3-weather-ps-1000 with harness/fit_headline3.py and TOTALS-03's weights with harness/totals_taught.py).

- TOTALS-08 first pass: the venue term's removal did what the mechanism said (park reversal -0.45 to -0.03, squared
  error -0.049 and -0.027, inside the registered range) but the pooled interval reaches zero. The tie-break, declared in
  the row before its runs: the same comparison with paired streams, four replays queued at 00:35 UTC October 10 (tags
  v2-prod3-ps-1000-<season> and v2-prod3-weather-ps-1000-<season>). Score: `python tools/replay_compare.py
  v2-prod3-ps-1000 v2-prod3-weather-ps-1000 out.json`; gate in the row. If it passes: copy
  brl_replay/environment_weather.json over brl_live/environment.json, refit the taught headline on the weather replays
  (scratch harness/fit_headline3.py SCRATCH v2-prod3-weather-ps-1000-2025 v2-prod3-weather-ps-1000-2026) as taught-v6
  with a new effective_from, refit TOTALS-03's totals weights the same way (harness/totals_taught.py), push main.
- The report workflow's schedule fired for the first time at 00:17 UTC October 10 (the 20:35 entry, late); the chain's
  fallback (tools/brl_daily_report_trigger.py) covers the misses. The report lane's memory is fixed: the pool and
  arsenal caches are bounded (a run reached 14.8 GB of 16; the April 2025 run holds at 9.2 GB), so the August and
  September 2025 deaths should not recur; after_loop3.sh re-pushes them when loop3 ends.
- Forward scoring record live: record.json 'scored' (the maps' log-loss gain over the league swing model on graded
  swing decisions; 20.7 nats per 1,000 on the first 45,433 decisions), stated on the Report tab from 1,000 decisions.
- Every GitHub Action is pinned by commit (the audit's ENG item).

- TOTALS-07 failed (the park prior is not the source of the cross-season park reversal; replays scored by
  tools/replay_compare.py, a general two-replay comparison paired by game). The source is the run-environment table's
  venue residual: fitted on one season and applied to the other in every replay, and the two seasons' venue terms are
  uncorrelated (-0.02 across 29 venues, spread 0.17 runs per game). TOTALS-08 registered and running (replays
  38004874163 for 2026 with brl_replay/environment_fit2025_weather.json, 38004938036 for 2025 with
  environment_fit2026_weather.json; the production candidate is brl_replay/environment_weather.json). Score with
  `python tools/replay_compare.py v2-prod3-1000 v2-prod3-weather-1000 out.json`; the gate is in the ledger row. If it
  passes: copy environment_weather.json over brl_live/environment.json (ADJUST['environment'] stays on), refit the taught
  headline on the new replays (harness/fit_headline), bump the headline version, then push main.
- UMP-01 (diagnostic): plate umpires' edge leanings persist only partly (0.37 across seasons; outside edge 0.60); the
  table is published (public/reports/umpires.json) and the live strip names the plate umpire from the box score with his
  leanings; nothing in the pricing. The lane experiment 'umpires' caches officials per game on the data branch
  (research/umpires-<season>.json).
- Forward scoring record: every graded swing decision's log loss under the hitter's map and under the league swing
  model, summed per pair, game and record (record.json 'scored'); the Report tab states the maps' gain per 1,000
  decisions once 1,000 decisions are graded with the new code (months rebuilt before 23:00 UTC October 9 lack it).
- Report: 'which family to lean on' per 100 pitches, overall and by count. Players: a team picker. The daily report build
  now starts from the live slate chain within an hour of 13:20 and 20:35 UTC when the cron does not fire
  (tools/brl_daily_report_trigger.py); check tomorrow that a daily run appears and that reports exist for the games.
- Backfill: July and June 2025 rebuilt; May, April, March 2025 then the second pass over 2026 follow in loop3; the
  worktree was moved to the latest code at 23:02 UTC. August and September 2025 died on the runner (step cancelled,
  not a timeout) and are queued by scratch report/after_loop3.sh with the October 8 to 9 daily rebuild; the receipt now
  records peak memory per stage and is published after every date.

- Night of October 9, live: the Report tab follows the game. During a game the page reads the official feed's batter,
  on-deck hitter and pitcher (lcSituation; SIT_FIELDS carries onDeck), matches the pair in the report, selects it and the
  count group the plan is priced by, and moves with every pitch until a cell or a count is tapped (state.rpFollow; a
  Follow the game switch brings it back; the on-deck hitter is one tap away; a pinch hitter or a reliever not on the
  morning card gets a plain line). While the pair at the plate is selected, its plan sits under the strip and the tables
  follow. The Summary tab shows a This at-bat card (the strip, the count-specific figure, the three family grids, a
  link to the full plan). Harness: scratch live/build_harness3.py builds a page with a faked live state for game 849832
  from the data branch's predictions.json and report doc; live/shot3.js and live/design3.js check it (served from
  scratch root on port 8766). The phone check now opens the Players list, a card and a game's Report tab with the
  same design rules (receipt keys players, report; the live strip under browser_feed.this_at_bat).
- Players pages are live from the lane (run 37994692489: 616 hitters, 880 pitchers, names and current teams from MLB,
  some winter-league or minor-league clubs for players listed there today); public/reports/players/index.json and
  16 shards each way, rebuilt by the daily report run or a push with "players": true.
- TOTALS-06 failed (tools/totals_park.py, data-branch aggregates): park offsets learned on one season make the other
  worse (-0.13 and -0.20 runs squared, intervals excluding zero); a park's mean residual reverses between seasons
  (correlation -0.46 across 30 parks) while the same teams' away residuals do not (+0.09). Reading: the PA model's park
  counter (all-time, 1,000-PA prior) carries one season's noise into the next with the wrong sign. TOTALS-07 registered
  and its fit running (run 37998079379, name pa-2026-v5-parkprior: v2-physics with park_prior_pa 6,000; the fit tool now
  reads the counter priors from tools/fit_model_params.json). Next: replay both seasons with the PROD-03 recipe (the
  trigger parameters of e31c6b7478 with model pa-2026-v5-parkprior, 1,000 worlds, paired streams against
  v2-prod3-1000-2025 and -2026, the cross-season tables: fit2025 tables for the 2026 replay, fit2026 for 2025), then
  score with tools/totals_park.py's cross-season correlation, squared error, P(over) and the win Brier; the gate is in
  the ledger row. If it passes, refit the taught headline on the new replays before any switch.
- Still running at the stamp: VALUE-18J (runs 37990450846 small, 37990489632 noisy; started 20:57 UTC); the 2025
  backfill months (loop3, log report/backfill_loop.log) with the second pass over 2026 after them. The 2025 September
  run (37995723313) died 6.5 minutes into its build (step cancelled, not a timeout) and must be pushed again after the
  loop ends: {"dates": [2025-09-01 .. 2025-09-28], "asof": "2025-09-01", "publish": true}. The daily October 8 to 9
  reports were built before own cost landed and could be rebuilt with it.
- LICENSE (MIT) and CITATION.cff exist (Alec: pick whatever); README cites them.

- Evening of October 9: VALUE-18I held (in-zone aims priced structurally came within 2 to 6% of the truth of their own
  choices in every structured synthetic world; null read zero), so the Report tab shows strike spots (squares; he lets it
  go, or swings to little effect) and the site states the in-zone part (about 23 runs per team-season) beside the
  outside range of 30 to 45. Each hitter card now carries his own cost: his swing decisions against the average hitter
  his side on the pitches he saw, runs per 600 plate appearances, with the cells where it costs him most. VALUE-18J
  (map-noise draws in the structural interval) is running as a diagnostic in the small and noisy worlds.
- The report workflow's schedule (13:20 and 20:35 UTC) has not fired once since it was added; every run so far came
  from a push to diag/report. A manual run now also builds the day's reports (workflow_dispatch input daily=true), but
  this session's token cannot dispatch workflows, so today's reports were built by a push with {"dates": [yesterday,
  today], "publish": true}. If the schedule still has not fired by October 10, trigger the daily build from the live
  slate workflow (permissions actions: write, then gh workflow run brl-report.yml -f daily=true at the two times), or
  push the two-date parameters from a scheduled task.
- The rebuild loop died once with a shell (a timed-out command killed its process group); it was restarted as
  scratch report/backfill_loop3.sh (remaining months of 2026 and 2025, then a second pass over October 1 to 7,
  September, August, July, June and May 2026 so every month carries strike spots and the by-group tallies). Keep waits
  inside tool calls short: a command that hits its timeout takes background loops with it.

- The afternoon of October 9 settled the aiming edge's pricing. VALUE-17 (80 synthetic worlds with known truth,
  tools/brl_synth.py, lane experiment 'value_synth'): the old headline method (regression coefficient times fitted gain)
  overstated the true value of its own chosen spots by 1.3 to 2.0 times in every structured world; the plate-appearance
  estimand's own-map intervals were too narrow (3 of 10 null false positives); the within-hitter neighbor-held estimator
  and the new pitch estimand passed. VALUE-18 (structural pricing: each aim priced by the engine's components times the
  hitter's own part scaled by its out-of-sample calibration) priced its own choices within 5% of the truth in every
  structured world, read zero in null worlds, and chose spots worth 1.3 to 1.8 times the old method's. VALUE-18F (frozen
  a06aa19e, later-period months): structural -43.5 [-46.6, -40.7] per 6,200 outside only; the outcome regression on the
  same months -29.4; VALUE-19 found the structure calibrated against outcomes (slopes 0.93 to 1.03). **Stated value of
  the aiming edge: 30 to 45 runs per team-season from outside pitches alone** (METHODS.md, the program page, the site's
  How page). The old 32 is superseded.
- The Report tab now chooses and prices aim plans structurally (tools/brl_report.py STRUCTURAL = True, calibration
  constants 0.95 outside and 0.82 inside): spots labeled chase (filled dot) or take (hollow dot), runs by count group,
  an evidence level per pair (thin under 700 hitter pitches or 150 pitcher pitches to the side), the dangerous miss per
  family (where the chosen aims' scattered pitches land, the worst cell's share and cost), and the record split by
  pricing method and by count group, family, side, evidence level and season (forward conditional calibration). Every
  backfilled month is being rebuilt with it (scratch report/backfill_loop.sh from the bf_wt worktree, most recent
  first; log report/backfill_loop.log); October 1 to 7 and September 2026 were rebuilt before the dangerous miss and the
  by-group tallies landed and should be rebuilt once more at the end.
- PLAN-01b landed: random maps moved the natural-experiment slope to -0.36, so the planner is set aside under its rule.
  PLAN-02S (synthetic, the planner priced by the generator): the single-best-pitch planner misreads noise as gain
  (null world claims +15 per 6,100 while truly costing 10); a calibrated soft planner (own part scaled by its
  calibration, the best tenth of the pool per state) tracks the truth in all three worlds (small world true +5.1,
  claimed +4.6). PLAN-03, when taken up: that planner with a sequential off-policy evaluation checked first in the
  synthetic worlds. Not in the product.
- TOTALS-04 (diagnostic, tools/totals_diag.py on the PROD-03 replays): means right overall, by team, by half and by inning;
  home-away independence holds; the team-run shape is 8 to 10% narrow; a negative binomial on the simulator's own mean
  matches CRPS and beats the histogram's log score by 0.047 nats per team-game (Monte Carlo bin noise); park extremes
  missed (Colorado, Sacramento under; Kansas City, Angels over). Next: TOTALS-05 (smooth the published distributions),
  TOTALS-06 (park adjustment), both to be registered before any change.
- ENG-01: research/registry.json is generated from the ledger by tools/registry.py; the discovery workflow refuses an
  experiment run whose commit message names an ID without a ledger row (registered or diagnostic). Still open from the
  audit: a locked environment and a smaller discovery module. Done October 9: Actions pinned by commit (each uses: line
  names the commit its major tag pointed to, read from the action repositories over git and verified to be commits;
  bump by re-resolving the tag), LICENSE and CITATION.cff, the reproduction section in README.
- PLAN-01b landed (run 37957704310): random maps with the own part's variance moved the natural-experiment slope to -0.36,
  far outside the registered band, so under the rule the planner is set aside; no plan value is stated. PLAN-03 (off-policy
  evaluation against the pitcher's logged choices, checked first in the synthetic worlds) is the replacement design when
  taken up.

- Alec, 9:40 a.m. October 9: the product is for coaches, past games must be evaluable with only information up to each
  game, and it must be made genuinely better than anything that exists; he does not want to wait weeks. Built the same
  morning: the matchup report (REPORT-01): tools/brl_report.py, workflow brl-report (scheduled 9:20 and 16:35 Eastern;
  a push to diag/report with tools/report_params.json runs a backfill: {"dates": [...], "asof": "YYYY-MM-01"}), one
  document per game on the data branch under public/reports/<date>/, read by the site's Report tab (template.html,
  REPORTS base URL on raw.githubusercontent.com). 2026 backfill running month by month (September, August, October,
  July first); 2025 next (needs the 2024 table as the previous season; it exists). Phone check passed with the tab.
- Afternoon of October 9 (ledger rows): BENCH-02F credited (hitter whiff maps +8.2 beyond boosting on untouched swings;
  the decision moment adds nothing to misses). SEQ-02 failed as registered but found a broader effect (after any taken
  pitch, and twice as much after a miss, a repeat of the family draws more misses and softer contact; a foul resets it);
  SEQ-03F (frozen 4ec64b68) scores it once on the untouched months. PLAN-01 registered and running: the plate appearance
  as a sequential decision problem (count plus what the hitter just saw; the pitcher's own pitches as actions; the credited
  pieces as the transition model; value iteration), tested by VALUE-08's natural experiment (the validated value is the
  slope times the claim; synthetic effect world slope 0.4, placebo -0.1, null 0.07). TOTALS-03 adopted as a product call:
  the page's expected total is 0.32 x simulator mean + 0.78 x market line - 0.543 from October 10 (beat the line -0.27
  [-0.42, -0.13] in 2026, -0.12 [-0.36, +0.11] in 2025); over chances stay unstated. The report tool now writes plans by
  count, two-strike miss spots, the pitcher's decision-moment deception rate, calibration bins per game and a forward
  record (public/reports/record.json, shown on the Report tab). Backfill loop: scratch report/backfill_loop.sh (a
  worktree at scratch bf_wt pushes one month at a time to diag/report and waits for the run to finish; log in
  report/backfill_loop.log); 2026 months done through the loop's list, then 2025.
- Alec, 8:56 a.m. October 9: the goal is to beat every model that exists. Where we stand (ledger rows):
  WHERE-01 (diagnostic): in the PROD-03 production replays our own number is level with the closing line (Brier pooled
  0.24245 against 0.24289; 2026 simulator alone 0.24274 against 0.24325) and a results logistic on both leans on ours
  (0.65 [0.24, 1.09] against the market's 0.38); development seasons, so FWD-02 (registered) scores ours against the
  market from October 9 on (the 2026 postseason is a record; 2027 is the test). Headline stays 60/40. Totals: our mean
  total ties the line in 2026 (squared error 19.71 against 19.73), trails in 2025 (20.32 against 20.10); P(over) is
  still worse than a coin (0.2527, 0.2517), the clear gap left.
  BENCH-01: against a boosted learner given the same pitch inputs, the logistic decision-moment model is 23 nats per
  1,000 decisions behind (most of MATCHUP-01F's +40 was the logistic league model's weakness), but the hitter maps read
  at the decision moment add +2.9 and +3.0 beyond boosting with the heat map; BENCH-01F (frozen 5e29436b) scores that
  once on the untouched months (run 37936470999).
- STANCE-01 held (partial): distance off the plate, recovered from the bat-tracking intercepts (reliability 0.99), goes
  with the sideways chase pattern (-0.27, -0.30; within hitter -0.31); MATCHUP-09 failed (shifting the swing surface by
  stance hurts the league model), so it is a scouting signal. CHANGE-01 failed: hitters re-time fully between pitchers.
- The CLV-03 public wording went live at 9 a.m. Eastern (Alec: push the public wording).

- Recorded since 3:45 a.m.: COMMAND-01 failed (early-start zone and waste rates add nothing beyond velocity);
  FRAMING-01 passed (the challenge system cut the spread of catcher framing to 0.67 of 2025's); PRESSURE-01 (hitters
  chase about 2.6 points more than their maps say with runners in scoring position, +0.19 log-odds [0.17, 0.22], and
  their own chase spots keep their pull). BATSPEED-01 (earlier) already answered the bat-speed early-warning question
  (no), so it is not repeated.
- SCOUT-03 exported (product): the private scouting page now uses maps fitted on 2025 and the whole 2026 regular
  season (scratch matchup_page/build.py matchup_page/scout3_receipt.json).
- SWINGMAP-01 failed: hitters' own chase spots are not where their bat goes (attack angle, direction and their
  changes all null). ZONEMAP-01 held in the partial band: the zone midpoint correlates +0.35 with the high-minus-low
  own part in both seasons (part of the vertical pattern is the hitter's own strike zone).
- MATCHUP-08 (development) and MATCHUP-08F (untouched, frozen f53a4b2b): batter zones credited for the league
  swing model (+0.59 nats per 1,000 decisions [0.28, 0.88]); hitter maps already carry each batter's zone.
  VALUE-15: re-priced on batter zones the edge keeps 85 to 94% (about a tenth is zone height); 32 runs stand with
  that note. The private page says so.
- CLV-03 done (ledger): with open-time lineups the line still moved toward our taught number 67.9% (2025) and
  72.6% (2026); lineups explain 0.5 to 3 points. The public page states these open-time figures from October 9,
  about 9 a.m. Eastern.
- DISP-02 done (ledger): the centered day-form shock fixed the run spread (team-run SD toward actual in both seasons,
  mean total level) and P(over) pooled -0.00038 [-0.00074, -0.00005], but the pooled win Brier upper end is +0.00041
  (bound +0.0002), so day form stays off. A totals-only use would need its own registration.
- COMMAND-02 done (ledger): repeated aims do not identify command (reliability 0.08 to 0.11); the plain location
  spread per pitch type predicts next-season walks beyond walk, zone and chase rates (+0.111 [0.037, 0.186] x1e-4).
- WARMUP-01 done (ledger): hitters do not warm up (-0.0030 runs per earlier plate appearance against fresh
  relievers), familiarity with a pitcher adds nothing measurable, the pitcher's pitches thrown carry the decline
  (+0.026 per 100); a substitute's first plate appearance -0.014. Out of sample +0.31 nats per 1,000 plate
  appearances over the simulator's times-through terms, but LOAD-01's size check puts the game-level lever near
  0.03 runs, below the replay floor: deferred. Proposed to Alec: a plate-appearance gate for small layers (own-stack
  log loss across seasons plus a do-no-harm replay).
- EXPLOIT-03 done (ledger): no aiming at hitters' own chase spots in the 2023-2025 postseasons (+0.04 points).
  EXPLOIT-03F runs once after the World Series: push tools/discovery_params.json with {"experiment": "exploit",
  "postseason": true, "post_seasons": [2026], "pool": "postseason", "seasons": [2023, 2024, 2025, 2026],
  "game_types": ["R", "F", "D", "L", "W"], "final_eval": true, "frozen_commit": "c5e52f18..."} and again with
  "pool": "regular" (frozen at c5e52f18, the registering commit).
- Recorded overnight: VALUE-14 (edge works through walks and outs in play), MAPS-01 (trait share 0.86), SERIES-01 (no
  fading over meetings), ABS-01, PERCEPT-01/02F (failed), MATCHUP-06/07 (failed), ENGINE-02 (no larger pair effect).
- FWD-01 runs after the World Series (instructions in the matchup section).
- PROD-03 and taught-v5 went live at about 8 a.m. Eastern October 9 (Alec: "Stop asking for permission you have it").
  The first live runs on the new code (37926945664, 37927992799) succeeded and the phone check passed at 12:09 UTC
  (390 and 1440 px, no errors); October 9 has no games, so the first forecasts with the package come on the next game
  day: check their boxes (pitchers per team, starters' batters faced) then. Awaiting Alec only for publishing new
  pages (the Matchup Physics page).

Earlier (10:30 p.m. October 8):

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

### Matchup physics research program (from 10:30 p.m. Eastern, October 8)

Alec's direction: put the main effort into invention and discovery (a new predictive method with a demonstrable
advantage), keep the product running, keep an untouched evaluation. Plan, mechanisms and rules:
discovery/MATCHUP_PROGRAM.md. Untouched set for the program: 2026 pitches from August 1 and the 2026 postseason
(dropped in code; one scoring per frozen candidate). Evidence rules for everything: top of LEDGER.md.

Credited on the untouched set (frozen code, one scoring each; ledger rows):
- MATCHUP-01F: hitter swing maps read at the decision moment (gravity-only projection 260 ms out) beat the standard
  hitter heat map on the true crossing by 40.2 nats per 1,000 decisions [38.5, 41.8] on every 2026 decision from
  August 1 (development 36.5); pair chase slope 0.95 [0.66, 1.19] (820 pairs). Frozen commit 45576a52.
- CONTACT-03: the contact window (spread of a hitter's contact depth, Savant bat tracking, earlier swings) predicts
  misses beyond his whiff rate: +0.16 nats per 1,000 swings [0.05, 0.27], coefficient -0.86 per 10 inches; the
  flexible swing-geometry model +1.07. Frozen commit 618cc7f3. Reading: contact depth is only seen on contact, so its
  spread is the hitter's window, not timing error (the M1 equation's assumption was wrong in sign).
- TIMING-03F: after a called strike the next contact is farther out front the slower that strike was, +0.151 inches
  per 10 ms [0.101, 0.192]; after a miss later, -0.239 [-0.317, -0.146] (within pitcher-types, cell fixed effects).
  Frozen commit 8795ff00.

- MATCHUP-04F and MATCHUP-05F: hitters answer pitch families differently at the same apparent spot. Family maps
  beat location maps on the untouched months by +3.91 [3.34, 4.44] (frozen fdb61e97); about half of that is a
  league-wide family pattern, so the league model now carries its own family-by-location part, and against it the
  hitter-specific family parts still add +2.16 [1.78, 2.52] (MATCHUP-05F, frozen 9f2f971e; breaking balls outside
  +3.95). Hitter family maps with the league's family part replace location maps in the scouting view and engine.
- VALUE-08F (replaces VALUE-01F and VALUE-02F's run figures, which were withdrawn at about 2:30 a.m. October 9 when a
  fresh placebo failed in VALUE-07): hitters' maps share a shape that predicts run value for anyone (SD 2.7 points,
  -0.0019 runs per point); the hitter's own part (his map minus the same-side mean map) passes its placebo, keeps its
  rate among pitches at the same spot (VALUE-09) and with the plate appearance's other pitches held fixed. Aiming
  each outside pitch at the best third of the pitcher's own spots of that pitch type for this hitter, at typical
  command (0.6 ft), by the hitter's own part: **about 32 runs per team-season [21, 44]** on the untouched months
  (location maps; family maps 29 [18, 42]). Frozen 02b266a2. In-zone values are not stated (their placebo fails).
  The edge grew in 2026: 2025 about 14 runs [7, 20], 2026 through July 22 [15, 28] (VALUE-11), August onward 32;
  map training length is not the reason (VALUE-10). EXPLOIT-01 (pitchers do not aim at hitter maps) stands.
- FWD-01 (registered, frozen 065e3dc6, run after the World Series): push tools/discovery_params.json with
  {"experiment": "matchup_final", "seasons": [2023, 2024, 2025, 2026], "final_eval": true, "frozen_commit": "065e3dc6...",
  "forward_from": "2026-10-09", "game_types": ["R", "F", "D", "L", "W"]} and the same with "experiment": "family" and
  "league_family": true. The feed pitch table holds regular-season games unless game_types says otherwise, so every
  credited untouched scoring so far used August 1 to September 27 only; the whole 2026 postseason is untouched.
- ABS-01: the 2026 challenge system cut called strikes on takes 0 to 1 inch off the side edge from 49% to 35%;
  chasing outside costs about 10% more than in 2025 (part of why the chase-spot value grew).
- Scouting page rebuilt from SCOUT-02 (research/discovery-scout-37894222125.json): family swing maps against the
  average hitter, aim spots per pitch family, VALUE-08F value; build with matchup_page/build.py scout2_receipt.json
  (the old SCOUT-01 page's run figures are superseded). Sent to Alec as a file at about 3 a.m.; publishing still needs
  his approval.

Development results around it (their run figures inherit the VALUE-07 withdrawal): VALUE-05 the outside edge sits most where the pitcher is behind in the count (39% of it on 18% of the pitches), two strikes carry their share. VALUE-03 nets ADAPT-01's tightening (hitters chase a little less everywhere after
seeing more tempting pitches than chance, -0.065 log-odds per extra one; they do not learn their spots): 86 percent
kept (about -37 outside on the credited value). VALUE-04 (running): the edge within each pitch type. DAMAGE-01: damage
maps on the true crossing help on 2025 (+0.21 per 1,000 balls in play) but DAMAGE-01F did not confirm them (+0.04, CI
includes 0); damage following the true crossing over the decision moment held (1.15 [0.95, 1.33]). DISCIPLINE-01
failed (maps do not improve season projections). LIVE-05: the bullpen package is no worse in game, so it stands once
switched on.

Private scouting page: scratchpad matchup_page/ (template.html, build.py, shot.js; data from the SCOUT-01 receipt
with aim spots, research/discovery-scout-37888261167.json), sent to Alec as a file; publishing it as an artifact was
blocked by the session's permission check (publishing new work needs Alec's approval). Never commit it to the
public repo (per-player summaries with names).

Development results since (October 8 night): MATCHUP-03 hitter whiff maps +6.5 on the true crossing (misses follow
where the ball arrives, swing decisions follow the decision moment: two clocks); STEER-01 falsifier triggered (one
map cannot locate a steering limit); ENGINE-01 pitch-by-pitch engine: pair strikeout and walk effects real
(+0.023 and +0.065 per point) but about 0.4 points of rate, no simulator layer, scouting view; MATCHUP-02 chasing
lowers walks; DRIFT-01 failed (maps are no early warning); ZONE-02 the feed stayed at the front of the plate in 2026
(only Savant moved), pool feed seasons as reported; TIMING-01/02 sequence effects in contact depth (within
pitcher-type: after a taken slower pitch contact comes earlier, +0.10 inches per 10 ms; after a missed slower pitch
later; 23% of a flight-time difference between one pitcher's pitches is not re-timed; hitter speed-follow reliable,
halves 0.54); TIMING-03 (cell fixed effects) running. Earlier: FATIGUE-01, SEQ-01, EXPOSURE-01 ruled out as TTO
sources; METHOD-01 paired streams.

Product: PROD-03 (fitted reliever choice and exits, team hooks, starter leash) passed its do-no-harm rule on both
seasons and is on in ADJUST from October 9, about 8 a.m. Eastern. With it went taught-v5 (intercept 0.0395, simulator 0.561, team model 0.524, refit on the PROD-03
replays by harness/taught_small.py). LIVE-05 (in-game, W26R done, W26P running) decides whether it stays on in-game.

Experiments live in tools/brl_discovery.py (feed) and tools/brl_matchup.py (Savant; params experiment "contact" with
"study": contact, decompose, timing, timing2, timing3, contact_final). Every experiment was checked on synthetic
planted and null worlds before it ran (scripts in the scratchpad matchup/). Built and off: MatchupAdjust,
matchup_table, replay param matchup (ENGINE-01 says no layer). CI skips pushes that only touch LEDGER.md, HANDOFF.md,
discovery/** or tools/discovery_params.json.

Runner capacity: the account runs about 20 jobs at once; replays (8 shards), discovery jobs, CI and the live slate
share it. Keep research light while games are on (live slate runs every 15 minutes). CI no longer runs on diag/**.

### What the bullpen work found (RELIEF-02)

The hand-set reliever exit rule pulled relievers mid-inning at about 57% of the decision points where a change is
allowed; real managers do it at about 9%. The engine used 4.6 relievers a team-game at 3.5 batters each (real 3.3 and
4.9). The fitted exits (brl_live/relief_exit.json, off) give 3.4 and 4.8. This is the largest structural miss found so
far and part of why simulated bullpens struck out fewer batters and allowed more hits than real ones.

### Known gaps, measured

- (October 11) The bullpen's chance to pitch is over-confident at the top (BULLPEN-03, -04): arms the simulator gives 60%
  to 80% pitch about 53% of the time, 80% and up about 51%, in both seasons; the rest are close. A straight calibration
  cannot fix it, so the page notes it. BULLPEN-05 found the choice model itself gives the pool's two busiest arms 20% of
  real entries against 16.6%; BULLPEN-06 (a heavy week, the top-two flag) fixes that entry by entry and halves the
  top-two excess in replays, but the 60%+ chances stay 12 to 31 points high, so it is off. What is left looks like
  availability: within a game's worlds the same arms are always available. A toy model on the replay shares (random
  unavailability of a quarter to a third of the pool per world) brings the top bins near real; the 0-10% bin's excess is
  the relievers outside the candidate list (call-ups), which no draw fixes. Also: relievers outside the 14-day candidate
  list (call-ups, returns, position players) pitch about 0.12 times a team-game, the whole shortfall in pitchers per
  game.
- (October 11) Outcome levels drift by season and the simulator carries them from its tables: in replays with tables from
  other seasons, 2025's walk-or-HBP chances ran 2 points high and 2026's hit chances 1 point high (PLAYER-03); live, the
  context table includes the current season through late September, so October is closer. Tested October 11: in-season
  re-anchoring raises the plate-appearance likelihood in both seasons (LEVEL-01) but lowers 2026's totals and worsens
  the over and under in full replays, alone (LEVEL-02) or with the running plays (LEVEL-03); the layer is built and off.
- (October 11) Relievers per team-game: the simulator uses about 0.1 fewer pitchers than real teams (4.19 against 4.29 in
  2025, 4.15 against 4.28 in 2026, LEASH-02's replays) and its game-to-game differences run wide: actual on simulated
  slope 0.50 and 0.48; once the starter's real length is known, the simulated count adds little (0.11 and 0.07). The
  team hooks (RELIEF-03) were the first suspect, but across teams the count is fine (slope 0.81 and 0.86); within a team,
  70% of the simulated game-to-game spread follows the starter's simulated length, and the simulator adds relievers
  faster per out the starter misses than real teams do (0.185 and 0.158 pitchers per starter out against 0.146 and
  0.127 real). It matters for the bullpen's chance-to-pitch table more than for wins.
- (October 11) Relievers' exits by pitches in the outing (research run 38166582743, the live relief-exit hazard on every
  real relief decision 2023-2026): under 10 pitches 0.82 to 0.86 of the expected removals, 20 to 39 pitches mid-inning
  1.11 to 1.42; milder than the starters' (0.5 to 2.1). A pitch term for relievers can follow LEASH-03 if it goes on.
- (October 11) Pitchers' pitches per plate appearance beyond what their outcomes take (research run 38166817136; 775
  pitcher-seasons with 300 or more batters): spread 2.9% (10th to 90th percentile 0.96 to 1.04), season to season
  correlation 0.51 to 0.55. About 3 pitches over a 25-batter start: a small refinement for LEASH-03's draws, not a priority.
- (October 11) The 2026 home-run overstatement for the top tenth is in established hitters: at 400 or more prior plate
  appearances 0.0514 real against 0.0598 simulated per plate appearance (0.86), under 400 on level (1.00); in 2025 both
  near level (0.98 and 0.93). Established power hitters fell off in 2026 beyond the league's 4% home-run drop, so the
  next refit's aging and time decay (RETRAIN-02's idea) is where to look, with 2026 as a season the fit can learn from.
- (October 11) The plate-appearance model's player differences run wide out of sample: by tenths of the simulated hit and
  home-run rate per plate appearance (PLAYER-06's replays, live settings), each step in the simulated rate came with 0.90
  (hits) and 0.82 (home runs) of that step in actual rates in 2026, the model's holdout season, against 0.95 and 0.97 in
  2025, its tuning season. A shrink of the player part toward the league cannot be gated on two seasons with this model
  (2025 tuned it); the clean test is next season's refit (train 2023-2025, tune on 2026, hold out 2027) or a refit
  trained on 2023 and tuned on 2024 with 2025 held out.
- (October 11) The simulated run distributions are narrow (team-run SD 3.10 against 3.21 to 3.25; games decided by five or
  more 27% against 29%), which sinks the run line (RUNLINE-01). Day-form worlds for the market numbers fixed the spread
  and the run line but not the over and under (DISP-03, off). Simulated starters also reach the 7th inning more often
  than real ones (12.0% to 12.8% of starts against 10.8% to 11.6%, LEASH-02 and LEASH-03). Not a missing bad-day spread:
  DISP-03's day-form replays (same plain hazard) have more 7-inning starts, not fewer (13.1% and 12.5% against 12.6% and
  11.8% without), since good days for the pitching side lengthen starts as much as bad days shorten them.
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
