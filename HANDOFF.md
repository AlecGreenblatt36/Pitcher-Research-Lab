# Baseball Research Lab handoff

## Current checkpoint — October 6, 2026, 5:40 p.m. Eastern

The box-score product sprint is deployed. Site: https://alecgreenblatt36.github.io/Pitcher-Research-Lab/

Two games have full pregame player distributions and five complete sampled games each: Dodgers–Braves and Brewers–Padres. There are seven preserved forecast versions across these games, two with full boxes, and zero final scored game/player forecasts at this checkpoint. Do not manufacture scores or borrow timestamps from older win-only versions. Latest implementation merge: `1fc11b2f2b2f78d1b6892c11756970125b714b9f` (PR #9). Browser-check correction: `aa37bf07bfc0b67ddaaa5f4c7b6d184558f16cff`.

## User's priority and next sprint

Box scores took priority over the prior sprint order. The next separate sprint is the conditional pitch bridge: generate each PA's pitches conditional on the outcome already chosen by the locked PA model, using only earlier pitch usage by count/hand and velocity distributions. Show count, type, speed, result; label pitches simulated, not predicted. Only subsequently test whether these pitch counts improve starter removal against the unchanged version on a separately declared evaluation. Do not change the live anchor or manager on the strength of attractive pitch sequences.

Then return to fitted postseason bullpen policy/active rosters; daily digest; track-record completion; through-2024 model copies and full 2025 replay; individual fitted engine/context upgrades. No 2025 replay has been executed. Repeated selection on 2025 is development, not untouched confirmation, even with through-2024 fitting. Both corrected Brier and log-loss differences with 95% intervals must meet the declared acceptance gate. Keep numerical MCSE separate.

## Working system and privacy

Repo `AlecGreenblatt36/Pitcher-Research-Lab`, default `main`. State branch `brl-live-data`. Workflow `.github/workflows/brl-live.yml` requests minute 7,22,37,52 each hour, plus manual and relevant push triggers. Timing is not guaranteed by GitHub; late output must be withheld. The user has installed the correct secret, encrypted release asset and Pages setting. No additional setup is needed for this sprint.

Secret name `BRL_PA_PACKAGE_KEY`: never print, request in chat, or commit its value. The pinned release remains `brl-live-runtime-20261006`, asset `brl-live-runtime-20261006.zip.enc`, configuration `brl_live/runtime.json`. Decrypted runtime is confined to RUNNER_TEMP and removed after execution. Do not publish the runtime, source-player rows, key or font files. Fonts are loaded from the public font provider; screenshot images are permitted.

Public code modules: `entrypoint.py` restores the runtime and calls `box_runner.main`; `history_refresh.py` manages immutable encrypted daily histories; `verified_store.py` checks object identity; `boxscore.py` observes the frozen engine and builds books/distributions; `box_page.py` renders the phone product. Old `live_extension.py` and `refreshed_page.py` are retained, but are not the new primary rendering path. Private runtime modules supply original cloud contracts/security/provider/manager/game rules.

## Proven live execution

- Run `37532008456` was genuinely triggered by `schedule`. It produced two win-only forecasts and completed at about 5:20 p.m. Eastern. This closes the earlier unobserved-cron gap; it does not guarantee all future ticks.
- Box run `37532752518` was push-triggered after PR #9. It read/authenticated the secret/runtime, assembled history through October 5 (1,228 PAs beyond seed), used 2,253 prior official PAs for independent bookkeeping annotations, and completed 10,000 worlds for each game. Both forecast and Pages jobs succeeded.
- Dodgers–Braves full box saved at 5:25:21 p.m. Eastern; public at 5:25:25, before 6 p.m. scheduled start. Forecast ID `3dbe8c97e573fb4e374bf91035548993a48fb2c1df43d6febbb702ae53cd8f63`; box SHA256 `3881b0909657c5a87d6a5ffedd6b0551aea741eb795cdefd7a109b076aaa6e3d`; commit `506bcfbefbfaa5c6a3e87c8ad90b4f2e94ed7850`.
- Brewers–Padres box saved at 5:29:41 p.m. Eastern; public at 5:29:45, before 9:30 p.m. scheduled start. ID `541e1017f3c46a6f35f32d13f8040b34a8b9df823dcbc872def93a5b3b509bd9`; box SHA256 `392d1a76072a4ef1352f9e6bd420348ae3d5a3155a7a67e65a163aaedd14ad9b`; commit `485af3ed1c71e0088888f234dcb1c711604a7615`.
- Actual-first-pitch eligibility and automatic player scoring remain pending final results. Do not call them completed merely because pregame publication worked.

## Box accounting and scoring

`ObservedSimulator` observes the original simulator without consuming its random stream. Scoring runner identities and responsible pitchers reconcile runs, hits and every PA across inning/batting/pitching books. Means and integer histograms include all 10,000 worlds, including zero innings for a nonappearing pitcher. Sample 0 has a modal exact score pair; four other samples use fixed world positions. This is a typical score, not the most probable detailed sequence. The five replayed seeds must reproduce those original score vectors.

New ledger keys: `box_scores`, `box_publications`, `actual_boxes`, `player_scores`; public JSON also has `view_scope`. Immutable public box output is `box_forecasts/<hash>.json`. The deployed file allowlist is still exactly `index.html`, `predictions.json`, `.nojekyll`. Every box has its own content hash and pre-first-pitch publication proof. Unchanged old win-only snapshots receive one new box-bearing version; unchanged box-bearing snapshots are not repeated. All old versions remain intact. The page only pairs a box with its exact forecast version.

Actual final boxes use official player IDs and reconciled team/inning/opponent totals. Unprojected substitutes remain visible and unscored. A projected nonparticipant in a complete final listing has zero actual opportunity; a missing field stays unavailable. Baseball IP 5.2 is 17 outs, not 5.2 decimal innings. Bat H/HR/K and pitcher K/IP use fair finite-ensemble CRPS; hit/HR occurrence also uses corrected binary Brier. These are dependent player-games, not independent evidence for model selection.

The inherited PA `pitch_number` is uniformly normalized to 1. It was rejected as a pitch-count source. Current displayed pitch counts and BB/HBP splits are empirical annotations from earlier actual official pitch events on an independent stream with declared fallbacks. They do not generate pitch sequences or influence removal. Small recent support is a limitation; do not claim pitch-level accuracy.

## Tests and screenshots

47 focused tests passed both locally and in the isolated cloud runtime (42 box-accounting/scoring tests plus five public-check tests). An additional Node test verifies exact-version box pairing and Preview/Live state labels. Do not add overlapping reruns as distinct tests.

Two local full 10,000-world experiments ran: a frozen pregame case exactly reproduced every original away/home/seed vector; another historical postseason case exercised a matching real final box. Seventeen genuine official final feeds passed actual parsing. Historical comparison previews have `view_scope=historical_replay`, real generation times, no publication records and zero live scores.

Normal HTTPS phone/desktop verification: workflow `37534342077`, corrected second attempt, job `112514427447`, artifact `11445552293`, passed at 390px and 1440px. Screens cover slate, game header/innings, batting, pitching, all five sample openings, half-inning PBP, track record and How. Required actual font faces loaded; every checked view has minimum 16px text, 44px tap targets, at most one status note and no horizontal overflow or JavaScript errors. This is viewport/browser evidence, not a physical-iPhone test. Final raw actual-comparison view was tested separately against real historical data, not yet after a new live final.

## Recorded failures and limits

First isolated cloud test picked up dashboard `app.py` via pytest configuration instead of the private runtime `app` package. Fixed with `-c /dev/null --import-mode=importlib`; no test was weakened and Flask was not installed to conceal the wrong import. First box-page phone check incorrectly tested unused Barlow default weight 400; corrected to explicitly load and verify the actual body 400 and heading 700 FontFace objects. The one corrected rerun passed. Local checker-test path initially did not exist; retrieved the actual repository test and reran successfully.

Earlier history-reader failure `Unsupported encrypted package` was a mismatched Contents payload. The repair checks returned bytes against size and Git object SHA, performs one immutable-blob recovery, then still authenticates AES-GCM. Keep this safeguard. Earlier direct-ref writes failed twice; PR merges preserved newer commits instead of forcing a rollback.

No fitted model, baseball rule or accuracy upgrade was adopted. Known gaps remain regular-season postseason bullpen logic, no pinch hitters/steals/WP/PB, legacy speed/defense/rest assumptions and no per-world talent uncertainty. The initial history's historical publication vintages remain unverified. Broad arbitrary-date coverage, clean 2025 replay, market comparison and the next-day refresh execution remain unproved. Preserve all failures and missing metrics in LEDGER.md.


## New parallel track: edge research and paired skill scoring

The conditional pitch bridge remains the next engine sprint. A parallel edge track now registers five requested ideas and supplies a richer scorecard without fitting or enabling any of them. Read `research/edge_track/README.md` and `experiments.json`. Each experiment needs matched-game/per-metric effects, not only winner Brier. Treat 2025 adaptive selection as development; do not call it untouched confirmation.

Implementation changes: `BoxAccumulator.finish` stores team and combined-run histograms from its exact score pairs. `BoxSimulator` freezes simple baselines from earlier fully reconciled official boxes and the unchanged fair team NB recipe. `box_runner` attaches `skill_baselines` inside the box BEFORE the existing immutable box publication and computes `skill_scores` after finals. An unchanged old box without those fields receives one new pregame version while the game is still Preview; no post-start backfill. Existing old wins/boxes and publication dates are untouched. `box_page` exposes seven How close rows and moves batting detail below the headline. Public top-level schema adds only `skill_scores`; the three-file Pages allowlist remains unchanged.

Pitching simple baseline: earlier team-box empirical starter lines and random uniform mapping of prior relief lines to the same current pregame bullpen, including zero nonappearance. It is not a readiness predictor. The initial cache has 31 complete prior games in the local source check, not a full season of role opportunities. Team and high-total forecasts use existing frozen NB parameters; high = 9+ combined runs. Count scores are fair CRPS for model draws versus exact fitted comparator CRPS. No correction is applied to the baseline as if its training sample were simulation draws.

Files: `brl_live/edge_metrics.py`; `research/edge_track/report.py`; `brl_live_tests/test_edge_metrics.py`. No raw inputs, private model or font files belong in the public patch. Raw baseline-source records remain in the encrypted history cache; only their hash, coverage and resulting forecast PMFs are published. Existing key/runtime asset remain unchanged.

New tests: 84 combined local tests passed (overlaps existing tests), including fixed-baseline/fair-score oracles, no-future/no-target inputs, time/publication/hash/lane gates, missing-stat and zero-opportunity cases, same-world totals and metric/report matching. Local real-source baseline construction passed with 31 games / 62 team boxes / 19 player baselines. A full local repeat-run harness hit two launch/setup failures and was not retried again. Cloud-run and deployment receipts are the remaining evidence for executing the installed new path. Do not convert null metric fields to zero or describe registration as an evaluated experiment.
