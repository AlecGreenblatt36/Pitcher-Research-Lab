# Baseball Research Lab — methods and evidence boundaries

## Purpose

The product forecasts full MLB game outcomes from information captured before the game. It publishes team win probabilities, expected runs, the capture/completion times, and every forecast version. The aim is an auditable forecasting system, not a claim of superior accuracy without evidence. Live forecasts and historical development results have separate scoreboards.

## What runs today

The live engine uses the existing locked PA provider and starter-removal model. It does not refit either model during a scheduled run. The encrypted runtime is pinned by ciphertext and plaintext SHA-256; its data-only inference export has its own pins. The historical anchor used 2025 calibration, so it cannot support a training-independent 2025 test.

A game path tracks innings, outs, runners, batting order, pitchers and score. Each forecast contains 10,000 completed paths. World seeds are unique within a game and repeat across lineup versions for reproducibility. A path that exceeds work limits is not silently dropped or replaced. The live contract withholds forecasts if a path remains unresolved. Mean runs are expectations, not a prediction of an exact integer final score.

Postseason uses **regular-season bullpen logic** until a separately fitted policy earns acceptance. It does not use the regular-season automatic runner in extra innings. This is a rules configuration, not a measured accuracy improvement. Other known limitations include heuristic advancement, unmodeled between-PA steals/wild pitches, incomplete speed/defense and reliever-rest inputs, and no integrated posterior talent sampling. These are research tasks, not implemented advantages.

## Information timing and refresh

Official MLB preview feeds provide teams, rosters, probable starters and posted lineups. When an official nine-hitter lineup is absent, the current implementation constructs an explicitly projected order from prior team appearances restricted to the active roster. That is a scenario heuristic, not a fitted lineup model. No final-boxscore lineup is used to create a live forecast.

The history extension processes public Statcast and official final-game data through the preceding Eastern-calendar day. It reconciles PA identities, terminal outcomes, dates and scores against the official source. Same-day/future history and conflicting or incomplete source records fail closed. Source responses carry capture times and hashes and are encrypted before persistence. A complete daily revision is reused on later checks that day; recent dates are revisited on the next daily refresh. The immutable seed history remains unchanged, and its original historical publication vintages are not independently established. New observations are timestamped when actually obtained, never backdated.

The input capture precedes simulation; a second preview check follows simulation. A run must finish before the original scheduled deadline and while the feed still has no pitches. GitHub publication times are retained separately. Once the game ends, observed first-pitch time is audited before the forecast is admitted to the live scoreboard. Early versions remain when lineups change. Identical input fingerprints are not resimulated just to increase the forecast count.

## Private storage and public output

The public repository contains source and aggregate predictions, not plaintext player history, model coefficients or raw source responses. Runtime/state ciphertext uses AES-256-GCM with purpose-specific authenticated data. The secret is supplied by repository Actions secrets, never a command argument or public file. Git object reads must match declared byte count and Git blob identity; an inconsistent or omitted Contents payload is fetched once by immutable blob SHA and checked again. AES authentication is still required afterward. Never delete or replace encrypted history merely to conceal a read failure.

The Pages artifact permits only `index.html`, `predictions.json`, and `.nojekyll`. It contains game cards and public summary data. A previously saved page may be served with a refresh-failed warning when a worker fails. A green deployment means the page was published, not that fresh forecasts were produced. Check the machine receipt's stage, forecast count and history coverage.

## Scoring and the fair baseline

Headline live scores select the **last eligible publicly saved pregame version per game**, not the version that happened to predict the result most favorably. Other versions retain their own scores. Actual scores must reconcile between official linescore and boxscore, teams must match, and publication must precede observed first pitch.

The inherited fair baseline is a team-strength negative-binomial run model with dispersion fitted on 2023–2024. It uses expanding strictly prior-date, league-shrunk team offense/defense and a separate home-win logit offset; the 20-game shrinkage rule is inherited, not chosen on the evaluation game. The live extension preserves its regular-season-only result recipe. Baseline and simulator are compared on identical eligible games and capture cutoffs.

For win probability p estimated from N independent simulated games and outcome y, corrected binary Brier is `(p-y)^2 - p*(1-p)/(N-1)`. Baseline Brier is exact for its supplied probability. Log loss is reported alongside the inherited second-order finite-simulation correction, explicitly an approximation. Numerical Monte Carlo error is separate from sampling uncertainty and model uncertainty. Future score comparisons must use paired games, confidence intervals that reflect dependence, and the same scoring definition. A simulation count is not a validation sample size. Market scores remain unavailable until licensed, same-cutoff odds have been sourced; closing odds must not be presented as a same-information early-forecast baseline.

## The winter evaluation protocol

Freeze separate PA, starter, preprocessing and calibration artifacts whose fitting/tuning use only data through 2024; select settings on a chronological 2024 development block. Keep the current model for live forecasts. Reconcile the official 2025 schedule, game identities, cancellations, resumptions and missing input archives; do not manufacture a 2,430-row total if the actual eligible manifest differs. Freeze the manifest and report coverage/exclusions.

Replay the full 2025 season with rolling pregame-only observations and the frozen models. Training-independent is not synonymous with untouched: 2025 has already influenced earlier project work. Repeatedly retaining features based on their 2025 results makes that replay a **development benchmark**. To obey the no-test-tuning rule, select new ideas on pre-2025 development folds, register the candidate before scoring, and keep an independently reserved or prospectively saved confirmation lane. Do not call repeated 2025 selection a clean confirmation test.

Each candidate gets a ledger entry with its hypothesis, training/tuning windows, input coverage, frozen artifact IDs, paired current-model comparison, corrected Brier/log-loss deltas and 95% intervals. Adoption requires both deltas below zero and both upper interval endpoints below zero on eligible evaluation data, plus input-integrity tests. Missing results are `not run`, not zero improvement. Failed or dropped ideas remain recorded. Public live confirmation is required for deployment claims.

## Research backlog and product plan

Fit and test one change at a time: advancement conditioned on grounder/fly-ball and runners; speed and defense; steals and wild pitches; bullpen rest/workload; postseason usage; talent uncertainty. Projected lineups, forecast weather/roof, umpires, travel, Bullpen Capacity Index, hook/bullpen coupling and shadow-index ideas remain hypotheses until targets, public inputs and tests are specified. No numerical proxy is silently treated as a measured biomechanical or psychological trait.

The product target is date/game selection, playback, projected-versus-actual game cards and track record on a phone. Current Pages is a scheduled static forecast page; it is not yet an arbitrary-date interactive simulation service. GitHub's scheduler can be delayed or drop jobs; polling frequency is not a guarantee of capturing every lineup before first pitch. Spring 2027 full-season operation is a planned milestone, not an established capability.

## Reference contracts

Implementation: `brl_live/entrypoint.py`, `history_refresh.py`, `live_extension.py`, `verified_store.py`; encrypted-runtime `cloud/contracts.py`, `cloud/runner.py`, `app/team_baseline.py` and the pinned engine.

Public source documentation: https://baseballsavant.mlb.com/csv-docs

GitHub schedule limits: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule


## Box-score sprint (October 6, 2026)

The box layer observes the locked simulator without consuming its random stream or changing any PA probability, runner transition, or pitcher-removal choice. It records the selected scoring runners and reconciles team runs/hits, batting PA/AB and opposing pitcher lines in every world. All projected means and count distributions include all 10,000 worlds, including zero opportunity for nonappearing pitchers. Five complete examples are retained: one with the most frequent exact score pair and four fixed world positions. This is a typical score, not a uniquely most probable sequence. Every extra inning is shown separately in the line score.

The inherited PA corpus normalizes pitch_number to 1 and cannot estimate pitch counts. Actual earlier official playEvents supply joint BB/HBP and pitch-count annotations on an independent stream conditional on pitcher, outcome and batter hand, with pitcher/outcome and league fallbacks. These are estimates from potentially small recent pools, not generated pitch sequences; they never feed the manager. The conditional pitch bridge is a separate next sprint.

Final actual boxes are parsed from the MLB Stats API only in the outcome path. Player IDs join actual and projected lines; totals reconcile with official inning scores and opposing pitching totals. An absent projected participant in a complete final listing has zero realized opportunity; a missing stat field is unavailable, not zero. Unprojected substitutes remain visible and unscored. Baseball innings x.2 convert to 3*x+2 outs.

Player hit/HR/K and pitcher K/innings distributions use fair finite-ensemble CRPS: mean absolute distance to the observation minus distinct-draw pair distance. Innings scores are outs scores divided by three. Hit/HR occurrence also use corrected Brier. The latest eligible public box version per game is selected without looking at outcomes. Its own content hash, commit and publication time must precede observed first pitch; a previous win-only forecast cannot backdate player distributions. Player rows within a game are dependent and are not independent confirmation samples.

The phone UI uses the prototype fonts/colors and Eastern times, with amber only for actuals. Complete player lines wrap into readable grids rather than shrinking wide tables. Definitions and limitations are in How to read this. Only forecast output distributions, five selected simulated paths and final box summaries are public; raw inputs, model files and keys remain encrypted/private.
