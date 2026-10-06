# Baseball Research Lab — methods and evidence boundaries

## Purpose and current product

The system forecasts MLB games using information captured before play. It publishes win probabilities, expected scores, inning lines, batting and pitching projections, five complete example games, forecast versions and their publication times. After a final result is reconciled, the product can compare players and teams with their saved forecasts and calculate proper scores. The mission is auditable forecasting; no superior-accuracy claim follows from a working interface or a large simulation count.

Live forecasting and historical diagnostics have separate records. The current product is a scheduled static website with game-card navigation, full box-score views, sample play-by-play, track record and a How to read this tab. It is not yet an arbitrary-date simulation service. Every displayed time is Eastern. The numerical scoring store retains timezone-aware timestamps.

## The locked game simulation

The worker uses existing locked PA probabilities and the retained starter-removal model. Neither is refit by a scheduled iteration. Runtime and data-only inference artifacts have cryptographic identity pins. Current model calibration used 2025, so this live lock is not eligible for a training-independent 2025 replay.

Each simulation tracks innings, outs, runners, batting order, current pitcher and score. A forecast requires 10,000 completed games. Seeds are unique within a game and reused across lineup versions for reproducibility. Unfinished paths cannot be discarded or replaced to obtain a more favorable distribution; unresolved work blocks publication. Expected runs need not be integers, and the team with more expected runs need not have the greater win chance.

Postseason games currently retain regular-season bullpen and removal logic, without the regular-season automatic runner. This is disclosed in How to read this and is not a fitted postseason advantage. No pinch-hitting, stolen-base, wild-pitch or passed-ball process is integrated yet. Speed, defense and reliever-rest assumptions remain limited; per-world talent uncertainty is not integrated. These are research tasks, not completed accuracy improvements.

## Pregame information and daily history

Official MLB preview feeds supply team identities, rosters, probable starters and posted lineups. An absent official batting order is explicitly projected using earlier team appearances restricted to the roster. The current order projection is a heuristic, not a fitted lineup model. A final box is never used to construct that same game's live lineup.

Public Statcast and official final data are reconciled through the previous Eastern-calendar day. Checks cover PA identities, outcomes, dates and scores. Conflicting or incomplete source data block new forecasts. Sources retain actual capture times and hashes and are encrypted before persistence. Complete daily revisions are reused on subsequent checks; recent dates are revisited on later daily refreshes. The original seed remains immutable, and its historical publication vintages are not independently authenticated. Newly obtained observations are never backdated.

A source capture precedes simulation. A second preview check after simulation must still show no pitches, and the computation must finish before its original scheduled deadline. Publication time is recorded separately. Final scoring additionally verifies the observed first-pitch time. Changed lineups receive new versions; unchanged input fingerprints are not repeatedly simulated just to increase the forecast count. Introducing box output to an older win-only fingerprint creates its own new version and deadline.

## Complete box-score accounting

The box layer observes the original simulation without consuming its random stream or changing probabilities, runner transitions or pitcher-removal choices. It records who actually scores and which pitcher is responsible. Team and inning runs/hits reconcile with batting and opposing pitching lines in every simulated game. PA/AB, walks, hit batters, strikeouts and home runs also reconcile. The old manager's own state is not modified by display accounting.

Player means and integer count distributions include all 10,000 worlds. A pitcher's expected innings and strikeouts therefore include zero opportunity in worlds where he does not appear. Appearance probability is shown separately. Hit and home-run chances mean at least one in the whole game. Projected innings are decimal expectations; actual and individual simulated IP use baseball notation. A line of 5.2 actual IP is 17 outs, not 5.2 decimal innings.

Every inning is displayed separately. An inning not played contributes zero to the expected mean; a missing inning in an individual box is not invented. The first example has a most frequent exact final score pair, with a declared tie-break rule; four other examples use fixed world positions. All five replay their original seeds. A typical final score is not a claim about the most probable detailed event sequence.

The inherited terminal-PA corpus has pitch_number normalized to 1, so it was rejected as a pitch-count source. Earlier official playEvents instead supply an empirical joint sample of BB/HBP identity and pitch count, conditional on pitcher, selected PA outcome and batter hand, with pitcher/outcome and league fallbacks. This independent annotation stream does not change the game's outcome or feed the removal model. Recent support can be small. Displayed pitch counts are estimates, not pitch sequences or demonstrated pitch-count accuracy. The conditional pitch bridge is the next separate sprint.

## Official actuals and player scoring

Final boxes are read only in the outcome path. Player IDs, not names or guessed batting spots, join projections and actuals. Batting totals, pitching totals and official inning scores must reconcile. Missing stat fields remain unavailable. A projected participant absent from a complete final listing has zero realized opportunity. Unprojected actual substitutes remain visible but unscored.

Every box projection has its own content hash, commit and publication timestamp. Player scores require that exact content to have been publicly saved before observed first pitch. An earlier win-only forecast cannot backdate a newly generated player distribution. The latest eligible box version per game is selected without inspecting results, while every version retains its own record.

Batting hits, home runs and strikeouts, plus pitching strikeouts and innings, use fair finite-ensemble CRPS. It is the mean absolute difference to the observed value minus the distinct-draw pair-distance term; self-pairs are excluded. Pitcher innings are first scored as outs and then divided by three. Hit and home-run occurrence also receive corrected Brier scores. Lower scores are better, but different count statistics have different scales. Many player rows from one game are dependent and do not create an equivalent number of independent games for significance testing.

## Game scores and the fair baseline

Headline live game scores use the last eligible publicly saved pregame version, not the version that most closely predicted the result. Official teams and final scores must match and reconcile; publication must precede observed first pitch. Versions remain separately inspectable.

The inherited baseline is a team-strength negative-binomial run model with dispersion fitted on 2023–2024. It uses expanding strictly prior-date, league-shrunk team offense/defense, inherited 20-game shrinkage and a separate home-win logit offset. Its result recipe remains regular-season-only. Baseline and simulator are compared on identical eligible games and input cutoffs.

For estimated win probability p from N independent worlds and observed binary outcome y, corrected Brier is `(p-y)^2 - p*(1-p)/(N-1)`. The supplied baseline probability is scored directly. Game log loss is shown with the inherited second-order finite-simulation correction, explicitly an approximation. Numerical Monte Carlo error is separate from model and sample uncertainty. Shared seeds across games must not be counted as independent game-path cells when computing aggregate numerical error.

Market scores stay unavailable until licensed, same-cutoff historical odds are sourced. Closing odds cannot be described as the same-information comparison for an early forecast. No licensed odds comparison has been executed.

## Privacy, storage and publication

Raw source responses, player history and runtime/model files remain encrypted or private. AES-256-GCM uses purpose-specific authenticated data; the key is an Actions secret, never a command argument or public file. Git reads must match declared size and blob identity. An inconsistent Contents payload may be fetched once by immutable blob SHA and checked again; AES authentication is still mandatory. Never erase encrypted history to conceal an IO failure.

The public artifact allows only index.html, predictions.json and .nojekyll. It contains forecast distributions, five selected simulated event paths, final-box summaries and scoring/publication records, not raw training sources. Forecast records are immutable. Rendering binds boxes to their exact forecast IDs rather than mixing an old box with a newer header. The prototype's fonts are loaded from the public provider; no font files are distributed.

A failed refresh preserves the previous page rather than replacing saved predictions. The page carries a machine-readable refresh-blocked flag, and its verification job rejects a stale fallback. A successful deployment alone is not a successful forecast. Read the run receipt's stage, forecast count and history coverage. GitHub's schedule can be delayed or dropped, so nominal polling frequency is not a guarantee of capturing every official lineup.

## Evidence and remaining evaluation

The box sprint's evidence is execution, accounting, reproducibility and rendered-product testing, not a new accuracy gain. An observed schedule-triggered run, two cloud box forecasts, 47 focused regression tests, full 10,000-world frozen-score reproduction, 17 real final-box parser checks and deployed 390px/1440px browser checks are recorded in LEDGER.md. The first prospective final player score is still pending at the October 6 checkpoint.

For the winter benchmark, separately freeze PA, starter, preprocessing and calibration models fitted/tuned only through 2024, selecting settings on chronological development data. Retain the current live lock. Reconcile the complete official 2025 schedule, cancellations/resumptions and input coverage rather than fabricating exactly 2,430 eligible rows. Report exclusions.

Training-independent does not mean untouched: earlier project choices already used 2025. Repeatedly selecting upgrades on the 2025 replay makes it a development benchmark. Register candidates before scoring and preserve independent or prospectively saved confirmation evidence. Both corrected Brier and log-loss differences must be negative with both 95% upper endpoints below zero under the declared paired evaluation. Otherwise retain the current model and record the candidate as dropped/deferred. Operational fixes do not get invented metric improvements.

## Reference contracts

Public implementation: brl_live/entrypoint.py, box_runner.py, boxscore.py, box_page.py, history_refresh.py and verified_store.py. The encrypted runtime contains the unchanged provider/manager/game engine, cloud contracts and baseline. Tests and milestone evidence are recorded in LEDGER.md; recovery instructions are in HANDOFF.md.

Public Statcast field documentation: https://baseballsavant.mlb.com/csv-docs

GitHub schedule behavior: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule


## Edge research and skill scorecards — October 6, 2026

The five edge research families are registered in `research/edge_track/experiments.json` with detailed protocols in its README. Registration is not fitting, validation, or adoption. The old contact-only residual cannot change K or pooled BB/HBP; the proposed stuff/arsenal models need separately declared outcome groups and BB/HBP handling. Bat-swing features require explicit coverage/measurement-regime checks, and in-game updating has a separate clock and scorebook.

`brl_live/edge_metrics.py` adds descriptive paired pregame diagnostics: pitcher K/BB/pitches/innings, reliever appearance, team runs and high/low combined runs. High is fixed at 9+ runs versus 8 or fewer; the complement is not scored again as another independent result. Pitcher count CRPS includes zero for nonappearance. Skill baselines are saved INSIDE the independently timestamped box before first pitch; no late backfill is admitted. Old boxes retain their original player scores but lack the new paired skill comparisons.

The simple pitcher baseline resamples earlier complete team boxes by starter/relief role and maps relief lines uniformly to the same pregame pool, preserving nonappearance zeros. It uses no matchup/stuff/rest information. Its support and capture dates are serialized; the initial accepted cache is short, not a claimed full historical season. Team-run and high-total baselines use the existing team NB means and frozen dispersion. The team CDF-sum CRPS returns a bound on numerical tail truncation. Baseline distributions are fitted exact distributions, not simulated ensembles, so they receive no finite-simulation correction.

The observer stores team and combined-run histograms from complete coherent worlds. It never reconstructs them by summing independently sampled player marginals. Within each skill metric, scores are averaged within game before averaging games. Unprojected actual pitchers and missing statistics are recorded as coverage gaps. Exact forecast-version pairing and immutable publication hashes remain required. Missing latest-baseline coverage does not revive an older favorable comparison. Pitch-count predictions remain the existing independent empirical bookkeeping annotations until the separate conditional pitch bridge sprint.

`research/edge_track/report.py` computes per-metric current/candidate/simple-baseline effects and paired calendar-date-block intervals from exact matched score records. It refuses changed comparator values and unmatched identities. It does not authorize adoption or certify supplied provenance; simulation MCSE unavailable from aggregate score rows remains null. No fresh through-2024 lock or full 2025 replay exists yet. Selecting repeatedly on 2025 turns it into adaptive development, even if every fit stops in 2024. Prospective confirmation remains separate.
