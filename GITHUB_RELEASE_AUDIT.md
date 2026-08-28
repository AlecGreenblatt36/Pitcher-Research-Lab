# GitHub Release Audit

Audit date: 2026-08-27

Source tree: `Pitcher-Research-Lab-main/Pitcher-Research-Lab-main`

Release tree: `Pitcher-Research-Lab-main/RELEASE_READY`

## Outcome

The incomplete GitHub download was reconstructed from the recovery build, reconciled with the latest backend, tested with deterministic fixtures and live MLB data, and copied into `RELEASE_READY` from the Git staging manifest. The clean-copy test installed only `requirements.txt` into a new virtual environment, started with no database, rendered the landing page, searched for and initialized Cade Horton, exercised all seven views, changed season and pitch type, and recorded no browser or HTTP failures.

## Bugs found and fixes made

1. The working download did not contain `templates/dashboard.html`, `static/`, `tests/`, or the complete Skenes case-study source. These were recovered and included in the staged release.
2. The recovered frontend still defaulted to Paul Skenes. The default MLBAM ID was removed; a fresh browser now opens in a neutral search/select state and no player API request is sent until selection.
3. The neutral page retained unresolved loading cards. Neutral mode now hides player-analysis panels until a pitcher is selected.
4. Release Profile was placeholder/future-work content. It is now a dynamic view backed by `/changes`, conditionally showing measured release X, release Z, extension, and arm angle only when values exist.
5. Release and change-detection copy contained roadmap language and implied conclusions. The live UI now documents the implemented sustained-deviation method, distinguishes sustained flags from a generic comparison window, and supports improvement, deterioration, stability, and mixed results.
6. Command & Location and Career / Timeline had content but no complete primary navigation path. Both are now first-class navigation entries.
7. The first uncached sync populated data but did not rebuild the season selector and research-window controls until another reload. Successful sync now re-renders both controls immediately.
8. A rookie/short-sample condition left the overview summary on `Loading...` when comparison periods were incomplete. It now renders an explicit sample-limited empty state.
9. Pitcher switching risked stale module output. Every view now waits on the shared selected-pitcher context, returns cleanly in neutral mode, and a selection change reloads the complete profile context.
10. Frontend tests depended on a packaged local database and did not cover the required data profiles. Tests now build a process-isolated temporary database with veteran, left-handed, rookie, reliever, short-sample, sparse/null, unusual-arsenal, one-pitch, and mixed-result fixtures.
11. The multi-pitcher test used a shared temporary filename, which could race when separate test runners were launched concurrently. Its database filename is now process-unique.
12. Initial and incremental ingestion behavior lacked isolated regression coverage. Mocked ingestion tests now verify initial creation, overlap-based incremental updates, official outings, and duplicate prevention.
13. README screenshots used unverified external attachments. They now reference four tested local screenshots in `docs/images/`.
14. Repository ignore rules did not cover all release-sensitive artifacts. Logs, archives, backups, SQLite variants, environment files, coverage output, and scratch/temp directories are now ignored.
15. There was no CI workflow or complete release validator. CI and validation now cover Python/JavaScript syntax, unit/browser tests, frontend dependencies, source guards, repository hygiene, secrets, and SQLite integrity.

The current backend modules, including `performance_routes.py`, already matched the latest overlaid GitHub versions. Database access was verified to use the shared `PRL_DATABASE` configuration across application route modules.

## Frontend dependency inventory

Direct stylesheets referenced by `templates/dashboard.html`:

- `static/style.css`
- `static/location_v2.css`
- `static/views.css`

Direct scripts referenced by `templates/dashboard.html`:

- `static/pitcher_context.js`
- `static/dashboard.js`
- `static/research.js`
- `static/location.js`
- `static/overview.js`
- `static/navigation.js`
- `static/performance.js`
- `static/career.js`
- `static/release.js`

Dynamically loaded frontend assets:

- `static/metric_guide.js`
- `static/metric_guide.css`
- `static/performance.css`
- `static/career.css`

All 16 static files exist with matching capitalization. Controlled Playwright tests and live-browser QA requested every dependency and received HTTP 200. No missing asset, console error, page error, failed request, or unexplained 4xx/5xx was recorded.

## Exact files added or changed

Application and configuration:

- `.github/workflows/ci.yml`
- `.gitignore`
- `README.md`
- `GITHUB_RELEASE_AUDIT.md`
- `validate_project.py`
- `scripts/live_release_qa.py`

Recovered and finalized frontend:

- `templates/dashboard.html`
- `static/style.css`
- `static/location_v2.css`
- `static/views.css`
- `static/pitcher_context.js`
- `static/dashboard.js`
- `static/research.js`
- `static/location.js`
- `static/overview.js`
- `static/navigation.js`
- `static/performance.js`
- `static/career.js`
- `static/release.js`
- `static/metric_guide.js`
- `static/metric_guide.css`
- `static/performance.css`
- `static/career.css`

Recovered and expanded tests:

- `tests/_test_environment.py`
- `tests/test_app.py`
- `tests/test_browser.py`
- `tests/test_data_conditions.py`
- `tests/test_ingestion.py`
- `tests/test_multi_pitcher.py`
- `tests/test_source_guards.py`

Recovered release content:

- `case_studies/skenes/README.md`
- `case_studies/skenes/prototype/README.md`
- all 29 Python research scripts under `case_studies/skenes/prototype/scripts/`
- `run_app.bat`
- `setup_and_run.bat`
- `docs/images/landing-neutral.png`
- `docs/images/overview-live.png`
- `docs/images/change-detection-live.png`
- `docs/images/release-profile-live.png`

The latest backend and operational files were preserved and staged: `app.py`, `comparison.py`, `pitcher_core.py`, `pitcher_routes.py`, `pitcher_sync.py`, `research_routes.py`, `location_routes.py`, `performance_routes.py`, `career_routes.py`, `official_sync.py`, `update_statcast.py`, `update_official_outings.py`, `update_all_pitchers.py`, `schema.sql`, all three requirements files, and the start/check/update batch scripts.

## Live ingestion and browser QA

The live database began empty. Three distinct real pitchers were searched and initialized:

| Pitcher | Profile | MLBAM | Pitches | Seasons | Games in pitch data | Pitch types |
|---|---|---:|---:|---|---:|---:|
| Mason Miller | RHP reliever | 695243 | 3,554 | 2023-2026 | 179 | 6 career types |
| Garrett Crochet | LHP starter | 676979 | 7,362 | 2020-2021, 2023-2026 | 142 | 8 career types |
| Cade Horton | RHP rookie / short sample | 690990 | 1,882 | 2025-2026 | 25 | 6 career types |

Initial ingestion verified identity resolution, full Statcast career download, schema/database creation, pitch insertion, official-outing synchronization, arsenal detection, and available seasons. The resulting live database contained 12,798 pitches, passed `PRAGMA integrity_check`, and had zero duplicate `(pitcher, game_pk, at_bat_number, pitch_number)` identities.

A second Mason Miller sync returned `mode=incremental`, fetched only the 2026-08-18 through 2026-08-27 overlap window (87 rows), kept the career total at 3,554, synchronized only needed official games, and produced zero duplicates. It did not rebuild the career.

For Mason Miller, Garrett Crochet, and Cade Horton, Playwright exercised search, selection, sync completion, Overview, Arsenal, Change Detection, Release Profile, Performance, Command & Location, Career / Timeline, season switching, and pitch-type switching. It then switched between pitchers and verified the prior name/data did not remain. Final arrays for console errors, page errors, HTTP errors, and request failures were all empty. No visible `undefined`, `NaN`, broken chart, endless loading state, or unexplained error response remained.

## Automated test results

- `python -m compileall -q -x "\\.venv|case_studies" .`: passed.
- `node --check` for all 10 `static/*.js` files: passed.
- `python -W error::ResourceWarning -m unittest discover -s tests -q`: 26 tests passed.
- `pytest -q`: 26 tests and 48 subtests passed.
- `python validate_project.py`: all eight gates passed.
- Playwright controlled-browser regression suite: passed desktop and mobile landing, all views, all frontend HTTP responses, custom-period apply/reset, season/pitch switching, and multi-pitcher stale-state checks.
- SQLite integrity and duplicate scans: passed.
- Secret, provider-trace, local-path, missing-template, missing-static-reference, and banned-copy scans: passed.

## Clean-copy test

`RELEASE_READY` was created from `git ls-files --cached`, containing only the 87 files intended for GitHub. A new `.venv` was created inside that copy and dependencies were installed solely from its `requirements.txt`. Flask started from the copy with no pre-existing database. A fresh browser loaded the neutral landing page and every static dependency. Searching for uncached Cade Horton created a new database with 1,882 pitches and 23 official outings; its SQLite integrity result was `ok` and duplicate count was zero. All seven views, 2026-to-2025 season switching, and pitch switching passed with no console, page, request, or HTTP errors.

After the test, the clean-copy virtual environment, generated database, and caches were removed from `RELEASE_READY`.

## Git staging audit

The working recovery folder was initialized locally because the supplied folder was not a clone. `git add .` stages 87 intended files. Explicit staging checks confirm inclusion of:

- `templates/dashboard.html`
- all 16 files under `static/`
- all seven test files
- all application Python modules
- `README.md`
- `requirements.txt`, `requirements-runtime.txt`, and `requirements-dev.txt`
- `schema.sql`
- `.github/workflows/ci.yml`
- the Skenes case-study source and local README screenshots

Explicit ignore/hygiene checks confirm `.venv`, generated databases, database sidecars, caches, logs, ZIP files, backups, environment files, temporary directories, and secrets are not staged. No push was attempted.

## Remaining limitations

- First-time ingestion requires network access to MLB/Statcast services and can be slow for veteran pitchers.
- Public tracking data can quantify observed pitch/release changes but cannot establish private biomechanical or medical causes.
- Statcast fields can be null or revised; the UI suppresses unavailable release measurements and the tests cover sparse/null samples.
- Official-outing availability depends on MLB schedule/boxscore endpoints and can lag the pitch feed.

## Final verdict

GITHUB READY
