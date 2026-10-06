# BRL cold-start handoff — October 6, 2026

## Mission and priorities

First finish automatic pregame postseason forecasts, lineup versions, Pages delivery and final scoring. Next maintain daily private history; create through-2024-only PA/starter copies for full 2025 replay; evaluate one fitted engine change at a time; then source licensed historical odds. Full spring-2027 live forecasting and arbitrary-date/game playback are future targets. Read METHODS.md and LEDGER.md before claiming accuracy.

## Repository and architecture

Repository: `AlecGreenblatt36/Pitcher-Research-Lab`. Default branch `main` has `.github/workflows/brl-live.yml`; schedule minutes 7,22,37,52 UTC each hour, plus manual and relevant push triggers. Source runs from `brl_live/entrypoint.py`. It restores the unchanged encrypted release, loads the extension, verifies/assembles prior-day history, runs game forecasts, persists versions and publishes the allowlisted static page. Data branch: `brl-live-data`.

Secret: `BRL_PA_PACKAGE_KEY`; never request its value in chat, log it, or commit it. Release tag `brl-live-runtime-20261006`; exact asset `brl-live-runtime-20261006.zip.enc`. Pins in `brl_live/runtime.json`. Do not replace them for an unrelated bug. Only ciphertext/aggregate predictions may persist publicly. Runtime files live under RUNNER_TEMP and are removed after the job.

Pages: https://alecgreenblatt36.github.io/Pitcher-Research-Lab/

## Failure, diagnosis and working repair

The user completed secret, encrypted-asset and Pages setup. Manual run `37518902595` at 19:25 UTC read the secret, restored the package and installed dependencies, but its worker emitted `"reason": "Unsupported encrypted package"` and made zero new forecasts. Pages successfully deployed the previous saved cards with a warning. This was not a missing GitHub setting.

Read-only diagnosis `37519489380` checked actual encrypted objects: history index and eight day objects decoded correctly; one small object's Contents payload decoded to 1,017 bytes although metadata declared 687. Its Git hash and encryption header did not match. Runtime package authentication passed. Do not interpret a green overall run as successful forecasting.

Fix `f8c5b59e36b66ebc1314bdeb4e3f90ec3baf16c1` adds VerifiedGitStore: verify byte count and blob identity; fetch once by immutable SHA if inline content is inconsistent/omitted; reject an inconsistent fallback; still require AES-GCM authentication. Existing encrypted objects, key, models and baseball rules are unchanged. The launcher records stage, secret-present and authenticated-decryption booleans, and a sanitized error location.

The first repaired cloud run `37520250698` completed: secret present, authenticated package decryption, encrypted history accepted through October 5, and **1 new 10,000-path forecast**. It retained both original forecasts and published Dodgers–Braves v2 with both official lineups. New forecast `45ecaeeec3618987b02a83adb8b412dc83c2195ef675a5b35d33c4e78e228616` was saved at 19:41:17 UTC and publicly committed at `b8bd0836165252c6cd41d5ac1229b1b79618500a` before the 22:00 scheduled start. Brewers–Padres inputs were unchanged and were not resimulated. Three versions across two games are now public; zero forecasted games have finished. Pages deployment succeeded. This run was triggered by a code push, not the cron scheduler. Existing encrypted daily history was reused and assembled (1,228 PAs beyond the frozen seed), not newly downloaded in this iteration.

## Tests actually executed

21 new object-transport tests passed locally: correct/empty/large content, byte-mismatch recovery, same-length tampering, missing base64, failed fallback, and 403/404/500 distinctions. The combined local refresh/launcher/storage suite passed 67 tests. Seven additional local tests cover prepared date-card rendering and public checker logic: **74 focused tests total**. Python compilation passed. These are not accuracy tests or a fresh runtime install claim.

With a decrypted runtime directory assigned to BRL_RUNTIME_DIR, the new public storage suite was run as `PYTHONPATH="$BRL_RUNTIME_DIR:." python -m pytest brl_live_tests -q`. It requires runtime dependencies; the key is not a test parameter. The original dashboard has an intermittent browser-selector CI regression. Its latest inspected run `37520250701` passed, but this repair did not claim to fix that separate root cause.

A read-only TinyFish normal-browser check completed (`d81c6249-270d-467d-89bb-e2c8f1d1890e`): the deployed HTTPS page loaded without a refresh warning, tabs worked, and both Dodgers forecast versions were visible. Viewport control was unavailable; this is desktop browser evidence, not a 390px/mobile-device test.

## Unapplied display patch — do not mistake prepared code for deployed code

The first healthy run ingested two prior-day finals without forecasts. They appear as two unnamed waiting cards on today's slate. A display-only wrapper fixes this while preserving every forecast, publication, actual and status in public JSON/scoring. A real public-document replay produced exactly two current-game cards with unchanged JSON.

That fix plus a public phone-check workflow is prepared in commit `e1c410c328c62d0de0ce839a36498d068a01e262`, parent `f8c5b59e36b66ebc1314bdeb4e3f90ec3baf16c1`. Moving main to that commit failed twice: ReadTimeout, then GitHub GraphQL server error. The head was checked after each failure and had not moved. No third retry was made. **The prepared display fix and public-check workflow are not active.** Separate METHODS/LEDGER/HANDOFF writes through the Contents API succeeded afterward. Preserve those newer documentation commits if applying the patch later; never force main backward to the prepared commit.

## Immediate continuation

Inspect the latest main run and machine receipt, not just its green conclusion. Confirm history through yesterday, new version count, per-game statuses and Pages content. Reusing complete cached history is valid and must not be called a new download. Skipping identical fingerprints is expected; never mint versions solely to inflate activity. Do not repeat the same failing step more than twice.

The Dodgers official-lineup revision is demonstrated with v1 retained. Continue checking other official-lineup revisions. After games finish, audit actual first pitch/publication order and reconcile scores automatically. Never backdate a forecast. Scoring a game that was genuinely forecast before play is still pending. Check that an actual **schedule-triggered** run occurs; a push/manual run does not establish that. The cron is installed, but its execution has not been observed in the verified receipts.

## Persistent objects and risk areas

`ledger.json` holds public forecasts, publication records, actuals and per-game status; `forecasts/<hash>.json` is immutable. `private/history-index.enc` points to encrypted `private/history-days/<hash>.enc`; inputs/results are encrypted separately. A failed refresh must not overwrite accepted history or forecasts. Pair the baseline to each frozen forecast, not a later pointer. History/model failure can still prevent the same worker from processing finals; decoupling result scoring is a follow-up reliability task.

The live path uses regular-season bullpen heuristics in postseason and disables the automatic runner. Fitted advancement/speed/defense, SB/WP, rest/availability, postseason policy and talent uncertainty remain unfinished. Do not use the 2025-calibrated live anchor for a clean 2025 test. Fit/tune separate artifacts only through 2024, freeze all preprocessing/calibration, reconcile the full game manifest, and disclose that repeated 2025 selection is development, not untouched confirmation. Current live Brier numbers remain unavailable until eligible finals exist. No accuracy upgrade was adopted in this repair.
