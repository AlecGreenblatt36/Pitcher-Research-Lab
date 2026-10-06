# BRL cold-start handoff — October 6, 2026

## Mission and order

Finish automatic pregame postseason forecasting, official-lineup revisions, Pages delivery and final scoring. Maintain private daily history. Next fit/freeze separate through-2024 PA and starter models, replay the full 2025 manifest, then test fitted engine/context improvements individually. Licensed historical odds come after a fair-baseline win. Full 2027 live forecasting and arbitrary-date/game playback remain product targets. Read METHODS.md and LEDGER.md before claiming accuracy.

## Where the working system lives

Repository: `AlecGreenblatt36/Pitcher-Research-Lab`; branch `main`. `.github/workflows/brl-live.yml` requests minutes 7,22,37,52 each UTC hour plus manual/relevant push triggers. `brl_live/entrypoint.py` restores the pinned encrypted runtime, verifies/assembles prior-day history, executes game forecasts, preserves versions and publishes allowlisted output. State branch: `brl-live-data`.

Pages: https://alecgreenblatt36.github.io/Pitcher-Research-Lab/

Secret name: `BRL_PA_PACKAGE_KEY`. Never request its value in chat, log it, or commit it. Release tag `brl-live-runtime-20261006`, asset `brl-live-runtime-20261006.zip.enc`; pins are in `brl_live/runtime.json`. The user installed all three prerequisites successfully. Do not replace the key or package for unrelated failures. Decrypted runtime lives under RUNNER_TEMP and is removed after the worker.

Public code: `entrypoint.py`, `bootstrap.py`, `history_refresh.py`, `live_extension.py`, `verified_store.py`, `refreshed_page.py`. Encrypted runtime includes `cloud/runner.py`, `cloud/contracts.py`, `cloud/security.py`, the locked engine/model/history and baseline. Do not publish that runtime in plaintext.

## Latest completed evidence

1. User's manual run `37518902595` read the secret, restored the runtime and deployed Pages, but failed history loading with `"reason": "Unsupported encrypted package"`. It preserved earlier cards and made zero new forecasts. A green setup/deploy result was not successful forecasting.
2. Read-only diagnosis `37519489380` authenticated the runtime/index and eight day objects. One Contents response declared 687 bytes but decoded to 1,017, with mismatched Git hash and encryption header. No user setting was missing.
3. Repair `f8c5b59e36b66ebc1314bdeb4e3f90ec3baf16c1` verifies each Git object's size/hash, fetches once by immutable blob SHA when inline bytes are inconsistent/absent, and still requires AES-GCM authentication. No key, stored source object, model coefficient or baseball rule was changed.
4. Cloud run `37520250698` succeeded: secret/authenticated decryption, history through October 5, and **one new 10,000-path official-lineup forecast** for Dodgers–Braves. v2 forecast ID `45ecaeeec3618987b02a83adb8b412dc83c2195ef675a5b35d33c4e78e228616`, saved 19:41:17 UTC, publication commit `b8bd0836165252c6cd41d5ac1229b1b79618500a`, before its 22:00 scheduled start. Both original v1 forecasts remain. Brewers–Padres inputs were unchanged and not resimulated.
5. Display patch PR #8 merged at `48fdd8c9b5dcd347db91fa991cc813e06f97ca67`. It removes prior-day unforecast results from today's cards without altering any public ledger/scoring records. Run `37522631086` reused accepted history, preserved three versions, generated no duplicates and redeployed the page.
6. Public browser run `37522795380` passed **normal deployed HTTPS navigation at 390px and 1440px**. Cards match public JSON, both tabs/version history work, no horizontal overflow or JavaScript errors. It has no key/private data access. This is browser viewport evidence, not testing a physical iPhone or all browsers.

There are **three versions across two games and zero scored games**. Player coverage is through October 5; 1,228 PAs beyond the frozen seed were already accepted and were reused, not freshly downloaded in these two cloud iterations. Prior-day finals were ingested but had no forecasts and did not enter scores. Healthy forecast runs were push-triggered; cron execution is still unobserved in checked receipts.

## Tests and deployment checks

74 unique focused local tests passed across refresh/storage/launcher/rendering/checker logic; Python compilation passed. The storage repair adds 21 tests; date-card/checker work adds seven. The public-check cloud job also passed its five checker tests and installed/runs Chromium normally. These counts overlap; do not add them together as unique tests.

The tested local public suite command was `PYTHONPATH="$BRL_RUNTIME_DIR:." python -m pytest brl_live_tests -q`, with BRL_RUNTIME_DIR pointing to the restored runtime and dependencies installed. No key is a test argument. The public checker command `python tools/check_brl_phone_page.py --output phone-check` ran successfully in GitHub Actions.

The independent original-dashboard CI last inspected at run `37522631072` passed. Its intermittent pitch-selector failure has been observed earlier; no root-cause repair to that old dashboard was claimed here.

`.github/workflows/brl-public-check.yml` now runs after completed successful BRL live slate workflows. It checks the deployed public page and uploads only public screenshots/receipt. It is not a second simulation loop or a cron execution witness.

## Failures retained

Two direct updates to advance main to the prepared UI commit failed (ReadTimeout, then GitHub GraphQL server error). Heads were checked and unchanged; no third direct-ref retry occurred. The tested commit was placed on `fix/public-slate-date-cards`, inspected as PR #8 and merged through the PR API while retaining newer documentation. This issue is resolved through the alternate merge path. Never force main backward to prepared commit `e1c410c328c62d0de0ce839a36498d068a01e262`.

## Immediate next steps

Observe an actual schedule-triggered live run. Continue official-lineup revisions with earlier versions kept. After the forecasted games finish, verify saved/publication times against actual first pitch and reconcile/score finals automatically. This last step cannot be claimed complete while those games are still pending. A lack of new forecasts on an unchanged fingerprint is correct; do not mint versions just to show activity. Do not retry the same failing step more than twice.

`ledger.json` holds forecasts/publications/actuals/status; immutable forecasts are `forecasts/<hash>.json`. `private/history-index.enc` points to encrypted day objects; inputs/results are encrypted separately. History failure must preserve prior data/forecasts. Baselines are paired to each frozen forecast. The shared worker can still fail before processing finals when history/model initialization fails; decoupling final reconciliation is a follow-up reliability task.

Postseason still uses regular-season bullpen logic, without the automatic runner. Fitted advancement/speed/defense, steals/wild pitches, rest/workload availability, postseason policy and talent uncertainty are unfinished. Current Pages is not yet arbitrary-date simulation/playback. No 2025 replay or model accuracy upgrade was performed here.

For winter, fit/tune PA, starter, preprocessing and calibration only through 2024; preserve the current live lock. Reconcile actual 2025 schedule coverage rather than inventing a 2,430-row total. Repeated 2025 candidate selection is development, not untouched confirmation; retain an independent/prospective confirmation lane. Keep both metrics' paired 95% intervals and MC numerical error separate in the ledger.


## Latest priority: box scores before postseason policy

New modules: brl_live/boxscore.py (observation-only accounting, distributions, actual parser and proper scores), box_runner.py (same live engine plus box publication), box_page.py (prototype-styled phone product). entrypoint now calls box_runner. Encrypted runtime/key/pins and PA/starter/game logic are unchanged.

New ledger members: box_scores keyed by forecast ID; box_publications containing immutable box SHA256, commit and publication time; actual_boxes by game PK; player_scores with eligible version rows and latest-version aggregates. Immutable outputs: box_forecasts/<hash>.json. Public JSON adds these plus view_scope; the Pages file allowlist remains index.html, predictions.json and .nojekyll. No raw source, decrypted runtime or font file is published.

An old win-only snapshot receives one new box-bearing version. An unchanged already-box-bearing snapshot is skipped. All 10,000 worlds must finish; five exact sample seeds are replayed for complete books. Publication of player distributions has its own deadline and cannot borrow an old win forecast's timestamp. Actual boxes are reconciled by player ID. Late or altered distributions do not score.

Local evidence: 42 box tests passed; 17 genuine final feeds parsed; a 10,000-world frozen pregame case reproduced every original score/seed vector; a separate 10,000-world historical postseason game supplied real matching actual comparisons. Five changed views at 390px passed offline component checks. Hosted HTTPS checks must pass after merge before claiming deployment. The first cloud test encountered dashboard pytest.ini/app.py shadowing and was fixed with -c /dev/null --import-mode=importlib; don't install Flask just to conceal that wrong import.

The inherited pitch_number field was unusable (all 1). Actual prior-date official pitch events now provide independent empirical bookkeeping counts and BB/HBP split; no pitch sequences or manager effects. Pitch-by-pitch is the NEXT sprint. Then resume fitted postseason policy and the master plan. Historical UI previews use actual generation times, view_scope=historical_replay, no publication receipts and zero live scores. Never backdate these.

After deployment: verify cloud box_forecasts_created, each distribution count and all five examples, actual first-pitch audit and final player scoring, the phone page screenshots and existing forecast-version preservation. Keep failures and missing metrics in the ledger. No 2025 model fitting/replay or accuracy upgrade occurred here.
