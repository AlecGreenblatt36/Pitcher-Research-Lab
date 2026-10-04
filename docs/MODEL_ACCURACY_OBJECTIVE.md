# Game Model Accuracy Objective

**Goal:** make the pregame game model as accurate as it can realistically be, judged by proper probabilistic scores rather than winner hit rate.

## Promotion targets

The game layer is not considered dependable until it satisfies all of the following on the same frozen historical replay:

1. Beat the fair negative-binomial team-strength baseline on finite-path-corrected winner Brier score.
2. Beat the same baseline on finite-path-corrected winner log loss.
3. Logistic calibration slope between **0.90 and 1.10**.
4. Empirical coverage of the 50% team-run interval within **48%–52%**.
5. Empirical coverage of the 80% team-run interval within **78%–82%**.

Winner accuracy is reported for readability but is not a target or promotion gate. Baseball winner prediction has a low ceiling; a high-50s hit rate can be excellent while still hiding poor probability calibration.

## Long-term market benchmark

Use a licensed historical odds source and score pregame moneylines on the exact same games.

Initial preferred source: **The Odds API historical MLB odds**, which documents paid historical moneyline snapshots for MLB. SportsDataIO historical odds is the commercial fallback if coverage, event matching, or licensing is better for the final portfolio use.

Market benchmark rules:

- use the final available snapshot strictly before first pitch;
- never use in-play or post-start odds;
- convert each book's prices to implied probabilities and remove vig within book;
- aggregate books with a documented robust rule such as the median de-vigged home probability;
- map events to MLB game IDs using teams and scheduled start time, then audit ambiguous matches;
- score market Brier and log loss on the same included games as the model;
- keep purchased/raw odds private and commit only code, hashes, coverage receipts, and aggregate scores unless the license explicitly permits redistribution.

The market is an external ceiling/reference, not a feature in the baseball-only model unless a separate market-blended product is explicitly tested.

## Build order after the clean benchmark and pinned PA reproduction pass

1. **Pregame bullpen roster and availability pipeline**
   - active roster as of first pitch;
   - transactions effective before first pitch;
   - rest, recent workload, availability, and role;
   - no target-game box-score or appearance information.

2. **Win-probability recalibration**
   - cross-fit within 2025 by month for an honest development report;
   - freeze the selected map before future Scorebook forecasts.

3. **Simulator/team-baseline blend**
   - cross-fit blend weights by the same monthly scheme;
   - report whether the blend beats both the unblended simulator and fair team baseline.

4. **Expected-run center**
   - diagnose largest residuals by park, weather, platoon, defense, catcher, and other forecast-valid context;
   - prioritize improvements to expected runs, because distribution shape alone is not enough.

5. **Event-keyed random streams**
   - key stochastic draws by game, path, event index, and draw type;
   - required before judging any small engine change where ordinary seed reuse would leave excessive paired Monte Carlo noise.

## Change-control process

Do not write a new protocol amendment unless something actually breaks or invalidates the current experiment.

Every modeling step gets one short receipt containing:

- what changed;
- exact source and artifact hashes;
- same 2,430 games and simulation count;
- paired corrected Brier, corrected log loss, calibration, run MAE/CRPS, and interval coverage versus the current engine;
- confidence interval for the paired difference;
- **ACCEPT**, **REJECT**, or **INCONCLUSIVE**;
- one-sentence reason.

A change is accepted only when it improves its intended metric without materially worsening the other proper-score and calibration gates. Rejected and inconclusive results remain in the research log.
