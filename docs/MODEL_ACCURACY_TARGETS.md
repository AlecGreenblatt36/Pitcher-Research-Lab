# Baseball Research Lab — Model Accuracy Targets

**Status:** controlling engineering target after the clean 2025 replay and locked-PA reproduction gate pass.  
**Principle:** optimize proper probabilistic scores and calibration, not raw winner accuracy.

## Promotion targets

A game-model version is not considered dependable until it satisfies all of the following on the same frozen 2,430-game 2025 development replay:

1. **Winner probability**
   - corrected Brier score beats the fair negative-binomial team-strength baseline;
   - finite-path-corrected log loss beats the same baseline;
   - paired uncertainty is reported against both the prior accepted engine and the fair baseline.

2. **Calibration**
   - logistic calibration slope is between **0.90 and 1.10**;
   - calibration intercept and decile/reliability table are reported;
   - winner accuracy is reported only as a descriptive metric and is never a promotion target.

3. **Run distributions**
   - empirical coverage of the nominal 50% team-run interval is between **48% and 52%**;
   - empirical coverage of the nominal 80% team-run interval is between **78% and 82%**;
   - run MAE, RMSE, CRPS, bias, and interval width are reported.

4. **External market benchmark**
   - score historical pregame MLB moneylines on the identical game set;
   - use only a licensed/authorized odds source;
   - select the final snapshot strictly before first pitch;
   - remove bookmaker vig before scoring;
   - report market Brier, log loss, calibration, coverage, and game-matching rate;
   - treat the market as an external benchmark, not as training input unless a later experiment explicitly says so.

The realistic long-run MLB winner-accuracy range is expected to be in the high 50s, but raw accuracy is not an objective because it ignores probability quality.

## Fair baseline

The team-strength baseline must use only information available before each 2025 game and must include:

- negative-binomial team-run marginals;
- home-field advantage estimated from prior data;
- shrunk team offense and defense;
- explicit extra-inning/tie handling;
- no simulated Monte Carlo noise when an analytic calculation is available.

## Build order after the clean gates pass

Each step is evaluated on the same 2,430 games. A step is accepted only if it improves the current accepted engine without violating calibration or interval-coverage gates.

1. **Pregame bullpen roster and availability pipeline**
   - active roster as of first pitch;
   - transactions effective before first pitch only;
   - recent workload and rest;
   - role and leverage history;
   - no target-game box-score or target-game participant leakage.

2. **Winner-probability recalibration**
   - cross-fit within 2025 by held-out month for reporting;
   - report raw and recalibrated Brier, corrected log loss, slope, intercept, and reliability bins;
   - after method selection, freeze one calibrator for prospective Scorebook forecasts.

3. **Simulator/team-baseline blend**
   - choose blend weights with the same month-wise cross-fitting scheme;
   - report simulator, team baseline, and blend side by side;
   - accept only if the blend beats both parents on corrected Brier and corrected log loss without harming run-distribution calibration.

4. **Expected-run center diagnosis**
   - identify the largest run-mean misses by park, weather, platoon, defense, catcher, starter/bullpen period, and home/away status;
   - change only components supported by a pregame-valid feature and a paired replay improvement;
   - do not confuse better interval shape with a sharper expected-run center.

5. **Event-keyed random streams**
   - implement deterministic streams keyed by game, path, event index, and draw type before evaluating any small engine change;
   - separate PA outcome, runner transition, pitcher removal, reliever selection, and between-pitch hazards;
   - verify that marginal distributions remain unchanged while paired-difference variance falls.

## Process

- No additional protocol amendments unless a real failure or leakage issue is discovered.
- Each engineering step produces one short receipt containing:
  - what changed;
  - source and artifact hashes;
  - paired result versus the current accepted engine;
  - result versus the fair team baseline where applicable;
  - calibration and interval-coverage effects;
  - **ACCEPT**, **REJECT**, or **INCONCLUSIVE**.
- Rejected and inconclusive changes remain recorded.
- 2026 is not reused for accept/reject development decisions.
- Final evidence comes from forecasts frozen before future games and scored afterward in the Scorebook.

## Historical moneyline source candidates

Preferred initial benchmark source: a licensed historical-odds API with timestamped bookmaker snapshots, such as The Odds API historical `baseball_mlb` head-to-head market or SportsDataIO historical MLB odds. The implementation must preserve vendor terms, source timestamps, bookmaker IDs, mapping coverage, and raw-to-no-vig transformation receipts.
