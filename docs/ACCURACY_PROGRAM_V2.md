# Baseball Research Lab Accuracy Program V2

**Status:** controlling engineering order for the research branch  
**Principle:** preserve the locked seven-outcome PA model; test every additional layer as a forecast-valid correction or downstream state model.

## Immediate defects verified in the current engine

1. `PlayerProfile.speed` defaults to `0.5`, so missing player speed is silently treated as identical average speed.
2. `TeamProfile.defense` and `TeamProfile.baserunning` default to `0.0`.
3. Runner advancement is governed by hand-selected transition coefficients, including `0.42 + 0.31 * speed` for a runner scoring from first on a double.
4. No between-PA stolen-base, caught-stealing, pickoff, wild-pitch, passed-ball, or balk process exists.
5. The automatic-runner option is configuration-only and ignores postseason `game_type`.
6. Fixture bullpen rest is explicitly set to `1.0` for all relievers.
7. Park and weather factors default to `1.0`; the locked PA provider uses its own historical park feature but no live weather correction.
8. The Flask route still allows an unvalidated ratings provider unless configuration is tightened.

These are higher-priority than a public-data ball-flight physics layer.

## Build order

### 0. Game picker and pregame snapshot pipeline

Build the product path required for prospective evidence:

- official schedule and game identifiers;
- probable starters and posted lineups;
- active roster and transactions effective before first pitch;
- roof status;
- plate umpire when posted;
- timestamped pregame weather forecast;
- saved forecast and immutable input receipt.

Start archiving these snapshots prospectively because historical pregame forecasts and exact posting times are not reliably reconstructable.

### 1. Real player speed, defense, and bullpen availability

- Sprint Speed as of the game date, shrunk toward position/league priors for low opportunity counts.
- Actual projected defensive lineup with prior-date OAA / Fielding Run Value.
- Pregame bullpen candidate set and rest from the previous five days.
- Never infer availability from target-game participants.

### 2. Empirical runner and between-PA state engine

Fit on 2023–2024 only:

`P(end base/out state, runs | start base/out state, PA outcome, outs, runner speed, fielder arm, game type)`

Use a hierarchical multinomial or equivalent shrunk transition model. Separately fit:

- stolen-base attempt;
- caught stealing conditional on attempt;
- pickoff;
- wild pitch;
- passed ball;
- balk.

A physical runner–throw race is a later candidate and must beat this empirical layer where it acts.

### 3. Recency-tempered empirical-Bayes talent candidate

Create PA v2 as a new lock, not an in-place modification. Estimate metric-specific decay rates and posterior uncertainty. Promote only after a complete chronological PA revalidation and full-game no-harm check.

### 4. Bullpen capacity and hook coupling

For reliever `r` before game `t`:

`fatigue_r = sum_{d=1..5} pitches_{r,t-d} * exp(-(d-1)/tau)`

`availability_r = sigmoid(alpha - beta*fatigue_r - gamma*consecutive_days_r)`

`BCI_t = sum_r availability_r * quality_r * expected_batters_r`

Starter-removal candidate:

`h(t|x) = h0(t) * exp(beta'x + gamma_BCI*BCI_t)`

Test the sign and incremental score improvement; do not assume it.

### 5. Per-path latent talent uncertainty

Draw player talent once per Monte Carlo path from the posterior distribution on the logit scale. Hypothesis: treating talent as known exactly contributes to winner overconfidence. Accept only if calibration improves without worse corrected Brier, corrected log loss, or run CRPS.

### 6. Recalibration and simulator/team blend

Use forward month cross-fitting within the development season. Report held-out corrected Brier and log loss. Freeze the selected calibrator and blend weight before prospective Scorebook use.

### 7. Expected-run center diagnostics

Rank residual miss by:

- park;
- weather;
- platoon;
- projected defense;
- catcher;
- starter context;
- bullpen state;
- travel/rest.

### 8. Small engine and frontier changes

Implement event-keyed random streams before small changes. Then consider shadow, travel, catcher/umpire, contact-vector, and park/weather modules.

## Oracle ladder

Before spending weeks on any layer, run a same-games/same-seeds diagnostic ladder. Each oracle is postgame information and is never a forecast.

1. actual lineup versus projected lineup;
2. actual starter batters faced versus manager model;
3. leave-one-game-out same-season realized player talent versus pregame talent;
4. actual bullpen order versus pregame bullpen pipeline;
5. observed weather versus archived pregame forecast, only where both exist.

Report corrected Brier, corrected log loss, and run CRPS changes. Use the ladder to rank engineering effort, not to claim forecast skill.

## Evaluation lanes

### Winner-moving asymmetric factors

Talent, lineup, starter quality/length, bullpen, travel, rest. Primary gates:

- corrected winner Brier;
- corrected winner log loss;
- calibration with uncertainty.

### Mostly symmetric run-environment factors

Weather, air density, roof, park carry. Primary gates:

- total-runs CRPS;
- team-run CRPS;
- randomized PIT;
- BIP-level proper scores.

A game-level winner result is a no-harm check, not the main acceptance metric.

## Data feasibility labels

- **Public and retrospective:** Statcast pitch/PA data, Sprint Speed, OAA, arm strength, schedules, play-by-play, rosters/transactions.
- **Public but prospective snapshot required:** forecast weather, posted lineup time, probable starter changes, roof status, umpire posting.
- **Unavailable publicly at required resolution:** per-ball batted-ball spin and complete per-play fielder starting coordinates.

Use average empirical carry conditional on EV, launch angle, spray, park, and air density rather than pretending per-ball lift is observed. Use OAA / Catch Probability and public alignment categories rather than invented coordinates.

## Naming and novelty

Use established names when applicable:

- conditional pitch bridge: Doob `h`-transform;
- probability residual tilt: KL-regularized exponential tilting;
- score-distribution calibration: minimum-KL / I-projection / maximum-entropy calibration;
- pitch tunnels: prior public baseball work exists.

The credible contribution is the integrated baseball application, pregame data boundary, and reproducible chronological validation. Any claim of original methodology requires a dedicated literature receipt.

## New hypothesis registry

- bullpen capacity index;
- starter hook–bullpen capacity coupling;
- shadow index from solar geometry and stadium orientation;
- travel-load decay with eastward time zones, day-after-night, and days since off day;
- starter context: rest, previous-start pitches, IL return, recent velocity change;
- runner send decision separated from safe/out execution;
- late-season clinched/eliminated state;
- separate postseason manager policy;
- opener and bulk-pitcher game state.

Each receives one short receipt: change, paired result, confidence interval, and `ACCEPT`, `REJECT`, or `INCONCLUSIVE`.
