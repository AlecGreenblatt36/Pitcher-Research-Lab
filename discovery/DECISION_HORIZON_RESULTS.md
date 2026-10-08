# Decision horizon: results (October 8, 2026)

Protocol and every prediction, each committed before the run it governs: `discovery/DECISION_HORIZON_PROTOCOL.md`.
Code: `tools/brl_discovery.py` (experiments `horizon` to `horizon7`), workflow `brl-discovery`. Receipts on the ledger
branch: `research/discovery-<experiment>-<run>.json`. Data: MLB's public pitch-by-pitch feeds, 2023-2026 regular
seasons, 2,847,795 pitches (2,827,719 swing-or-take decisions after dropping bunts, pitchouts, intentional balls and
hit batters), processed inside Actions; only model results and per-player summaries left the runner.

## What was measured

Each pitch's flight is rebuilt with constant acceleration from 50 ft (speed loss from start and end speed, spin
acceleration from pfx over the last 40 ft, gravity), and the crossing a hitter would expect if he stopped taking in
the flight `tau` seconds before the plate is projected under different expectations. Swing decisions are modeled on
that projected crossing (count, pitch type, velocity, batter's prior swing rate held fixed) and the held-out fit is
profiled over `tau`. Rebuild checks: median flight from 50 ft 0.383 s; four-seam spin lift 17.7 ft/s^2 (expected 15
to 18); 25.7% of pitches at 0-0 (counts are pre-pitch).

## Results

| Run | Question | Result |
|---|---|---|
| 37734449007 | Which commit time explains swings, one shared location map | Gravity-only projection from 225 ms beats the true crossing by 40.3 nats per 1,000 decisions on 2026 (log loss 0.5062 to 0.4659); expecting the pitcher's fastball: 24.5 at 200 ms; knowing the type: 4.0 at 250 ms. Shuffling each pitch's own movement within its pitcher-type keeps 32.8 of 40.3: mostly a pitch-type-level effect. |
| 37736387462 | Decisive: separate location map per pitch type, only the pitch-by-pitch movement surprise | Direct estimate tau = 262 ms on 2026 (tau squared 0.0684, 95% interval 0.0664 to 0.0702), 260 ms on 2025, 259 ms on primary fastballs alone. Profile improves to 4.6 nats per 1,000 at 250 ms. The frozen rule asked for 100 to 250 ms: strong effect, just outside the window, recorded as not a clean pass. |
| 37737624724 | Mechanism checks (predictions written first) | Sideways 258 ms, up-and-down 264 ms. 79 to 97 mph: 254 to 266 ms with no trend, so a fixed time before arrival, not a fixed distance. Counts 258 to 265 ms; innings 259 to 264; pitch types 248 to 279; 2025 and 2026 260 and 262. Speed surprise barely matters (0.018 log-odds per mph). Long profile peaks at 250 to 300 ms and turns back by 400 ms. All as predicted. |
| 37737930311 | Does facing a pitcher again move it (times through the order) | No: 259, 263, 268 ms the first, second and third time; flat with pitch-type exposure. Prediction (falls with exposure) failed. Hitters do not read late movement better as the game goes on. |
| 37736961636 | Per-hitter horizon: new information or chase rate restated | Very repeatable (odd against even days 0.84) and sorts hitters as scouts would (latest: Mike Trout 149 ms, George Springer, Lourdes Gurriel Jr., Juan Soto 166 ms; earliest: Javier Baez and Elly De La Cruz at the 300 ms grid edge, Adam Duvall, Oneil Cruz). But measured on 2023-2024 it adds nothing to 2025-2026 strikeout, walk, chase or breaking-ball whiff rates beyond 2023-2024 plate-discipline numbers (R squared changes in the fourth decimal, intervals span zero). Bat speed correlates +0.13 (the protocol predicted negative). Dropped as a scouting number. |
| 37738096684 | Pitcher side: does more shape surprise win decisions | Median 1.1 in of unexpected movement per pitch after the horizon, stable year to year (0.75). No effect on future chase or whiff; more walks (+0.32 pts per SD, interval excludes zero) and fewer pitches in the zone. It is wildness, not deception, exactly the pattern the pre-run rule named. Dropped. |
| 37738373322 | After deciding, how late is the bat steered (balls in play) | Launch angle rises 0.248 degrees per unit of up-and-down surprise (SE 0.003); sideways placebo 0.033. With an assumed 16 to 25 degrees per inch of bat-ball offset, the bat is steered until about 100 to 125 ms before the plate (111 ms at 20). If it were set at the decision, the slope would be 1.08 to 1.69. Balls in play exclude the largest misses, which biases this number short. |

## Reading

Two clocks: the swing decision stops using the flight about 260 ms before the plate (about 33 ft out for a 92 mph
pitch), at a fixed time regardless of speed, count, pitch type or how often the hitter has seen the pitcher; the bat
keeps being aimed until roughly 100 to 125 ms. The 150 to 175 ms decision point that public tunneling and deception
metrics assume is neither moment. At the pitch-type level, decisions follow a gravity-only projection: a four-seamer's
last 5.4 inches of ride after 225 ms do not enter the decision, which is the rising fastball measured at league scale.

What this is not: a per-hitter scouting number (redundant with chase rate) or a pitcher variability number
(wildness). Nothing here is causal about training. The flight is rebuilt, not tracked along its path; the zone is a
fixed box; the contact horizon rests on an assumed geometry.

## Decision

Continue, with the target changed: the league-level measurement holds; the hitter and pitcher uses we expected are
dropped. Next experiment: re-score consecutive pitch pairs at the measured horizons (separation at 260 ms, at 175 ms
and at the plate) and test which predicts chase and whiff on the second pitch beyond its stuff and location, fitted
on 2023-2024 and scored on 2025-2026. If 260 ms wins, the field's deception metrics are measured at the wrong moment
and the corrected version is a pitch-design tool.
