# Decision horizon: results (October 8, 2026)

Protocol and every prediction, each committed before the run it governs: `discovery/DECISION_HORIZON_PROTOCOL.md`.
Code: `tools/brl_discovery.py` (experiments `horizon` to `horizon14`), workflow `brl-discovery`. Receipts on the ledger
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
| 37740908440 | Blind window or expectation pull (reviewer's strongest rival): split each pitch's miss from the pitcher's usual spot into movement surprise and the line it left the hand on | Movement part discounted 0.413 (a blind window from 261 ms predicts 0.467); line part -0.043 (predicted 0); difference 0.450 to 0.465. A pull toward an expected spot would discount both the same: rejected. By type: four-seam 0.46, sinker 0.50, changeup 0.42, curveball 0.33, slider 0.30 (sliders partly anticipated). Implied commit with pitcher-season intercepts: 245 ms. |
| 37738936962, 37741128823 | Tunneling scored at the measured horizon (2,104,443 consecutive pitch pairs) | The previous-pitch separation (visual angle from the batter's eye) best predicts a chase at 260 ms: 1.31 nats per 1,000 against 1.03 at the public 175 ms and 0.05 at the plate; head to head on 526,205 out-of-zone pitches, 260 ms beats 175 ms by 0.27 nats per 1,000 (0.22 to 0.32). Swings at strikes peak at the same moment. Whiffs get almost nothing from tunneling at any time (prediction for whiffs failed). |
| 37741959956, 37742304034 | Lever: within pitcher, does a change in a pitch's early separation from its usual predecessors change its chase rate (2,715 pitch types, consecutive seasons 2023-2026 pooled) | More separation at the decision moment, fewer chases: -0.51 points of chase rate per typical change (-0.74 to -0.27), with changes in its own speed, movement, zone rate and height held fixed; 175 ms alone -0.64 (-0.91 to -0.37). Together they cannot be separated (260: -0.03, -0.57 to 0.46; 175: -0.61, -1.20 to 0.02), so this test supports the lever's direction, not its timing. Whiffs: no relation. |
| 37738373322 | After deciding, how late is the bat steered (balls in play) | Launch angle rises 0.248 degrees per unit of up-and-down surprise (SE 0.003); sideways placebo 0.033. With an assumed 16 to 25 degrees per inch of bat-ball offset, the bat is steered until about 100 to 125 ms before the plate (111 ms at 20). If it were set at the decision, the slope would be 1.08 to 1.69. Balls in play exclude the largest misses, which biases this number short. |
| 37743322069 | Per-hitter steering limit (launch-angle slope on vertical surprise, 2023-2024, 206 hitters) | Repeats weakly: odd against even days 0.18, under the 0.2 set in advance. Bat speed -0.13 (the predicted sign), swing length -0.05. The apparent late steerers (Eugenio Suarez, Corey Seager, Aaron Judge, Ronald Acuna Jr.) are mostly power hitters who miss rather than mishit when fooled, which hides the effect in their balls in play. Dropped. |
| 37751569525 | Negative control: umpires' called strikes (1,477,048 taken pitches) through the identical instrument, heights standardized by each batter's zone recovered from the official zone numbers | Umpires: tau squared 0.0002 on 2025 (interval -0.0007 to 0.0014, about 15 ms) and -0.0012 on 2026; their held-out fit is best at the true crossing, and the hitters' value fits 19.6 nats per 1,000 calls worse. Their calls are 3.6 times sharper in location than swings, so tracking error would show more clearly in them. Hitters on the same pitches through the same code: 0.0659 and 0.0653 (257 and 256 ms). The horizon belongs to the hitters, not to the tracking system. Umpires are not fooled by late movement (a small positive for curveballs in 2025, 0.0057 with SE 0.0023, does not repeat in 2026); their calls follow the ball slightly past the front of the plate (plate-velocity term positive; its size is not recoverable). All three predictions held. |
| 37836217655 | Ball-strike challenges (2026) as a second decision about the same pitch: 370,808 called pitches; 3,468 batter challenges of called strikes, 4,294 catcher and 133 pitcher challenges of called balls (addendum 13) | Batters' decisions to challenge follow the crossing, not the 260 ms percept: tau squared 0.0044 (0.0009 to 0.0079), the held-out profile best at the crossing and 3.5 nats per 1,000 worse at the hitters' swing value; on synthetic pitches batters judging the crossing give 0.004 and batters judging from 260 ms 0.034. Catchers -0.0056 (-0.0081 to -0.0032), the value synthetic catchers on the crossing give (-0.005). Prediction 1 failed; prediction 2 held in size (below 0.015) but its interval excludes zero on the negative side, as the synthetic null does. Reading: the same hitters who commit their swings about 260 ms out judge where a taken pitch crossed as well as the catcher does (by watching it in or by seeing the catch; the data cannot tell which), so the horizon governs when the swing is committed, not what the hitter can know about the pitch. Overturned: batters 57.8%, catchers 55.3%, pitchers 39.9%; batters' challenges of pitches that moved late toward the zone (top third) are overturned 5.0 points less often than the bottom third (SE 2.1), not interpreted. |
| 37775965368 | Pitcher-level decision-moment tunneling: each pitcher-season's average 260 ms separation of back-to-back pitches of different types (1,340,514 pairs, 1,347 pitcher-seasons) against chase above expected (swings outside the zone beyond a league model of location, count, pitch type, speed and the batter's swing rate) | A stable trait (odd against even days 0.91, season to season 0.82). Measured on one half of the days, closer pairs go with more chases above expected on the other half, weakly: -0.209 points of chase rate per SD (-0.401 to -0.001), with plate separation, mix, fastball speed and rise, breaking sweep and zone rate fixed. Nothing for next season beyond this season's chase above expected (+0.06, -0.09 to 0.23): that prediction failed. 175 ms is indistinguishable (correlation 0.93), as predicted. A description of how an arsenal hides, not a forecasting number. Closest in 2026: Cam Schlittler, Jacob Misiorowski, Louis Varland; farthest: Sean Newcomb, Matt Boyd, Spencer Arrighetti, Framber Valdez. |

## Reading

Two clocks: the swing decision stops using the flight about 260 ms before the plate (about 33 ft out for a 92 mph
pitch), at a fixed time regardless of speed, count, pitch type or how often the hitter has seen the pitcher; the bat
keeps being aimed until roughly 100 to 125 ms. The 150 to 175 ms decision point that public tunneling and deception
metrics assume is neither moment. At the pitch-type level, decisions follow a gravity-only projection: a four-seamer's
last 5.4 inches of ride after 225 ms do not enter the decision, which is the rising fastball measured at league scale.

It is not a tracking artifact: umpires calling the same tracked pitches, through the same code, judge the true crossing (15 ms against the hitters' 257 ms), although their calls depend on location about 3.6 times as sharply as swings do.

What this is not: a per-pitcher tunneling forecast (stable, weakly tied to chases, nothing for next season), a per-hitter scouting number (redundant with chase rate) or a pitcher variability number
(wildness). Nothing here is causal about training. The flight is rebuilt, not tracked along its path; the zone is a
fixed box; the contact horizon rests on an assumed geometry.

## Instrument

`Decision Horizon` page (private artifact, built from the receipts): the flight at the commit point, every profile
and check, hitters (with the failure), pitchers (with the failure), the steering limit, tunneling at the right
moment, and a pitch pair explorer for 679 pitchers' 2026 arsenals (average flights, visual angle from the batter's
eye, against the decision and steering lines).

## Decision

Continue, with the target changed: the league-level measurement holds, including against the reviewer's strongest
rival and against tracking error (umpires on the same pitches show no horizon), and tunneling scored at it beats the field's 175 ms choice for chases; the hitter and pitcher uses we expected
are dropped. The lever test ran tonight: within pitcher, early separation moves chase rate in the predicted direction, but
season-to-season changes cannot separate 260 from 175 ms. Next experiment: a planned prospective test. Before the
2027 season, freeze the pitchers and pitch types whose spring shapes moved their decision-moment separation most, with
the predicted direction and size of each chase-rate change, and score it in May against a speed-and-movement-only
forecast and against the 175 ms version on pitchers where the two separations moved differently. Cheaper side step:
replace the assumed bat-ball geometry in the steering limit with Statcast's measured swing timing and miss distance.
