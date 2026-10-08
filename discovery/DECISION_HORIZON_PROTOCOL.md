# Decision horizon: when does a hitter stop using the ball's flight? (protocol, frozen before results)

Written 2026-10-08 and committed at 01:50 ET (commit 70807e93), before any outcome from this experiment was looked at.

## The question

Every public deception and tunneling measure assumes one fixed moment when the hitter commits (Baseball
Prospectus used 175 ms before the plate in 2017 and 150 ms in 2018; their 2025 Pitch Type Probability keeps a
fixed decision point). Nobody has measured that moment from game data, and nobody knows whether it differs from
hitter to hitter. If it can be measured, it is a new scouting quantity (how late a hitter can wait), a pitch-design
target (where a pitch has to separate from its look-alikes for this hitter) and a check on every tunneling metric
built on the assumed number.

## Idea in one line

A swing decision can only use what the hitter saw before he committed. Rebuild each pitch's flight, project where
the hitter would have expected it to cross the plate if he committed `tau` seconds before it arrived, and find the
`tau` at which his swing decisions are best explained. If hitters really commit before the plate, the projected
location at the right `tau` explains swings better than the true location does.

## Data (inside Actions only; nothing pitch-level leaves the runner)

Sealed pitch-by-pitch seasons 2023-2026 from MLB's public game feeds (private/statcast/season-<year>.enc): pitch
type, call code, count, start and end speed, spin, pfx movement, plate location, position at y = 50 ft,
extension. Regular season only. Bunts, pitchouts, intentional balls and hit batters are dropped.

## Measurement: rebuilt flight (reconstructed, not measured)

Constant-acceleration flight from y = 50 ft to the front of the plate (y = 17/12 ft): the y deceleration from start
and end speed, the flight time from those, spin accelerations from pfx over the last 40 ft, gravity, and the initial
x and z velocities solved so the flight ends at the measured plate location. Checked on the runner: rebuilt flight
times and the implied release distance against extension.

Projected crossing at commit time `tau` (seconds before the plate): position and velocity at commit, carried to the
plate with an expected acceleration. Four expectations (observer models):

- O1 straight: gravity only (no spin).
- O2 fastball: the pitcher's average fastball acceleration (his most-thrown of four-seam, sinker, cutter that season).
- O3 own type: the pitcher's average acceleration for this pitch's type that season (the hitter recognized the type).
- O4 mixture: a posterior over the pitcher's types from position and velocity at commit (Gaussian per type) and his
  usage of each type in that count to that batter hand; the projection is the posterior-weighted mix.

At `tau = 0` every observer gives the true location.

## Primary test (decisive)

Swing or take, logistic regression. Controls held fixed across `tau`: count (12), pitch type group (fastball,
sinker, cutter, slider or sweeper, curveball, changeup or splitter, other), velocity (spline), the batter's swing
rate on earlier dates (shrunk), batter hand. Location enters only through the projected crossing: splines of signed
distance to the rulebook zone edge (approximate fixed zone, 1.5 to 3.5 ft high, 0.83 ft half width), of the
inside/outside coordinate and of height, each by strikes (0, 1, 2).

- Fit on 2023-2025 (a random half million pitches per season), score on all 2026 pitches (log loss per pitch).
- Profile: `tau` in {0, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30} for each observer.

Reading:

- Supports a commit point: the 2026 log loss at some `tau > 0` is lower than at `tau = 0` by more than the
  bootstrap interval (game-clustered), with a maximum between 0.10 and 0.25 s.
- Against: the profile is lowest at `tau = 0` (decisions follow the true location; the hitter effectively sees the
  whole flight, or the rebuild is too noisy to tell).
- Which observer wins at its best `tau` tells what hitters expect (straight, fastball, the right type, or a mix).

## Checks that could show the idea is wrong

1. Shuffle: give each pitch the spin acceleration of a random pitch of the same pitcher, type and season. The
   projections keep their spread but lose their link to the pitch actually thrown. If the gain at `tau > 0` survives
   the shuffle, it is not about late movement.
2. Both: actual location and projected location together. At the true `tau` the projected terms should keep weight
   when the actual location is also in the model.
3. Split decisions: chases (true location outside the zone) and takes of strikes (inside) fitted separately should
   point to similar `tau`.

## Per-hitter horizon (secondary, only if the primary test supports a commit point)

With the global model fixed at its best observer and `tau`, each hitter's own log likelihood over a fine `tau` grid
(0 to 0.30 s by 0.025) on his 2023-2025 pitches; his horizon is the maximum of a smoothed profile. Reliability:
odd against even game dates. External check that the fit never saw: Statcast bat speed and swing length
(public bat-tracking leaderboard, 2024-2025). Prediction written now: faster bats and shorter swings go with later
commits (smaller `tau`). No correlation, or the opposite sign, counts against a hitter-level reading.

## What would not count

A better fit at `tau > 0` that disappears under the shuffle; a best `tau` above 0.30 s (before the ball is
halfway); per-hitter horizons with split-half reliability under 0.2.

## Second track (if time): fatigue or familiarity

Times-through-the-order split with measured pitcher physical change against his own baseline, pitch-type exposure
and batters seeing a starter for the first time late in his outing. Prior work (Baseball Prospectus 2021-2025,
Brill et al. 2023) covers much of it, so it is second.

## Addendum, committed 02:12 ET (90a5ddd7), before any real-data outcome was seen (reviewer's challenge)

The reviewer's strongest alternative: with one location map shared by all pitch types, the fastball observer
slides each off-speed type's swing map by a fixed amount, so a type-by-location difference in swing behavior
(breaking balls below the zone chased more, for any reason) shows up as a best tau above zero with no commit at
all. Synthetic check: when hitters use the true location but swing at breaking balls as if 3 inches higher, the
shared-map profile has a false minimum at 0.10 s.

Added tests (experiment 'horizon2'), which become the decisive ones:

- Selection on 2025, confirmation on 2026 (the first run picked its best cell on 2026).
- Baseline with a separate location map for each pitch type group.
- Within-type surprise only: the pitch's spin acceleration minus the pitcher's usual for that exact pitch type that
  season (leave-one-out), projected over tau, with type maps. A type-level effect cannot produce it.
- Primary fastballs only (fastball observer and own-type observer coincide).
- Direct estimate of tau squared: the type-map model fixed at the true location, then the coefficient on the
  displacement along its decision gradient, with a game-clustered interval. Under no commit it is zero.

Synthetic recovery: planted 0.17 s commit gives 0.177-0.184 s (interval excludes zero); the null with the
type-by-location effect gives an interval that includes zero for the within-type estimate while the between-type
estimate is spuriously positive. Decision rule: a commit is supported only if the within-type estimate's 2026
interval excludes zero and its point estimate falls between 0.10 and 0.25 s.

## Addendum 2, committed 02:26 ET (f1fc7503), after the within-type result and before these checks ran

Result of the decisive test (run 37736387462): the within-type surprise moves swing decisions with tau = 0.262 s on
2026 (tau squared 0.0684, 95% interval 0.0664 to 0.0702), 0.260 s on 2025, 0.259 s on primary fastballs alone.
The interval excludes zero by a wide margin, but the point estimate is just above the 0.25 s ceiling written above,
so by the frozen rule this is not a clean pass; it is recorded as such.

Mechanism checks that follow from the extrapolation reading (experiment 'horizon4'), with predictions written now:

- Horizontal and vertical surprises fitted separately give the same commit time (within about 0.03 s). A generic
  "nastier pitch" effect has no reason to scale the same way in both directions.
- By pitch speed: a hitter who commits a fixed time before arrival shows the same tau at 80 and 95 mph; a hitter
  who commits at a fixed distance shows tau falling as speed rises (tau times speed constant). Either result is
  informative; no prediction about which.
- A faster-than-usual pitch at the same place (speed surprise) should not shift swing decisions much once the
  movement surprise is in; a large speed effect would point to an effort or intent confound.
- The long profile (to 0.40 s) should turn back up past the commit time; a fit that keeps improving to 0.40 s
  (before the ball is halfway) would mean the projection is tracking something other than a commit, such as
  where the pitcher aimed.

## Addendum 3, before the familiarity run (horizon5)

Question: does facing a pitcher again change the horizon? This bridges to the times-through-the-order debate.
Within-type commit time (type maps) and type-level projection reliance (one shared map, gravity-only projection),
by the number of earlier plate appearances against this pitcher today (0, 1, 2, 3+) and by how many pitches of
this exact type the batter has already seen from him today. Written before the run: if hitters learn a pitcher,
both measures fall with exposure (later commit, less reliance on the projected path); if the times-through-the-order
penalty is the pitcher tiring, they stay flat. Selection (only better starters reach a third time through) is a
known weakness of the times-faced split; the pitch-type exposure split within a first plate appearance is cleaner.

## Addendum 4, before the pitcher-side run (horizon6)

The hitter-level horizon failed its incremental test (horizon3: no gain over 2023-2024 plate-discipline numbers for
any 2025-2026 outcome). Pitcher side: since hitters cannot use a pitch's movement surprise after the horizon, a
pitcher whose pitches vary more around their own average shape may win more swing decisions. Measure per pitcher on
2023-2024: usage-weighted spread of within-type spin acceleration, as inches of surprise at 0.26 s. Test against
2025-2026 strikeout, walk, chase, whiff and zone rates with his 2023-2024 rates, fastball velocity, fastball ride and
overall movement as controls. Written before the run: a useful pitcher number needs a bootstrap interval for its
effect that excludes zero on chase or whiff and a lower leave-one-out error. A positive effect on walks with no gain
on chase or whiff would mean the spread is wildness, not deception.

## Addendum 5, before the contact-horizon run (horizon7)

Results so far (runs 37737624724 and 37737930311): horizontal and vertical surprises give the same commit time
(0.258 and 0.264 s); the commit time is the same from 79 to 97 mph (0.254 to 0.266 s), so it is a fixed time before
arrival, not a fixed distance; the same in every count, inning and season; the long profile peaks at 0.25 to 0.30 s
and turns back by 0.40 s. Familiarity: the within-type horizon does not fall with times faced (0.259, 0.263, 0.268)
or with pitch-type exposure, so the prediction that hitters learn a pitcher's late movement within a game failed.

Next question: after deciding to swing, how late can the hitter still steer the bat? Launch angle on balls in play
against the pitch's vertical movement surprise; the implied contact horizon depends on an assumed bat-ball geometry
(k degrees of launch angle per inch of offset, reported for 12 to 25). Written before the run: a two-stage hitter
(decide early, steer later) shows a contact horizon clearly shorter than 0.26 s for any k in 16 to 25; a contact
horizon equal to or longer than the decision horizon would mean the bat path is set when the swing decision is made.
The horizontal surprise is a placebo for launch angle and should give a slope near zero.

## Addendum 6, before the tunneling run (horizon8)

Contact horizon result (run 37738373322): launch angle follows the vertical surprise with a slope implying the bat is
steered until about 100 to 125 ms (assumed geometry), against 1.08 to 1.69 if it were set at the decision; prediction
met. Per-hitter and pitcher-variability uses failed (DISC-02).

Tunneling test: for each pitch following another in the same plate appearance, the visual-angle separation between
the two flights, seen from the batter's eye, at the same time before each reaches the plate; separation entered as a
flexible term on top of the second pitch's own type, speed, movement and location, the first pitch's type and
location, the count and the batter. Fit 2023-2024, score 2025-2026, profile over the time. Predictions written now:
for chases the most useful separation is near 260 ms, clearly earlier than the 150-175 ms used publicly; for whiffs
the useful separation is later (nearer the 100-125 ms steering limit). If both peak at 150-175 ms the public choice
stands; if separation adds nothing at any time beyond the pitches themselves, pairwise tunneling carries no
information our pitch-level model does not already have.

## Addendum 7, before the blind-window test (horizon9)

Reviewer's strongest remaining alternative after the mechanism checks: expectation pull. Hitters' calls could be
pulled toward where they expect this pitcher to put this pitch in this count; the within-type surprise is correlated
with the miss from that spot, so "decides as if it moved normally" could appear with no blind window at all. Test
(reviewer's design): split each pitch's miss from the pitcher's usual spot (same pitch type, batter side, count
bucket, season, leave-one-out) into the movement-surprise part over the flight from 50 ft and the rest (the line it
left the hand on); both as displacements along the type-map gradient with pitcher-season intercepts, on 2025-2026.
Predictions written now: a blind window discounts the movement part by about (0.261 / t_f)^2, near 0.45, and the line
part near zero; expectation pull discounts both by the same amount. Synthetic check with a planted blind window: 0.186
for movement (0.193 expected) and 0.009 for the line.

## Addendum 8, before the within-pitcher change test (horizon11)

Results since addendum 7: blind window confirmed (movement discounted 0.413 against 0.467 predicted, line -0.043);
tunneling separation predicts chases best at 260 ms and beats 175 ms head to head by 0.27 nats per 1,000 (0.22 to
0.32); tunneling adds nothing for whiffs.

Lever test: for each pitcher's pitch type thrown at least 150 times in both 2025 and 2026, its average flight's
visual-angle separation from the flights of the pitches that usually precede it (weighted by how often), at 260 ms,
at 175 ms and at the plate; change from 2025 to 2026 against the change in its chase rate, with changes in its speed,
movement, zone rate and height as controls, weighted by pitches, intervals by resampling pitchers. Predictions written
now: more separation at the decision moment means fewer chases (negative coefficient), and the 260 ms change predicts
better than the 175 ms change. Power is the main risk: most pitchers change little, so an interval spanning zero is
the likely outcome if the effect is small, and would be reported as not established rather than as no effect.
