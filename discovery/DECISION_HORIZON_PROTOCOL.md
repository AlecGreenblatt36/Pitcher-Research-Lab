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

## Addendum 9, pooling all season pairs (same test, more power)

The 2025 to 2026 change test (run 37741959956, 927 pitch types) supported the direction (more separation at 260 ms,
fewer chases: -0.040 per degree, interval -0.069 to -0.008) but could not separate 260 from 175 ms (both negative
alone; in one model neither interval excludes zero). The same frozen specification is now run on 2023-2024,
2024-2025 and 2025-2026 changes pooled, with intervals resampling pitchers. Same predictions as addendum 8.

## Addendum 10, before the per-hitter steering run (horizon12)

Pooled change test (run 37742304034, 2,715 pitch types over three season pairs): direction supported (-0.51 points of
chase per typical change at 260 ms, -0.74 to -0.27), timing not separable from 175 ms by this design.

The second clock per hitter: each hitter's slope of launch angle on the vertical movement surprise, 2023-2024 balls
in play (at least 500), shrunk. Predictions written now: split-half reliability of at least 0.2 or it is not a
measurement; faster bat speed goes with a smaller slope (steering later) and longer swings with a larger one; and it
counts as new information only if, beyond the 2023-2024 values, it improves a 2025-2026 contact outcome (whiffs per
swing, sweet-spot rate, hard-hit rate, launch-angle spread) with an interval that excludes zero.

## Addendum 11, before the umpire run (horizon13)

Result since addendum 10: per-hitter steering limit failed its reliability bar (0.18 against 0.2; dropped).

Negative control for the main finding. If the 260 ms horizon came from tracking error rather than perception (the
measured crossing and the measured movement erring together, so that subtracting part of the movement surprise
cleans the location), any decision made on the true crossing would show the same positive tau squared through the
same estimator. Umpires' calls on taken pitches are such a decision: the umpire watches the ball to the glove. The
same instrument as the decisive test (pitch-type location maps, within-type movement surprise, displacement along
the decision gradient) is applied to called strikes, with heights standardized by each batter's own zone (recovered
from the official zone numbers) and count-specific maps including three-ball counts; hitters' swings go through the
identical code for comparison. Trained on 2023-2024, measured on 2025 (primary) and 2026 (secondary: 2026 calls may
include challenge reviews).

Predictions written now:
1. Umpires: within-type tau squared below 0.01 on 2025 (at most a seventh of the hitters' 0.0684). A negative value
   is expected if calls follow the ball past the front of the plate toward the glove: on synthetic pitches, 20 ms of
   travel past the plate gives -0.005 to -0.008. At 0.034 or above (half the hitters' value) the hitters' horizon
   cannot be separated from tracking error and the finding is withdrawn until it can.
2. Hitters through the same code: tau squared between 0.05 and 0.09.
3. Umpires' held-out profile is best at tau squared of 0.01 or less, and the hitters' value (0.0684) fits worse than
   the true crossing by at least 5 nats per 1,000 calls.
The plate-velocity term is reported but not interpreted: on synthetic pitches it recovers 0.002 s for a planted
0.020 s (the type maps absorb it), so the glove shows up in the surprise term instead.
Synthetic recovery (tools/brl_discovery.py horizon13; 335,000 pitches): umpires on the true crossing 0.0003 and
-0.0007 (SE 0.001); umpires anticipating 120 ms (0.0144 planted) 0.0136 and 0.0121; hitters at 260 ms (0.0676
planted) 0.061 to 0.064; batter zone tops and bottoms recovered (league 3.393 and 1.614 ft against 3.393 and 1.611).

## Addendum 12, before the pitcher-level tunneling run (horizon14)

Result since addendum 11: umpires judge the true crossing (tau squared 0.0002 on 2025 against the hitters' 0.0659
through the same code); the horizon is not a tracking artifact.

Can the pitch-level tunneling result become a scouting number for pitchers? For every consecutive pair of different
pitch types in a plate appearance, the visual-angle separation of the two flights from the batter's eye 260 ms and
175 ms before each reaches the plate and at the plate. A pitcher-season's decision-moment tunneling is the average
260 ms separation of his different-type pairs. Outcome: chase above expected (swing rate on pitches outside the zone
minus a league model's expectation for those pitches from location, count, pitch type, speed and the batter's prior
swing rate, fitted on 2023-2024). Pitcher-seasons with at least 400 pitches outside the zone and 300 different-type
pairs; halves by odd and even days need half of that.

Predictions written now:
1. The 260 ms separation is a stable pitcher trait: odd against even days at least 0.8, season to season at least 0.6.
2. Cross-sample validity: separation measured on one half of the days against chase above expected on the other half,
   with plate separation, the share of different-type pairs, fastball speed and rise, breaking-ball sweep and zone rate
   held fixed: closer pairs at 260 ms go with more chases above expected (negative coefficient, interval excluding zero).
   If not, the pitch-level effect does not make a pitcher-level number and it is dropped.
3. 260 against 175 ms cannot be decided at this level: the two correlate about 0.9, and on synthetic pitches an effect
   planted on release-point spread alone makes the earlier moment win with no decision moment involved. Both are
   reported; neither is read as timing.
4. Next season: beyond this season's chase above expected and the same controls, the 260 ms separation predicts next
   season's (negative coefficient, interval excluding zero). The R squared gain is reported but cannot test this (its
   interval is bounded at zero).
Pipeline check on synthetic pitches (planted effect of 0.25 log-odds per SD of release closeness): cross-sample
coefficient -1.22 points of chase per SD (-1.62 to -0.68); null scenario 0.02 (-0.15 to 0.25); next season -0.25
(-0.49 to -0.01) planted, 0.08 (-0.06 to 0.22) null.

## Addendum 13, before the challenge run (challenges)

Since addendum 12: pitcher-level tunneling is stable but descriptive only (DISC-06).

In 2026 batters, catchers and pitchers can challenge the umpire's ball or strike, and the official play-by-play records
each challenge on the pitch with who made it and whether the call was overturned (probe run 37835384951: 36 player
challenges on pitches in 12 late-September games). A challenge is a second decision about the same pitch, made after the
ball is caught, by people who saw it from different places. If hitters' perception of the pitch freezes about 260 ms
before the plate (DISC-01), a batter's decision to challenge a called strike should follow where the pitch looked from
there, not where it crossed. A catcher catches the ball and sees where it ends: his challenges of called balls should
follow the crossing, like the umpires' calls (DISC-05).

Instrument: the umpire-test code (pitch-type location maps with each batter's own zone, edge distance, count, inning,
pitch type, handedness and the challenging side's challenges left as controls); the decision is displaced along the
within-type late movement and tau squared estimated with the offset-logit gradient, cross-fitted on even and odd games
(every 2026 regular-season game, fetched on the runner; aggregates only leave it).

Predictions written now:
1. Batters (called strikes, outcome: the batter challenged): tau squared above zero with the interval excluding zero,
   between 0.02 and 0.06, and the held-out profile better at 0.02 or 0.0684 than at the crossing.
2. Catchers (called balls, outcome: the catcher challenged): the interval includes zero and the estimate is below 0.015.
3. Pitchers (called balls): no prediction; reported.
The overturn rate by late movement toward the zone is reported for each role but not predicted (on synthetic pitches its
sign depends on the selection of taken pitches as much as on perception).
Synthetic recovery (tools/brl_discovery.py challenges; 1,200 games, about 300 to 400 challenges per role): batters
judging from 260 ms (0.0676 planted) 0.034 (0.027 to 0.042) — the selection of taken pitches, which hitters also choose at
260 ms, halves the estimate; batters judging the crossing 0.004 (-0.006 to 0.015); catchers on the crossing -0.005
(-0.015 to 0.006) in both runs; pitchers at 150 ms (0.0225 planted) 0.032 (0.011 to 0.052). So the test separates the two
readings for batters (0.034 against 0.004) though it does not recover the full 0.0676.
