# Decision horizon: when does a hitter stop using the ball's flight? (protocol, frozen before results)

Written 2026-10-08 01:55 ET, before any outcome from this experiment was looked at.

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
