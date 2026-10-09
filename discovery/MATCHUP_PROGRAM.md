# Matchup physics: research program (from October 8, 2026)

Objective: a predictive method for what happens when a specific hitter meets a specific pitch, built from geometry,
timing and perception, that predicts outcomes the standard models misjudge, survives tests designed to break it, and
changes what the simulator predicts and what a scout can act on.

## What the strongest existing work assumes

- **Pitch-quality models** (Stuff+, Pitching+, PitchingBot) score a pitch from its own physics, location and count. The
  hitter is generic: one pitch has one value against everyone. Flexible learners can find any function of a pitch's own
  physics, so a new transform of those same inputs is not new information; new information has to come from the hitter,
  the sequence or the pitcher's state.
- **Matchup models** combine a batter's and a pitcher's outcome rates (log5 or odds ratio, hierarchical Bayesian
  versions; e.g. arXiv 2511.17733, PMC6192592) or learn player embeddings from outcomes (batter2vec, pitcher2vec). They
  work at the plate-appearance level and have no mechanism for why a pairing deviates from the two averages.
- **Swing-process work with bat tracking** (Baseball Prospectus, 2025: swing initiation from early, uncertain
  information; tilt and intercept as "under" and "ahead") models swing decisions and timing, not whiffs or contact
  quality, and leaves individual differences to future work.
- **Our Decision Horizon results** (discovery/DECISION_HORIZON_RESULTS.md): swing decisions use the flight up to about
  260 ms before the plate, the bat is steered until about 100 to 125 ms; per-hitter and per-pitcher versions of these
  times added nothing to forecasts.

## Candidate mechanisms

**M1. Contact tolerance (swing plane against pitch plane).** In the vertical plane through the plate, the barrel's
sweet spot moves toward the pitcher at speed v_s and attack angle AA; the ball arrives at speed v_b, descending at
approach angle VAA. Let theta = AA - |VAA| (zero when the barrel travels along the ball's line). If the swing is
early or late by dt, the closest approach of barrel and ball is

    d = u * |sin(theta)| * |dt|,   u = v_s * v_b / |v_s - v_b|  (about 55 to 60 ft/s),

and the meeting point moves along the flight by u * dt. So a hitter's spread of contact depth (Savant's intercept point,
toward the pitcher), sigma_y = u * sigma_t, measures his timing spread in the same units, and the vertical miss caused
by timing has spread |sin(theta)| * sigma_y, with no unknown speed. Adding the aiming error e_z (mean m, spread
sigma_e; it includes the late break he cannot see after his steering limit, the Decision Horizon result), the chance of
meeting the ball within h (ball and barrel radii, about 2.75 in for any contact) is

    P(contact | swing) = Phi((h - m) / s) - Phi((-h - m) / s),   s^2 = sigma_e^2 + (sin(theta) * sigma_y)^2.

Testable consequences: (a) whiffs on swings rise with |sin(theta)| * sigma_y, beyond the hitter's whiff rate and the
pitch's own quality; (b) at fixed |theta|, hitters with a wide contact-depth spread suffer more; (c) uppercut swings
miss flat fastballs at the top of the zone and level swings miss steep breaking balls more than their averages say;
(d) for a 10 degree mismatch a 10 ms timing error costs about 1.2 inches, for 3 degrees 0.35 inches, so the plane
match should matter most against pitches hitters time poorly. Prediction comes from each hitter's usual attack angle
and contact-depth spread on earlier swings (the angle on the swing itself is partly a reaction to the pitch, so it
tests the mechanism, not the forecast). Inputs: Savant bat tracking per swing and the fitted flight of each pitch
(approach angle at the contact depth). Baselines: the hitter's whiff rate, the pitch's own whiff value from a flexible
model of its physics, location and count, and both together.

**M2. Matchups seen at the decision moment.** Each hitter swings according to where the pitch appears to be headed at
his decision moment, not where it ends; a pitcher's arsenal is a distribution of those appearances. Represent the hitter
as a swing function over the decision-moment picture and the pitcher as a distribution over it; their overlap predicts
chases and called strikes for that pairing. Baseline: the same league swing map with additive hitter and pitcher
terms (the pitch-level log5). The interaction term is the test.

**M3. The pitcher's state inside the game.** Velocity, spin and release height drift as a start goes on. If drift
against the pitcher's own early-game level predicts the next hitters' results beyond pitch count and times through
the order, it measures fatigue before results show it, and it belongs in the live simulator and its hook model.

**M4. Sequence and expectation.** What the hitter has just seen sets his expectation for the next pitch; the decision
moment results already show previous-pitch separation at 260 ms predicting chases. A count- and sequence-conditioned
expectation model scores how surprising each pitch is to that hitter at that moment.

## How a result earns its place

1. Each experiment is registered in LEDGER.md before it runs: the mechanism, the competing explanations, the
   prediction that would distinguish them, and the decision rule.
2. **Untouched evaluation set for this program, fixed now:** every pitch of the 2026 regular season from August 1 on and
   every 2026 postseason pitch. No model in this program is fitted, tuned or inspected on it. A candidate is frozen
   (code commit and parameters) before it is scored there, once; a second look requires a new registration that says
   so. Development uses 2023 to 2025 and 2026 through July 31. The prospective test is 2027.
3. Pitch locations are put on one reference before seasons are pooled (2026 plate_x and plate_z are at the middle of
   the plate; ZONE-01 and the Savant reference check measure it).
4. A mechanism is credited only for what the flexible baseline with the same inputs cannot do: predicting held-out
   hitter-pitch combinations, needing fewer parameters, or extrapolating to pitch shapes and hitters it was not fitted
   on.
5. What survives goes into the simulator as a measured layer and is replayed like every other change; what fails is
   recorded with what it rules out.

## First steps

- DATA-02: Savant pitch-by-pitch with bat tracking and fitted flights, 2024 to 2026 regular seasons, sealed on the
  ledger branch; coverage and the plate-reference check in the receipt.
- ZONE-01: the definition change in the feed's own locations (running).
- M1 first test after DATA-02, M3 on the feed data in parallel.
