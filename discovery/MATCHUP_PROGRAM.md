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
2. **Evaluation sets.** Development: 2023 to 2025 and 2026 through July 31. Later-period validation: the 2026 regular
   season from August 1 (set aside as untouched on October 8; after the pricing correction of October 9 it counts as
   validation with adaptive reuse, since VALUE-08F re-scored months VALUE-01F and VALUE-02F had used). Forward
   confirmation: the 2026 postseason and 2027, with candidates frozen before any of their games and a predeclared
   stopping rule. A second look at any block requires a registration that says so.
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

## Results through October 9, 2026, 12:30 p.m. Eastern (ledger rows carry every number and interval)

Credited (frozen code, scored once per candidate on the later-period validation months, August 1 to September 27, 2026;
see evidence rule 6 on their reuse). The decision-relevant comparisons are the ones against a boosted learner with the
same inputs (BENCH-01F, BENCH-02F); the aiming value is estimated policy potential under stated assumptions, with an
adversarial stress test of its estimator registered (VALUE-17) before the figure is used again:

- **Swing decisions are read at the decision moment** (MATCHUP-01F). A hitter's own swing map in the coordinates of
  where the pitch appears to be headed 260 ms before the plate (gravity-only projection) predicts his swings 40.2 nats
  per 1,000 decisions better than his standard heat map of where pitches cross the plate, and predicts his chasing
  against a specific pitcher's arsenal with a calibration slope of 0.95.
  That gain was measured against the standard heat map inside a logistic model. Against a gradient-boosted learner
  given the same pitch inputs (BENCH-01), most of it belongs to the league part, which boosting learns from the raw
  inputs (it beats our logistic model by 23 nats); the hitter-specific maps read at the decision moment still add
  about 3 nats per 1,000 decisions beyond boosting with the heat map, in 2025 and in 2026 through July, and +3.6
  [3.2, 4.2] on the untouched months (BENCH-01F, credited). The best swing model is boosting with both maps as inputs.
  For misses the same test credits each hitter's whiff map on the true crossing at +8.2 nats per 1,000 swings beyond
  boosting with his whiff rate (BENCH-02F), with the decision moment adding nothing to misses: the two clocks hold
  against a flexible baseline. The program's swing and miss models are boosting with the maps as inputs.
- **Misses and contact follow where the ball arrives** (MATCHUP-03, CONTACT-03, DAMAGE-01F). The spread of a hitter's
  contact depth from bat tracking is his contact window (contact depth is only seen on contact, so the spread is the
  range of timing he survives, not his timing error) and predicts misses beyond his whiff rate; damage on contact is
  better described on the true crossing than at the decision moment. Hitter-specific damage maps were not confirmed.
- **Sequence effects in timing** (TIMING-03F). After a called strike the next pitch is met farther out front the
  slower that strike was (0.15 inches per 10 ms); after a miss, later (0.24); a foul carries nothing.
- **Sequencing reaches the outcome** (SEQ-03F, credited on untouched swings): after a taken pitch, and about twice
  as much after a miss, coming back with the same pitch family draws more misses than changing it, beyond what the
  pitch itself predicts (about +2 to +3 points of whiff rate; +0.11 to +0.22 log-odds), and softer contact (-0.4 to
  -0.7 mph); a slower next pitch adds to it; a foul resets it. The registered mechanism (the timing carried after a
  called strike, SEQ-02) was not what reaches misses: the effect is the same after a taken ball.
- **Hitters answer pitch families differently at the same apparent spot** (MATCHUP-04F, MATCHUP-05F): maps with a
  family part beat location-only maps on the untouched months (+3.9 nats per 1,000 decisions); about half is a
  league-wide family pattern, and hitter-specific family parts add +2.2 [1.8, 2.5] beyond a league model that has it.
- **An unexploited edge, priced on each hitter's own part** (EXPLOIT-01, VALUE-08F, VALUE-09). Pitchers do not aim at
  a hitter's decision-moment chase spots. Hitters' maps share a shape that predicts run value for anyone; the first
  pricing (VALUE-01F, VALUE-02F) mixed it in and was withdrawn when a fresh placebo failed. Each hitter's own part (his
  map minus the average same-side map) passes the placebo, holds among pitches at the same spot and with the plate
  appearance's other pitches fixed. Aiming each outside pitch at the best third of the pitcher's own spots of that
  pitch type for this hitter, at typical command, is worth about 32 runs over a team's season (21 to 44) on the
  untouched months; in-zone aiming is not established.

- **Each batter's own zone** (MATCHUP-08F): the league swing model on each batter's zone beats the fixed zone on the
  untouched months (+0.59 nats per 1,000 decisions [0.28, 0.88]); hitter maps already carry it. Part of a hitter's
  high-versus-low own part is his zone height (ZONEMAP-01, correlation 0.35), and re-priced on batter zones the edge
  keeps 85 to 94% of its value (VALUE-15), so about a tenth of it is zone height.
- **Hitters judge a pitch's side against their own body** (STANCE-01, development): each hitter's distance off the
  plate, recovered from the bat-tracking contact points (reliability 0.99), goes with his sideways chase pattern: an inch
  farther off the plate, about 0.75 points less chasing away relative to inside (r -0.27 and -0.30), and hitters who
  moved between 2025 and 2026 shifted their pattern the same way (-0.31). Moving the whole swing surface by his position
  does not improve the model (MATCHUP-09), so it is a scouting signal: a stance change announces a chase-pattern change.
- **The times-through-the-order penalty is the pitcher's pitch count** (WARMUP-01, development): against fresh
  relievers hitters do not warm up (-0.003 runs per earlier plate appearance), familiarity with a pitcher adds nothing
  measurable, and the decline tracks pitches thrown (+0.026 runs per plate appearance per 100); a substitute's first
  plate appearance runs 0.014 below his level. Postseason staffs do not aim at hitters' own chase spots either
  (EXPLOIT-03, 2023-2025; EXPLOIT-03F scores 2026 after the World Series).

What did not hold: maps as an early warning (DRIFT-01) or as season projections (DISCIPLINE-01); the swing-plane
formula as the source of the geometry signal (CONTACT-01, -02); locating the steering limit from misses with one map
(STEER-01); pitch-type surprise (SEQ-01); velocity, flight reading and pitch-type familiarity as the
times-through-the-order penalty (FATIGUE-01, EXPOSURE-01); pair-specific strikeout effects large enough for the
simulator (ENGINE-01: real, about 0.4 points of strikeout rate); command measured from repeated aims (COMMAND-02);
chase spots explained by the bat path (SWINGMAP-01); timing carried across a pitching change (CHANGE-01: hitters re-time
fully between pitchers); the swing surface shifted by where the hitter stands (MATCHUP-09).

## What is original against the closest published work (checked October 9, 2026)

- Public swing-decision metrics (Statcast swing/take run values; SEAGER; SOTO) score decisions by where the pitch
  crosses the plate, mostly with population models; SOTO adds a hitter-specific damage zone. None represents the
  hitter at an estimated decision moment, and none tests hitter maps on held-out months.
- Baseball Prospectus's 2025 swing-process work (bat and pitch tracking) models swing initiation and swing-or-take from
  trajectory information up to a decision point with population-level models and leaves personalization to future
  work; it does not price anything in runs or examine pitchers.
- Pitch tunneling work (Baseball Prospectus, 2017 onward) assumes a tunnel point and scores pitchers; it does not
  estimate the decision moment from swings or fit hitters.
- New here: the decision moment estimated from swings (Decision Horizon) and used as the coordinate system for
  per-hitter maps, credited on untouched data; the measurement that pitchers do not aim at these maps; that fact used
  as a natural experiment to put a run value on hitter-specific maps, with command scatter, within-game adaptation and
  within-pitch-type limits; the contact window reading of bat-tracking depth spread; the timing sequence effects.
