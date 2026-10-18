# Baseball Research Lab: methods and evidence boundaries (current)

This file describes the system as it runs now. The experiment ledger (LEDGER.md) is the record of evidence: every
experiment with its registered prediction, result and decision, including the failed ones. The matchup research
program is described in discovery/MATCHUP_PROGRAM.md. HANDOFF.md carries the operating state. Where this file and
the ledger disagree, the ledger is right and this file is stale.

## What is published

A static site (predictions, box-score distributions, in-game win chances, a track record, and a matchup report for
every game). Only model outputs, per-player summaries and public box-score facts are published. Plate-appearance
and pitch data stay sealed on the runner.

## The forecast and its three streams

- The simulator: a seven-outcome plate-appearance model (logistic, with pitch-physics features; fitted on 2023-2024,
  tuned on 2025, tested on 2026 before use) played 10,000 times per game with fitted layers (context offsets, run
  environment, team offsets, base running, reliever choice, relief exits with team hooks, starter leash). It is
  trained on results only and never sees a betting line.
- Our model: a fixed combination of the simulator's and a team model's log-odds. Its weights were taught by the
  market's closing lines of past seasons (least squares on the closing log-odds). It never sees the line of the game
  it forecasts. It is not a market-free forecast: its weights carry past market information.
- The headline: our model combined with the market's pregame line (60% line, 40% ours on the log-odds scale) when
  a line has been captured. It uses the current line.

The track record scores each stream separately on the same games (Brier score and log loss, against a coin and
against always picking the home team), with the market's own pregame line beside them.

Before a team posts its lineup, the simulator and the game plans use the last lineup the team ran out against a
starter who throws with the same hand as today's starter, within 30 days, else its last lineup (LINEUP-01, from
October 10, 2026; brl_live/lineups.py). On every 2025 and 2026 team-game it named 7.3 of the 9 starters against 6.8
for the last lineup alone (+0.42 and +0.50, both 95% intervals above zero) and put more of them in the right spot;
where the opposing starter's hand changed from the last game (40 to 44% of team-games) it named about one more. The
team's usual lineup against that hand (LINEUP-02) named 0.1 more still but scrambled the batting order and stays off;
putting a regular back in for a fill-in (LINEUP-03) named 0.07 more but put 0.1 fewer in the right spot and missed its
gate. On 2024, which no lineup rule was chosen on, the same-hand lineup named 0.45 more starters than the last lineup.
The graded forecast is the last version before first pitch, which nearly always has the posted lineups, so this
changes the early versions and the plans built before lineups come out.

## What the replays say, and what they do not

Full replays of the 2025 and 2026 regular seasons (1,000 worlds per game, each season with tables fitted on the
other season) are development evidence: the settings were chosen on them. On those replays our model's Brier score
is about level with the closing line (pooled 0.24245 against 0.24289, paired difference -0.00044 with an interval
from -0.00165 to +0.00072). That interval includes zero; it supports similar development-replay performance, not
equivalence and not prospective parity. The forward record from October 9, 2026 (FWD-02 in the ledger) is what will
decide.

Line movement: in those replays, when our model and the opening line differed, the line moved toward our number
about 7 times in 10 by first pitch, and 68% (2025) and 73% (2026) of the time when the replay used only the previous
game's lineup and the probable starters (assumed known at the open; the opening line's time is the source's). From
October 2026 the record keeps actual opening snapshots as captured, with the full denominator and signed magnitude.

Totals: the simulator's total-run distributions are close to calibrated (middle-half coverage 0.48), but its
over/under chances are no better than a coin and the market's are only slightly better. From October 10, 2026 the
page's expected total is a taught combination (0.32 x simulator mean + 0.78 x market line - 0.543), which beat the
line on squared error in 2026 replays (-0.27, interval excluding zero) and was within noise in 2025 (-0.12). No over
chance of our own is stated. The totals error is being diagnosed as a distribution problem (first five innings
against full games, team runs against combined runs, tails, extras, park and weather subsets) before any feature is
added.

## The matchup research program

The pitch-level work represents each pitch where it appears to be headed about 260 ms before the plate (a gravity-
only projection). That is an effective decision horizon under this model, not a claim about what hitters perceive.
Credited results are listed in discovery/MATCHUP_PROGRAM.md with their ledger rows. The decision-relevant result is
that each hitter's swing map read at that moment adds about 3.6 nats per 1,000 decisions to a gradient-boosted
model given the same pitch inputs and the standard heat map (BENCH-01F), and each hitter's whiff map on the true
crossing adds about 8.2 nats per 1,000 swings to a boosted miss model (BENCH-02F). The larger +40 figure compares
representations inside one logistic model and is the easier comparison.

Evidence labels. Development: 2023 to 2025 and 2026 through July. Later-period validation with adaptive reuse:
August 1 to September 27, 2026. These months were set aside as untouched, but the pricing of the aiming edge was
corrected there (VALUE-01F and VALUE-02F were withdrawn and VALUE-08F replaced them on the same months), and earlier
horizon work inspected 2026, so a new registration cannot restore them as untouched. Forward confirmation: the 2026
postseason and 2027, scored under predeclared rules.

From October 10, 2026 the report's league swing model also reads where the pitch crossed (SWING-CROSS-01): the
decision-moment model alone over-called swings on pitches that ended outside by about 8 to 11%, growing with distance
outside (CAL-01), and the crossing term brought that within 1 to 2.4% on later months with lower log loss for the league
and the maps. The aiming value was re-priced with it (VALUE-18X): -42.3 [-45.2, -39.4] runs per 6,200 plate appearances by
the engine's components against VALUE-18F's -43.5, the realized-outcome regression unchanged at -29.4, the inside part
-22.7, the calibration slopes 0.967 and 0.822. The stated range is now about 30 to 40 (30 to 45 before, from rounding 43.5).

From October 10, 2026 the called-strike model also carries each season's own profile at the zone edge (PROD-05, a
product call; no registered gate passed). The model mixed seasons, and the calls at the edge move from season to season: on
the 30 days after each monthly as-of date, the mixed model called 13% too many strikes within 0.1 ft outside the zone in
2025, 47% too many in 2026 and 4% too few in 2024; with each season's profile those came within 2 to 5%, with log loss on
the takes no worse in 2024 (+0.03 nats per 1,000) and lower in 2025 and 2026 (by 1.4 and 3.7). The gates it missed: a
two-month horizon in 2026, where the 2026 profile fit through July called 13% too few strikes 0.1 to 0.25 ft outside in
August and September (CS-SEASON-01), and log loss in 2024 (CS-SEASON-02). Weighting recent takes more (CS-RECENT-01) also
missed, on the band a quarter to half a foot outside. Re-priced with it (VALUE-18Y), the aiming value is -42.6
[-45.5, -39.8] by the engine's components and -23.2 inside (the study fits on 2023 to 2025, so it prices with 2025's profile,
close to the mixed one); the stated range stays about 30 to 40, and about 23 inside.

The whole chain a plan is priced with was then checked against what the same pitches produced (CAL-03): the engine's
value of a swing and of a take, from the miss, foul, called-strike and ball-in-play pieces and the count values, against
the value of the state each swing and take led to. Outside pitches, where the aiming value lives, were priced right (within
0.15 to 0.21 runs per 100 swings, 0.01 per 100 takes). Inside the zone swings were priced as too costly to the hitter, in
sample as well, and the cause was two errors in the engine (CAL-03b): the foul model, fit with the hitter's whiff
propensity, was evaluated with it at zero, so fouls came out 16 to 18% high and balls in play 15 to 17% low (FOUL-01); and a
two-strike foul tip that ends the at-bat, a strikeout, was priced as a ball in play (FOUL-02). With both corrected the
training rows match within 0.05 runs per 100 swings and the later 2025 rows within 0.16. The value of a ball in play still
moves with the period (6% low on July to September 2025, 8% high on August and September 2026), which shifts every
swing alike and is left open. Re-priced with both corrections (VALUE-18Z), the aiming value is -42.9 [-45.9, -40.0] by the
engine's components and -23.9 inside; the stated range stays about 30 to 40, and about 24 inside.

Why the engine's figure sits above the realized-outcome one was then traced (CAL-04): splitting pitches out of the zone by
the hitter's own swing tendency at the pitch, the swing maps are right in every group, but the whiff maps are too flat:
a hitter who rarely chases a spot misses more when he does swing there than his map says (about 10%), and a hitter who
chases a spot misses a little less (4 to 6%). So the engine prices the extra chases a hitter's tendency adds as costlier
than they are. Looser whiff maps (WHIFF-LAM-01) and the swing tendency as an input to the whiff model (WHIFF-OWN-01 to
04, the last with separate slopes for laying off and chasing) each fixed the low end, but overcorrected the heavy chasers'
misses or cost accuracy inside the zone, and none met its gate, though in run values the input tracked what the swings
produced at both ends. The line is closed for this season: the engine keeps the location-only whiff model and the stated
range keeps both figures.

The aiming value is estimated policy potential, not observed runs saved. It is stated as a range, about 30 to 40
runs over a team's season from outside pitches alone on the later-period months (VALUE-18X; the description that follows
is VALUE-18F's, whose figures the re-pricing moved by about a run): 43.5 [40.7, 46.6] when
each aim is priced by the engine's own components (the hitter's calibrated own swing part times the value of a swing
against a take from the whiff, called-strike, foul and contact models and the count values), 29 [25, 34] when the
same kind of choice is priced by a regression of realized pitch outcomes on the own part. The two disagree by about
38% in the per-point value of an extra swing, and the range carries that disagreement. The figure depends on
identification (pitchers do not aim at hitter-specific spots on average, which does not by itself establish that
exposure to the map is as good as random), an assumed 0.6 ft of command scatter per axis, limited hitter adaptation,
and intervals that do not propagate every source of uncertainty (the structural interval resamples hitters and the
calibration, not the maps' own noise or the component models). The estimator was stress-tested in synthetic seasons
with known truth (VALUE-17, VALUE-18): the earlier regression-times-gain figure overstated the true value of its
chosen spots by 1.3 to 2.0 times, the structural pricing priced its own choices within 5% in every world with
location structure and read zero in worlds with none, and its choices were worth 1.3 to 1.8 times the old method's.
The earlier stated figure of 32 runs was the regression's and is superseded. The in-zone part, priced the same way, adds
about 23 runs per team-season on the later-period months (VALUE-18I: in the synthetic worlds the inside structural pricing
came within 2 to 6% of the truth of its own choices, and the Report tab shows those strike spots beside the chase and take
spots). The maps' own noise, measured by refitting the maps on resampled training sets (VALUE-18J), is about 13% of the
structural interval's width (0.4 runs per 6,200 outside pitches against 0.85 to 1.1 from hitters and calibration); in the
synthetic worlds with noisy hitters the structural estimate runs about 5% below the true value of its own choices, the
conservative direction.

The plate umpire is a note, not an input (UMP-01): umpires' edge leanings, measured against league called-strike rates for
the same locations by season, persist only partly from one season to the next (correlation 0.37 at the edges, 0.60 at the
outside edge, a true spread of about 1 strike per 100 taken pitches at the edges), below the bar set in advance for
moving a strike spot. The live report names the plate umpire from the box score with his leanings and says so.

## The matchup report

Each game's report is built from pitches thrown before its as-of date and states that date. Backfilled reports are
retrospective reconstructions: the maps use only earlier pitches, but the method was chosen in October 2026. The
grade keeps three things apart: prediction quality (chases against the league's and the maps' expectations),
location alignment (pitches in the recommended cells against the pitcher's usual rate, which is not evidence of
intent) and policy value (not measured by a grade). Intent logging and randomized plan comparisons are the evidence
that would settle the last link; they need a team.

The report also carries plain counts for the standard scouting views (PROD-06, a display change with no claim): each
hitter's results over MLB's 13 zones (the per-pitch zone, catcher's view) by the pitcher's hand and pitch type, each
pitcher's by the batter's side, each hitter's batted-ball direction (field thirds at 15 degrees from the gameday hit
coordinates) and trajectory, and each planned arm's pitches on the seven days before the game (the bullpen card shows
the team's latest game days among them). Zone cells also count balls in play hit 95 mph or more and balls in play with
a measured speed. All of it comes from the same training pitches as the maps. The page colors a zone against the
league's rate for that batter side and pitcher hand; SLG counts a triple as a double, and zones with few at-bats are
shown in gray.

What changed from last season is said plainly and only when it is large (display, no claim): for a pitcher with 300 or
more pitches in each season, a pitch type added or dropped (5% or more of his pitches one season, under 2% the other),
a fastball 1 mph or more faster or slower, or a pitch used 10 points more or less; for a hitter with 200 or more plate
appearances in each, a chase, miss, strikeout, walk or hard-hit rate that moved by more than three times its sampling
error and by 3 points (5 for hard-hit). Box scores of real games show each hitter's average and each pitcher's ERA through
the game, as MLB's feed prints them. The Players page ranks the leaders on each percentile row and shows a team's
players in one table, shaded by those ranks (the players index carries each player's rates and ranks).

Player cards add percentile ranks (PROD-08): each player's rate among all players with 300 or more plate appearances
(batters faced for a pitcher) on the same training pitches, counted from those pitches. Walks include hit batters, a
sacrifice fly counts as an at-bat, and bunt plate appearances are left out, as in the hot zones. Hitter plans add the
head-to-head line against each arm, which the plan does not use. The running game (PROD-09) comes from public season
statistics (steals, caught stealing, times on first and sprint speed; steals allowed per pitcher), using a season only
once it is over.

The hitter tags are levels against the league with fixed thresholds, and they were checked on the next season's
pitches (TAGS-01, TAGS-02): every tag held in both test seasons, at about 80 to 95% of its training difference (the
relative reads below keep about half to nine-tenths). The
two-strike tags compare a hitter's two-strike chase rate with his own chase rate in the other counts, net of the
league's change on his side; that change correlates 0.64 and 0.72 from one season to the next. The pitch-type tags
(TAGS-03) compare his misses per swing on breaking balls or offspeed with his misses on fastballs, net of the league's
same difference on his side (correlations 0.52 to 0.62 from one season to the next). The pull tags (TAGS-05) compare
his share of ground balls and of balls in the air hit to his pull-side third with the league's (correlations 0.58 to
0.70), and the ground-ball and fly-ball tags (TAGS-06) his ground-ball share (0.73 and 0.75). The up-or-down tags
compare the top third of his zone (MLB zones 1-3) with the bottom third (7-9), net of the league's same difference on
his side: misses per swing (TAGS-07, 0.46 and 0.71, tagged at 9 points) and hard-hit balls among measured balls in play
(TAGS-08, 0.54 and 0.55, tagged at 10 points). Across the plate (inside third against away third, read with the side
he takes against each pitcher hand), hard contact carries over 0.60 and 0.64 (TAGS-10, both directions tagged at 13
points) but misses only 0.29 and 0.35 (TAGS-09): only the away direction held in both seasons and is tagged (Misses
away, 4.3 points). These relative reads keep about half to nine-tenths of their training difference a season later.
Switch hitters' charts, league comparisons and pull reads follow the side he takes against each pitcher hand
(cards: side_by_hand, spray_pull). A velocity read (extra misses on fastballs at 96 mph and up) did not carry over (TAGS-04:
0.26 and 0.15) and is not shown. Pitcher tags (PTAGS-01) were checked the same way: first-pitch strikes, pitches in the zone, chases,
fastballs when behind and on the first pitch; seven held in both test seasons and are shown, two (falling behind first,
few chases) are not. Height tags (PTAGS-02: fastballs in the top third of the fixed zone or above, in the bottom third or
below, breaking balls below the zone; 8 points over the league) carry over 0.73 to 0.78 and all three held. Ground-ball
and fly-ball pitchers (PTAGS-03: ground balls among batted balls against, 8 points either way) carry over 0.75 and 0.82
and both held. Inside and away (PTAGS-04: fastballs on the inner or outer third of the plate or beyond, each pitcher read
against the league's rate for the pitcher-hand and batter-side matchups he faced, 6 points over) carry over 0.57 to 0.61
and both held. The plan sheets' lineup notes are drawn from these tags only.

## The game page's props and previews (October 11, 2026)

Hitter chances on the Odds tab are each hitter's own: every event of his lineup slot in a simulated game is his with his
spot's share of the slot's plate appearances (PLAYER-02, 2023-2025 shares), the same rule the Summary's meters use. Scored
on full replays of both seasons with the live settings (PLAYER-03), the chance of at least one hit came within 1 point of
the actual share, a home run within 0.6 and a walk or hit-by-pitch within about 2; a logistic map fitted on one season did
not hold up on the other, because each replay's level comes from the other seasons' tables, so the page shows the raw
chances. Part-timers bat less often than their slot's share allows (95% to 96% of it under 100 prior plate appearances);
factors for that fixed their chances but the gain over all hitters was within noise (PLAYER-05, not adopted). Starters'
strikeouts are shown as a prop ladder at the lines 2.5 to 8.5, where the replays put each chance within about 2 points of
how often starters got there (PLAYER-04, described after looking).

The run line, the first inning and the first five innings were checked the same way before being shown (RUNLINE-01,
NRFI-01, FIRST5-01) and none passed its rule: given a win, the simulator's margins run tight (the narrowness DISP-01 found in
its run distributions), and the first-inning and five-inning chances held on level but missed a per-fifth bound set near
the noise. DISP-03 took the market numbers alone from worlds with a centered day-form shock while win chances stayed plain:
the spread and the run line came right in both seasons, but the total's over and under improved only within noise, so it
stays off. Two or more total bases held on level but its top tenth ran 3.4 points high in 2026 (PLAYER-06), so that column
stays off too: in 2026, the model's first season unseen in fitting or tuning, the best hitters' rates run high (by tenths
of the simulated rate, each step up in the simulated rate came with 0.90 of that step in actual hits and 0.82 in home runs,
against 0.95 and 0.97 in 2025, the season the model was tuned on).

The run charts and ranges the page states (each team's and the total's middle range, the chance of 10 or more total
runs) keep the simulator's mean but take a negative binomial with its variance times 1.095, the residual over the
predictive variance of a team's runs in the replays (TOTALS-05): the raw histograms held the real score up to 2.7 points
less often than their mass claimed, the widened ones within 1.4 points in both seasons, with CRPS and the 10-or-more
Brier no worse.
Starters' strikeouts get the same treatment (PLAYER-08): each start's simulated distribution is kept and spread by a
one-strikeout step to the residual variance, 1.07 times the simulated one; the ladder's largest gap fell from 2.35 to
1.92 points (2025) and 1.39 to 1.20 (2026), and the middle range held the real count within 0.4 points of its own mass.
Starters' hits allowed were checked the same way before getting a ladder (PLAYER-09): every line from 2.5 to 7.5 came
within 1.7 points of how often starters went over, with the corrected starter exits and with the plain ones, so the Odds
tab shows it for every game.

Starters' outs recorded (the pitcher-outs prop) were scored the same way (PLAYER-07): right on average and at the lines
near each start's median, but the simulator ended too many starts in the 4th inning and too few in the 5th (more than
12.5 outs 3.8 points short in 2025). The cause was the exit hazard: a logistic fit whose inning terms are a straight
line and a step from the 6th on has nothing for the 5th, so it cannot hold managers' habit of letting a starter finish
the 5th, and after the 4th ends it expected about half again as many removals as happen, in every season (LEASH-02). A
correction on its log-odds by inning, its end and the score, fitted on 2023 and 2024 decisions, lowered the decision log
loss on 2025 and 2026 and, in full replays of both seasons, put every outs line from 12.5 to 20.5 within 2.4 points with
win chances and totals unchanged. It is on for regular-season games (the postseason keeps the plain hazard with its
measured x0.91), and the outs ladder shows for boxes simulated with it. Real pitch counts would sharpen it further on
real decisions (LEASH-03: the exit hazard's log loss falls by about 0.006), but in the simulator, pitches drawn by
outcome one plate appearance at a time did not fix the middle outs lines, so the engine's pitch draws are built and
idle.

Relievers got the same treatment (RELIEF-04): the relief-exit hazard was right on average but pulled relievers who came
in by the 3rd inning about two thirds more often than real managers do (long relief stays in) and 6th-inning arrivals
too seldom, so the simulator used more relievers per out its starter missed than real teams. A correction by entry
inning and batters faced, fitted on other seasons, cut that gap from 0.039 and 0.031 pitchers per out to 0.017 and 0.006
with win chances and totals unchanged; it is on for regular-season games.

The bullpen's chance to pitch was scored per reliever on the same replays (BULLPEN-03): right on average and within
about 3 points below 60%, but the arms the simulator calls likeliest pitch far less often than it says (60% to 80%
shown, about 53% pitched, in both seasons). The choice model itself is calibrated entry by entry, and unavailable arms
explain little of it, so the miss is in how often the simulated games call for the top relievers. A straight calibration
could not pull those chances down (BULLPEN-04), so the page keeps them and says so under the tables.

A game to come now opens with what the simulated games say in plain words (KEYS-01, display). Each box counts, over
its own simulated games, how often each team scores first, leads after five, gets 18 outs from its starter, and how
often each hitter homers or gets two hits, with the team's wins in each case. The page shows the few that swing the
game most (the swing in the win chance times the square root of share x (1 - share)), with both chances shifted by the
same amount on the log-odds scale so that, mixed by the share, they give the published pick (which also uses the team
model and the betting line). KEYS-02 checked the simulator's own conditional chances on every game of 2025 and 2026
re-run: teams that scored first won 67.4% and 66.8% of the time against 67.2% simulated, and the starter's 18 outs, a
hitter's home run and a hitter's two hits came within about a point both when they happened and when they did not.

## The season's level, re-anchored in-season (October 11, 2026)

Outcome levels move from season to season (more walks and fewer extra-base hits and home runs in 2026 than in 2025), and
the simulator carries its level from tables fitted on earlier seasons. LEVEL-01 re-anchored each outcome's league level
every day from the season's real plate appearances on earlier dates only: per class, log((observed + m x share) /
(predicted + m x share)), with m = 10,000 pseudo plate appearances at the predicted shares. On every real plate
appearance of 2025 and 2026 (tables from the other season), it raised the log-likelihood by 0.11 and 0.15 nats per 1,000
plate appearances, both intervals above zero, with m picked on the other season. In full replays (LEVEL-02) it put
2025's walk chances and 2026's hit and home-run chances closer to level, but it lowered 2026's totals to 0.11 runs under
the real ones and made the over and under worse: with each outcome's rate right, the 2026 mix (more walks, fewer
extra-base hits) turns into fewer runs in the simulator than in real games, a shortfall the tables' level errors had
been hiding. The layer (LevelAdjust) stays off. Adding the running plays (wild pitches, passed balls, balks, pickoffs)
to it put 2026's totals within 0.04 runs of the real ones and moved 2025's from 0.04 under to 0.09 over (LEVEL-03), but
made the over and under worse in both, so neither is used: the simulator's season level is not what limits how it calls
totals against the line.

The Summary's team table and the Matchups tab's career lines against the starter are MLB's public numbers, read by the
browser (PROD-14). How much a career line says was measured on 2023-2026 plate appearances (BVP-01): beyond each player's
own level, a 20 plate-appearance record carried between none and about 15% of its on-base edge into later meetings, and no
strikeout or slugging edge could be told from noise, so the lines are shown as facts and never colored as a read.

## Privacy, storage and publication

Raw source responses, player history, pitch tables and model files stay encrypted on the data branch; the key is an
Actions secret. Research lanes write metrics-only receipts. The public site's files are an allowlist (index.html,
predictions.json, .nojekyll, days/*.json); the matchup reports are model outputs on the public data branch.

## Superseded methods (October 6, 2026)

The text below described the system before the October 7 to 9 work (market comparison, taught headline, matchup
program). It is kept for the record; its statement that no market comparison had been executed is no longer true.

### Methods as of October 6, 2026

## Purpose and current product

The system forecasts MLB games using information captured before play. It publishes win probabilities, expected scores, inning lines, batting and pitching projections, five complete example games, forecast versions and their publication times. After a final result is reconciled, the product can compare players and teams with their saved forecasts and calculate proper scores. The mission is auditable forecasting; no superior-accuracy claim follows from a working interface or a large simulation count.

Live forecasting and historical diagnostics have separate records. The current product is a scheduled static website with game-card navigation, full box-score views, sample play-by-play, track record and a How to read this tab. It is not yet an arbitrary-date simulation service. Every displayed time is Eastern. The numerical scoring store retains timezone-aware timestamps.

## The locked game simulation

The worker uses existing locked PA probabilities and the retained starter-removal model. Neither is refit by a scheduled iteration. Runtime and data-only inference artifacts have cryptographic identity pins. Current model calibration used 2025, so this live lock is not eligible for a training-independent 2025 replay.

Each simulation tracks innings, outs, runners, batting order, current pitcher and score. A forecast requires 10,000 completed games. Seeds are unique within a game and reused across lineup versions for reproducibility. Unfinished paths cannot be discarded or replaced to obtain a more favorable distribution; unresolved work blocks publication. Expected runs need not be integers, and the team with more expected runs need not have the greater win chance.

Postseason games currently retain regular-season bullpen and removal logic, without the regular-season automatic runner. This is disclosed in How to read this and is not a fitted postseason advantage. No pinch-hitting, stolen-base, wild-pitch or passed-ball process is integrated yet. Speed, defense and reliever-rest assumptions remain limited; per-world talent uncertainty is not integrated. These are research tasks, not completed accuracy improvements.

## Pregame information and daily history

Official MLB preview feeds supply team identities, rosters, probable starters and posted lineups. An absent official batting order is explicitly projected using earlier team appearances restricted to the roster. The current order projection is a heuristic, not a fitted lineup model. A final box is never used to construct that same game's live lineup.

Public Statcast and official final data are reconciled through the previous Eastern-calendar day. Checks cover PA identities, outcomes, dates and scores. Conflicting or incomplete source data block new forecasts. Sources retain actual capture times and hashes and are encrypted before persistence. Complete daily revisions are reused on subsequent checks; recent dates are revisited on later daily refreshes. The original seed remains immutable, and its historical publication vintages are not independently authenticated. Newly obtained observations are never backdated.

A source capture precedes simulation. A second preview check after simulation must still show no pitches, and the computation must finish before its original scheduled deadline. Publication time is recorded separately. Final scoring additionally verifies the observed first-pitch time. Changed lineups receive new versions; unchanged input fingerprints are not repeatedly simulated just to increase the forecast count. Introducing box output to an older win-only fingerprint creates its own new version and deadline.

## Complete box-score accounting

The box layer observes the original simulation without consuming its random stream or changing probabilities, runner transitions or pitcher-removal choices. It records who actually scores and which pitcher is responsible. Team and inning runs/hits reconcile with batting and opposing pitching lines in every simulated game. PA/AB, walks, hit batters, strikeouts and home runs also reconcile. The old manager's own state is not modified by display accounting.

Player means and integer count distributions include all 10,000 worlds. A pitcher's expected innings and strikeouts therefore include zero opportunity in worlds where he does not appear. Appearance probability is shown separately. Hit and home-run chances mean at least one in the whole game. Projected innings are decimal expectations; actual and individual simulated IP use baseball notation. A line of 5.2 actual IP is 17 outs, not 5.2 decimal innings.

Every inning is displayed separately. An inning not played contributes zero to the expected mean; a missing inning in an individual box is not invented. The first example has a most frequent exact final score pair, with a declared tie-break rule; four other examples use fixed world positions. All five replay their original seeds. A typical final score is not a claim about the most probable detailed event sequence.

The inherited terminal-PA corpus has pitch_number normalized to 1, so it was rejected as a pitch-count source. Earlier official playEvents instead supply an empirical joint sample of BB/HBP identity and pitch count, conditional on pitcher, selected PA outcome and batter hand, with pitcher/outcome and league fallbacks. This independent annotation stream does not change the game's outcome or feed the removal model. Recent support can be small. Displayed pitch counts are estimates, not pitch sequences or demonstrated pitch-count accuracy. The conditional pitch bridge is the next separate sprint.

## Official actuals and player scoring

Final boxes are read only in the outcome path. Player IDs, not names or guessed batting spots, join projections and actuals. Batting totals, pitching totals and official inning scores must reconcile. Missing stat fields remain unavailable. A projected participant absent from a complete final listing has zero realized opportunity. Unprojected actual substitutes remain visible but unscored.

Every box projection has its own content hash, commit and publication timestamp. Player scores require that exact content to have been publicly saved before observed first pitch. An earlier win-only forecast cannot backdate a newly generated player distribution. The latest eligible box version per game is selected without inspecting results, while every version retains its own record.

Batting hits, home runs and strikeouts, plus pitching strikeouts and innings, use fair finite-ensemble CRPS. It is the mean absolute difference to the observed value minus the distinct-draw pair-distance term; self-pairs are excluded. Pitcher innings are first scored as outs and then divided by three. Hit and home-run occurrence also receive corrected Brier scores. Lower scores are better, but different count statistics have different scales. Many player rows from one game are dependent and do not create an equivalent number of independent games for significance testing.

## Game scores and the fair baseline

Headline live game scores use the last eligible publicly saved pregame version, not the version that most closely predicted the result. Official teams and final scores must match and reconcile; publication must precede observed first pitch. Versions remain separately inspectable.

The inherited baseline is a team-strength negative-binomial run model with dispersion fitted on 2023–2024. It uses expanding strictly prior-date, league-shrunk team offense/defense, inherited 20-game shrinkage and a separate home-win logit offset. Its result recipe remains regular-season-only. Baseline and simulator are compared on identical eligible games and input cutoffs.

For estimated win probability p from N independent worlds and observed binary outcome y, corrected Brier is `(p-y)^2 - p*(1-p)/(N-1)`. The supplied baseline probability is scored directly. Game log loss is shown with the inherited second-order finite-simulation correction, explicitly an approximation. Numerical Monte Carlo error is separate from model and sample uncertainty. Shared seeds across games must not be counted as independent game-path cells when computing aggregate numerical error.

Market scores stay unavailable until licensed, same-cutoff historical odds are sourced. Closing odds cannot be described as the same-information comparison for an early forecast. No licensed odds comparison has been executed.

## Privacy, storage and publication

Raw source responses, player history and runtime/model files remain encrypted or private. AES-256-GCM uses purpose-specific authenticated data; the key is an Actions secret, never a command argument or public file. Git reads must match declared size and blob identity. An inconsistent Contents payload may be fetched once by immutable blob SHA and checked again; AES authentication is still mandatory. Never erase encrypted history to conceal an IO failure.

The public artifact allows only index.html, predictions.json and .nojekyll. It contains forecast distributions, five selected simulated event paths, final-box summaries and scoring/publication records, not raw training sources. Forecast records are immutable. Rendering binds boxes to their exact forecast IDs rather than mixing an old box with a newer header. The prototype's fonts are loaded from the public provider; no font files are distributed.

A failed refresh preserves the previous page rather than replacing saved predictions. The page carries a machine-readable refresh-blocked flag, and its verification job rejects a stale fallback. A successful deployment alone is not a successful forecast. Read the run receipt's stage, forecast count and history coverage. GitHub's schedule can be delayed or dropped, so nominal polling frequency is not a guarantee of capturing every official lineup.

## Evidence and remaining evaluation

The box sprint's evidence is execution, accounting, reproducibility and rendered-product testing, not a new accuracy gain. An observed schedule-triggered run, two cloud box forecasts, 47 focused regression tests, full 10,000-world frozen-score reproduction, 17 real final-box parser checks and deployed 390px/1440px browser checks are recorded in LEDGER.md. The first prospective final player score is still pending at the October 6 checkpoint.

For the winter benchmark, separately freeze PA, starter, preprocessing and calibration models fitted/tuned only through 2024, selecting settings on chronological development data. Retain the current live lock. Reconcile the complete official 2025 schedule, cancellations/resumptions and input coverage rather than fabricating exactly 2,430 eligible rows. Report exclusions.

Training-independent does not mean untouched: earlier project choices already used 2025. Repeatedly selecting upgrades on the 2025 replay makes it a development benchmark. Register candidates before scoring and preserve independent or prospectively saved confirmation evidence. Both corrected Brier and log-loss differences must be negative with both 95% upper endpoints below zero under the declared paired evaluation. Otherwise retain the current model and record the candidate as dropped/deferred. Operational fixes do not get invented metric improvements.

## Reference contracts

Public implementation (as of October 6; brl_engine/entrypoint.py has launched the live runs since October 7): box_runner.py, boxscore.py, box_page.py, history_refresh.py and verified_store.py. The encrypted runtime contains the unchanged provider/manager/game engine, cloud contracts and baseline. Tests and milestone evidence are recorded in LEDGER.md; recovery instructions are in HANDOFF.md.

Public Statcast field documentation: https://baseballsavant.mlb.com/csv-docs

GitHub schedule behavior: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule


## Edge research and skill scorecards — October 6, 2026

The five edge research families are registered in `research/edge_track/experiments.json` with detailed protocols in its README. Registration is not fitting, validation, or adoption. The old contact-only residual cannot change K or pooled BB/HBP; the proposed stuff/arsenal models need separately declared outcome groups and BB/HBP handling. Bat-swing features require explicit coverage/measurement-regime checks, and in-game updating has a separate clock and scorebook.

`brl_live/edge_metrics.py` adds descriptive paired pregame diagnostics: pitcher K/BB/pitches/innings, reliever appearance, team runs and high/low combined runs. High is fixed at 9+ runs versus 8 or fewer; the complement is not scored again as another independent result. Pitcher count CRPS includes zero for nonappearance. Skill baselines are saved INSIDE the independently timestamped box before first pitch; no late backfill is admitted. Old boxes retain their original player scores but lack the new paired skill comparisons.

The simple pitcher baseline resamples earlier complete team boxes by starter/relief role and maps relief lines uniformly to the same pregame pool, preserving nonappearance zeros. It uses no matchup/stuff/rest information. Its support and capture dates are serialized; the initial accepted cache is short, not a claimed full historical season. Team-run and high-total baselines use the existing team NB means and frozen dispersion. The team CDF-sum CRPS returns a bound on numerical tail truncation. Baseline distributions are fitted exact distributions, not simulated ensembles, so they receive no finite-simulation correction.

The observer stores team and combined-run histograms from complete coherent worlds. It never reconstructs them by summing independently sampled player marginals. Within each skill metric, scores are averaged within game before averaging games. Unprojected actual pitchers and missing statistics are recorded as coverage gaps. Exact forecast-version pairing and immutable publication hashes remain required. Missing latest-baseline coverage does not revive an older favorable comparison. Pitch-count predictions remain the existing independent empirical bookkeeping annotations until the separate conditional pitch bridge sprint.

`research/edge_track/report.py` computes per-metric current/candidate/simple-baseline effects and paired calendar-date-block intervals from exact matched score records. It refuses changed comparator values and unmatched identities. It does not authorize adoption or certify supplied provenance; simulation MCSE unavailable from aggregate score rows remains null. No fresh through-2024 lock or full 2025 replay exists yet. Selecting repeatedly on 2025 turns it into adaptive development, even if every fit stops in 2024. Prospective confirmation remains separate.
