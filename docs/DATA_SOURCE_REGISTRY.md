# Dynamic Engine Data Source Registry

This registry separates public inputs, derived features, and unavailable/proprietary information. All production snapshots must be timestamped and stored outside the public repository when bulk redistribution is restricted.

## Core event and roster data

### MLB Stats API

Use for schedule, game feed, play-by-play, box score, roster by date, transactions, venue, and umpire assignment. Endpoint behavior is public but not formally documented by MLB; the open-source `MLB-StatsAPI` project is a reference wrapper, not an MLB guarantee.

Required cutoff rule: roster and transaction state as of first pitch. Never infer the pregame bullpen from pitchers who appeared in the target game's box score.

## Statcast / Baseball Savant

### Statcast Search CSV

Official field documentation covers pitch velocity/movement, release extension, spin, exit velocity, launch angle, game/player IDs, catcher/fielder IDs, and base state.

Primary uses:

- pitch type, location, movement, velocity, extension;
- count transitions and pitch sequence;
- EV/LA/spray contact distribution;
- official runner/fielding context;
- catcher and fielder identity.

### Official Savant leaderboards

- Sprint Speed: runner speed.
- Arm Strength: fielder throwing ability.
- Outs Above Average / Fielding Run Value: range and total measurable defense.
- Catcher Framing / Fielding Run Value: framing, blocking, throwing.
- Statcast Park Factors / Venue pages: hit-type and batted-ball carry context.

Use prior-date or prior-season values only; apply hierarchical shrinkage for small samples.

## Weather and physics

### NOAA / National Weather Service

Use the official forecast API for temperature, humidity, pressure, wind speed/direction, and forecast issuance time. Preserve the forecast actually available before first pitch; do not substitute observed postgame weather in a pregame replay.

Derived quantities:

- air density;
- wind component from home plate toward each spray sector;
- roof/open-air status;
- weather uncertainty scenarios.

## Umpire and catcher zone effects

Use MLB umpire assignment from the game feed/jobs endpoint. Derive called-strike effects from prior called pitches in Statcast, with shrinkage and location controls. Catcher framing can use Savant framing data or a fitted called-strike model.

## Public research foundations

- Sidhu & Caffo, *MONEYBaRL* (AOAS, 2014): pitch decision/count state as a Markov decision process.
- Brill, Deshpande & Wyner (JQAS, 2023): model TTO/fatigue continuously rather than assuming a special third-time cutoff.
- Jensen, McShane & Wyner (Bayesian Analysis, 2009): hierarchical partial pooling for baseball talent.
- Gneiting & Raftery (JASA, 2007): proper scoring rules.
- Czado, Gneiting & Held (Biometrics, 2009): forecast diagnostics for discrete counts, including PIT-style checks.

## Not publicly available or not safe to assume

- private Hawk-Eye skeletal/swing trajectories;
- intended pitch location;
- private medical or wearable data;
- bullpen warm-up state;
- private scouting grades;
- team proprietary park/positioning models.

Use an explicit proxy or mark unavailable. Never fabricate these inputs.
