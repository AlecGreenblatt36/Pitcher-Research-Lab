# Paul Skenes Origin Case Study

Pitcher Research Lab began as a focused study of Paul Skenes during the 2026 season. The original question came from reporting that his delivery and horizontal release position had changed during the year, with June 9 identified as an important point in the discussion.

The project was built to test a broader question: **can public pitch-tracking data identify when a pitcher's profile moves, quantify the size of that movement, and show what changed around the same period?**

The Skenes case remains the origin example, but the application no longer assumes that a pitcher is declining or that a specific mechanical explanation is correct. The same workflow can now be applied to any cached MLB pitcher, any available season, and either an automatic or custom research window.

## Original reporting

- Tom Verducci, Sports Illustrated, August 11, 2026: https://www.si.com/mlb/pirates/paul-skenes-struggles-slight-windup-change-breakdown-of-the-week
- Follow-up, August 25, 2026: https://www.si.com/mlb/pirates/paul-skenes-another-major-adjustment-but-something-is-still-off-verduccis-view

## Research boundaries

Statcast can directly measure release position, velocity, movement, spin, extension, pitch location and pitch outcomes. Those measurements can be compared with video observations, but the application does not treat a statistical association as proof of a mechanical cause, injury, fatigue or organizational intent.

## From case study to application

The one-player investigation established the research questions and metric definitions that informed the generalized application. The main branch now contains the reusable production system and this case-study context rather than the early exploratory scripts.
