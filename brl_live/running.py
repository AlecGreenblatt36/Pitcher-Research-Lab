"""Base running for the simulator: each runner's speed from Statcast sprint speed, and stolen-base attempts and
success from the runner's and the pitcher's records.

The engine used one speed (0.5) for every runner and had no stolen bases. Run conversion (runs scored minus the runs
a team's plate-appearance outcomes are worth) differs by team beyond noise and carries from season to season
(DESERVED-01 in LEDGER.md); stealing and running are part of it.

Inputs are public season statistics (tools/brl_running_data.py, copied to running.json.gz here): sprint speed, stolen
bases, caught stealing, times on first base (singles, walks, hit batters) and plate appearances per runner, and
stolen bases and caught stealing allowed per pitcher with batters faced. A model built "through" a season uses that
season and the two before it, weighted 1, 0.5 and 0.25, so a regular-season replay of 2026 is built through 2025 and a
2026 postseason game through 2026.

  speed     0.5 + (sprint speed - league mean) / 7.5, clipped to 0..1. The league mean is weighted by plate
            appearances (27.35 ft/s in 2023-2026, SD 1.33), so the average runner keeps the 0.5 the engine's advancement
            rates were set for. Unknown runners get 0.5.
  attempt   per plate appearance with the next base open: the runner's attempts per time on first, shrunk toward the
            league by 40 times on base, times `per_pa`, times the pitcher's attempts allowed per batter faced relative
            to the league (shrunk by 400 batters faced); steals of third at `third` of that rate; none from the 7th
            inning on when the margin is `late_margin` runs or more
  success   the runner's stolen bases per attempt (shrunk toward the league by 25 attempts) moved on the log-odds scale
            by the pitcher's success allowed (shrunk by 30 attempts); +0.04 for third

per_pa = 0.85 gives 0.90 attempts per team-game at 78% success against a league-average plate-appearance model with
2026 regulars in the lineups (MLB: 0.91 in 2025, 0.88 in 2026, success 0.78 and 0.77).
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import replace
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / 'running.json.gz'
_LOADED: dict = {}


def load_running(path=DATA_PATH) -> dict:
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)


def steal_model(through_season: int, path=DATA_PATH, **kw) -> 'StealModel':
    """A StealModel from the data file, cached per (file, season, settings)."""
    key = (str(path), int(through_season), tuple(sorted(kw.items())))
    model = _LOADED.get(key)
    if model is None:
        model = _LOADED[key] = StealModel(load_running(path), int(through_season), **kw)
    return model


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1.0 - p))


class StealModel:
    def __init__(self, doc: dict, through_season: int, per_pa: float = 0.85, third: float = 0.18,
                 k_attempt: float = 40.0, k_success: float = 25.0, k_bf: float = 400.0, k_pitcher_success: float = 30.0,
                 late_margin: int = 4, speed_scale: float = 7.5):
        self.per_pa, self.third, self.late_margin = float(per_pa), float(third), int(late_margin)
        seasons = sorted((int(y) for y in (doc.get('seasons') or {}) if int(y) <= int(through_season)), reverse=True)[:3]
        if not seasons:
            raise ValueError(f'no base running data through {through_season}')
        runners: dict = {}
        pitchers: dict = {}
        sprint: dict = {}
        weighted = [0.0, 0.0]
        for i, y in enumerate(seasons):
            w = 0.5 ** i
            season = doc['seasons'][str(y)]
            for pid, r in (season.get('runners') or {}).items():
                a = runners.setdefault(pid, [0.0, 0.0, 0.0])
                a[0] += w * r.get('sb', 0); a[1] += w * r.get('cs', 0); a[2] += w * r.get('on1', 0)
                if r.get('sprint') is not None:
                    if pid not in sprint:
                        sprint[pid] = float(r['sprint'])          # the most recent season with a measured sprint speed
                    weighted[0] += w * r.get('pa', 0) * float(r['sprint']); weighted[1] += w * r.get('pa', 0)
            for pid, r in (season.get('pitchers') or {}).items():
                a = pitchers.setdefault(pid, [0.0, 0.0, 0.0])
                a[0] += w * r.get('sb', 0); a[1] += w * r.get('cs', 0); a[2] += w * r.get('bf', 0)
        sb = sum(a[0] for a in runners.values()); cs = sum(a[1] for a in runners.values()); on1 = sum(a[2] for a in runners.values())
        self.league_attempt = (sb + cs) / on1 if on1 > 0 else 0.10
        self.league_success = sb / (sb + cs) if sb + cs > 0 else 0.78
        self.sprint_mean = weighted[0] / weighted[1] if weighted[1] > 0 else 27.35
        self.speed_scale = float(speed_scale)
        self.attempts = {pid: (s + c + k_attempt * self.league_attempt) / (o + k_attempt) for pid, (s, c, o) in runners.items()}
        self.successes = {pid: (s + k_success * self.league_success) / (s + c + k_success) for pid, (s, c, o) in runners.items()}
        psb = sum(a[0] for a in pitchers.values()); pcs = sum(a[1] for a in pitchers.values()); pbf = sum(a[2] for a in pitchers.values())
        per_bf = (psb + pcs) / pbf if pbf > 0 else 0.024
        p_success = psb / (psb + pcs) if psb + pcs > 0 else self.league_success
        self.pitcher_attempt = {pid: ((s + c + k_bf * per_bf) / (bf + k_bf)) / per_bf for pid, (s, c, bf) in pitchers.items()}
        self.pitcher_success = {pid: _logit((s + k_pitcher_success * p_success) / (s + c + k_pitcher_success)) - _logit(p_success)
                                for pid, (s, c, bf) in pitchers.items()}
        self.sprint = sprint
        self.through_season = int(through_season)
        self.seasons = seasons

    def speed(self, pid) -> float:
        s = self.sprint.get(str(pid))
        return 0.5 if s is None else min(1.0, max(0.0, 0.5 + (s - self.sprint_mean) / self.speed_scale))

    def attempt(self, pid, to_base: int, inning: int = 1, margin: int = 0, pitcher=None) -> float:
        """Chance of an attempt during the next plate appearance; to_base 1 is second, 2 is third."""
        if inning >= 7 and abs(margin) >= self.late_margin:
            return 0.0
        a = self.attempts.get(str(pid), self.league_attempt) * self.per_pa
        if pitcher is not None:
            a *= self.pitcher_attempt.get(str(pitcher), 1.0)
        return min(0.6, a * (self.third if to_base == 2 else 1.0))

    def success(self, pid, to_base: int, pitcher=None) -> float:
        s = self.successes.get(str(pid), self.league_success)
        if pitcher is not None:
            shift = self.pitcher_success.get(str(pitcher), 0.0)
            if shift:
                s = 1.0 / (1.0 + math.exp(-(_logit(s) + shift)))
        return min(0.97, s + (0.04 if to_base == 2 else 0.0))

    def with_speeds(self, matchup):
        """The matchup with every hitter's speed from his sprint speed (pitchers and the rest unchanged)."""
        teams = {}
        for side in ('away', 'home'):
            t = getattr(matchup, side)
            teams[side] = replace(t, lineup=tuple(replace(p, speed=self.speed(p.player_id)) for p in t.lineup))
        return replace(matchup, away=teams['away'], home=teams['home'])

    def describe(self) -> str:
        return (f'stolen bases and runner speed through {self.through_season} (league attempts per time on first '
                f'{self.league_attempt:.3f}, success {self.league_success:.2f})')
