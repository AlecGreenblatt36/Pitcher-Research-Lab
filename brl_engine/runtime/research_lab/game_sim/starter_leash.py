"""How long a regular-season start runs, beyond the starter's own record (LEASH-01).

The fitted hazard (starter_hazard.FittedStarterPolicy) pulls a starter against p_exp, his shrunk average batters faced
over every earlier start. That average knows nothing about the start in front of it. Against actual starts in full
replays, the simulator ran openers and starters on short rest about 8 batters long, starts that follow a relief outing
4.7 long, the first start back from a layoff of 20 days or more 1.5 long (the second 0.7), March and September starts
1.0 long, and regular starts about 0.6 short.

A table gives a shift in batters faced per start from these facts (estimated on the simulated-minus-actual starters of a
full replay of another season, tools/brl_leash.py), and the shift is delivered through p_exp along the hazard's measured
response: RESPONSE lists the hazard's mean batters faced at each p_exp with league-average hitting, so a shift of
-7 batters for an opener moves p_exp along that curve rather than one for one.

Facts come from appearances before the game's date only: days since the pitcher's previous appearance (short rest is
four days or fewer), whether that appearance was a start, starts since the last layoff of 20 days or more (the first
start of a season follows the winter, so it counts as the first back), and the month.

doc: {'name': ..., 'terms': {'base', 'short', 'relief', 'lay1', 'lay2', 'mar', 'sep'}, 'response': [[p_exp, mean bf],
...]} (brl_live/leash.json; brl_replay/leash_fit_YYYY.json).
"""
from __future__ import annotations

from datetime import date as _date

import numpy as np

SHORT_REST_DAYS = 4
LAYOFF_DAYS = 20
TERMS = ("base", "short", "relief", "lay1", "lay2", "mar", "sep")


def _ordinal(d) -> int:
    return _date.fromisoformat(str(d)[:10]).toordinal()


class AppearanceIndex:
    """Each pitcher's appearances (date ordinals and whether each was a start), for facts before a date."""

    def __init__(self, rows):
        """rows: iterable of (pitcher, date, start) for every appearance (one per game and pitcher)."""
        by: dict[int, list] = {}
        for pid, d, start in rows:
            by.setdefault(int(pid), []).append((_ordinal(d), bool(start)))
        self.by = {}
        for pid, items in by.items():
            items.sort()
            self.by[pid] = (np.asarray([x[0] for x in items], np.int64), np.asarray([x[1] for x in items], bool))

    @classmethod
    def from_frame(cls, app) -> "AppearanceIndex":
        """From an appearance table with columns pitcher, date and start (brl_live.live_feed.appearances layout)."""
        return cls(zip(app["pitcher"].to_numpy(), app["date"].astype(str).str[:10].to_numpy(), app["start"].to_numpy()))

    def facts(self, pid, game_date) -> dict:
        """Facts for a start on game_date from appearances strictly before it."""
        day = _ordinal(game_date)
        month = int(str(game_date)[5:7])
        days_arr, start_arr = self.by.get(int(pid), (np.zeros(0, np.int64), np.zeros(0, bool)))
        i = int(np.searchsorted(days_arr, day, "left"))
        if i == 0:
            return {"days": None, "prev_start": None, "since_layoff": 1, "month": month}
        days = int(day - days_arr[i - 1])
        since = 1 if days >= LAYOFF_DAYS else None
        if since is None:
            # the latest earlier start: was it the first after a layoff?
            starts = np.flatnonzero(start_arr[:i])
            if len(starts):
                j = int(starts[-1])
                gap = None if j == 0 else int(days_arr[j] - days_arr[j - 1])
                since = 2 if (gap is None or gap >= LAYOFF_DAYS) else 3
            else:
                since = 3
        return {"days": days, "prev_start": bool(start_arr[i - 1]), "since_layoff": since, "month": month}


def indicators(f: dict) -> dict:
    days = f.get("days")
    return {"base": 1.0,
            "short": float(days is not None and days <= SHORT_REST_DAYS),
            "relief": float(f.get("prev_start") is False),
            "lay1": float(f.get("since_layoff") == 1),
            "lay2": float(f.get("since_layoff") == 2),
            "mar": float(f.get("month") == 3),
            "sep": float((f.get("month") or 0) >= 9)}


class Leash:
    def __init__(self, doc: dict):
        self.name = str(doc.get("name") or "starter leash")
        self.terms = {k: float(doc["terms"].get(k, 0.0)) for k in TERMS}
        r = np.asarray(doc["response"], float)
        if r.ndim != 2 or r.shape[1] != 2 or np.any(np.diff(r[:, 0]) <= 0) or np.any(np.diff(r[:, 1]) <= 0):
            raise ValueError("leash: the response must rise with p_exp")
        self.x, self.f = r[:, 0], r[:, 1]
        self._lo = (self.f[1] - self.f[0]) / (self.x[1] - self.x[0])
        self._hi = (self.f[-1] - self.f[-2]) / (self.x[-1] - self.x[-2])

    def response(self, p_exp: float) -> float:
        """The hazard's mean batters faced at p_exp (linear beyond the measured range)."""
        p = float(p_exp)
        if p < self.x[0]:
            return float(self.f[0] + (p - self.x[0]) * self._lo)
        if p > self.x[-1]:
            return float(self.f[-1] + (p - self.x[-1]) * self._hi)
        return float(np.interp(p, self.x, self.f))

    def inverse(self, bf: float) -> float:
        """The p_exp whose mean batters faced is bf."""
        b = float(bf)
        if b < self.f[0]:
            return float(self.x[0] + (b - self.f[0]) / self._lo)
        if b > self.f[-1]:
            return float(self.x[-1] + (b - self.f[-1]) / self._hi)
        return float(np.interp(b, self.f, self.x))

    def shift(self, facts: dict) -> float:
        """Batters faced to add to the start (negative: shorter)."""
        ind = indicators(facts)
        return float(sum(self.terms[k] * ind[k] for k in TERMS))

    def adjust(self, p_exp: float, facts: dict) -> float:
        """p_exp moved along the hazard's response so the start's expected length moves by shift(facts)."""
        target = max(3.0, self.response(p_exp) + self.shift(facts))
        return self.inverse(target)
