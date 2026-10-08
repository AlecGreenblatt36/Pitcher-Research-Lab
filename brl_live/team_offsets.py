"""Team offsets: what a team's hitters and its pitchers-and-fielders do beyond the plate-appearance model.

The PA model rates batters and pitchers one at a time. Its 2026 residuals, summed by team, keep a team-level part
that persists from one half of the season to the other (LEDGER.md TEAM-01): fielding teams that turn more balls
in play into outs or get more strikeouts than their pitchers' own rates say (defense, catching, game plans), and
batting teams that do better or worse than their hitters' rates (mostly age: young lineups improve on their
history, older ones decline). An offset is, per outcome class, the log of observed over predicted counts from the
team's earlier games, pulled toward zero by k pseudo plate appearances at the league rate and optionally decayed
with a half-life in days:

    offset[team, side, class] = log((observed + k q) / (expected + k q))

side is 'bat' (the team at the plate) or 'fld' (the team in the field). The expected counts of one side include
the other side's offsets of the opponent at that time, so a team is not credited for its opponents. A plate
appearance's probabilities are multiplied by exp(bat offset of the batting team + fld offset of the fielding team)
and renormalized (brl_live/provider_adjust.TeamAdjust).

Input rows are team-level sums only (one per date, team and side: plate appearances, observed counts and the
model's predicted counts per class), written by the research lane (research/team-resid-*.json.gz).
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import numpy as np

LABELS = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')
# League shares the counts start from (2025-2026 regular seasons, rounded), worth 2,000 plate appearances, so the
# first days of a season never meet a class with no events yet.
PRIOR_SHARE = np.array([0.453, 0.221, 0.100, 0.142, 0.044, 0.031, 0.009])
PRIOR_PA = 2000.0
TABLE_PATH = Path(__file__).resolve().parent / 'team_offsets.json'


def _ord(ymd: str) -> int:
    return _date.fromisoformat(str(ymd)[:10]).toordinal()


def read_rows(path) -> list[dict]:
    raw = Path(path).read_bytes()
    doc = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
    if tuple(doc.get('labels') or LABELS) != LABELS:
        raise ValueError('team residual rows have a different label order')
    return doc['rows']


def _opponents(rows: list[dict]) -> dict:
    """(date, team, side) -> opponent: a team's batting rows on a date are the same plate appearances as its opponent's
    fielding rows, so they pair by date and identical observed counts (and predicted sums)."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r['date'], r['side'], tuple(r['obs']), round(sum(r['pred']), 2))].append(r['team'])
    out = {}
    for r in rows:
        other = 'fld' if r['side'] == 'bat' else 'bat'
        cands = [t for t in by_key.get((r['date'], other, tuple(r['obs']), round(sum(r['pred']), 2)), []) if t != r['team']]
        if len(cands) == 1:
            out[(r['date'], r['team'], r['side'])] = cands[0]
    return out


class OffsetState:
    """Chronological offsets: absorb one date's rows after that date, read offsets for any later date."""

    def __init__(self, k: float = 4000.0, half_life: float | None = None, sides=('bat', 'fld')):
        self.k, self.half_life, self.sides = float(k), (float(half_life) if half_life else None), tuple(sides)
        self.O = {s: defaultdict(lambda: np.zeros(7)) for s in ('bat', 'fld')}
        self.E = {s: defaultdict(lambda: np.zeros(7)) for s in ('bat', 'fld')}
        self.league = np.zeros(7)
        self.last_day = None

    def _decay_to(self, day: int) -> None:
        if self.half_life and self.last_day is not None and day > self.last_day:
            f = 0.5 ** ((day - self.last_day) / self.half_life)
            for s in ('bat', 'fld'):
                for team in self.O[s]:
                    self.O[s][team] *= f; self.E[s][team] *= f
        self.last_day = day if self.last_day is None else max(self.last_day, day)

    def share(self) -> np.ndarray:
        return (self.league + PRIOR_PA * PRIOR_SHARE / PRIOR_SHARE.sum()) / (self.league.sum() + PRIOR_PA)

    def offsets(self, team: str, side: str, day: int | None = None) -> np.ndarray:
        if side not in self.sides:
            return np.zeros(7)
        q = self.share()
        f = 1.0
        if self.half_life and self.last_day is not None and day is not None and day > self.last_day:
            f = 0.5 ** ((day - self.last_day) / self.half_life)
        o = self.O[side].get(team); e = self.E[side].get(team)
        if o is None:
            return np.zeros(7)
        return np.log((o * f + self.k * q) / (e * f + self.k * q))

    def absorb(self, day_rows: list[dict], opponents: dict) -> None:
        if not day_rows:
            return
        day = _ord(day_rows[0]['date'])
        self._decay_to(day)
        # Offsets as they stood before this date, for the opponent adjustment.
        before = {}
        for r in day_rows:
            other = 'fld' if r['side'] == 'bat' else 'bat'
            opp = opponents.get((r['date'], r['team'], r['side']))
            before[(r['team'], r['side'])] = self.offsets(opp, other) if opp is not None else np.zeros(7)
        for r in day_rows:
            obs, pred = np.asarray(r['obs'], float), np.asarray(r['pred'], float)
            adj = pred * np.exp(before[(r['team'], r['side'])])
            if adj.sum() > 0:
                adj *= pred.sum() / adj.sum()
            self.O[r['side']][r['team']] += obs
            self.E[r['side']][r['team']] += adj
            if r['side'] == 'bat':
                self.league += obs

    def table(self, day: int | None = None) -> dict:
        teams = sorted(set(self.O['bat']) | set(self.O['fld']))
        return {t: {s: [round(float(v), 5) for v in self.offsets(t, s, day)] for s in ('bat', 'fld')} for t in teams}


def center_table(tb: dict) -> dict:
    """Offsets with the league's average (the plain mean over teams, per side and outcome class) taken out, so the
    offsets move teams against each other and leave the league's level to the context offsets (CENTER-01)."""
    if not tb:
        return tb
    out = {t: dict(v) for t, v in tb.items()}
    for side in ('bat', 'fld'):
        vals = [np.asarray(v[side], float) for v in tb.values() if v.get(side) is not None]
        if not vals:
            continue
        m = np.mean(vals, axis=0)
        for t, v in tb.items():
            if v.get(side) is not None:
                out[t][side] = [round(float(x), 5) for x in np.asarray(v[side], float) - m]
    return out


def by_date(rows: list[dict], dates, k: float = 4000.0, half_life: float | None = None, sides=('bat', 'fld'), center: bool = False) -> dict:
    """{date: {team: {'bat': [7], 'fld': [7]}}}: offsets at the start of each requested date (earlier rows only),
    centered on the league when center is set."""
    want = sorted(set(str(d)[:10] for d in dates))
    days = defaultdict(list)
    for r in rows:
        days[r['date']].append(r)
    opponents = _opponents(rows)
    st = OffsetState(k, half_life, sides)
    out = {}
    keys = sorted(set(days) | set(want))
    for d in keys:
        if d in want:
            out[d] = center_table(st.table(_ord(d))) if center else st.table(_ord(d))
        st.absorb(days.get(d, []), opponents)
    return out


def current_table(rows: list[dict], through: str | None = None, k: float = 4000.0, half_life: float | None = None,
                  sides=('bat', 'fld'), source: str = '', center: bool = False) -> dict:
    """The production table: offsets after every row dated on or before `through` (default: all rows)."""
    through = through or max(r['date'] for r in rows)
    use = [r for r in rows if r['date'] <= through]
    days = defaultdict(list)
    for r in use:
        days[r['date']].append(r)
    opponents = _opponents(use)
    st = OffsetState(k, half_life, sides)
    for d in sorted(days):
        st.absorb(days[d], opponents)
    counts = {t: {s: {'observed': [round(float(v), 4) for v in st.O[s][t]], 'expected': [round(float(v), 4) for v in st.E[s][t]]}
                  for s in ('bat', 'fld') if t in st.O[s]} for t in sorted(set(st.O['bat']) | set(st.O['fld']))}
    return {'schema': 'brl.team-offsets.v1', 'labels': list(LABELS), 'estimated_through': through, 'k': k, 'half_life': half_life,
            'sides': list(sides), 'last_day': st.last_day, 'league_share': [round(float(v), 6) for v in st.share()],
            'source': source, 'center': bool(center), 'teams': center_table(st.table()) if center else st.table(), 'counts': counts}


def load_table(path: Path = TABLE_PATH) -> dict:
    doc = json.loads(Path(path).read_text())
    if tuple(doc['labels']) != LABELS:
        raise ValueError('team offsets label order mismatch')
    return doc


def game_log_multipliers(table: dict, away: str, home: str, day: str | None = None) -> dict:
    """{'away': log-multipliers while the away team bats, 'home': while the home team bats}; unknown teams get zeros.
    With a half-life, the stored counts are decayed from the table's last day to the game's day before the offsets
    are taken, exactly as the chronological state would."""
    teams, counts = table.get('teams') or {}, table.get('counts') or {}
    sides = set(table.get('sides') or ('bat', 'fld'))
    hl, last, k = table.get('half_life'), table.get('last_day'), float(table.get('k') or 4000.0)
    q = np.asarray(table.get('league_share') or [0.0] * 7, float)
    f = 0.5 ** (max(0, _ord(day) - int(last)) / float(hl)) if (hl and last and day) else 1.0

    def get(team, side):
        if side not in sides:
            return np.zeros(7)
        c = (counts.get(str(team)) or {}).get(side)
        if c is not None and q.sum() > 0:
            o, e = np.asarray(c['observed'], float), np.asarray(c['expected'], float)
            return np.log((f * o + k * q) / (f * e + k * q))
        v = (teams.get(str(team)) or {}).get(side)
        return np.asarray(v, float) if v is not None else np.zeros(7)
    if table.get('center'):
        names = sorted(set(counts) | set(teams))
        mean = {side: np.mean([get(t, side) for t in names], axis=0) if names else np.zeros(7) for side in ('bat', 'fld')}
        raw = get
        get = lambda team, side: raw(team, side) - mean[side]   # noqa: E731
    return {'away': (get(away, 'bat') + get(home, 'fld')).tolist(), 'home': (get(home, 'bat') + get(away, 'fld')).tolist()}
