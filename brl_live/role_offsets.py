"""Role offsets: what starters (by times through the order) and relievers do beyond the plate-appearance model.

The context offsets (brl_live/provider_adjust.ContextAdjust) set the league's level by batting side and inning. Within
an inning the model can still be off by role: the replays found starters' strikeouts 3.6% short of their actual lines
(PLAYER-01). An offset is, per outcome class, the log of observed over predicted counts (predictions after the context
offsets) from earlier dates, pulled toward zero by k pseudo plate appearances at the role's own predicted shares and
optionally decayed with a half-life in days:

    offset[role, class] = log((observed + k q) / (predicted + k q))

Roles: 'starter_1', 'starter_2', 'starter_3' (the starter facing the order the first, second, third or later time) and
'reliever'. Input rows are group-level sums only (research/role-resid-*.json.gz: one per date, role, times through the
order, batting side and inning bucket).
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import numpy as np

LABELS = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')
ROLES = ('starter_1', 'starter_2', 'starter_3', 'reliever')
TABLE_PATH = Path(__file__).resolve().parent / 'role_offsets.json'


def _ord(ymd: str) -> int:
    return _date.fromisoformat(str(ymd)[:10]).toordinal()


def role_key(role: str, tto: int) -> str:
    return 'reliever' if role != 'starter' else 'starter_' + str(min(max(int(tto or 1), 1), 3))


def read_rows(path) -> list[dict]:
    raw = Path(path).read_bytes()
    doc = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
    if tuple(doc.get('labels') or LABELS) != LABELS:
        raise ValueError('role residual rows have a different label order')
    return doc['rows']


class RoleState:
    """Chronological sums per role: absorb one date's rows after that date, read offsets for any later date."""

    def __init__(self, k: float = 2000.0, half_life: float | None = None):
        self.k, self.half_life = float(k), (float(half_life) if half_life else None)
        self.O = defaultdict(lambda: np.zeros(7)); self.E = defaultdict(lambda: np.zeros(7))
        self.last_day = None

    def _decay_to(self, day: int) -> None:
        if self.half_life and self.last_day is not None and day > self.last_day:
            f = 0.5 ** ((day - self.last_day) / self.half_life)
            for key in list(self.O):
                self.O[key] *= f; self.E[key] *= f
        self.last_day = day if self.last_day is None else max(self.last_day, day)

    def absorb(self, day_rows: list[dict]) -> None:
        if not day_rows:
            return
        self._decay_to(_ord(day_rows[0]['date']))
        for r in day_rows:
            key = role_key(r['role'], r.get('tto', 1))
            self.O[key] += np.asarray(r['obs'], float)
            self.E[key] += np.asarray(r.get('pred_ctx') or r['pred'], float)

    def offsets(self, key: str, day: int | None = None) -> np.ndarray:
        o, e = self.O.get(key), self.E.get(key)
        if o is None or e.sum() <= 0:
            return np.zeros(7)
        f = 1.0
        if self.half_life and self.last_day is not None and day is not None and day > self.last_day:
            f = 0.5 ** ((day - self.last_day) / self.half_life)
        q = e / e.sum()
        return np.log((o * f + self.k * q) / (e * f + self.k * q))

    def table(self, day: int | None = None) -> dict:
        return {key: [round(float(v), 5) for v in self.offsets(key, day)] for key in ROLES}


def by_date(rows: list[dict], dates, k: float = 2000.0, half_life: float | None = None) -> dict:
    """{date: {role: [7]}}: offsets at the start of each requested date (earlier rows only)."""
    want = sorted(set(str(d)[:10] for d in dates))
    days = defaultdict(list)
    for r in rows:
        days[str(r['date'])[:10]].append(r)
    st = RoleState(k, half_life)
    out = {}
    for d in sorted(set(days) | set(want)):
        if d in want:
            out[d] = st.table(_ord(d))
        st.absorb(days.get(d, []))
    return out


def current_table(rows: list[dict], through: str | None = None, k: float = 2000.0, half_life: float | None = None, source: str = '') -> dict:
    """The production table: offsets after every row dated on or before `through`."""
    through = through or max(str(r['date'])[:10] for r in rows)
    days = defaultdict(list)
    for r in rows:
        if str(r['date'])[:10] <= through:
            days[str(r['date'])[:10]].append(r)
    st = RoleState(k, half_life)
    for d in sorted(days):
        st.absorb(days[d])
    return {'schema': 'brl.role-offsets.v1', 'labels': list(LABELS), 'estimated_through': through, 'k': k, 'half_life': half_life,
            'last_day': st.last_day, 'source': source, 'roles': st.table(),
            'counts': {key: {'observed': [round(float(v), 3) for v in st.O[key]], 'expected': [round(float(v), 3) for v in st.E[key]]} for key in ROLES if key in st.O}}


def load_table(path: Path = TABLE_PATH) -> dict:
    doc = json.loads(Path(path).read_text())
    if tuple(doc['labels']) != LABELS:
        raise ValueError('role offsets label order mismatch')
    return doc


def log_multipliers(table: dict, day: str | None = None) -> dict:
    """{role: seven log-multipliers} at a game's date (stored counts decayed from the table's last day when it has a
    half-life, exactly as the chronological state would)."""
    counts = table.get('counts') or {}
    hl, last, k = table.get('half_life'), table.get('last_day'), float(table.get('k') or 2000.0)
    f = 0.5 ** (max(0, _ord(day) - int(last)) / float(hl)) if (hl and last and day) else 1.0
    out = {}
    for key in ROLES:
        c = counts.get(key)
        if c is not None:
            o, e = np.asarray(c['observed'], float), np.asarray(c['expected'], float)
            q = e / e.sum() if e.sum() > 0 else np.zeros(7)
            out[key] = np.log((f * o + k * q) / (f * e + k * q)).tolist() if e.sum() > 0 else [0.0] * 7
        else:
            out[key] = list((table.get('roles') or {}).get(key) or [0.0] * 7)
    return out
