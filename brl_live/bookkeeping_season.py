"""Season-long pitch sequences and batted-ball records for box-score bookkeeping.

tools/brl_bookkeeping_backfill.py seals one file per season on the ledger branch
(private/bookkeeping/season-<year>.enc: AES-256-GCM, purpose bookkeeping-season-<year>,
gzip JSON inside). This module reads it back into the same rows the day cache produces, so
the pitch bridge sees the whole season: a pitcher's own count paths, pitch mix and speeds,
a batter's own batted-ball shapes and directions. Rows are limited to games played before
the bookkeeping cutoff; the day cache remains the source for the most recent days and wins
on any overlap.
"""
from __future__ import annotations

import gzip
import json

from app.safety import Blocked
from cloud.security import unseal

SCHEMA = 'brl.bookkeeping-season.v1'
STUDY_SCHEMA = 'brl.pitch-physics-season.v1'     # the research dataset sealed next to it (never read by the live runtime)
MAX_AGE_DAYS = 400                                # season rows older than this are left to the day cache / ignored


def season_path(year: int) -> str:
    return f'private/bookkeeping/season-{int(year)}.enc'


def season_purpose(year: int) -> str:
    return f'bookkeeping-season-{int(year)}'


def receipt_path(year: int) -> str:
    return f'bookkeeping/season-{int(year)}.json'


def study_path(year: int) -> str:
    return f'private/statcast/season-{int(year)}.enc'


def study_purpose(year: int) -> str:
    return f'pitch-physics-season-{int(year)}'


def physics_path(year: int) -> str:
    """Per-plate-appearance physics table (gzip CSV inside) derived from the study: what the live provider reads."""
    return f'private/statcast/physics-{int(year)}.enc'


def physics_purpose(year: int) -> str:
    return f'physics-table-season-{int(year)}'


def load_physics_table(store, key: bytes, year: int):
    """The sealed per-PA physics table for a season as a DataFrame, or None when none is sealed."""
    import io
    import pandas as pd
    saved = store.read(physics_path(year))
    if saved is None:
        return None
    raw = gzip.decompress(unseal(saved[0], key, physics_purpose(year)))
    frame = pd.read_csv(io.BytesIO(raw))
    frame['date_key'] = frame['date_key'].astype(str).str[:10]
    return frame


def load_season(store, key: bytes, year: int) -> dict | None:
    """The sealed season document, or None when no backfill has been made for that year."""
    saved = store.read(season_path(year))
    if saved is None:
        return None
    doc = json.loads(gzip.decompress(unseal(saved[0], key, season_purpose(year))))
    if doc.get('schema') != SCHEMA or int(doc.get('year', -1)) != int(year) or not isinstance(doc.get('games'), dict):
        raise Blocked('Season bookkeeping identity mismatch')
    return doc


def season_rows(doc: dict, before: str, seen: set, max_age_days: int = MAX_AGE_DAYS) -> list[dict]:
    """Bookkeeping rows for games dated before `before` and at most `max_age_days` old, skipping (game_pk, at-bat) keys already seen."""
    from datetime import date, timedelta
    rows = []
    oldest = (date.fromisoformat(before[:10]) - timedelta(days=max_age_days)).isoformat()
    for pk, game in doc['games'].items():
        day = str(game.get('date') or '')[:10]
        if not day or day >= before or day < oldest:
            continue
        for r in game.get('rows') or []:
            key = (int(pk), int(r['i']))
            if key in seen:
                continue
            seen.add(key)
            rows.append({'date_key': day, 'outcome': r['o'], 'terminal_event': r['e'], 'pitcher': int(r['p']),
                         'batter': None if r.get('b') is None else int(r['b']), 'stand': str(r.get('s') or 'R'),
                         'pitch_number': int(r.get('n') or 0), 'available_at': game.get('fetched_at'),
                         'pitches': r.get('pt') or None, 'contact': r.get('c') or None})
    return rows
