"""The per-PA pitch physics table a physics-aware PA model needs at forecast time.

Sources, in order: the sealed season tables made by the backfill (private/statcast/physics-
<year>.enc, the previous and the current season), then the cached official feeds of every
history day dated after the last season table's coverage, so the table runs through yesterday
exactly like the plate-appearance history. Rows are keyed by (game_pk, at_bat_number); the
backfill's rows win on overlap. Only needed when brl_engine/model.json selects a model with
physics features; a bundle without them never reads it.
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from app.safety import Blocked
from research_lab.pa_model.physics import table_from_feed, TABLE_COLUMNS, RUN_VALUE
from research_lab.pa_model.outcomes import map_event
from .bookkeeping_season import load_physics_table


def assemble_physics_table(cache, index: dict, origin: datetime, years: tuple | None = None) -> tuple[pd.DataFrame, dict]:
    """Season tables plus day-cache feeds after their coverage; (table, receipt)."""
    today = origin.astimezone(ZoneInfo('America/New_York')).date()
    years = tuple(years or (today.year - 1, today.year))
    parts, receipt = [], {'seasons': {}, 'day_cache_days': 0, 'day_cache_rows': 0}
    latest = ''
    for year in years:
        table = load_physics_table(cache.store, cache.key, year)
        if table is None:
            continue
        receipt['seasons'][str(year)] = {'rows': int(len(table)), 'through': str(table['date_key'].max())}
        latest = max(latest, str(table['date_key'].max()))
        parts.append(table)
    if not parts:
        raise Blocked('No sealed physics table for the selected model')
    for day, ref in sorted(index.get('days', {}).items()):
        if day <= latest or day >= today.isoformat():
            continue
        doc = cache.load_day(ref['id'])
        for value in doc['sources']:
            url = value.get('url', '')
            if '/feed/live' not in url or '?' in url:
                continue
            feed = json.loads(value['body'])
            if feed['gameData']['status']['abstractGameState'] != 'Final':
                continue
            rows = table_from_feed(feed, map_event)
            if len(rows):
                outcomes = {}
                for play in feed['liveData']['plays']['allPlays']:
                    o = map_event((play.get('result') or {}).get('eventType', ''))
                    if o is not None and (play.get('about') or {}).get('isComplete'):
                        outcomes[int(play['about']['atBatIndex']) + 1] = o
                rows['outcome'] = rows['at_bat_number'].map(outcomes)
                rows['run_value'] = rows['outcome'].map(RUN_VALUE).fillna(0.0)
                parts.append(rows); receipt['day_cache_rows'] += int(len(rows))
        receipt['day_cache_days'] += 1
    table = pd.concat(parts, ignore_index=True)
    table = table.drop_duplicates(['game_pk', 'at_bat_number'], keep='first')
    table = table.loc[table['date_key'].astype(str) < today.isoformat()].reset_index(drop=True)
    missing = [c for c in TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise Blocked('Physics table missing columns: ' + ', '.join(missing))
    receipt['rows'] = int(len(table)); receipt['through'] = str(table['date_key'].max()) if len(table) else None
    return table, receipt
