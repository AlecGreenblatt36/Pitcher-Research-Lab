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

# Which seasons the physics features are built from. 'live': the previous and the current season (the original
# runtime). 'fit': every season the selected model's fit used (its manifest's physics_seasons) plus the current one,
# keeping only rows of plate appearances that are in the plate-appearance history, as the fit's chronological builder
# does; with 'live' the count features and the shrunk rates differ from the ones the model was fitted on (SKEW-01).
SEASONS = 'live'


def seasons_for(manifest: dict, year: int, mode: str | None = None) -> tuple:
    """The seasons to load for a forecast in `year`."""
    mode = SEASONS if mode is None else mode
    fitted = [int(y) for y in (manifest.get('physics_seasons') or []) if int(y) <= int(year)]
    if mode == 'fit' and fitted:
        return tuple(sorted(set(fitted) | {int(year) - 1, int(year)}))
    return (int(year) - 1, int(year))


def join_history(table: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Only physics rows whose game, plate appearance, batter and pitcher are in the history (as the fit's builder)."""
    keys = history[['game_pk', 'at_bat_number', 'batter', 'pitcher']].dropna().astype('int64').drop_duplicates(['game_pk', 'at_bat_number'])
    t = table.astype({'game_pk': 'int64', 'at_bat_number': 'int64', 'batter': 'int64', 'pitcher': 'int64'})
    kept = t.merge(keys, on=['game_pk', 'at_bat_number', 'batter', 'pitcher'], how='inner').reset_index(drop=True)
    return kept, int(len(table) - len(kept))


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
            try:
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
            except Exception as exc:
                # One unreadable feed costs that game's pitch physics, never the day's forecasts.
                receipt['day_cache_errors'] = receipt.get('day_cache_errors', 0) + 1
                receipt.setdefault('day_cache_error', type(exc).__name__ + ': ' + str(exc)[:160])
        receipt['day_cache_days'] += 1
    table = pd.concat(parts, ignore_index=True)
    table = table.drop_duplicates(['game_pk', 'at_bat_number'], keep='first')
    table = table.loc[table['date_key'].astype(str) < today.isoformat()].reset_index(drop=True)
    missing = [c for c in TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise Blocked('Physics table missing columns: ' + ', '.join(missing))
    receipt['rows'] = int(len(table)); receipt['through'] = str(table['date_key'].max()) if len(table) else None
    return table, receipt
