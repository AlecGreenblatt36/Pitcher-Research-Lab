"""LONG-01: the plate-appearance history a long-history model's counters read at forecast time.

A model fitted on the history back to 2015 (its manifest lists history_seasons) weights every prior plate appearance by
a half-life, so its player, split, park and league counters need the sealed HIST-01 seasons (private/history/pa-<year>.enc,
regular season only, as in the fit) in front of the live history. Only the provider reads this file; everything else
(lineups, bullpens, the starter policy) keeps the live history. Without history_seasons nothing changes.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd

from app.safety import Blocked
from cloud.security import unseal

COLUMNS = ['date_key', 'game_pk', 'at_bat_number', 'batter', 'pitcher', 'stand', 'p_throws', 'park', 'outcome', 'age_bat', 'age_pit']


def history_path(year: int) -> str:
    return f'private/history/pa-{int(year)}.enc'


def history_purpose(year: int) -> str:
    return f'pa-history-season-{int(year)}'


def older_seasons(store, key: bytes, years) -> pd.DataFrame:
    frames = []
    for year in sorted(int(y) for y in years):
        saved = store.read(history_path(year))
        if saved is None:
            raise Blocked(f'Sealed history for {year} is missing')
        frame = pd.read_csv(io.BytesIO(gzip.decompress(unseal(saved[0], key, history_purpose(year)))), low_memory=False)
        frame = frame[frame['game_type'].astype(str) == 'R']
        frames.append(frame.loc[:, COLUMNS])
    out = pd.concat(frames, ignore_index=True)
    out['date_key'] = out['date_key'].astype(str).str[:10]
    return out


def write(store, key: bytes, years, history: pd.DataFrame, target: Path) -> dict:
    """The older seasons followed by the live history (provider columns), gzip CSV at target; a counts-only receipt."""
    older = older_seasons(store, key, years)
    recent = history.loc[:, COLUMNS].copy()
    recent['date_key'] = recent['date_key'].astype(str).str[:10]
    first = str(recent['date_key'].min())
    older = older[older['date_key'] < first]
    combined = pd.concat([older, recent], ignore_index=True)
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target, index=False, compression='gzip')
    return {'seasons': sorted(int(y) for y in years), 'older_rows': int(len(older)), 'rows': int(len(combined)), 'through': str(combined['date_key'].max())}
