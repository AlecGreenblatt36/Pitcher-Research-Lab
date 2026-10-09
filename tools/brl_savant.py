"""Baseball Savant pitch-by-pitch data with bat tracking, for discovery experiments inside Actions.

Savant's search export (statcast_search/csv, one day at a time) carries what the official feed does not: the fitted
flight of every pitch (vx0, vy0, vz0, ax, ay, az at y = 50 ft), the ABS zone top and bottom from 2026, arm angle,
the pitch's run value, spray coordinates, and bat tracking on swings (bat speed, swing length, attack angle and
direction, swing path tilt, where the bat met the ball relative to the batter). The data is fetched on the runner,
kept as compact numpy columns, sealed with the package key and stored on the ledger branch under private/savant/;
only coverage counts and league summaries leave the runner in receipts.

Documented definition change (Statcast CSV documentation): plate_x and plate_z are measured at the front of the plate
through 2025 and at the middle of the plate from 2026; sz_top and sz_bot are the ABS zone from 2026. With the fitted
flight every pitch's crossing can be recomputed at one reference (front_crossing below), so seasons can be pooled.
"""
from __future__ import annotations

import gzip, io, json, time
from datetime import date, timedelta
from urllib.request import Request, urlopen

import numpy as np

URL = ('https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&player_type=pitcher'
       '&hfGT={gt}%7C&hfSea={year}%7C&game_date_gt={d}&game_date_lt={d}')
SEASONS = {2024: ('2024-03-20', '2024-09-30'), 2025: ('2025-03-18', '2025-09-28'), 2026: ('2026-03-25', '2026-09-27')}
POST = {2026: ('2026-09-29', '2026-11-05')}
F16 = ('release_speed', 'release_spin_rate', 'spin_axis', 'release_pos_x', 'release_pos_z', 'release_extension', 'arm_angle',
       'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot',
       'launch_speed', 'launch_angle', 'hit_distance_sc', 'hc_x', 'hc_y', 'estimated_woba_using_speedangle', 'delta_run_exp',
       'bat_speed', 'swing_length', 'attack_angle', 'attack_direction', 'swing_path_tilt',
       'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches')
I32 = ('game_pk', 'batter', 'pitcher', 'fielder_2', 'at_bat_number', 'pitch_number', 'balls', 'strikes', 'outs_when_up', 'inning',
       'zone', 'bat_score', 'fld_score', 'n_thruorder_pitcher', 'age_bat', 'age_pit')
FLAGS = ('on_1b', 'on_2b', 'on_3b')
CATS = ('pitch_type', 'description', 'events', 'stand', 'p_throws', 'bb_type', 'inning_topbot', 'game_type', 'home_team', 'away_team')
SWING_DESC = ('swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip')
BAT_FIELDS = ('bat_speed', 'swing_length', 'attack_angle', 'attack_direction', 'swing_path_tilt',
              'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches', 'arm_angle')


def fetch_day(year: int, d: str, gt: str = 'R', tries: int = 4):
    import pandas as pd
    url = URL.format(gt=gt, year=year, d=d)
    last = None
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0 (BRL research)', 'Accept': 'text/csv,*/*'}), timeout=180) as r:
                text = r.read().decode('utf-8-sig', 'replace')
            if not text.strip() or not text.lstrip().startswith(('"pitch_type"', 'pitch_type')):
                return None if not text.strip() or '<html' in text[:200].lower() else _frame(pd, text)
            return _frame(pd, text)
        except Exception as exc:          # throttled or a timeout: wait and retry
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'savant {d}: {type(last).__name__}: {str(last)[:120]}')


def _frame(pd, text):
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    return df if len(df) else None


def pull(year: int, start: str, end: str, gt: str = 'R', workers: int = 3, log=print):
    """Every day from start to end; returns (columns dict, notes)."""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    days = [(d0 + timedelta(k)).isoformat() for k in range((d1 - d0).days + 1)]
    frames, notes = [], {'days': len(days), 'days_with_rows': 0, 'errors': []}

    def one(d):
        try:
            return d, fetch_day(year, d, gt)
        except Exception as exc:
            return d, exc
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (d, df) in enumerate(pool.map(one, days)):
            if isinstance(df, Exception):
                notes['errors'].append(f'{d}: {str(df)[:100]}')
            elif df is not None:
                frames.append(df); notes['days_with_rows'] += 1
            if i % 30 == 0:
                log(f'savant {year} {gt} {d}: {sum(len(f) for f in frames)} rows')
    if not frames:
        return None, notes
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(['game_pk', 'at_bat_number', 'pitch_number'])
    return pack(df), notes


def pack(df) -> dict:
    import pandas as pd
    out = {}
    cols = set(df.columns)
    for c in F16:
        if c in cols:
            out[c] = pd.to_numeric(df[c], errors='coerce').to_numpy(np.float32)
    for c in I32:
        if c in cols:
            out[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).to_numpy(np.int64).astype(np.int32)
    for c in FLAGS:
        if c in cols:
            out[c] = df[c].notna().to_numpy(np.int8)
    for c in CATS:
        if c in cols:
            values = df[c].fillna('').astype(str).to_numpy()
            vocab, codes = np.unique(values, return_inverse=True)
            out[c] = codes.astype(np.int16); out[c + '__vocab'] = vocab.astype('U32')
    out['day'] = np.asarray([date.fromisoformat(str(x)[:10]).toordinal() for x in df['game_date']], np.int32)
    return out


def to_bytes(cols: dict) -> bytes:
    buf = io.BytesIO()
    half = {k: (v.astype(np.float16) if v.dtype == np.float32 else v) for k, v in cols.items()}
    np.savez_compressed(buf, **half)
    return buf.getvalue()


def from_bytes(raw: bytes) -> dict:
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    return {k: (z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k]) for k in z.files}


def path(year: int, part: str) -> str:
    return f'private/savant/pitches-{int(year)}-{part}.enc'


def purpose(year: int, part: str) -> str:
    return f'savant-pitches-{int(year)}-{part}'


def split_parts(cols: dict, year: int) -> dict:
    """Two halves of a regular season (before and from July 1) and the postseason, each small enough to store."""
    cut = date(year, 7, 1).toordinal()
    gt = cols['game_type__vocab'][cols['game_type']] if 'game_type' in cols else np.full(len(cols['day']), 'R')
    masks = {'a': (cols['day'] < cut) & (gt == 'R'), 'b': (cols['day'] >= cut) & (gt == 'R'), 'post': gt != 'R'}
    parts = {}
    for name, m in masks.items():
        if m.any():
            parts[name] = {k: (v[m] if not k.endswith('__vocab') else v) for k, v in cols.items()}
    return parts


def label(cols: dict, name: str) -> np.ndarray:
    return cols[name + '__vocab'][cols[name]]


def coverage(cols: dict) -> dict:
    """Counts and league summaries only: rows by month, which bat-tracking fields exist and how often they are filled
    on swings, by month."""
    desc = label(cols, 'description')
    swing = np.isin(desc, SWING_DESC)
    month = np.asarray([date.fromordinal(int(d)).month for d in cols['day']])
    out = {'rows': int(len(desc)), 'swings': int(swing.sum()), 'columns': sorted(k for k in cols if not k.endswith('__vocab')),
           'rows_by_month': {int(m): int((month == m).sum()) for m in np.unique(month)}}
    filled = {}
    for c in BAT_FIELDS:
        if c in cols:
            v = np.isfinite(cols[c])
            ref = swing if c != 'arm_angle' else np.ones_like(swing)
            filled[c] = {int(m): round(float(v[ref & (month == m)].mean()), 4) for m in np.unique(month) if (ref & (month == m)).any()}
    out['filled_share_by_month'] = filled
    for c in ('attack_angle', 'swing_path_tilt', 'bat_speed'):
        if c in cols:
            v = cols[c][swing]; v = v[np.isfinite(v)]
            if len(v):
                out[c + '_league'] = {'mean': round(float(v.mean()), 3), 'sd': round(float(v.std()), 3), 'n': int(len(v))}
    return out


# ---------------------------------------------------------------- geometry from the fitted flight
FRONT, MIDDLE = 17.0 / 12.0, 8.5 / 12.0          # distance from the plate's back tip (ft)


def _time_to(cols, y):
    """Time from y = 50 ft to distance y (negative before 50 ft) along the fitted flight."""
    vy0, ay = cols['vy0'].astype(np.float64), cols['ay'].astype(np.float64)
    a, b, c = 0.5 * ay, vy0, 50.0 - y
    with np.errstate(invalid='ignore'):
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


def anchor(cols: dict, year: int, y_ref: float | None = None) -> None:
    """Adds the flight's position at y = 50 ft (_x50, _z50), solved from the reported crossing at the season's documented
    reference: the front of the plate through 2025, the middle from 2026 (or y_ref when given)."""
    y_ref = (FRONT if year <= 2025 else MIDDLE) if y_ref is None else y_ref
    t = _time_to(cols, y_ref)
    cols['_x50'] = (cols['plate_x'] - cols['vx0'] * t - 0.5 * cols['ax'] * t * t).astype(np.float64)
    cols['_z50'] = (cols['plate_z'] - cols['vz0'] * t - 0.5 * cols['az'] * t * t).astype(np.float64)


def at(cols: dict, y: float):
    """Position (x, z), time from 50 ft and velocity of the fitted flight at distance y from the plate's back tip;
    needs anchor() first."""
    t = _time_to(cols, y)
    x = cols['_x50'] + cols['vx0'] * t + 0.5 * cols['ax'] * t * t
    z = cols['_z50'] + cols['vz0'] * t + 0.5 * cols['az'] * t * t
    v = {'vx': cols['vx0'] + cols['ax'] * t, 'vy': cols['vy0'] + cols['ay'] * t, 'vz': cols['vz0'] + cols['az'] * t}
    return x, z, t, v


def reference_check(cols: dict) -> dict:
    """Which plate reference the reported plate_x and plate_z use: for each assumption (front, middle), anchor the flight
    on the reported crossing, run it back to the release (y = 60.5 ft minus extension) and compare with the reported
    release height and side. The right assumption leaves the same offset in every season; the wrong one moves it by
    the ball's drop (or drift) over the 8.5 inches between the two references."""
    out = {}
    ok = np.isfinite(cols['vy0']) & np.isfinite(cols['plate_z']) & np.isfinite(cols['release_pos_z']) & np.isfinite(cols['release_extension'])
    for name, y_ref in (('front', FRONT), ('middle', MIDDLE)):
        c = {k: cols[k] for k in ('vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'plate_x', 'plate_z')}
        anchor(c, 0, y_ref)
        y_rel = 60.5 - cols['release_extension'].astype(np.float64)
        t = _time_to(c, y_rel)
        z = c['_z50'] + c['vz0'] * t + 0.5 * c['az'] * t * t
        x = c['_x50'] + c['vx0'] * t + 0.5 * c['ax'] * t * t
        dz = (z - cols['release_pos_z'])[ok]; dx = (x - cols['release_pos_x'])[ok]
        dz, dx = dz[np.isfinite(dz)], dx[np.isfinite(dx)]
        out[name] = {'release_height_offset_in': round(float(np.median(dz)) * 12.0, 3), 'sd_in': round(float(np.std(dz)) * 12.0, 3),
                     'release_side_offset_in': round(float(np.median(dx)) * 12.0, 3), 'n': int(len(dz))}
    return out
