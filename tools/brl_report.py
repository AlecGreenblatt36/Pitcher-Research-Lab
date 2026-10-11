"""The matchup report: for every game of a date, what the credited pitch-level models say about each hitter against
the pitchers he will face, built only from pitches thrown before the report's as-of date, and, once the game is
played, graded against what happened.

Runs inside Actions (brl-report workflow) on the sealed pitch tables; only model outputs, per-player summaries and
public box-score facts leave the runner. Output: public/reports/<date>.json on the ledger branch (and an index), which
the live site reads straight from the public data branch.

Per hitter against each pitcher (the starter and the relievers used most in the last 14 days):
- chase_points: extra chases per 100 outside pitches from his own swing map at the decision moment against this
  pitcher's own pitches (MATCHUP-01F representation with the league's family part and his family maps, MATCHUP-05F),
  with the pair chase calibration (0.95);
- k_points and bb_points: the count-by-count chain's strikeout and walk change against the league (ENGINE-01 scale);
- aim: where to put outside pitches of each family (the best third of the pitcher's own spots by the hitter's own part,
  under 0.6 ft of command scatter) and the runs per 100 plate appearances that is worth (VALUE-08F);
- the hitter's own chase map by family and his whiff map (7 by 7 grids), his zone top and bottom.

Grade (finished games, every pitch in the table): for each pitcher-hitter pair, the pitches thrown outside the zone,
how many landed in the recommended cells against the pitcher's usual share, the hitter's chases against what his map
predicted on those very pitches (the league alone and with his map), and the plate appearance results. The as-of date
is stated on every report; a backfilled month uses the maps as of the first of that month, so no report ever uses a
pitch from its own game or later.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('brl_discovery', ROOT / 'tools' / 'brl_discovery.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)

SCHEMA = 'brl.report.v1'
GU = np.array([-1.25, -0.83, -0.42, 0.0, 0.42, 0.83, 1.25]); GZ = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
FAMILIES = (('fastball', (0, 1, 2, 6), 0, 94.0), ('breaking', (3, 4), 3, 85.0), ('offspeed', (5,), 5, 86.0))
B_OUT_FAMILY, N_OUT = -0.000657, 1.876          # VALUE-08F (family maps): runs per point of the own part, outside pitches per plate appearance
STRUCTURAL = True                                # VALUE-18: aims chosen and priced by the engine's components (whiff, called strike, foul, contact value, count values)
SWING_CROSS = True                               # SWING-CROSS-01 (gate passed October 10, 2026): the league swing model also reads where the pitch crossed; params swing_cross false turns it off
CS_SEASON = True                                 # PROD-05 (product call October 10, 2026; CS-SEASON-01 and -02 gates not met): each season's own edge profile for the called-strike model (params cs_season false turns it off)
CS_RECENT = None                                 # CS-RECENT-01: the called-strike model fit on every training take weighted toward the latest days, half-life in days (params cs_recent)
FOUL_FIX = True                                   # FOUL-01 and FOUL-02 (check held October 10, 2026): the engine's foul model reads the hitter's whiff propensity, and a two-strike foul tip ending the at-bat is a strikeout (params foul_fix false turns them off)
LAM_W = 30.0                                      # shrinkage of the hitter whiff maps toward the league (report and engine; params lam_w; WHIFF-LAM-01: 10 did not pass)
WHIFF_OWN = 0                                     # WHIFF-OWN-01 (1, gate not met) and -02 (2: one weight out of the zone, one inside): the engine's whiff model reads the hitter's own swing tendency at the pitch (params whiff_own)
STRIKE_SPOTS = True
POOL_CACHE = 800            # pools kept in memory at once (pitcher, side, count group, type group, zone); see Fitted._pool                              # VALUE-18I: in-zone aims priced the same way (strike spots); its synthetic verdict held on October 9, 2026
N_IN = 2.04                                      # inside pitches per plate appearance (VALUE-18F)
OWN_PART_CALIBRATION = {'outside': 0.95, 'inside': 0.82}   # VALUE-18 (2025 development run): how much of a fitted point of the own part shows up in actual swings
CHASE_SLOPE, K_SCALE, BB_SCALE = 0.95, 0.40, 0.53   # MATCHUP-01F pair slope; ENGINE-01 coefficients over calibrated
sig = lambda v: 1 / (1 + np.exp(-v))


BINS = ('under -2', '-2 to 0', '0 to 2', 'over 2')


def bin_of(chase_points: float) -> str:
    return BINS[0] if chase_points < -2 else BINS[1] if chase_points < 0 else BINS[2] if chase_points < 2 else BINS[3]


def record_from(days: dict) -> dict:
    """The forward record over every graded game: chases against the league's and the maps' expectations, by what the
    map predicted for the pair, and staffs' aiming against their usual rates."""
    rec = {'schema': 'brl.report-record.v1', 'built_at': datetime.now(timezone.utc).isoformat(), 'games': 0, 'dates': 0, 'first_date': None, 'last_date': None,
           'outside_pitches': 0, 'in_recommended': 0, 'usual_expected': 0.0, 'chases': 0, 'chases_expected_league': 0.0, 'chases_expected_map': 0.0,
           'scored': {'decisions': 0, 'log_loss_map': 0.0, 'log_loss_league': 0.0, 'games': 0},
           'bins': {k: {'pairs': 0, 'outside': 0, 'chases': 0, 'league': 0.0, 'map': 0.0} for k in BINS}, 'by_month': {}, 'by_pricing': {}}
    for day in sorted(days):
        doc = days[day]; used = False
        pricing = doc.get('pricing') or 'regression'      # how the day's aim plans were chosen (VALUE-18: structural from October 9, 2026)
        for g in (doc.get('games') or {}).values():
            gr = g.get('grade')
            if not gr or not gr.get('pairs'):
                continue
            used = True; rec['games'] += 1
            for k in ('outside_pitches', 'in_recommended', 'usual_expected', 'chases', 'chases_expected_league', 'chases_expected_map'):
                rec[k] += gr.get(k, 0)
            if gr.get('log_loss_map') is not None and gr.get('log_loss_league') is not None and gr.get('outside_pitches'):
                sc = rec['scored']; sc['games'] += 1; sc['decisions'] += gr['outside_pitches']; sc['log_loss_map'] += gr['log_loss_map']; sc['log_loss_league'] += gr['log_loss_league']
            bp = rec['by_pricing'].setdefault(pricing, {'games': 0, 'outside_pitches': 0, 'in_recommended': 0, 'usual_expected': 0.0})
            bp['games'] += 1
            for k in ('outside_pitches', 'in_recommended', 'usual_expected'):
                bp[k] += gr.get(k, 0)
            mo = rec['by_month'].setdefault(day[:7], {'games': 0, 'outside_pitches': 0, 'in_recommended': 0, 'usual_expected': 0.0, 'chases': 0, 'chases_expected_league': 0.0, 'chases_expected_map': 0.0})
            mo['games'] += 1
            for k in ('outside_pitches', 'in_recommended', 'usual_expected', 'chases', 'chases_expected_league', 'chases_expected_map'):
                mo[k] += gr.get(k, 0)
            for k, bn in (gr.get('bins') or {}).items():
                if k in rec['bins']:
                    for kk in ('pairs', 'outside', 'chases', 'league', 'map'):
                        rec['bins'][k][kk] += bn.get(kk, 0)
            groups_ = dict(gr.get('by_group') or {})
            if 'chases' in gr:
                groups_[f'season:{day[:4]}'] = {'outside': gr.get('outside_pitches', 0), 'chases': gr['chases'], 'league': gr.get('chases_expected_league', 0.0), 'map': gr.get('chases_expected_map', 0.0)}
            for gk, gv in groups_.items():
                g_ = rec.setdefault('by_group', {}).setdefault(gk, {'outside': 0, 'chases': 0, 'league': 0.0, 'map': 0.0})
                for kk in ('outside', 'chases', 'league', 'map'):
                    g_[kk] += gv.get(kk, 0)
        if used:
            rec['dates'] += 1; rec['first_date'] = rec['first_date'] or day; rec['last_date'] = day
    for d_ in [rec] + list(rec['by_month'].values()):
        for k in ('usual_expected', 'chases_expected_league', 'chases_expected_map'):
            d_[k] = round(d_[k], 1)
    sc = rec['scored']
    if sc['decisions']:
        sc['log_loss_map'] = round(sc['log_loss_map'], 2); sc['log_loss_league'] = round(sc['log_loss_league'], 2)
        sc['map_gain_nats_per_1000'] = round(1000.0 * (sc['log_loss_league'] - sc['log_loss_map']) / sc['decisions'], 2)
    for d_ in rec['by_pricing'].values():
        d_['usual_expected'] = round(d_['usual_expected'], 1)
    for g_ in (rec.get('by_group') or {}).values():
        g_['league'] = round(g_['league'], 1); g_['map'] = round(g_['map'], 1)
    for bn in rec['bins'].values():
        bn['league'] = round(bn['league'], 1); bn['map'] = round(bn['map'], 1)
    return rec


ZONES13 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)   # MLB's zones, catcher's view: 1-9 the strike zone by thirds, 11-14 outside


def zone_counts(T: dict, rows, swing, whiff) -> dict:
    """The standard hot-zone counts on these rows (MLB's zone number per pitch, catcher's view): for all pitches and for
    each pitch family, per zone [zone, pitches, swings, misses, at-bats ended there, hits, total bases], and when the
    table has exit speeds two more: balls in play hit 95 mph or more, and balls in play with a measured speed (TAGS-08).
    Counts only."""
    out = {}
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return out
    z = T['zone'][rows]; g = T['group'][rows]
    sw = swing[rows]; wh = whiff[rows]
    last = T['last_in_pa'][rows] == 1; o7 = T['out7'][rows]
    ab = last & np.isin(o7, (0, 1, 3, 4, 5, 6)); hit = last & np.isin(o7, (3, 4, 5))
    tb = np.where(last & (o7 == 3), 1.0, 0.0) + np.where(last & (o7 == 4), 2.0, 0.0) + np.where(last & (o7 == 5), 4.0, 0.0)
    speeds = 'ls' in T
    if speeds:
        ls = T['ls'][rows].astype(np.float64)
        inplay = np.isfinite(T['spray'][rows].astype(np.float64)) if 'spray' in T else (last & (T['call'][rows] == 1))
        meas = inplay & np.isfinite(ls); hard = meas & (np.nan_to_num(ls) >= 95)
    fams = [('all', np.ones(len(rows), bool))] + [(fname, np.isin(g, codes)) for fname, codes, _, _ in FAMILIES]
    for fname, fm in fams:
        cells = []
        for zz in ZONES13:
            m = fm & (z == zz)
            n = int(m.sum())
            cell = [zz, n, int(sw[m].sum()), int(wh[m].sum()), int(ab[m].sum()), int(hit[m].sum()), int(round(float(tb[m].sum())))]
            if speeds:
                cell += [int(hard[m].sum()), int(meas[m].sum())]
            cells.append(cell)
        out[fname] = cells
    return out


def zone_split(T: dict, rows, swing, whiff, by: str = 'throw_r') -> dict:
    """zone_counts against right-handed and against left-handed opponents, the split every scouting sheet shows:
    {'R': {...}, 'L': {...}} keyed by the pitcher's hand for a hitter (by='throw_r') or the batter's side for a pitcher
    (by='stand_r'); a hand never faced is left out."""
    rows = np.asarray(rows, dtype=np.int64)
    out = {}
    for hand, code in (('R', 1), ('L', 0)):
        rr = rows[T[by][rows] == code] if len(rows) else rows
        if len(rr):
            out[hand] = zone_counts(T, rr, swing, whiff)
    return out


COUNT_GROUPS = ('first', 'ahead', 'behind', 'two')      # the pitcher's view: 0-0; 0-1 and 1-1; more balls than strikes; two strikes


def count_group(balls, strikes) -> np.ndarray:
    """0 the first pitch, 1 the pitcher ahead or even (0-1, 1-1), 2 the pitcher behind (1-0, 2-0, 3-0, 2-1, 3-1), 3 two strikes."""
    b = np.asarray(balls); s = np.asarray(strikes)
    return np.where(s == 2, 3, np.where((b == 0) & (s == 0), 0, np.where(b > s, 2, 1)))


def count_tend(T: dict, rows, swing, whiff) -> dict:
    """A hitter's swings by count group, the usual scouting split: {group: [pitches, swings, misses, pitches out of the
    zone, chases]}, out of the zone by MLB's zone (11-14). Counts only."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return {}
    g = count_group(T['balls'][rows], T['strikes'][rows]); out_ = T['zone'][rows] >= 11
    sw = swing[rows]; wh = whiff[rows]
    res = {}
    for k, name in enumerate(COUNT_GROUPS):
        m = g == k
        res[name] = [int(m.sum()), int(sw[m].sum()), int(wh[m].sum()), int((m & out_).sum()), int(sw[m & out_].sum())]
    return res


def usage_by_count(T: dict, rows) -> dict:
    """A pitcher's pitch types by count group and batter side: {side: {group: [fastballs, breaking balls, offspeed]}}."""
    rows = np.asarray(rows, dtype=np.int64)
    out = {}
    for side, code in (('R', 1), ('L', 0)):
        rr = rows[T['stand_r'][rows] == code] if len(rows) else rows
        if len(rr) < 50:
            continue
        g = count_group(T['balls'][rr], T['strikes'][rr]); grp = T['group'][rr]
        fam = np.where(np.isin(grp, (0, 1, 2, 6)), 0, np.where(np.isin(grp, (3, 4)), 1, 2))
        out[side] = {name: [int(((g == k) & (fam == f)).sum()) for f in range(3)] for k, name in enumerate(COUNT_GROUPS)}
    return out


def spray_counts(T: dict, rows) -> dict | None:
    """Where a hitter's batted balls go, in field thirds (left, center, right, split at 15 degrees either side of
    straightaway), ground balls and balls in the air (line drives and fly balls) apart, plus pop-ups and hard-hit balls
    (95 mph and up, of those measured). Counts only."""
    rows = np.asarray(rows, dtype=np.int64)
    if 'spray' not in T or not len(rows):
        return None
    ang = T['spray'][rows]; tr = T['traj'][rows]; ls = T['ls'][rows]
    ok = np.isfinite(ang)
    if not ok.any():
        return None
    third = np.digitize(np.clip(np.where(ok, ang, 0.0), -45, 45), [-15.0, 15.0])
    air = (tr == 1) | (tr == 2)
    meas = ok & np.isfinite(ls)
    return {'gb': [int(((tr == 0) & ok & (third == k)).sum()) for k in range(3)],
            'air': [int((air & ok & (third == k)).sum()) for k in range(3)],
            'popups': int(((tr == 3) & ok).sum()), 'hard': [int((ls[meas] >= 95).sum()), int(meas.sum())]}


# The running game (PROD-09): public season statistics the simulator already uses (brl_live/running.json.gz, from MLB's
# season statistics and Savant's sprint speed leaderboard): each runner's steals, caught stealing, times on first and
# sprint speed; each pitcher's steals and caught stealing allowed. A season's numbers are used once it is over (as of
# October 1), else the season before, so a past plan never shows numbers from after its date.
RUNNING_PATH = ROOT / 'brl_live' / 'running.json.gz'
_RUNNING: dict = {}


def running_season(asof_day: int) -> tuple[int | None, dict]:
    if 'doc' not in _RUNNING:
        try:
            _RUNNING['doc'] = json.loads(gzip.decompress(RUNNING_PATH.read_bytes()))
        except Exception:
            _RUNNING['doc'] = {}
    seasons = (_RUNNING['doc'] or {}).get('seasons') or {}
    d = date.fromordinal(int(asof_day))
    y = d.year if d >= date(d.year, 10, 1) else d.year - 1
    while y >= 2015 and str(y) not in seasons:
        y -= 1
    return (y, seasons[str(y)]) if str(y) in seasons else (None, {})


def running_league(season: dict) -> dict:
    run = list((season.get('runners') or {}).values()); pit = list((season.get('pitchers') or {}).values())
    att = sum(r.get('sb', 0) + r.get('cs', 0) for r in run); on1 = sum(r.get('on1', 0) for r in run)
    sb = sum(r.get('sb', 0) for r in run)
    return {'att_per_on1': round(att / on1, 4) if on1 else None, 'sb_pct': round(sb / att, 3) if att else None,
            'att_per_bf': round(sum(r.get('sb', 0) + r.get('cs', 0) for r in pit) / max(sum(r.get('bf', 0) for r in pit), 1), 5) if pit else None}


# Percentile ranks the way the public player pages show them (PROD-08): each player's rate among every player with PCT_MIN
# or more plate appearances (hitters) or batters faced (pitchers) over the training window, as the share of those players
# he does better than (a hitter's strikeouts and a pitcher's walks count lower as better). Counts and ranks only.
PCT_MIN = 300
PCT_HITTER = (('avg', 1), ('slg', 1), ('k', -1), ('bb', 1), ('chase', -1), ('whiff', -1), ('hard', 1))
PCT_PITCHER = (('k', 1), ('bb', -1), ('whiff', 1), ('chase', 1), ('hard', -1), ('avg', -1), ('velo', 1))


def league_arsenal(T: dict, rows, swing, whiff, outside) -> dict:
    """Each pitch type's league rates on these rows, the reference the arsenal tables show next to a pitcher's: speed,
    misses per swing, chases, share in the zone and put-aways (strikeouts per two-strike pitch of the type)."""
    if 'sub' not in T:
        return {}
    idx = np.flatnonzero(rows) if np.asarray(rows).dtype == bool else np.asarray(rows, dtype=np.int64)
    sub = T['sub'][idx]; sw = np.asarray(swing)[idx] == 1; wh = np.asarray(whiff)[idx] == 1; out = np.asarray(outside)[idx]
    two = T['strikes'][idx] == 2; k_end = (T['last_in_pa'][idx] == 1) & (T['out7'][idx] == 1); v0 = T['v0'][idx].astype(np.float64)
    res = {}
    for k, code in enumerate(D.SUBTYPES):
        m = sub == k
        if m.sum() < 2000:
            continue
        res[code] = {'pitches': int(m.sum()), 'speed': round(float(np.nanmean(v0[m])), 1),
                     'whiff_rate': round(float((wh & m).sum() / max((sw & m).sum(), 1)), 3),
                     'chase_rate': round(float((sw & m & out).sum() / max((m & out).sum(), 1)), 3),
                     'zone_rate': round(float((m & ~out).sum() / m.sum()), 3),
                     'putaway': round(float((m & two & k_end).sum() / max((m & two).sum(), 1)), 3)}
    return res


def player_lines(T: dict, rows, swing, whiff, outside, key: str) -> dict:
    """Each player's line over these rows as counts, keyed by player id (key 'batter' or 'pitcher'): plate appearances
    (batters faced for a pitcher), at-bats, hits, total bases, home runs, strikeouts, walks (with hit batters; the outcome
    table lumps them), pitches, swings, misses, pitches out of the zone, chases, batted balls with a measured exit speed
    and those at 95 mph or more; for a pitcher also his main fastball's speed (four-seam or sinker, whichever he throws
    more, from 50 or more of them)."""
    rows = np.asarray(rows)
    idx = np.flatnonzero(rows) if rows.dtype == bool else rows.astype(np.int64)
    if not len(idx):
        return {}
    ids, inv = np.unique(T[key][idx], return_inverse=True)
    n = len(ids)
    cnt = lambda m: np.bincount(inv, weights=np.asarray(m, dtype=np.float64), minlength=n)
    last = T['last_in_pa'][idx] == 1; o7 = T['out7'][idx]
    ls = T['ls'][idx].astype(np.float64)
    inplay = np.isfinite(T['spray'][idx]) if 'spray' in T else (last & (T['call'][idx] == 1))
    meas = inplay & np.isfinite(ls)
    tb = np.where(o7 == 3, 1.0, 0.0) + np.where(o7 == 4, 2.0, 0.0) + np.where(o7 == 5, 4.0, 0.0)
    sw = np.asarray(swing)[idx] == 1; out = np.asarray(outside)[idx]
    cols = {'pa': cnt(last), 'ab': cnt(last & np.isin(o7, (0, 1, 3, 4, 5, 6))), 'h': cnt(last & np.isin(o7, (3, 4, 5))),
            'tb': cnt(np.where(last, tb, 0.0)), 'hr': cnt(last & (o7 == 5)), 'k': cnt(last & (o7 == 1)), 'bb': cnt(last & (o7 == 2)),
            'pitches': np.bincount(inv, minlength=n).astype(np.float64), 'swings': cnt(sw), 'misses': cnt(np.asarray(whiff)[idx]),
            'outside': cnt(out), 'chases': cnt(out & sw), 'bbe': cnt(meas), 'hard': cnt(meas & (np.nan_to_num(ls) >= 95))}
    velo = None
    if key == 'pitcher' and 'sub' in T:
        sub = T['sub'][idx]; v0 = T['v0'][idx].astype(np.float64); okv = np.isfinite(v0)
        ff, si = D.SUBTYPES.index('FF'), D.SUBTYPES.index('SI')
        n_ff, n_si = cnt(sub == ff), cnt(sub == si)
        s_ff, c_ff = cnt(np.where((sub == ff) & okv, v0, 0.0)), cnt((sub == ff) & okv)
        s_si, c_si = cnt(np.where((sub == si) & okv, v0, 0.0)), cnt((sub == si) & okv)
        use_ff = n_ff >= n_si
        num, den = np.where(use_ff, s_ff, s_si), np.where(use_ff, c_ff, c_si)
        velo = np.where(den >= 50, num / np.maximum(den, 1.0), np.nan)
    res = {}
    for i, pid in enumerate(ids):
        row = {k: int(round(float(v[i]))) for k, v in cols.items()}
        if velo is not None and np.isfinite(velo[i]):
            row['velo'] = round(float(velo[i]), 1)
        res[int(pid)] = row
    return res


def line_rates(L: dict) -> dict:
    """The rates a line's counts give (None where the denominator is zero)."""
    div = lambda a, b: (L[a] / L[b]) if L.get(b) else None
    r = {'avg': div('h', 'ab'), 'slg': div('tb', 'ab'), 'k': div('k', 'pa'), 'bb': div('bb', 'pa'), 'chase': div('chases', 'outside'),
         'whiff': div('misses', 'swings'), 'hard': div('hard', 'bbe')}
    if L.get('velo') is not None:
        r['velo'] = float(L['velo'])
    return r


def pct_reference(lines: dict, keys) -> dict:
    """metric -> the sorted rates of every player with PCT_MIN or more plate appearances (or batters faced)."""
    q = [line_rates(L) for L in lines.values() if L.get('pa', 0) >= PCT_MIN]
    return {k: np.sort(np.array([x[k] for x in q if x.get(k) is not None], dtype=np.float64)) for k, _ in keys}


def pct_ranks(L: dict | None, ref: dict, keys) -> dict | None:
    """A ranked player's percentile on each metric (1 to 100, higher better for him), or None below PCT_MIN."""
    if not L or L.get('pa', 0) < PCT_MIN:
        return None
    r = line_rates(L); out = {}
    for k, sgn in keys:
        v, a = r.get(k), ref.get(k)
        if v is None or a is None or len(a) < 20:
            continue
        below = int(np.searchsorted(a, v, 'left')); above = int(len(a) - np.searchsorted(a, v, 'right'))
        tie = len(a) - below - above
        p = ((below if sgn > 0 else above) + 0.5 * tie) / len(a)
        out[k] = int(min(100, max(1, round(100 * p))))
    return out or None


def put(repo, token, path, text, branch, message, tries=14):
    """put_text with patience: several backfills commit to the same branch at once, so a 409 is ordinary; and many runs
    writing at once can hit GitHub's secondary rate limit (403 or 429 with a retry hint), which is waited out."""
    import random
    from urllib.error import HTTPError
    limited = 0
    for attempt in range(tries):
        try:
            return D.put_text(repo, token, path, text, branch, message)
        except HTTPError as exc:
            if exc.code in (403, 429):
                try:
                    body = exc.read().decode(errors='replace').lower()
                except Exception:
                    body = ''
                hdr = exc.headers or {}
                rate = 'rate limit' in body or hdr.get('Retry-After') or hdr.get('X-RateLimit-Remaining') == '0'
                if not rate or limited >= 8:
                    raise
                limited += 1
                wait = float(hdr.get('Retry-After') or 0) or 60.0 * (1 + limited / 2)
                print(f'write rate-limited ({exc.code}); waiting {wait:.0f}s', flush=True)
                time.sleep(min(wait, 300) + random.uniform(0, 10))
                continue
            if exc.code not in (409, 422) or attempt == tries - 1:
                raise
            time.sleep(random.uniform(2, 6) * (1 + attempt / 3))


def _rate_wait(exc, limited: int):
    """Seconds to wait out GitHub's rate limit for this error, or None when it is not a rate limit."""
    if exc.code not in (403, 429):
        return None
    try:
        body = exc.read().decode(errors='replace').lower()
    except Exception:
        body = ''
    hdr = exc.headers or {}
    if not ('rate limit' in body or hdr.get('Retry-After') or hdr.get('X-RateLimit-Remaining') == '0'):
        return None
    return float(hdr.get('Retry-After') or 0) or 60.0 * (1 + limited / 2)


def put_many(repo, token, files: dict, branch, message, tries=16, fast_tries=8):
    """Many files in one commit through the Git Data API, so a report day is one commit instead of one per game (fewer
    commits on the ledger branch, fewer 409s for the live runs writing beside it). Each file becomes a blob (blobs do
    not move the branch), then one tree on the branch head, one commit and a fast-forward of the branch; when another
    writer moved the head in between (422 or 409) the tree goes on the new head. When the branch keeps moving faster
    than that (other runs committing every second), the files go one at a time through put() after fast_tries.
    Rate limits are waited out."""
    import random
    from urllib.error import HTTPError
    if not files:
        return None
    git = f'https://api.github.com/repos/{repo}/git'
    limited = 0

    def call(url, method='GET', payload=None, conflicts=False):
        nonlocal limited
        for attempt in range(tries):
            try:
                return D.api(url, token, method, payload)
            except HTTPError as exc:
                wait = _rate_wait(exc, limited)
                if wait is not None and limited < 8:
                    limited += 1
                    print(f'write rate-limited ({exc.code}); waiting {wait:.0f}s', flush=True)
                    time.sleep(min(wait, 300) + random.uniform(0, 10)); continue
                if exc.code >= 500 and attempt < tries - 1:
                    time.sleep(2 + 3 * attempt); continue
                raise
        raise RuntimeError('GitHub API kept failing: ' + url)

    blobs = {path: call(f'{git}/blobs', 'POST', {'content': text, 'encoding': 'utf-8'})['sha'] for path, text in files.items()}
    entries = [{'path': p_, 'mode': '100644', 'type': 'blob', 'sha': sha} for p_, sha in sorted(blobs.items())]
    for attempt in range(fast_tries):
        head = call(f'{git}/ref/heads/{branch}')['object']['sha']
        base_tree = call(f'{git}/commits/{head}')['tree']['sha']
        tree = call(f'{git}/trees', 'POST', {'base_tree': base_tree, 'tree': entries})['sha']
        commit = call(f'{git}/commits', 'POST', {'message': message, 'tree': tree, 'parents': [head]})['sha']
        try:
            call(f'{git}/refs/heads/{branch}', 'PATCH', {'sha': commit, 'force': False})
            return commit
        except HTTPError as exc:
            if exc.code not in (409, 422):
                raise
            time.sleep(random.uniform(0.2, 1.0))
    print(f'branch busy: writing {len(files)} files one at a time ({message})', flush=True)
    for path, text in sorted(files.items(), key=lambda kv: kv[0].endswith('/index.json')):       # a day's index last
        put(repo, token, path, text, branch, message, tries=tries)
    return None


def rebuild_index(repo, token, branch) -> tuple[dict, dict]:
    """The reports index from what is on the branch (never a read-modify-write, which parallel backfills would race):
    every public/reports/<date>/index.json, as {date: summary} and the index document."""
    listing = D.api(f'https://api.github.com/repos/{repo}/contents/public/reports?ref={branch}', token)
    days = {}
    for item in listing if isinstance(listing, list) else []:
        if item.get('type') == 'dir' and len(item.get('name', '')) == 10:
            raw = D.read_blob(repo, token, f"public/reports/{item['name']}/index.json", branch)
            if raw is not None:
                try:
                    days[item['name']] = json.loads(raw.decode())
                except ValueError:
                    pass
    idx = {'schema': 'brl.reports.v1', 'dates': {}}
    for day, doc in sorted(days.items()):
        games = doc.get('games') or {}
        idx['dates'][day] = {'asof': doc.get('asof'), 'games': len(games), 'graded': sum(1 for g in games.values() if (g.get('grade') or {}).get('pairs'))}
    return idx, days


def famof(g: int) -> str:
    return 'fastball' if g in (0, 1, 2, 6) else ('breaking' if g in (3, 4) else 'offspeed')


def clean(o):
    """JSON-safe: NaN and infinities become null (a browser's JSON parser rejects NaN, and one bad number would break a whole day's page)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class Fitted:
    """Everything fitted on the pitches before the as-of date: league models, hitter maps, pools of pitcher spots."""

    def __init__(self, T: dict, asof_day: int, stage, min_pitches=400, min_swings=200):
        F = D.rebuild(T)
        keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
        self.T = T = D.take(T, keep); F = {k: v[keep] for k, v in F.items()}
        self.train = tr = T['day'] < asof_day
        self.asof_day = asof_day
        self.n_train = int(tr.sum())
        swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64); whiff = (T['call'] == 2).astype(np.float64)
        self.swing, self.whiff = swing, whiff
        self.xp, self.zp = D.projected(T, F, None, 'straight', 0.26)
        self.xt, self.zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
        u_t = np.where(T['stand_r'] == 1, self.xt, -self.xt)
        self.outside = (np.abs(u_t) > D.ZONE_HALF) | (self.zt > D.ZONE_TOP) | (self.zt < D.ZONE_BOT)
        rng = self.rng = np.random.default_rng(11)
        # propensities use earlier dates only (as the maps will); the report's games sit after every training pitch
        Tp = dict(T); Tp['day'] = np.where(tr, T['day'], asof_day)
        prop_s = D.swing_propensity(Tp); prop_p = D.pitcher_propensity(Tp)
        self.Ls = D.location_block(self.xp, self.zp, T['stand_r'], T['strikes'])
        self.Bh0 = D.hitter_basis(self.xp, self.zp, T['stand_r'], T['strikes']); nh = self.nh = self.Bh0.shape[1] - 1
        lf = [(self.Bh0[:, :-1] * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32), (self.Bh0[:, :-1] * (T['group'] == 5)[:, None]).astype(np.float32)]
        Xs = np.hstack([self.Ls] + lf + [D.control_block(T, prop_s), prop_p[:, None].astype(np.float32)] +
                       ([D.cross_block(self.xt, self.zt, T['stand_r'], T['group'])] if SWING_CROSS else [])); del lf
        self.i_ps = self.Ls.shape[1] + 2 * nh + 32
        idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
        self.m_s = D.fit_logistic(Xs[idx], swing[idx]); self.off_s = self.m_s.decision_function(Xs); del Xs
        cf_ = self.m_s.coef_[0].astype(np.float64); nL = self.Ls.shape[1]
        self.wL, self.wB, self.wO = cf_[:nL], cf_[nL:nL + nh], cf_[nL + nh:nL + 2 * nh]
        self.wC = cf_[-24:] if SWING_CROSS else None              # SWING-CROSS-01: the crossing block's weights (the last 24 columns)
        self.Bs = D.family_basis(self.Bh0, T['group'])
        stage('league swing model')
        gb_tr = self.gb_tr = D._groups(T['batter'], tr)
        self.maps_s = D._hitter_maps(self.Bs, swing, self.off_s, gb_tr, D.FAMILY_SHRINKAGE_LEAGUE_FAMILY['location_by_family'], min_pitches)
        self.side = {h: int(np.round(T['stand_r'][r].mean())) for h, r in gb_tr.items()}
        Ssum = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
        for h, m in self.maps_s.items():
            Ssum[self.side[h]] = Ssum[self.side[h]] + m; Nn[self.side[h]] += 1
        self.mbar = {h: (Ssum[self.side[h]] - m) / max(Nn[self.side[h]] - 1, 1) for h, m in self.maps_s.items()}
        self.mbar_side = {sd: Ssum[sd] / max(Nn[sd], 1) for sd in (0, 1)}
        stage(f'hitter swing maps {len(self.maps_s)}')
        # whiff model on the true crossing (MATCHUP-03's representation)
        tw = dict(Tp); tw['call'] = np.where(whiff == 1, 1, np.where(swing == 1, 0, 3)); prop_w = D.swing_propensity(tw, 200.0)
        grp = np.zeros((len(swing), 7), np.float32); grp[np.arange(len(swing)), np.clip(T['group'], 0, 6)] = 1
        Lw = D.location_block(self.xt, self.zt, T['stand_r'], T['strikes'])
        Xw = np.hstack([Lw, grp, D.hats(T['v0'].astype(np.float64), D.V_KNOTS), (T['strikes'] == 2)[:, None].astype(np.float32), prop_w[:, None].astype(np.float32),
                        (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)])
        sw_tr = tr & (swing == 1)
        idx = np.flatnonzero(sw_tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
        self.m_w = D.fit_logistic(Xw[idx], whiff[idx]); self.off_w = self.m_w.decision_function(Xw)
        fam = np.column_stack([np.isin(T['group'], (0, 1, 2)), np.isin(T['group'], (3, 4)), np.isin(T['group'], (5,))]).astype(np.float64)
        self.Bw = np.hstack([D.hitter_basis(self.xt, self.zt, T['stand_r'], T['strikes']), fam, D.hats(self.zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
        self.maps_w = D._hitter_maps(self.Bw, whiff, self.off_w, D._groups(T['batter'], sw_tr), float(LAM_W), min_swings)
        Wsum = {0: 0.0, 1: 0.0}; Wn = {0: 0, 1: 0}
        for h, m in self.maps_w.items():
            sd_ = self.side.get(h, 1); Wsum[sd_] = Wsum[sd_] + m; Wn[sd_] += 1
        self.mbar_w = {h: (Wsum[self.side.get(h, 1)] - m) / max(Wn[self.side.get(h, 1)] - 1, 1) for h, m in self.maps_w.items()}
        # called strikes on takes and fouls on contact (league), for the count chain
        plat_ = (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)
        tk = tr & (T['call'] == 0); idx = np.flatnonzero(tk); idx = rng.choice(idx, min(len(idx), 400000), replace=False)
        self.cs_target = None
        if CS_SEASON:
            # CS-SEASON-01: each season's own edge profile; the count chain prices today's pitches with the target season's,
            # and the calibration check reads each row with its own season's
            seas = sorted(set(int(v) for v in np.unique(T['season'][tk])))
            self.cs_target = D.cs_target_season(T['season'], tk, date.fromordinal(asof_day).year)
            e_c = D.edge_distance(self.xt, self.zt, T['stand_r'])
            Xc = np.hstack([Lw, plat_, D.cs_season_block(e_c, T['season'], seas)])
            m_c = D.fit_logistic(Xc[idx], (T['cs'] == 1).astype(np.float64)[idx])
            self.p_cs_own = m_c.predict_proba(Xc)[:, 1]
            Xc[:, -len(seas) * len(D.E_KNOTS):] = D.cs_season_block(e_c, np.full(len(e_c), self.cs_target), seas)
            self.p_cs = m_c.predict_proba(Xc)[:, 1]
        elif CS_RECENT:
            # CS-RECENT-01: every training take, weighted toward the latest days; every row (training or later) read with today's model
            Xc = np.hstack([Lw, plat_]); ia = np.flatnonzero(tk)
            m_c = D.fit_logistic(Xc[ia], (T['cs'] == 1).astype(np.float64)[ia], w=D.recency_weights(T['day'][ia], float(CS_RECENT)))
            self.p_cs = m_c.predict_proba(Xc)[:, 1]; self.p_cs_own = self.p_cs
        else:
            Xc = np.hstack([Lw, plat_])
            self.p_cs = D.fit_logistic(Xc[idx], (T['cs'] == 1).astype(np.float64)[idx]).predict_proba(Xc)[:, 1]
            self.p_cs_own = self.p_cs
        ct = tr & (T['call'] == 1); foul = ((T['call'] == 1) & (T['last_in_pa'] == 0)).astype(np.float64)
        idx = np.flatnonzero(ct); idx = rng.choice(idx, min(len(idx), 400000), replace=False)
        self.p_fo = D.fit_logistic(Xw[idx], foul[idx]).predict_proba(Xw)[:, 1]
        del Xw, Xc
        stage(f'whiff maps {len(self.maps_w)}')
        # batter zones from the training rows' official zone numbers (MATCHUP-08)
        top_b, bot_b, _ = D.batter_zones(T, tr)
        self.zone = {}
        for h, r in gb_tr.items():
            self.zone[h] = (float(top_b[r[0]]), float(bot_b[r[0]]))
        # league reference rates on the training rows, for the cards
        u_p = np.where(T['stand_r'] == 1, self.xp, -self.xp)
        looks_in = ~((np.abs(u_p) > D.ZONE_HALF) | (self.zp > D.ZONE_TOP) | (self.zp < D.ZONE_BOT))
        self.looks_in_ends_out = looks_in & self.outside
        self.league = {'chase_rate': round(float(swing[tr & self.outside].mean()), 3), 'whiff_rate': round(float(whiff[tr & (swing == 1)].sum() / max((tr & (swing == 1)).sum(), 1)), 3),
                       'looks_in_ends_out': round(float(self.looks_in_ends_out[tr].mean()), 3)}
        # the standard hot-zone counts for every hitter of each side together, the reference the cards are colored against
        # (by the hitter's side, then the pitcher's hand)
        self.league['zones'] = {sd_: zone_split(T, np.flatnonzero(tr & (T['stand_r'] == (1 if sd_ == 'R' else 0))), swing, whiff) for sd_ in ('R', 'L')}
        self.league['spray'] = {sd_: spray_counts(T, np.flatnonzero(tr & (T['stand_r'] == (1 if sd_ == 'R' else 0)))) for sd_ in ('R', 'L')}
        self.league['counts'] = {sd_: count_tend(T, np.flatnonzero(tr & (T['stand_r'] == (1 if sd_ == 'R' else 0))), swing, whiff) for sd_ in ('R', 'L')}
        self.league['arsenal'] = league_arsenal(T, tr, swing, whiff, self.outside)         # each pitch type's league rates, for the arsenal tables
        # every player's line on the training rows and the percentile references (PROD-08)
        self.lines_h = player_lines(T, tr, swing, whiff, self.outside, 'batter')
        self.lines_p = player_lines(T, tr, swing, whiff, self.outside, 'pitcher')
        self.pct_ref_h, self.pct_ref_p = pct_reference(self.lines_h, PCT_HITTER), pct_reference(self.lines_p, PCT_PITCHER)
        self.league['ranked'] = {'min': PCT_MIN, 'hitters': int(sum(1 for L in self.lines_h.values() if L['pa'] >= PCT_MIN)),
                                 'pitchers': int(sum(1 for L in self.lines_p.values() if L['pa'] >= PCT_MIN))}
        # pitcher tag counts (PTAGS-01) on the training rows, and the league's rates
        ptc = pitcher_tag_counts(T, tr)
        self.ptags = {int(pid): {k: [int(ptc[k][0][i]), int(ptc[k][1][i])] for k in ('fps', 'zone', 'chase', 'fb_behind', 'fb_first')} for i, pid in enumerate(ptc['ids'])}
        self.league['ptags'] = {k: round(float(ptc[k][0].sum() / max(ptc[k][1].sum(), 1)), 4) for k in ('fps', 'zone', 'chase', 'fb_behind', 'fb_first')}
        self.run_year, self.run_season = running_season(asof_day)
        if self.run_year:
            self.league['run'] = dict(running_league(self.run_season), season=self.run_year)
        # the pitcher's training pitches
        self.gp = D._groups(T['pitcher'], tr)
        self.cgrp = np.where(T['strikes'] == 2, 2, np.where(T['balls'] > T['strikes'], 1, 0))
        self.cidx = T['balls'].astype(np.int64) * 3 + T['strikes'].astype(np.int64)
        K = 16; jit = np.random.default_rng(3).standard_normal((K, 2)); self.jit = (jit - jit.mean(0)) / jit.std(0); self.K = K
        self.pools = {}; self.chains = {}
        # VALUE-18: the engine's components for pricing aims (the planner's fitted pieces) and the training count values
        self.PM = None
        if STRUCTURAL:
            self.PM = D.PAModels(T, tr, rng, {'league_n': 500000, 'min_pitches': min_pitches, 'min_swings': min_swings, 'swing_cross': SWING_CROSS,
                                               'cs_season': CS_SEASON, 'cs_target': self.cs_target, 'cs_recent': CS_RECENT, 'foul_prop': FOUL_FIX, 'foul_tip': FOUL_FIX, 'lam_w': LAM_W, 'whiff_own': WHIFF_OWN}, stage)
            ci_all = np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)
            fin_all = np.where(T['out7'] >= 0, D.LW7[np.clip(T['out7'], 0, 6)], np.nan)
            self.cv = np.array([float(np.nanmean(fin_all[tr & (ci_all == c_)])) if (tr & (ci_all == c_) & np.isfinite(fin_all)).any() else 0.0 for c_ in range(12)])
            self.xoff, self.zoff = self.xp - self.xt, self.zp - self.zt
        # grids
        UU, ZZ = np.meshgrid(GU, GZ); self.uu, self.zz = UU.ravel(), ZZ.ravel()
        self.Bg = D.hitter_basis(self.uu, self.zz, np.ones(len(self.uu), np.int64), np.zeros(len(self.uu), np.int64))
        self.Bgw = np.hstack([self.Bg, np.tile(np.array([1.0, 0.0, 0.0]), (len(self.uu), 1)), D.hats(self.zz, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
        self.out_cells = [k for k in range(len(self.uu)) if (abs(self.uu[k]) > D.ZONE_HALF or self.zz[k] > D.ZONE_TOP or self.zz[k] < D.ZONE_BOT)]
        stage('fitted')

    def league_loc(self, LBm, Bh_m, g):
        return LBm.astype(np.float64) @ self.wL + np.isin(g, (3, 4)) * (Bh_m[:, :-1] @ self.wB) + (g == 5) * (Bh_m[:, :-1] @ self.wO)

    def hitter_card(self, h: int) -> dict | None:
        if h not in self.maps_s:
            return None
        T = self.T; r = self.gb_tr.get(h)
        sd = self.side[h]
        own = {}
        for fname, _, gcode, _ in FAMILIES:
            dgf = D.family_basis(self.Bg, np.full(len(self.uu), gcode, np.int64)) @ (self.maps_s[h] - self.mbar[h])
            own[fname] = {'grid': [round(float(v), 2) for v in dgf],
                          'top': [[float(self.uu[k]), float(self.zz[k]), round(float(dgf[k]), 3)] for k in sorted(self.out_cells, key=lambda k: -dgf[k])[:3] if dgf[k] > 0]}
        card = {'side': 'R' if sd == 1 else 'L', 'pitches': int(len(r)) if r is not None else 0, 'own_chase': own,
                'chase_rate': round(float(self.swing[r][self.outside[r]].mean()), 3) if r is not None and self.outside[r].any() else None,
                'whiff_rate': round(float(self.whiff[r].sum() / max(self.swing[r].sum(), 1)), 3) if r is not None else None}
        if h in self.maps_w:
            dw = self.Bgw @ self.maps_w[h]
            card['whiff_grid'] = [round(float(v), 2) for v in dw]
        if h in self.zone:
            card['zone_top_ft'], card['zone_bottom_ft'] = round(self.zone[h][0], 2), round(self.zone[h][1], 2)
        if r is not None and len(r):
            card['zones'] = zone_split(T, r, self.swing, self.whiff)        # by the pitcher's hand; the page adds the two for all pitchers
            sp = spray_counts(T, r)
            if sp:
                card['spray'] = sp
            card['counts'] = count_tend(T, r, self.swing, self.whiff)
        rn = (getattr(self, 'run_season', {}) or {}).get('runners', {}).get(str(int(h)))
        if rn:
            card['run'] = {'season': self.run_year, **{k: rn[k] for k in ('sb', 'cs', 'on1', 'sprint') if k in rn}}
        L = getattr(self, 'lines_h', {}).get(int(h))
        if L:
            card['line'] = L
            pr = pct_ranks(L, self.pct_ref_h, PCT_HITTER)
            if pr:
                card['pct'] = pr
        if self.PM is not None and r is not None and len(r) >= 300:
            # VALUE-18 from the hitter's side: what his own swing tendencies cost him on the pitches he actually saw, against the average
            # hitter his side at the same pitches (his calibrated own part times the value of a swing against a take), per 600 plate
            # appearances, and the cells where it costs him most (outside the zone: chases; inside: strikes he lets go or swings at to little effect)
            T = self.T
            blk = self.PM.blocks_at(self.xp[r], self.zp[r], self.xt[r], self.zt[r], T['stand_r'][r], T['throw_r'][r], T['group'][r], T['v0'][r].astype(np.float64), T['balls'][r], T['strikes'][r])
            pids = T['pitcher'][r]
            ps_ = np.array([self.PM.p_scalar.get(int(q), (0.0, self.PM.lg_bip))[0] for q in pids]); pb_ = np.array([self.PM.p_scalar.get(int(q), (0.0, self.PM.lg_bip))[1] for q in pids])
            tau = 0.01 * self.PM.swing_minus_take(blk, h, (ps_, pb_), T['balls'][r], T['strikes'][r], self.cv)
            dev = (sig(self.off_s[r] + self.Bs[r] @ self.maps_s[h]) - sig(self.off_s[r] + self.Bs[r] @ self.mbar[h])) * 100
            lam_r = np.where(self.outside[r], OWN_PART_CALIBRATION['outside'], OWN_PART_CALIBRATION['inside'])
            cost = dev * lam_r * tau                   # runs per pitch, negative = the hitter loses against the average hitter his side
            n_pa = max(int((T['pitch_no'][r] == 0).sum()), 1)
            uu_ = np.where(sd == 1, self.xt[r], -self.xt[r])
            cell = np.argmin(np.abs(uu_[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[r][:, None] - GZ[None, :]), 1)
            by_cell = np.bincount(cell, weights=cost, minlength=len(GU) * len(GZ)) / n_pa * 600
            worst = [k for k in np.argsort(by_cell)[:4] if by_cell[k] < -0.05]
            card['own_cost'] = {'runs_per_600_pa': round(float(cost.sum() / n_pa * 600), 2),
                                'outside_runs_per_600_pa': round(float(cost[self.outside[r]].sum() / n_pa * 600), 2),
                                'inside_runs_per_600_pa': round(float(cost[~self.outside[r]].sum() / n_pa * 600), 2),
                                'cells': [[float(GU[k % len(GU)]), float(GZ[k // len(GU)]), round(float(by_cell[k]), 2),
                                           ('chase' if (abs(GU[k % len(GU)]) > D.ZONE_HALF or GZ[k // len(GU)] > D.ZONE_TOP or GZ[k // len(GU)] < D.ZONE_BOT) else 'take' if dev[cell == k].mean() < 0 else 'weak')] for k in worst],
                                'pitches': int(len(r))}
        return card

    def pitcher_card(self, p: int) -> dict | None:
        r = self.gp.get(p)
        if r is None or len(r) < 150:
            return None
        T = self.T; mix = {}
        for fname, groups, _, _ in FAMILIES:
            mm = np.isin(T['group'][r], groups)
            if mm.any():
                mix[fname] = {'share': round(float(mm.mean()), 3), 'speed': round(float(np.nanmean(T['v0'][r][mm])), 1)}
                # each pitch type in a scout's terms: misses per swing, chases, strikes by location, and where it goes to each side
                rr = r[mm]; sw = self.swing[rr]; out = self.outside[rr]
                if sw.sum() >= 25:
                    mix[fname]['whiff_rate'] = round(float(self.whiff[rr].sum() / sw.sum()), 3)
                if out.sum() >= 30:
                    mix[fname]['chase_rate'] = round(float(sw[out].mean()), 3)
                if len(rr) >= 40:
                    mix[fname]['zone_rate'] = round(float((~out).mean()), 3)
                # put-away rate: strikeouts on this pitch over the times he threw it with two strikes
                two = T['strikes'][rr] == 2
                if two.sum() >= 30:
                    mix[fname]['putaway'] = round(float((two & (T['last_in_pa'][rr] == 1) & (T['out7'][rr] == 1)).sum() / two.sum()), 3)
                for sd, key in ((1, 'to_righties'), (0, 'to_lefties')):
                    m2 = T['stand_r'][rr] == sd
                    if m2.sum() < 40:
                        continue
                    uu = np.where(sd == 1, self.xt[rr][m2], -self.xt[rr][m2]); zz = self.zt[rr][m2]
                    cell = np.argmin(np.abs(uu[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(zz[:, None] - GZ[None, :]), 1)
                    cnt = np.bincount(cell, minlength=len(GU) * len(GZ))
                    mix[fname][key] = [[float(GU[k % len(GU)]), float(GZ[k // len(GU)]), round(float(cnt[k] / m2.sum()), 3)] for k in np.argsort(-cnt)[:3] if cnt[k] > 0]
        # the arsenal by pitch type, the way the public pitch pages list it: share, speed, spin, movement (PITCHf/x inches over
        # the last 40 feet; horizontal positive to his arm side), misses per swing, chases, strikes and put-aways
        # the latest season's pitches when there are 300 or more of them (speeds and mixes move from season to season, and the
        # public pitch pages list each season), else both seasons
        yr = date.fromordinal(self.asof_day).year if hasattr(self, 'asof_day') else None
        r_cur = r[T['season'][r] == yr] if yr is not None and 'season' in T else r[:0]
        ra = r_cur if len(r_cur) >= 300 else r
        arsenal = []
        sub = T['sub'][ra]; rh = T['throw_r'][r][0] == 1
        for k, code in enumerate(D.SUBTYPES):
            mm = sub == k
            if mm.sum() < max(20, 0.01 * len(ra)):
                continue
            rr = ra[mm]; sw = self.swing[rr]; out = self.outside[rr]; two = T['strikes'][rr] == 2
            fin = lambda v: None if not np.isfinite(v) else v
            spin = fin(float(np.nanmean(np.where(T['spin'][rr] > 0, T['spin'][rr], np.nan)))) if np.isfinite(T['spin'][rr]).any() else None
            arsenal.append({'type': code, 'pitches': int(mm.sum()), 'share': round(float(mm.mean()), 3),
                            'speed': round(float(np.nanmean(T['v0'][rr])), 1),
                            'spin': int(round(spin)) if spin else None,
                            'v_mov': round(float(np.nanmean(T['pfx_z'][rr])), 1) if np.isfinite(T['pfx_z'][rr]).any() else None,
                            'h_arm': round(float(np.nanmean(T['pfx_x'][rr])) * (-1.0 if rh else 1.0), 1) if np.isfinite(T['pfx_x'][rr]).any() else None,
                            'whiff_rate': round(float(self.whiff[rr].sum() / sw.sum()), 3) if sw.sum() >= 25 else None,
                            'chase_rate': round(float(sw[out].mean()), 3) if out.sum() >= 30 else None,
                            'zone_rate': round(float((~out).mean()), 3),
                            'putaway': round(float((two & (T['last_in_pa'][rr] == 1) & (T['out7'][rr] == 1)).sum() / two.sum()), 3) if two.sum() >= 30 else None})
        arsenal.sort(key=lambda a: -a['share'])
        # how he is used, for the bullpen card: appearances, starts, the inning he comes in and batters per relief outing
        games_ = T['game'][r]; inn_ = T['inning'][r]
        ug, gi = np.unique(games_, return_inverse=True)
        entry = np.full(len(ug), 99); np.minimum.at(entry, gi, inn_)
        bf = np.bincount(gi, weights=(T['last_in_pa'][r] == 1).astype(np.float64), minlength=len(ug))
        rel = entry > 1
        role = {'apps': int(len(ug)), 'starts': int((~rel).sum()), 'relief': int(rel.sum())}
        if rel.sum():
            role.update({'ninth_share': round(float((entry[rel] >= 9).mean()), 3), 'entry_inning': round(float(entry[rel].mean()), 2),
                         'bf_per_relief': round(float(bf[rel].mean()), 2)})
        card = {'throws': 'R' if T['throw_r'][r][0] == 1 else 'L', 'pitches': int(len(r)), 'mix': mix, 'arsenal': arsenal, 'role': role,
                'arsenal_season': int(yr) if ra is r_cur else None,
                'chase_rate_against': round(float(self.swing[r][self.outside[r]].mean()), 3) if self.outside[r].any() else None,
                'looks_in_ends_out': round(float(self.looks_in_ends_out[r].mean()), 3),
                'zones': zone_split(T, r, self.swing, self.whiff, by='stand_r'),      # where he throws, by the batter's side
                'usage': usage_by_count(T, r)}                                        # what he throws, by count and batter side
        pt = getattr(self, 'ptags', {}).get(int(p))
        if pt:
            card['ptags'] = pt
        rp = (getattr(self, 'run_season', {}) or {}).get('pitchers', {}).get(str(int(p)))
        if rp:
            card['run_against'] = {'season': self.run_year, **{k: rp[k] for k in ('sb', 'cs', 'bf') if k in rp}}
        L = getattr(self, 'lines_p', {}).get(int(p))
        if L:
            card['line'] = L
            pr = pct_ranks(L, self.pct_ref_p, PCT_PITCHER)
            if pr:
                card['pct'] = pr
        return card

    def _chain(self, ps_, pw_, pcs_, pfo_, ci_):
        use_c = np.bincount(ci_, minlength=12) >= 15; ki = ci_ % 3
        def agg(w):
            return np.where(use_c, np.bincount(ci_, weights=w, minlength=12), np.bincount(ki, weights=w, minlength=3)[np.arange(12) % 3])
        n_ = agg(np.ones(len(ps_))); s_ = agg(ps_); sw_ = agg(ps_ * pw_); t_ = agg(1 - ps_); tc_ = agg((1 - ps_) * pcs_)
        c_ = agg(ps_ * (1 - pw_)); cf_ = agg(ps_ * (1 - pw_) * pfo_)
        d = lambda v: {(b, k): float(v[b * 3 + k]) for b in range(4) for k in range(3)}
        return D._count_chain(d(s_ / np.maximum(n_, 1e-9)), d(sw_ / np.maximum(s_, 1e-9)), d(tc_ / np.maximum(t_, 1e-9)), d(cf_ / np.maximum(c_, 1e-9)))

    def _arsenal(self, p, sd):
        key = (p, sd)
        if key in self.chains:
            return self.chains[key]
        T = self.T; r = self.gp[p]; side = T['stand_r'][r] == sd
        r = r[side] if side.sum() >= 100 else r
        A = {'r': r, 'ci': self.cidx[r], 'pcs': self.p_cs[r], 'pfo': self.p_fo[r], 'os': self.off_s[r], 'ow': self.off_w[r], 'Bs': self.Bs[r], 'Bw': self.Bw[r], 'out': self.outside[r]}
        A['league'] = self._chain(sig(A['os']), sig(A['ow']), A['pcs'], A['pfo'], A['ci'])
        if len(self.chains) >= POOL_CACHE // 2:
            for k_ in list(self.chains)[:POOL_CACHE // 8]:
                del self.chains[k_]
        self.chains[key] = A
        return A

    def _pool(self, p, sd, c3, tg, zone='outside'):
        key = (p, sd, c3, tg, zone)
        if key in self.pools:
            return self.pools[key]
        T = self.T; r = self.gp[p]
        zm = self.outside[r] if zone == 'outside' else ~self.outside[r]
        r = r[(T['stand_r'][r] == sd) & (self.cgrp[r] == c3) & zm & (np.clip(T['group'][r], 0, 6) == tg)]
        if len(r) < 15:
            self.pools[key] = None; return None
        n_all = len(r)
        if len(r) > 250:
            r = self.rng.choice(r, 250, replace=False)
        pl_ = sig(self.off_s[r]); third = np.searchsorted(np.percentile(pl_, [33.3, 66.7]), pl_)
        lb0 = self.league_loc(self.Ls[r], self.Bh0[r], T['group'][r]); K = self.K; sgm = 0.6
        xj = (self.xp[r][:, None] + sgm * self.jit[None, :, 0]).ravel(); zj = (self.zp[r][:, None] + sgm * self.jit[None, :, 1]).ravel()
        sj = np.repeat(np.full(len(r), sd), K); kj = np.repeat(T['strikes'][r], K); gj = np.repeat(T['group'][r], K)
        Bj0 = D.hitter_basis(xj, zj, sj, kj)
        offj = np.repeat(self.off_s[r] - lb0, K) + self.league_loc(D.location_block(xj, zj, sj, kj), Bj0, gj)
        if self.wC is not None:
            # the scattered pitch crosses where it lands: swap the crossing part of the league swing logit
            xtj_ = xj - np.repeat(self.xp[r] - self.xt[r], K); ztj_ = zj - np.repeat(self.zp[r] - self.zt[r], K)
            c0 = D.cross_block(self.xt[r], self.zt[r], T['stand_r'][r], T['group'][r]).astype(np.float64) @ self.wC
            offj = offj - np.repeat(c0, K) + D.cross_block(xtj_, ztj_, sj, gj).astype(np.float64) @ self.wC
        uu_ = np.where(sd == 1, self.xt[r], -self.xt[r])
        cell = np.argmin(np.abs(uu_[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[r][:, None] - GZ[None, :]), 1)
        struct = None
        if self.PM is not None:
            # the scattered pitch's true crossing (the same scatter), its side of the zone and the engine's blocks there
            xtj = xj - np.repeat(self.xoff[r], K); ztj = zj - np.repeat(self.zoff[r], K)
            uj = np.where(sd == 1, xtj, -xtj); e_j = np.maximum(np.maximum(np.abs(uj) - D.ZONE_HALF, ztj - D.ZONE_TOP), D.ZONE_BOT - ztj)
            lam_j = np.where(e_j > 0, OWN_PART_CALIBRATION['outside'], OWN_PART_CALIBRATION['inside'])
            bj_ = np.repeat(T['balls'][r], K)
            blk = self.PM.blocks_at(xj, zj, xtj, ztj, sj, np.repeat(T['throw_r'][r], K), gj, np.repeat(T['v0'][r].astype(np.float64), K), bj_, kj)
            land = np.argmin(np.abs(uj[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(ztj[:, None] - GZ[None, :]), 1)   # where each scattered pitch lands
            struct = (blk, bj_, kj, lam_j, e_j > 0, land)
        if len(self.pools) >= POOL_CACHE:
            # the pools of a pitcher are reused across the hitters of one lineup, which are consecutive calls; a month of pitchers would
            # otherwise hold several gigabytes (a run reached 14.8 GB of 16 on the runner), so the oldest entries go first
            for k_ in list(self.pools)[:POOL_CACHE // 4]:
                del self.pools[k_]
        self.pools[key] = (offj, D.family_basis(Bj0, gj), third, cell, len(r), n_all, struct)
        return self.pools[key]

    def head_to_head(self, hitters, staff) -> dict:
        """(hitter, pitcher) -> [plate appearances, at-bats, hits, total bases, home runs, strikeouts, walks] over the
        training pitches, for every pair of these hitters and arms that met (the standard head-to-head line). Counts only."""
        T = self.T
        if not len(hitters) or not len(staff):
            return {}
        m = self.train & (T['last_in_pa'] == 1) & np.isin(T['batter'], np.asarray(hitters, dtype=np.int64)) & np.isin(T['pitcher'], np.asarray(staff, dtype=np.int64))
        idx = np.flatnonzero(m)
        if not len(idx):
            return {}
        o7 = T['out7'][idx]
        u, inv = np.unique(np.stack([T['batter'][idx], T['pitcher'][idx]], 1), axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        cnt = lambda mm: np.bincount(inv, weights=np.asarray(mm, dtype=np.float64), minlength=len(u))
        tb = np.where(o7 == 3, 1.0, 0.0) + np.where(o7 == 4, 2.0, 0.0) + np.where(o7 == 5, 4.0, 0.0)
        cols = [np.bincount(inv, minlength=len(u)).astype(np.float64), cnt(np.isin(o7, (0, 1, 3, 4, 5, 6))), cnt(np.isin(o7, (3, 4, 5))), cnt(tb),
                cnt(o7 == 5), cnt(o7 == 1), cnt(o7 == 2)]
        return {(int(b), int(q)): [int(round(float(c_[i]))) for c_ in cols] for i, (b, q) in enumerate(u)}

    def pair(self, h: int, p: int) -> dict | None:
        """One hitter against one pitcher: chase, strikeout and walk changes, and the aim plan with its value."""
        if h not in self.maps_s or p not in self.gp or len(self.gp[p]) < 150:
            return None
        sd = self.side[h]; A = self._arsenal(p, sd)
        ph = sig(A['os'] + A['Bs'] @ self.maps_s[h]); pl = sig(A['os'])
        wmap = self.maps_w.get(h)
        hit = self._chain(ph, sig(A['ow'] + (A['Bw'] @ wmap if wmap is not None else 0.0)), A['pcs'], A['pfo'], A['ci'])
        chase = float((ph - pl)[A['out']].mean()) * 100 * CHASE_SLOPE if A['out'].any() else 0.0
        out = {'chase_points': round(chase, 2), 'k_points': round((hit[0] - A['league'][0]) * 100 * K_SCALE, 2), 'bb_points': round((hit[1] - A['league'][1]) * 100 * BB_SCALE, 2),
               'arsenal_pitches': int(len(A['r']))}
        T = self.T; rp = self.gp[p][(T['stand_r'][self.gp[p]] == sd) & self.outside[self.gp[p]]]
        # how much the numbers rest on: the hitter's training pitches behind his map, the pitcher's pitches to this side (the audit's insufficient-evidence state)
        n_h = int(len(self.gb_tr.get(h, ()))); n_ps = int((T['stand_r'][self.gp[p]] == sd).sum())
        out['evidence'] = {'hitter_pitches': n_h, 'pitcher_pitches_to_side': n_ps, 'whiff_map': bool(h in self.maps_w),
                           'level': 'thin' if (n_h < 700 or n_ps < 150) else 'ok'}
        if len(rp) < 40:
            return out
        tot = 0.0; wsum = 0.0; tot_runs = 0.0; cells = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
        # where each chosen spot's value comes from: the scattered pitches that stay outside (his extra chases) or land inside (strikes he takes)
        v_out = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}; v_in = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
        land_n = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}; land_v = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
        by_count = {}
        # by family: the value per pitch of aiming that family's outside pitches at its best third (which family to lean on, per count and overall)
        fam_runs = {f_: 0.0 for f_, *_ in FAMILIES}; fam_w = {f_: 0.0 for f_, *_ in FAMILIES}

        def labeled(c_, vo_, vi_):
            return [[float(GU[k % len(GU)]), float(GZ[k // len(GU)]), 'chase' if vo_[k] <= vi_[k] else 'take'] for k in np.argsort(c_)[::-1][:3] if c_[k] > 0]
        p_scalar = self.PM.p_scalar.get(p, (0.0, self.PM.lg_bip)) if self.PM is not None else None
        for c3, cname in ((0, 'even_or_ahead'), (1, 'behind'), (2, 'two_strikes')):
            ct = 0.0; cw = 0.0; ct_runs = 0.0; ccells = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
            cv_out = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}; cv_in = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
            cfam_runs = {f_: 0.0 for f_, *_ in FAMILIES}; cfam_w = {f_: 0.0 for f_, *_ in FAMILIES}
            for tg in range(7):
                P_ = self._pool(p, sd, c3, tg)
                if P_ is None:
                    continue
                offj, Bj, third, cell, npool, n_all, struct = P_
                dev_jk = (sig(offj + Bj @ self.maps_s[h]) - sig(offj + Bj @ self.mbar[h])) * 100
                dj = dev_jk.reshape(npool, self.K).mean(1)
                if struct is not None:
                    # VALUE-18: the structural value of each aim, runs per pitch (negative is good for the pitcher): the calibrated own part
                    # times the value if the hitter swings minus if he takes, averaged over the scatter
                    blk, bj_, kj, lam_j, out_j, land = struct
                    tau_j = 0.01 * self.PM.swing_minus_take(blk, h, p_scalar, bj_, kj, self.cv)
                    vjk = dev_jk * lam_j * tau_j
                    vj = vjk.reshape(npool, self.K).mean(1)
                    vo_j = (vjk * out_j).reshape(npool, self.K).mean(1); vi_j = (vjk * ~out_j).reshape(npool, self.K).mean(1)
                gains = []; gains_runs = []
                for t3 in range(3):
                    sel = np.flatnonzero(third == t3)
                    if len(sel) >= 3:
                        k3 = max(1, len(sel) // 3)
                        best = sel[np.argsort(vj[sel])[:k3]] if struct is not None else sel[np.argsort(dj[sel])[::-1][:k3]]
                        gains.append(float(dj[best].mean() - dj[sel].mean()))
                        if struct is not None:
                            gains_runs.append(float(vj[best].mean() - vj[sel].mean()))
                            np.add.at(v_out[famof(tg)], cell[best], n_all * vo_j[best]); np.add.at(v_in[famof(tg)], cell[best], n_all * vi_j[best])
                            np.add.at(cv_out[famof(tg)], cell[best], n_all * vo_j[best]); np.add.at(cv_in[famof(tg)], cell[best], n_all * vi_j[best])
                            # the landing cloud of the chosen aims: where the scattered pitches land and what each landing cell costs
                            pts = (best[:, None] * self.K + np.arange(self.K)[None, :]).ravel()
                            np.add.at(land_n[famof(tg)], land[pts], n_all / len(pts)); np.add.at(land_v[famof(tg)], land[pts], n_all * vjk[pts] / len(pts))
                        np.add.at(cells[famof(tg)], cell[best], n_all / len(rp)); np.add.at(ccells[famof(tg)], cell[best], n_all)
                if gains:
                    tot += n_all * float(np.mean(gains)); wsum += n_all; ct += n_all * float(np.mean(gains)); cw += n_all
                    if gains_runs:
                        tot_runs += n_all * float(np.mean(gains_runs)); ct_runs += n_all * float(np.mean(gains_runs))
                        fam_runs[famof(tg)] += n_all * float(np.mean(gains_runs)); fam_w[famof(tg)] += n_all
                        cfam_runs[famof(tg)] += n_all * float(np.mean(gains_runs)); cfam_w[famof(tg)] += n_all
            if cw > 0:
                by_count[cname] = {'gain_points': round(ct / cw, 2), 'pitches': int(cw),
                                   'cells': {f_: labeled(c_, cv_out[f_], cv_in[f_]) for f_, c_ in ccells.items()}}
                if self.PM is not None:
                    by_count[cname]['runs_per_100_pa'] = round(float(ct_runs / cw * N_OUT) * 100, 2)
                    by_count[cname]['by_family'] = {f_: {'runs_per_100_pitches': round(float(cfam_runs[f_] / cfam_w[f_]) * 100, 2), 'pitches': int(cfam_w[f_])} for f_ in cfam_w if cfam_w[f_] > 0}
        if wsum > 0:
            g_ = tot / wsum
            out['aim'] = {'runs_per_100_pa': round(float(tot_runs / wsum * N_OUT) * 100, 2) if self.PM is not None else round(float(B_OUT_FAMILY * g_ * N_OUT) * 100, 2),
                          'gain_points': round(g_, 2),
                          'cells': {f_: labeled(c_, v_out[f_], v_in[f_]) for f_, c_ in cells.items()},
                          'by_count': by_count, 'pricing': 'structural' if self.PM is not None else 'regression'}
            if self.PM is not None:
                out['aim']['runs_per_100_pa_regression'] = round(float(B_OUT_FAMILY * g_ * N_OUT) * 100, 2)
                out['aim']['by_family'] = {f_: {'runs_per_100_pitches': round(float(fam_runs[f_] / fam_w[f_]) * 100, 2), 'pitches': int(fam_w[f_])} for f_ in fam_w if fam_w[f_] > 0}
                # the dangerous miss, by family: among where the chosen aims' scattered pitches land, the cell that gives the hitter the most
                # (its share of landings and its cost per 100 plate appearances), and the landing cloud itself (share by cell)
                dm = {}
                for f_ in land_n:
                    tot_f = float(land_n[f_].sum())
                    if tot_f <= 0:
                        continue
                    k_ = int(np.argmax(land_v[f_]))
                    if land_v[f_][k_] <= 0:
                        continue
                    dm[f_] = {'cell': [float(GU[k_ % len(GU)]), float(GZ[k_ // len(GU)])], 'share': round(float(land_n[f_][k_] / tot_f), 3),
                              'runs_per_100_pa': round(float(land_v[f_][k_] / tot_f * N_OUT) * 100, 2),
                              'cloud': [round(float(v_ / tot_f), 3) for v_ in land_n[f_]]}
                if dm:
                    out['aim']['dangerous_miss'] = dm
        # VALUE-18I: strike spots, the same pricing over the pitcher's in-zone spots (he takes them, or swings at them to little effect)
        if STRIKE_SPOTS and self.PM is not None:
            rpi = self.gp[p][(T['stand_r'][self.gp[p]] == sd) & ~self.outside[self.gp[p]]]
            if len(rpi) >= 40:
                tot_i = 0.0; wsum_i = 0.0; cells_i = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}; dev_i = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
                by_count_i = {}
                for c3, cname in ((0, 'even_or_ahead'), (1, 'behind'), (2, 'two_strikes')):
                    ct_i = 0.0; cw_i = 0.0; ccells_i = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}; cdev_i = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
                    for tg in range(7):
                        P_ = self._pool(p, sd, c3, tg, 'inside')
                        if P_ is None or P_[6] is None:
                            continue
                        offj, Bj, third, cell, npool, n_all, struct = P_
                        dev_jk = (sig(offj + Bj @ self.maps_s[h]) - sig(offj + Bj @ self.mbar[h])) * 100
                        dj = dev_jk.reshape(npool, self.K).mean(1)
                        blk, bj_, kj, lam_j, out_j, land = struct
                        tau_j = 0.01 * self.PM.swing_minus_take(blk, h, p_scalar, bj_, kj, self.cv)
                        vj = (dev_jk * lam_j * tau_j).reshape(npool, self.K).mean(1)
                        gains_i = []
                        for t3 in range(3):
                            sel = np.flatnonzero(third == t3)
                            if len(sel) >= 3:
                                k3 = max(1, len(sel) // 3); best = sel[np.argsort(vj[sel])[:k3]]
                                gains_i.append(float(vj[best].mean() - vj[sel].mean()))
                                np.add.at(cells_i[famof(tg)], cell[best], n_all / len(rpi)); np.add.at(ccells_i[famof(tg)], cell[best], n_all)
                                np.add.at(dev_i[famof(tg)], cell[best], n_all * dj[best]); np.add.at(cdev_i[famof(tg)], cell[best], n_all * dj[best])
                        if gains_i:
                            tot_i += n_all * float(np.mean(gains_i)); wsum_i += n_all; ct_i += n_all * float(np.mean(gains_i)); cw_i += n_all
                    if cw_i > 0:
                        by_count_i[cname] = {'runs_per_100_pa': round(float(ct_i / cw_i * N_IN) * 100, 2), 'pitches': int(cw_i),
                                             'cells': {f_: [[float(GU[k % len(GU)]), float(GZ[k // len(GU)]), 'take' if cdev_i[f_][k] < 0 else 'weak'] for k in np.argsort(c_)[::-1][:3] if c_[k] > 0] for f_, c_ in ccells_i.items()}}
                if wsum_i > 0:
                    out['strike'] = {'runs_per_100_pa': round(float(tot_i / wsum_i * N_IN) * 100, 2), 'pricing': 'structural',
                                     'cells': {f_: [[float(GU[k % len(GU)]), float(GZ[k // len(GU)]), 'take' if dev_i[f_][k] < 0 else 'weak'] for k in np.argsort(c_)[::-1][:3] if c_[k] > 0] for f_, c_ in cells_i.items()},
                                     'by_count': by_count_i}
        # miss spots (unpriced): among the pitcher's two-strike pitches to this side, where this hitter's own whiff map
        # (his map minus the same-side mean) says he misses most, by family
        if wmap is not None and h in self.mbar_w:
            r2 = self.gp[p][(T['stand_r'][self.gp[p]] == sd) & (T['strikes'][self.gp[p]] == 2)]
            if len(r2) >= 40:
                dev = self.Bw[r2] @ (wmap - self.mbar_w[h])
                uu_ = np.where(sd == 1, self.xt[r2], -self.xt[r2])
                cell2 = np.argmin(np.abs(uu_[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[r2][:, None] - GZ[None, :]), 1)
                miss = {}
                for fname, groups, _, _ in FAMILIES:
                    mm = np.flatnonzero(np.isin(T['group'][r2], groups))
                    if len(mm) < 12:
                        continue
                    k3 = max(1, len(mm) // 3); best = mm[np.argsort(dev[mm])[::-1][:k3]]
                    acc = np.bincount(cell2[best], minlength=len(GU) * len(GZ))
                    miss[fname] = {'cells': [[float(GU[k % len(GU)]), float(GZ[k // len(GU)])] for k in np.argsort(acc)[::-1][:3] if acc[k] > 0],
                                   'own_whiff_points': round(float((sig(self.off_w[r2[best]] + self.Bw[r2[best]] @ wmap) - sig(self.off_w[r2[best]] + self.Bw[r2[best]] @ self.mbar_w[h])).mean()) * 100, 1)}
                if miss:
                    out['miss_spots'] = miss
        return out

    def grade(self, game: int, h: int, p: int, aim_cells: dict | None) -> dict | None:
        """The pitches p threw to h in this game against the report: outside pitches in the recommended cells against
        the pitcher's usual share, and the hitter's chases against the league's and his map's predictions."""
        T = self.T; r = np.flatnonzero((T['game'] == game) & (T['batter'] == h) & (T['pitcher'] == p))
        if len(r) == 0:
            return None
        out = {'pitches': int(len(r)), 'outside': int(self.outside[r].sum())}
        o = r[self.outside[r]]
        if len(o) and h in self.maps_s:
            pl = sig(self.off_s[o]); ph = sig(self.off_s[o] + self.Bs[o] @ self.maps_s[h])
            out['chases'] = int(self.swing[o].sum()); out['chases_expected_league'] = round(float(pl.sum()), 2); out['chases_expected_map'] = round(float(ph.sum()), 2)
            # the forward scoring record: the log loss of each swing decision under the map and under the league rates (sums; nats)
            y = self.swing[o].astype(float); e_ = 1e-6
            out['log_loss_map'] = round(float(-np.sum(y * np.log(np.clip(ph, e_, 1)) + (1 - y) * np.log(np.clip(1 - ph, e_, 1)))), 4)
            out['log_loss_league'] = round(float(-np.sum(y * np.log(np.clip(pl, e_, 1)) + (1 - y) * np.log(np.clip(1 - pl, e_, 1)))), 4)
            # the same by count group, pitch family and batter side (conditional calibration, accumulated in the record)
            side_lab = 'R' if T['stand_r'][o][0] == 1 else 'L'
            sub = {}
            cg_o = self.cgrp[o]; fam_o = np.array([famof(int(g)) for g in T['group'][o]])
            for name, keys, labels in (('count', cg_o, {0: 'even_or_ahead', 1: 'behind', 2: 'two_strikes'}), ('family', fam_o, None)):
                for kv in np.unique(keys):
                    m = keys == kv; lab = labels[int(kv)] if labels else str(kv)
                    sub[f'{name}:{lab}'] = {'outside': int(m.sum()), 'chases': int(self.swing[o][m].sum()), 'league': round(float(pl[m].sum()), 2), 'map': round(float(ph[m].sum()), 2)}
            sub[f'side:{side_lab}'] = {'outside': int(len(o)), 'chases': out['chases'], 'league': out['chases_expected_league'], 'map': out['chases_expected_map']}
            out['by_group'] = sub
        if len(o) and aim_cells:
            sd = self.side.get(h, 1); uu_ = np.where(sd == 1, self.xt[o], -self.xt[o])
            cell = np.argmin(np.abs(uu_[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[o][:, None] - GZ[None, :]), 1)
            rec = np.zeros(len(o), bool)
            for j, g in enumerate(T['group'][o]):
                cells = aim_cells.get(famof(int(g))) or []
                want = {int(np.argmin(np.abs(GU - c[0]))) + len(GU) * int(np.argmin(np.abs(GZ - c[1]))) for c in cells}
                rec[j] = cell[j] in want
            out['in_recommended_cells'] = int(rec.sum())
            # the pitcher's usual share: his training outside pitches to this side in those cells
            rp = self.gp.get(p)
            if rp is not None:
                rp = rp[(T['stand_r'][rp] == sd) & self.outside[rp]]
                if len(rp) >= 40:
                    uu2 = np.where(sd == 1, self.xt[rp], -self.xt[rp])
                    cell2 = np.argmin(np.abs(uu2[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[rp][:, None] - GZ[None, :]), 1)
                    hits = 0
                    for fname, groups, _, _ in FAMILIES:
                        want = {int(np.argmin(np.abs(GU - c[0]))) + len(GU) * int(np.argmin(np.abs(GZ - c[1]))) for c in (aim_cells.get(fname) or [])}
                        mm = np.isin(T['group'][rp], groups)
                        hits += int(np.isin(cell2[mm], list(want)).sum())
                    out['usual_share_in_cells'] = round(hits / len(rp), 3)
        last = r[T['last_in_pa'][r] == 1]
        if len(last):
            o7 = T['out7'][last]; names = D.OUT7
            out['results'] = [names[int(v)] for v in o7 if 0 <= v < 7]
        return out


def mlb(url: str):
    import urllib.request
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'BRL-report/1.0', 'Accept': 'application/json'}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 + 3 * attempt)


def schedule(day: str) -> list:
    doc = mlb(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day}&gameType=R,F,D,L,W&hydrate=probablePitcher,lineups,team')
    out = []
    for d in doc.get('dates', []):
        for g in d.get('games', []):
            st = g.get('status') or {}
            teams = g.get('teams') or {}
            row = {'game_pk': int(g['gamePk']), 'game_type': g.get('gameType'), 'status': st.get('detailedState'), 'final': st.get('abstractGameState') == 'Final',
                   'start': g.get('gameDate'), 'teams': {}, 'lineups': {}}
            for side in ('away', 'home'):
                t = teams.get(side, {}); team = t.get('team') or {}
                row['teams'][side] = {'id': int(team.get('id') or 0), 'name': team.get('name'), 'abbr': team.get('abbreviation'),
                                      'probable': int(((t.get('probablePitcher') or {}).get('id')) or 0) or None}
            lu = g.get('lineups') or {}
            for side, key in (('away', 'awayPlayers'), ('home', 'homePlayers')):
                row['lineups'][side] = [int(x['id']) for x in (lu.get(key) or []) if x.get('id')]
            out.append(row)
    return out


def boxscore_players(game_pk: int) -> dict:
    """Finished games: who batted and pitched for each side, in batting order (public box score)."""
    doc = mlb(f'https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore')
    out = {}
    for side in ('away', 'home'):
        t = (doc.get('teams') or {}).get(side) or {}
        out[side] = {'batters': [int(v) for v in t.get('batters') or []], 'pitchers': [int(v) for v in t.get('pitchers') or []], 'battingOrder': [int(v) for v in t.get('battingOrder') or []]}
    return out


def recent_players(T: dict, game_pks_by_team: dict, team_games: dict) -> tuple[dict, dict, dict]:
    """From the pitch table: each team's hitters (by plate appearances) and relievers in its recent games, and the batting
    order of its latest game in the table (the simulator projects a lineup the same way: the team's most recent lineup).
    Relievers are every pitcher after a game's first one, ranked by games pitched in relief, then batters faced, then the
    latest appearance (before October 10, 2026 only the last game in the list counted, so a plan showed whoever pitched
    in relief the night before)."""
    hitters, relievers, last_order = {}, {}, {}
    g = T['game']; h = T['half']; b = T['batter']; p = T['pitcher']; ab = T['ab']; pn = T['pitch_no']; dy = T['day']
    first = pn == 0
    for tid, pks in game_pks_by_team.items():
        hc = {}; pc = {}; pbf = {}; plast = {}; latest = None
        for pk in pks:
            side = team_games[(pk, tid)]            # 'away' or 'home'
            bat_half = 0 if side == 'away' else 1; fld_half = 1 - bat_half
            m = (g == pk) & (h == bat_half) & first
            for x in b[m]:
                hc[int(x)] = hc.get(int(x), 0) + 1
            if m.any():
                key = (int(dy[m].max()), int(pk))
                if latest is None or key > latest[0]:
                    o = np.argsort(ab[m], kind='stable')
                    latest = (key, list(dict.fromkeys(int(x) for x in b[m][o]))[:9])
            m2 = (g == pk) & (h == fld_half)
            if m2.any():
                order = np.argsort(ab[m2] * 100 + pn[m2], kind='stable'); ps = p[m2][order]
                st = int(ps[0]); day_ = int(dy[m2].max())
                for x in {int(v) for v in ps} - {st}:
                    pc[x] = pc.get(x, 0) + 1; plast[x] = max(plast.get(x, 0), day_)
                for x in p[m2 & first]:
                    if int(x) != st:
                        pbf[int(x)] = pbf.get(int(x), 0) + 1
        if latest is not None and len(latest[1]) == 9:
            last_order[tid] = latest[1]
        hitters[tid] = sorted(hc, key=lambda k: -hc[k])[:13]
        relievers[tid] = sorted(pc, key=lambda k: (-pc[k], -pbf.get(k, 0), -plast.get(k, 0), k))[:6]
    return hitters, relievers, last_order


def team_sides(first_day: int, last_day: int) -> dict:
    """game_pk -> ({'away': team id, 'home': team id}, game type), from the public schedule, a month a call."""
    out = {}
    d = date.fromordinal(int(first_day)).replace(day=1); end = date.fromordinal(int(last_day))
    while d <= end:
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        doc = mlb(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={d.isoformat()}&endDate={(nxt - timedelta(days=1)).isoformat()}&gameType=R,F,D,L,W')
        for dd in doc.get('dates', []):
            for g in dd.get('games', []):
                tm = g.get('teams') or {}
                ids = {s_: int((((tm.get(s_) or {}).get('team')) or {}).get('id') or 0) for s_ in ('away', 'home')}
                if ids['away'] and ids['home']:
                    out[int(g['gamePk'])] = (ids, g.get('gameType'))
        d = nxt
    return out


def team_game_rows(T: dict, sides: dict, pks=None) -> dict:
    """team id -> its games in the pitch table as brl_live.lineups prior games (day, game_pk, nine starters in batting
    order, the opposing starter's hand, everyone who batted), with the game type and season after them."""
    keep = np.isin(T['game'], np.fromiter(pks, dtype=np.int64)) if pks is not None else np.ones(len(T['game']), bool)
    g = T['game'][keep]; h = T['half'][keep]; b = T['batter'][keep]; ab = T['ab'][keep]; pn = T['pitch_no'][keep]
    dy = T['day'][keep]; tr = T['throw_r'][keep]; ss = T['season'][keep]
    o = np.lexsort((pn, ab, h, g))
    key = g[o].astype(np.int64) * 2 + h[o]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(o) else np.array([], int)
    ends = np.r_[starts[1:], len(o)]
    out = {}
    for s_, e_ in zip(starts, ends):
        i0 = o[s_]; pk = int(g[i0])
        if pk not in sides:
            continue
        ids, gtype = sides[pk]
        bb = [int(x) for x in b[o[s_:e_]]]
        out.setdefault(ids['away' if int(h[i0]) == 0 else 'home'], []).append(
            (int(dy[i0]), pk, list(dict.fromkeys(bb))[:9], 'R' if int(tr[i0]) == 1 else 'L', set(bb), gtype, int(ss[i0])))
    return out


ZONE_M_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
ZONE_METRICS = (('avg', 'h', 'ab'), ('swing', 'sw', 'n'), ('miss', 'wh', 'sw'), ('slg', 'tb', 'ab'))


def zone_tables(T: dict, mask) -> dict:
    """Per (batter, zone) counts on the masked pitches: n, swings, misses, at-bats ended there, hits, total bases; plus each
    batter's side (the majority of his pitches). MLB's 13 zones in ZONES13 order."""
    rows = np.flatnonzero(mask)
    zmap = np.full(32, -1, np.int64)
    for i, zz in enumerate(ZONES13):
        zmap[zz] = i
    zi = zmap[np.clip(T['zone'][rows], 0, 31)]
    rows = rows[zi >= 0]; zi = zi[zi >= 0]
    bats, bi = np.unique(T['batter'][rows], return_inverse=True)
    key = bi * 13 + zi; nb = len(bats)
    sw = (T['call'][rows] == 1) | (T['call'][rows] == 2); wh = T['call'][rows] == 2
    last = T['last_in_pa'][rows] == 1; o7 = T['out7'][rows]
    ab = last & np.isin(o7, (0, 1, 3, 4, 5, 6)); hit = last & np.isin(o7, (3, 4, 5))
    tb = np.where(last & (o7 == 3), 1.0, 0.0) + np.where(last & (o7 == 4), 2.0, 0.0) + np.where(last & (o7 == 5), 4.0, 0.0)
    cnt = lambda w: np.bincount(key, weights=w.astype(np.float64), minlength=nb * 13).reshape(nb, 13)
    side = np.bincount(bi, weights=T['stand_r'][rows].astype(np.float64), minlength=nb) / np.maximum(np.bincount(bi, minlength=nb), 1)
    pa = np.bincount(bi, weights=last.astype(np.float64), minlength=nb)
    return {'bats': bats, 'side': (side >= 0.5).astype(np.int64), 'pa': pa, 'n': cnt(np.ones(len(rows), bool)), 'sw': cnt(sw), 'wh': cnt(wh),
            'ab': cnt(ab), 'h': cnt(hit), 'tb': cnt(tb)}


def zone_estimates(tr: dict, num: str, den: str, m: float, k_all: float = 30.0) -> dict:
    """The candidates for each training hitter's zone rates: raw, league (his side), scaled (league shape at his own level)
    and shrunk (his counts plus m pseudo-events at the scaled rate)."""
    N, Dn = tr[num], tr[den]
    lg_z = np.zeros((2, 13)); lg_all = np.zeros(2)
    for sd in (0, 1):
        sel = tr['side'] == sd
        lg_z[sd] = N[sel].sum(0) / np.maximum(Dn[sel].sum(0), 1e-9); lg_all[sd] = N[sel].sum() / max(Dn[sel].sum(), 1e-9)
    lz = lg_z[tr['side']]; la = lg_all[tr['side']]
    h_all = (N.sum(1) + k_all * la) / (Dn.sum(1) + k_all)          # his overall rate, lightly shrunk
    scaled = lz * (h_all / np.maximum(la, 1e-9))[:, None]
    if num != 'tb':
        scaled = np.clip(scaled, 0.005, 0.995)
    raw = np.where(Dn > 0, N / np.maximum(Dn, 1e-9), lz)
    shrunk = (N + m * scaled) / (Dn + m) if m > 0 else raw
    return {'raw': raw, 'league': lz, 'scaled': scaled, 'shrunk': shrunk}


def zone_score(pred, num_t, den_t, binary: bool):
    """Per-hitter loss sums and event counts on the test season: binary log loss, or squared error weighted by events."""
    if binary:
        p = np.clip(pred, 1e-4, 1 - 1e-4)
        loss = -(num_t * np.log(p) + (den_t - num_t) * np.log(1 - p))
    else:
        obs = np.where(den_t > 0, num_t / np.maximum(den_t, 1e-9), 0.0)
        loss = den_t * (pred - obs) ** 2
    return loss.sum(1), den_t.sum(1)


def zone_shrink_study(T: dict, stage, spec: dict) -> dict:
    """ZONES-01 (LEDGER): expected hot zones. m chosen on (fit_train -> fit_test), then every candidate scored once on
    (train -> test); hitters with min_pa plate appearances in training and a test season. Aggregates only."""
    reg = (T['post'] == 0) if 'post' in T else np.ones(len(T['season']), bool)
    fit_train = tuple(spec.get('fit_train', (2023, 2024))); fit_test = int(spec.get('fit_test', 2025))
    train = tuple(spec.get('train', (2024, 2025))); test = int(spec.get('test', 2026)); min_pa = float(spec.get('min_pa', 100))
    rng = np.random.default_rng(int(spec.get('seed', 20261011)))

    # ZONES-02: an optional subset of pitches (a pitcher hand or a pitch family), the same on both sides of the split
    sub = spec.get('subset') or {}
    base = reg.copy()
    if 'throw_r' in sub:
        base &= T['throw_r'] == int(sub['throw_r'])
    if 'family' in sub:
        base &= np.isin(T['group'], {'fastball': (0, 1, 2, 6), 'breaking': (3, 4), 'offspeed': (5,)}[sub['family']])

    def joined(tr_years, te_year):
        tr = zone_tables(T, base & np.isin(T['season'], tr_years)); te = zone_tables(T, base & (T['season'] == te_year))
        pos = {int(b): i for i, b in enumerate(te['bats'])}
        keep = np.array([pa >= min_pa and int(b) in pos for b, pa in zip(tr['bats'], tr['pa'])], bool)
        ti = np.array([pos[int(b)] for b in tr['bats'][keep]], np.int64)
        trk = {k: (v[keep] if hasattr(v, '__len__') and len(v) == len(keep) else v) for k, v in tr.items()}
        tek = {k: v[ti] for k, v in te.items() if k in ('n', 'sw', 'wh', 'ab', 'h', 'tb')}
        return trk, tek

    out = {'fit': {'train': list(fit_train), 'test': fit_test}, 'score': {'train': list(train), 'test': test}, 'min_pa': min_pa, 'grid': list(ZONE_M_GRID), 'metrics': {}}
    tr1, te1 = joined(fit_train, fit_test); stage('zone study: fit tables')
    tr2, te2 = joined(train, test); stage('zone study: score tables')
    out['hitters'] = {'fit': int(len(tr1['bats'])), 'score': int(len(tr2['bats']))}
    for name, num, den in ZONE_METRICS:
        binary = num != 'tb'
        curve = []
        for m in ZONE_M_GRID:
            est = zone_estimates(tr1, num, den, m)
            l, e = zone_score(est['shrunk'], te1[num], te1[den], binary)
            curve.append(float(l.sum() / max(e.sum(), 1)))
        m_best = ZONE_M_GRID[int(np.argmin(curve))]
        est = zone_estimates(tr2, num, den, m_best)
        per = {}
        for cand in ('raw', 'league', 'scaled', 'shrunk'):
            per[cand] = zone_score(est[cand], te2[num], te2[den], binary)
        events = per['raw'][1]; tot = events.sum()
        res = {'m': m_best, 'fit_curve': [round(c, 6) for c in curve], 'events': int(tot),
               'loss': {c: round(float(per[c][0].sum() / max(tot, 1)), 6) for c in per}}
        nh = len(events); draws = rng.integers(0, nh, (2000, nh))
        for a, b in (('shrunk', 'raw'), ('shrunk', 'scaled'), ('shrunk', 'league'), ('scaled', 'raw')):
            d = per[a][0] - per[b][0]
            boot = d[draws].sum(1) / np.maximum(events[draws].sum(1), 1)
            res[f'{a}_minus_{b}'] = [round(float(d.sum() / max(tot, 1)), 6), round(float(np.percentile(boot, 2.5)), 6), round(float(np.percentile(boot, 97.5)), 6)]
        out['metrics'][name] = res
        stage(f'zone study: {name}')
    return out


# TAGS-01 (LEDGER): the hitter tags the plans print (the page's hitterTags), each a level against the league with the
# page's threshold: name, rate, threshold (positive: at or above the league by that much; negative: at or below).
TAG_RULES = (('Chases a lot', 'chase', 0.04), ('Patient', 'chase', -0.06), ('Swings and misses', 'whiff', 0.05),
             ('Puts the bat on the ball', 'whiff', -0.07), ('Takes the first pitch', 'first', -0.12),
             ('Swings at the first pitch', 'first', 0.12), ('Expands with two strikes', 'two', 0.08),
             ('Shrinks the zone with two strikes', 'two', -0.10))
# TAGS-02: the coaching line for hitter's counts (countPlan): his swing rate with the pitcher behind, 10 points over his side's league
TAG_RULES_EXTRA = (('Aggressive in hitter\'s counts', 'behind', 0.10),)
TAG_FLOOR = {'chase': 1, 'whiff': 1, 'first': 80, 'two': 80, 'behind': 80}          # the page reads count rows with 80 or more pitches


def tag_counts(T: dict, mask) -> dict:
    """Per batter on the masked pitches: his side and the counts behind each tag: chases (true crossing outside the fixed
    zone, as the cards), misses per swing, first-pitch swings, and chases out of MLB's zone (11-14) with two strikes and in
    the other counts. Each rate is a (numerator, denominator) pair of arrays in batter order."""
    rows = np.flatnonzero(mask)
    bats, bi = np.unique(T['batter'][rows], return_inverse=True)
    nb = len(bats)
    c = lambda m: np.bincount(bi, weights=np.asarray(m, dtype=np.float64), minlength=nb)
    px, pz = T['px'][rows].astype(np.float64), T['pz'][rows].astype(np.float64)
    outside = (np.abs(px) > D.ZONE_HALF) | (pz > D.ZONE_TOP) | (pz < D.ZONE_BOT)
    sw = (T['call'][rows] == 1) | (T['call'][rows] == 2); wh = T['call'][rows] == 2
    g = count_group(T['balls'][rows], T['strikes'][rows]); ooz = T['zone'][rows] >= 11
    first, two, behind = g == 0, g == 3, g == 2
    grp = T['group'][rows]; fb, br, os_ = np.isin(grp, (0, 1, 2, 6)), np.isin(grp, (3, 4)), grp == 5
    v0 = np.nan_to_num(T['v0'][rows].astype(np.float64), nan=0.0) if 'v0' in T else np.zeros(len(rows))
    if 'spray' in T and 'traj' in T:
        ang = T['spray'][rows].astype(np.float64); okb = np.isfinite(ang); trj = T['traj'][rows]
        pulled = okb & np.where(T['stand_r'][rows] == 1, np.nan_to_num(ang) < -15.0, np.nan_to_num(ang) > 15.0)
        gb_, air_, pop_ = okb & (trj == 0), okb & ((trj == 1) | (trj == 2)), okb & (trj == 3)
    else:
        pulled = gb_ = air_ = pop_ = np.zeros(len(rows), bool)
    # TAGS-07 and TAGS-08: the top third of his zone (MLB zones 1-3) and the bottom third (7-9); balls in play with a
    # measured exit speed and those at 95 mph or more
    z_ = T['zone'][rows]; up_, dn_ = (z_ >= 1) & (z_ <= 3), (z_ >= 7) & (z_ <= 9)
    if 'ls' in T:
        ls_ = T['ls'][rows].astype(np.float64)
        hit_ = np.isfinite(ls_) & (np.isfinite(T['spray'][rows].astype(np.float64)) if 'spray' in T else ((T['call'][rows] == 1) & (T['last_in_pa'][rows] == 1)))
        hard_ = hit_ & (np.nan_to_num(ls_) >= 95)
    else:
        hit_ = hard_ = np.zeros(len(rows), bool)
    n = c(np.ones(len(rows), bool))
    return {'bats': bats, 'side': (c(T['stand_r'][rows] == 1) / np.maximum(n, 1) >= 0.5).astype(np.int64), 'n': n,
            'wh_up': (c(wh & up_), c(sw & up_)), 'wh_dn': (c(wh & dn_), c(sw & dn_)),
            'hh_up': (c(hard_ & up_), c(hit_ & up_)), 'hh_dn': (c(hard_ & dn_), c(hit_ & dn_)),
            'chase': (c(sw & outside), c(outside)), 'whiff': (c(wh), c(sw)), 'first': (c(sw & first), c(first)),
            'two': (c(sw & two & ooz), c(two & ooz)), 'other': (c(sw & ~two & ooz), c(~two & ooz)), 'behind': (c(sw & behind), c(behind)),
            # TAGS-03: misses per swing by pitch family
            'wh_fb': (c(wh & fb), c(sw & fb)), 'wh_br': (c(wh & br), c(sw & br)), 'wh_os': (c(wh & os_), c(sw & os_)),
            # TAGS-04: misses per swing on hard fastballs (96 mph and up) and on softer ones (under 93)
            'wh_hi': (c(wh & fb & (v0 >= 96)), c(sw & fb & (v0 >= 96))), 'wh_lo': (c(wh & fb & (v0 < 93)), c(sw & fb & (v0 < 93))),
            # TAGS-05: pulled ground balls and pulled balls in the air (field third on his pull side, 15 degrees off center)
            'pull_gb': (c(gb_ & pulled), c(gb_)), 'pull_air': (c(air_ & pulled), c(air_)),
            # TAGS-06: ground balls among his batted balls with a trajectory
            'gb_share': (c(gb_), c(gb_ | air_ | pop_))}


def _tag_league(tc: dict, rate: str, side_wise: bool) -> np.ndarray:
    """The league's rate for each batter: overall (chase, whiff: the cards' league) or for his side (count habits)."""
    num, den = tc[rate]
    if not side_wise:
        return np.full(len(num), num.sum() / max(den.sum(), 1e-9))
    lg = np.array([num[tc['side'] == s].sum() / max(den[tc['side'] == s].sum(), 1e-9) for s in (0, 1)])
    return lg[tc['side']]


def tag_study(T: dict, stage, spec: dict) -> dict:
    """TAGS-01 (LEDGER): do the hitter tags hold up on later pitches? Tags from training seasons exactly as the page draws
    them (hitters with min_pitches or more training pitches); for each tag, the tagged hitters' rates on the test pitches
    against the league's on the same pitches, weighted by each hitter's test pitches for that rate, with intervals from
    resampling hitters; plus the two-strike change relative to his own chase in the other counts (persistence and how
    often the level tags repeat Chases a lot / Patient). Aggregates only."""
    reg = (T['post'] == 0) if 'post' in T else np.ones(len(T['season']), bool)
    ok = reg & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
    ok &= np.isfinite(T['px'].astype(np.float64)) & np.isfinite(T['pz'].astype(np.float64))
    if 'bunt_pa' in T:
        ok &= ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    min_pitches = int(spec.get('min_pitches', 300)); min_test = int(spec.get('min_test', 200)); boot = int(spec.get('boot', 1000))
    rng = np.random.default_rng(int(spec.get('seed', 20261011)))
    out = {'splits': []}
    for split in spec.get('splits', [{'train': [2024, 2025], 'test': 2026, 'test_end': '2026-08-01'}, {'train': [2023, 2024], 'test': 2025}]):
        tr_m = ok & np.isin(T['season'], split['train'])
        te_m = ok & (T['season'] == int(split['test']))
        if split.get('test_end'):
            te_m &= T['day'] < date.fromisoformat(split['test_end']).toordinal()
        a, b = tag_counts(T, tr_m), tag_counts(T, te_m)
        pos = {int(x): i for i, x in enumerate(b['bats'])}
        res = {'train': list(split['train']), 'test': int(split['test']), 'test_end': split.get('test_end'), 'tags': {}}
        card = a['n'] >= min_pitches
        ti = np.array([pos.get(int(x), -1) for x in a['bats']], np.int64)
        has_test = (ti >= 0) & card
        has_test[has_test] &= b['n'][ti[has_test]] >= min_test
        res['hitters'] = {'cards': int(card.sum()), 'with_test': int(has_test.sum())}
        tagged = {}
        for name, rate, thr in TAG_RULES + (TAG_RULES_EXTRA if spec.get('extra') else ()):
            side_wise = rate in ('first', 'two', 'behind')
            num, den = a[rate]
            r_tr = np.where(den >= TAG_FLOOR[rate], num / np.maximum(den, 1e-9), np.nan)
            lg_tr = _tag_league(a, rate, side_wise)
            d_tr = r_tr - lg_tr
            on = card & np.isfinite(d_tr) & ((d_tr >= thr) if thr > 0 else (d_tr <= thr))
            tagged[name] = on
            sel = np.flatnonzero(on & has_test)
            tnum, tden = b[rate]
            lg_te = _tag_league(b, rate, side_wise)
            j = ti[sel]
            w = tden[j]; d_te = np.where(w > 0, tnum[j] / np.maximum(w, 1e-9) - lg_te[j], 0.0)
            wd = w * d_te
            mean = float(wd.sum() / max(w.sum(), 1e-9)) if len(sel) else None
            lo = hi = None
            if len(sel) >= 5:
                bs = rng.integers(0, len(sel), size=(boot, len(sel)))
                means = wd[bs].sum(1) / np.maximum(w[bs].sum(1), 1e-9)
                lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
            wt = den[sel]
            res['tags'][name] = {'rate': rate, 'threshold': thr, 'hitters': int(on.sum()), 'with_test': int(len(sel)),
                                 'train_diff': float((wt * d_tr[sel]).sum() / max(wt.sum(), 1e-9)) if len(sel) else None,
                                 'test_diff': mean, 'test_ci': [lo, hi], 'test_pitches': int(w.sum()),
                                 'held': bool(mean is not None and lo is not None and (mean >= thr / 2 if thr > 0 else mean <= thr / 2) and (lo > 0 if thr > 0 else hi < 0))}
        # how often a two-strike tag repeats the chase tag of the same direction
        res['overlap'] = {'expands_also_chases': [int((tagged['Expands with two strikes'] & tagged['Chases a lot']).sum()), int(tagged['Expands with two strikes'].sum())],
                          'shrinks_also_patient': [int((tagged['Shrinks the zone with two strikes'] & tagged['Patient']).sum()), int(tagged['Shrinks the zone with two strikes'].sum())]}
        # the two-strike change against his own chase in the other counts, net of the league's change on his side
        def rel(tc):
            tn, td = tc['two']; on_, od = tc['other']
            ok_ = (td >= 80) & (od >= 150)
            r2 = np.where(ok_, tn / np.maximum(td, 1e-9), np.nan); ro = np.where(ok_, on_ / np.maximum(od, 1e-9), np.nan)
            lg2, lgo = _tag_league(tc, 'two', True), _tag_league(tc, 'other', True)
            return (r2 - ro) - (lg2 - lgo), ro - lgo
        rel_a, lvl_a = rel(a); rel_b, _ = rel(b)
        both = has_test & np.isfinite(rel_a)
        both[both] &= np.isfinite(rel_b[ti[both]])
        x, y = rel_a[both], rel_b[ti[both]]
        res['relative_two_strike'] = {'hitters': int(both.sum()), 'sd_train': float(np.std(x)) if len(x) else None,
                                      'corr_train_test': float(np.corrcoef(x, y)[0, 1]) if len(x) >= 20 else None,
                                      'corr_with_chase_level': float(np.corrcoef(x, lvl_a[both])[0, 1]) if len(x) >= 20 else None}
        if len(x) >= 20:
            sd = float(np.std(x)); rtags = {}
            for nm_, sgn in (('expands_relative', 1), ('shrinks_relative', -1)):
                sel = np.flatnonzero(both)[(x * sgn) >= sd]
                if not len(sel):
                    continue
                j = ti[sel]; w = b['two'][1][j]; d = rel_b[j]
                bs = rng.integers(0, len(sel), size=(boot, len(sel)))
                means = (w * d)[bs].sum(1) / np.maximum(w[bs].sum(1), 1e-9)
                m_ = float((w * d).sum() / max(w.sum(), 1e-9))
                rtags[nm_] = {'threshold': sgn * sd, 'hitters': int(len(sel)), 'train_diff': float(rel_a[sel].mean()), 'test_diff': m_,
                              'test_ci': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
            res['relative_two_strike']['tags'] = rtags
        if spec.get('velocity'):
            # TAGS-04: his misses on hard fastballs minus on softer ones, net of the league's same difference on his side
            def relv(tc):
                hn, hd = tc['wh_hi']; ln, ld = tc['wh_lo']
                ok_ = (hd >= 80) & (ld >= 150)
                rh_ = np.where(ok_, hn / np.maximum(hd, 1e-9), np.nan); rl_ = np.where(ok_, ln / np.maximum(ld, 1e-9), np.nan)
                return (rh_ - rl_) - (_tag_league(tc, 'wh_hi', True) - _tag_league(tc, 'wh_lo', True))
            va, vb = relv(a), relv(b)
            bothv = has_test & np.isfinite(va)
            bothv[bothv] &= np.isfinite(vb[ti[bothv]])
            xv, yv = va[bothv], vb[ti[bothv]]
            ve = {'hitters': int(bothv.sum()), 'sd_train': float(np.std(xv)) if len(xv) else None,
                  'corr_train_test': float(np.corrcoef(xv, yv)[0, 1]) if len(xv) >= 20 else None}
            if len(xv) >= 20:
                sdv = float(np.std(xv)); vt = {}
                for nm_, sgn in (('late_on_velocity', 1), ('handles_velocity', -1)):
                    sel = np.flatnonzero(bothv)[(xv * sgn) >= sdv]
                    if not len(sel):
                        continue
                    j = ti[sel]; w = b['wh_hi'][1][j]; d = vb[j]
                    bs = rng.integers(0, len(sel), size=(boot, len(sel)))
                    means = (w * d)[bs].sum(1) / np.maximum(w[bs].sum(1), 1e-9)
                    vt[nm_] = {'threshold': sgn * sdv, 'hitters': int(len(sel)), 'train_diff': float(va[sel].mean()),
                               'test_diff': float((w * d).sum() / max(w.sum(), 1e-9)), 'test_ci': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
                ve['tags'] = vt
            res['relative_velocity_whiff'] = ve
        if spec.get('spray'):
            # TAGS-05: his pull share on ground balls and in the air against the league's on his side; tags at fixed thresholds
            res['pull'] = {}
            for key, thr, floor in (('pull_gb', 0.10, 60), ('pull_air', 0.08, 60)) + ((('gb_share', 0.08, 120),) if spec.get('gb_share') else ()):
                pn, pd_ = a[key]
                r_tr = np.where(pd_ >= floor, pn / np.maximum(pd_, 1e-9), np.nan)
                d_tr = r_tr - _tag_league(a, key, True)
                tn, tdn = b[key]
                lg_te = _tag_league(b, key, True)
                okp = has_test & np.isfinite(d_tr)
                okp[okp] &= tdn[ti[okp]] >= floor
                x_ = d_tr[okp]; j_all = ti[okp]; y_ = tn[j_all] / np.maximum(tdn[j_all], 1e-9) - lg_te[j_all]
                ent = {'hitters': int(okp.sum()), 'corr_train_test': float(np.corrcoef(x_, y_)[0, 1]) if len(x_) >= 20 else None, 'tags': {}}
                for nm_, sgn in (('pulls', 1), ('goes_the_other_way', -1)):
                    sel = np.flatnonzero(okp)[(x_ * sgn) >= thr]
                    if not len(sel):
                        continue
                    j = ti[sel]; w = tdn[j]; d = tn[j] / np.maximum(w, 1e-9) - lg_te[j]
                    bs = rng.integers(0, len(sel), size=(boot, len(sel)))
                    means = (w * d)[bs].sum(1) / np.maximum(w[bs].sum(1), 1e-9)
                    ent['tags'][nm_] = {'threshold': sgn * thr, 'hitters': int(len(sel)), 'train_diff': float(d_tr[sel].mean()),
                                        'test_diff': float((w * d).sum() / max(w.sum(), 1e-9)), 'test_ci': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
                res['pull'][key] = ent
        if spec.get('families'):
            # TAGS-03: his misses per swing on breaking balls (and offspeed) minus on fastballs, net of the league's same
            # difference on his side: season-to-season correlation and tags at one standard deviation
            res['relative_family_whiff'] = {}
            for fam, key, floor in (('breaking', 'wh_br', 100), ('offspeed', 'wh_os', 60)):
                def relf(tc):
                    fn, fd = tc[key]; bn, bd = tc['wh_fb']
                    ok_ = (fd >= floor) & (bd >= 150)
                    rf = np.where(ok_, fn / np.maximum(fd, 1e-9), np.nan); rb = np.where(ok_, bn / np.maximum(bd, 1e-9), np.nan)
                    lgf, lgb = _tag_league(tc, key, True), _tag_league(tc, 'wh_fb', True)
                    return (rf - rb) - (lgf - lgb)
                ra_, rb_ = relf(a), relf(b)
                bothf = has_test & np.isfinite(ra_)
                bothf[bothf] &= np.isfinite(rb_[ti[bothf]])
                xf, yf = ra_[bothf], rb_[ti[bothf]]
                entry = {'hitters': int(bothf.sum()), 'sd_train': float(np.std(xf)) if len(xf) else None,
                         'corr_train_test': float(np.corrcoef(xf, yf)[0, 1]) if len(xf) >= 20 else None}
                if len(xf) >= 20:
                    sdf = float(np.std(xf)); ft = {}
                    for nm_, sgn in (('misses_more', 1), ('misses_less', -1)):
                        sel = np.flatnonzero(bothf)[(xf * sgn) >= sdf]
                        if not len(sel):
                            continue
                        j = ti[sel]; w = b[key][1][j]; d = rb_[j]
                        bs = rng.integers(0, len(sel), size=(boot, len(sel)))
                        means = (w * d)[bs].sum(1) / np.maximum(w[bs].sum(1), 1e-9)
                        ft[nm_] = {'threshold': sgn * sdf, 'hitters': int(len(sel)), 'train_diff': float(ra_[sel].mean()),
                                   'test_diff': float((w * d).sum() / max(w.sum(), 1e-9)), 'test_ci': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
                    entry['tags'] = ft
                res['relative_family_whiff'][fam] = entry
        if spec.get('vertical'):
            # TAGS-07 (misses) and TAGS-08 (hard contact): the top third of his zone against the bottom third, relative to
            # himself and net of the league's same difference on his side; season-to-season correlation and tags at one
            # standard deviation, scored on the test pitches (weights: the smaller of his two test rows)
            res['vertical'] = {}
            for read, ku, kd, floor in (('miss', 'wh_up', 'wh_dn', 80), ('hard', 'hh_up', 'hh_dn', 50)):
                def relz(tc, ku=ku, kd=kd, floor=floor):
                    un, ud = tc[ku]; dn2, dd = tc[kd]
                    ok_ = (ud >= floor) & (dd >= floor)
                    ru = np.where(ok_, un / np.maximum(ud, 1e-9), np.nan); rd = np.where(ok_, dn2 / np.maximum(dd, 1e-9), np.nan)
                    return (ru - rd) - (_tag_league(tc, ku, True) - _tag_league(tc, kd, True))
                za, zb = relz(a), relz(b)
                bz = has_test & np.isfinite(za)
                bz[bz] &= np.isfinite(zb[ti[bz]])
                xz, yz = za[bz], zb[ti[bz]]
                lg = {}
                for sd_, nm_s in ((1, 'R'), (0, 'L')):
                    m_ = a['side'] == sd_
                    lg[nm_s] = [float(a[ku][0][m_].sum() / max(a[ku][1][m_].sum(), 1e-9)), float(a[kd][0][m_].sum() / max(a[kd][1][m_].sum(), 1e-9))]
                ent = {'hitters': int(bz.sum()), 'floor': floor, 'league_train_up_down': lg, 'sd_train': float(np.std(xz)) if len(xz) else None,
                       'corr_train_test': float(np.corrcoef(xz, yz)[0, 1]) if len(xz) >= 20 else None}
                if len(xz) >= 20:
                    sdz = float(np.std(xz)); zt = {}
                    for nm_, sgn in (('up', 1), ('down', -1)):
                        sel = np.flatnonzero(bz)[(xz * sgn) >= sdz]
                        if not len(sel):
                            continue
                        j = ti[sel]; w = np.minimum(b[ku][1][j], b[kd][1][j]); d = zb[j]
                        bs = rng.integers(0, len(sel), size=(boot, len(sel)))
                        means = (w * d)[bs].sum(1) / np.maximum(w[bs].sum(1), 1e-9)
                        m_ = float((w * d).sum() / max(w.sum(), 1e-9)); lo_, hi_ = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
                        zt[nm_] = {'threshold': sgn * sdz, 'hitters': int(len(sel)), 'train_diff': float(za[sel].mean()), 'test_diff': m_,
                                   'test_ci': [lo_, hi_], 'held': bool(sgn * m_ >= sdz / 3 and (lo_ > 0 if sgn > 0 else hi_ < 0))}
                    ent['tags'] = zt
                res['vertical'][read] = ent
        out['splits'].append(res)
        stage(f"tag study {split['train']} -> {split['test']}")
    return out


# PTAGS-01 (LEDGER): pitcher tags, each a level against the league with a fixed threshold: name, rate, threshold
PTAG_RULES = (('Throws first-pitch strikes', 'fps', 0.05), ('Falls behind first', 'fps', -0.05),
              ('Lives in the zone', 'zone', 0.04), ('Works off the plate', 'zone', -0.04),
              ('Gets chases', 'chase', 0.04), ('Few chases', 'chase', -0.04),
              ('Fastballs when behind', 'fb_behind', 0.12), ('Spins it when behind', 'fb_behind', -0.12),
              ('Starts with spin', 'fb_first', -0.15))


def pitcher_tag_counts(T: dict, mask) -> dict:
    """Per pitcher on the masked pitches: first-pitch strikes (anything but a ball on 0-0), pitches in MLB's zone (1-9),
    chases on pitches out of it (11-14), and fastballs with the count behind him and on 0-0."""
    rows = np.flatnonzero(mask)
    ids, pi = np.unique(T['pitcher'][rows], return_inverse=True)
    n_ = len(ids)
    c = lambda m: np.bincount(pi, weights=np.asarray(m, dtype=np.float64), minlength=n_)
    g = count_group(T['balls'][rows], T['strikes'][rows]); call = T['call'][rows]
    sw = (call == 1) | (call == 2); z = T['zone'][rows]; inz = (z >= 1) & (z <= 9); ooz = z >= 11
    fb = np.isin(T['group'][rows], (0, 1, 2, 6)); first, behind = g == 0, g == 2
    ball = (call == 0) & (T['cs'][rows] != 1) if 'cs' in T else (call == 0)
    return {'ids': ids, 'n': c(np.ones(len(rows), bool)), 'fps': (c(first & ~ball), c(first)), 'zone': (c(inz), c(inz | ooz)),
            'chase': (c(sw & ooz), c(ooz)), 'fb_behind': (c(fb & behind), c(behind)), 'fb_first': (c(fb & first), c(first))}


def pitcher_tag_study(T: dict, stage, spec: dict) -> dict:
    """PTAGS-01 (LEDGER): do pitcher tags hold on later pitches? Tags from the training seasons (pitchers with min_pitches
    or more), each tagged group's rate minus the league's on the test pitches, weighted by the pitchers' test pitches,
    with intervals from resampling pitchers; plus each rate's season-to-season correlation. Aggregates only."""
    reg = (T['post'] == 0) if 'post' in T else np.ones(len(T['season']), bool)
    ok = reg & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
    min_pitches = int(spec.get('min_pitches', 500)); min_test = int(spec.get('min_test', 300)); boot = int(spec.get('boot', 1000))
    rng = np.random.default_rng(int(spec.get('seed', 20261011)))
    floors = {'fps': 100, 'zone': 300, 'chase': 150, 'fb_behind': 100, 'fb_first': 100}
    out = {'splits': []}
    for split in spec.get('splits', [{'train': [2024, 2025], 'test': 2026, 'test_end': '2026-08-01'}, {'train': [2023, 2024], 'test': 2025}]):
        tr_m = ok & np.isin(T['season'], split['train']); te_m = ok & (T['season'] == int(split['test']))
        if split.get('test_end'):
            te_m &= T['day'] < date.fromisoformat(split['test_end']).toordinal()
        a, b = pitcher_tag_counts(T, tr_m), pitcher_tag_counts(T, te_m)
        pos = {int(x): i for i, x in enumerate(b['ids'])}
        ti = np.array([pos.get(int(x), -1) for x in a['ids']], np.int64)
        has = (ti >= 0) & (a['n'] >= min_pitches)
        has[has] &= b['n'][ti[has]] >= min_test
        res = {'train': list(split['train']), 'test': int(split['test']), 'pitchers': int(has.sum()), 'tags': {}, 'corr': {}}
        for rate, floor in floors.items():
            an, ad = a[rate]; bn, bd = b[rate]
            okr = has & (ad >= floor)
            okr[okr] &= bd[ti[okr]] >= floor
            x = an[okr] / ad[okr] - an.sum() / max(ad.sum(), 1); y = bn[ti[okr]] / bd[ti[okr]] - bn.sum() / max(bd.sum(), 1)
            res['corr'][rate] = {'pitchers': int(okr.sum()), 'r': float(np.corrcoef(x, y)[0, 1]) if okr.sum() >= 20 else None}
        for name, rate, thr in PTAG_RULES:
            an, ad = a[rate]; bn, bd = b[rate]
            lg_a, lg_b = an.sum() / max(ad.sum(), 1), bn.sum() / max(bd.sum(), 1)
            d_tr = np.where(ad >= floors[rate], an / np.maximum(ad, 1e-9) - lg_a, np.nan)
            on = has & np.isfinite(d_tr) & ((d_tr >= thr) if thr > 0 else (d_tr <= thr))
            sel = np.flatnonzero(on)
            j = ti[sel]; w = bd[j]; d = np.where(w > 0, bn[j] / np.maximum(w, 1e-9) - lg_b, 0.0); wd = w * d
            mean = float(wd.sum() / max(w.sum(), 1e-9)) if len(sel) else None
            lo = hi = None
            if len(sel) >= 5:
                bs = rng.integers(0, len(sel), size=(boot, len(sel)))
                ms = wd[bs].sum(1) / np.maximum(w[bs].sum(1), 1e-9)
                lo, hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
            res['tags'][name] = {'rate': rate, 'threshold': thr, 'pitchers': int(len(sel)), 'train_diff': float(np.nanmean(d_tr[sel])) if len(sel) else None,
                                 'test_diff': mean, 'test_ci': [lo, hi],
                                 'held': bool(mean is not None and lo is not None and (mean >= thr / 2 if thr > 0 else mean <= thr / 2) and (lo > 0 if thr > 0 else hi < 0))}
        out['splits'].append(res)
        stage(f"pitcher tag study {split['train']} -> {split['test']}")
    return out


def matchup_read_study(T: dict, stage, spec: dict) -> dict:
    """MATCHUPREAD-01 (LEDGER): a pitcher's own locations laid over a hitter's expected zones (ZONES-01 shrinkage) against
    the league's locations, scored on the test season's at-bats. Aggregates only."""
    reg = (T['post'] == 0) if 'post' in T else np.ones(len(T['season']), bool)
    train = tuple(spec.get('train', (2024, 2025))); test = int(spec.get('test', 2026))
    m = float(spec.get('m', 160.0)); min_fp = int(spec.get('min_pitches', 200)); min_pa = float(spec.get('min_pa', 100)); k_all = 30.0
    zmap = np.full(32, -1, np.int64)
    for i, zz in enumerate(ZONES13):
        zmap[zz] = i
    zi = zmap[np.clip(T['zone'], 0, 31)]
    g = T['group']; fam = np.where(np.isin(g, (0, 1, 2, 6)), 0, np.where(np.isin(g, (3, 4)), 1, np.where(g == 5, 2, -1)))
    last = T['last_in_pa'] == 1; o7 = T['out7']
    ab = last & np.isin(o7, (0, 1, 3, 4, 5, 6))
    tb = np.where(last & (o7 == 3), 1.0, 0.0) + np.where(last & (o7 == 4), 2.0, 0.0) + np.where(last & (o7 == 5), 4.0, 0.0)
    ok = reg & (zi >= 0) & (fam >= 0)
    tr = ok & np.isin(T['season'], train); te = reg & (T['season'] == test) & ab & (fam >= 0)
    pit, bat, st, th = T['pitcher'], T['batter'], T['stand_r'].astype(np.int64), T['throw_r'].astype(np.int64)
    # the pitcher's location counts by (pitcher, batter side, family) and the league's by (hand, side, family)
    ptr = np.flatnonzero(tr)
    pkey = {}
    pk_codes = (pit[ptr].astype(np.int64) * 2 + st[ptr]) * 3 + fam[ptr]
    uk, inv = np.unique(pk_codes, return_inverse=True)
    pcounts = np.zeros((len(uk), 13)); np.add.at(pcounts, (inv, zi[ptr]), 1.0)
    for j, kc in enumerate(uk):
        pkey[int(kc)] = j
    lcounts = np.zeros((2, 2, 3, 13)); np.add.at(lcounts, (th[ptr], st[ptr], fam[ptr], zi[ptr]), 1.0)
    lshare = lcounts / np.maximum(lcounts.sum(-1, keepdims=True), 1.0)
    stage('matchup read: locations')
    # each hitter's at-bats and total bases by zone against each hand on each family, and the league's by (side, hand, family)
    atr = np.flatnonzero(tr & ab)
    hk_codes = (bat[atr].astype(np.int64) * 2 + th[atr]) * 3 + fam[atr]
    uh, hinv = np.unique(hk_codes, return_inverse=True)
    hab = np.zeros((len(uh), 13)); htb = np.zeros((len(uh), 13))
    np.add.at(hab, (hinv, zi[atr]), 1.0); np.add.at(htb, (hinv, zi[atr]), tb[atr])
    hside = np.zeros(len(uh)); np.add.at(hside, hinv, st[atr].astype(np.float64)); hside = (hside / np.maximum(np.bincount(hinv, minlength=len(uh)), 1) >= 0.5).astype(np.int64)
    lab = np.zeros((2, 2, 3, 13)); ltb = np.zeros((2, 2, 3, 13))
    np.add.at(lab, (st[atr], th[atr], fam[atr], zi[atr]), 1.0); np.add.at(ltb, (st[atr], th[atr], fam[atr], zi[atr]), tb[atr])
    hfam = (uh % 3).astype(np.int64); hhand = ((uh // 3) % 2).astype(np.int64)
    lz = ltb[hside, hhand, hfam] / np.maximum(lab[hside, hhand, hfam], 1.0)
    la = ltb[hside, hhand, hfam].sum(1) / np.maximum(lab[hside, hhand, hfam].sum(1), 1.0)
    h_all = (htb.sum(1) + k_all * la) / (hab.sum(1) + k_all)
    scaled = lz * (h_all / np.maximum(la, 1e-9))[:, None]
    xslg = (htb + m * scaled) / (hab + m)
    hkey = {int(kc): j for j, kc in enumerate(uh)}
    pa_tr = {}
    for b_, n_ in zip(*np.unique(bat[np.flatnonzero(reg & np.isin(T['season'], train) & last)], return_counts=True)):
        pa_tr[int(b_)] = int(n_)
    stage('matchup read: hitter zones')
    # the test at-bats
    ev = np.flatnonzero(te)
    E, B, Y, P = [], [], [], []
    for i in ev:
        pkc = (int(pit[i]) * 2 + int(st[i])) * 3 + int(fam[i]); hkc = (int(bat[i]) * 2 + int(th[i])) * 3 + int(fam[i])
        pj = pkey.get(pkc); hj = hkey.get(hkc)
        if pj is None or hj is None or pcounts[pj].sum() < min_fp or pa_tr.get(int(bat[i]), 0) < min_pa:
            continue
        x = xslg[hj]; sp = pcounts[pj] / pcounts[pj].sum(); sl = lshare[int(th[i]), int(st[i]), int(fam[i])]
        E.append(float(sp @ x)); B.append(float(sl @ x)); Y.append(float(tb[i])); P.append(int(pit[i]))
    E, B, Y, P = map(np.asarray, (E, B, Y, P))
    stage('matchup read: test at-bats')
    out = {'train': list(train), 'test': test, 'm': m, 'min_pitches': min_fp, 'min_pa': min_pa, 'at_bats': int(len(Y)), 'pitchers': int(len(np.unique(P))) if len(P) else 0}
    if len(Y) < 1000:
        out['note'] = 'too few test at-bats'
        return out
    d = E - B; r = Y - B
    out.update({'mse_composite': round(float(((Y - E) ** 2).mean()), 5), 'mse_baseline': round(float(((Y - B) ** 2).mean()), 5),
                'sd_composite_minus_baseline': round(float(d.std()), 4), 'mean_actual': round(float(Y.mean()), 4), 'mean_composite': round(float(E.mean()), 4), 'mean_baseline': round(float(B.mean()), 4)})
    slope = float((d * r).sum() / max((d * d).sum(), 1e-12))
    up, pinv = np.unique(P, return_inverse=True)
    rng = np.random.default_rng(int(spec.get('seed', 20261011)))
    se_d = ((Y - E) ** 2 - (Y - B) ** 2)
    sum_dr = np.bincount(pinv, weights=d * r, minlength=len(up)); sum_dd = np.bincount(pinv, weights=d * d, minlength=len(up))
    sum_se = np.bincount(pinv, weights=se_d, minlength=len(up)); cnt = np.bincount(pinv, minlength=len(up)).astype(np.float64)
    draws = rng.integers(0, len(up), (2000, len(up)))
    bs = sum_dr[draws].sum(1) / np.maximum(sum_dd[draws].sum(1), 1e-12); bm = sum_se[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1.0)
    out['slope'] = [round(slope, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    out['mse_composite_minus_baseline'] = [round(float(se_d.mean()), 6), round(float(np.percentile(bm, 2.5)), 6), round(float(np.percentile(bm, 97.5)), 6)]
    return out


def lineup_study(T: dict, stage, spec: dict) -> dict:
    """LINEUP-01 and LINEUP-02 (LEDGER): for every team-game of the evaluated seasons, how many of the nine starters each
    pre-lineup rule names (and how many in the right spot), from the team's games on earlier days only. Aggregates only."""
    from brl_live import lineups as LU
    evaluate = sorted(int(x) for x in (spec.get('evaluate') or (2025, 2026)))
    sides = team_sides(int(T['day'].min()), int(T['day'].max())); stage('schedule')
    by_team = team_game_rows(T, sides); stage('team games')
    rng = np.random.default_rng(20261010)
    pairs = (('hand', 'last'), ('freq', 'hand'), ('freq', 'last'), ('swap', 'hand'))

    def summarize(rows):
        n = len(rows)
        if not n:
            return {'team_games': 0}
        out = {'team_games': n}
        for rule in LU.RULES:
            ov = np.array([r[rule][0] for r in rows]); sp = np.array([r[rule][1] for r in rows])
            out[rule] = {'overlap': round(float(ov.mean()), 4), 'spots': round(float(sp.mean()), 4), 'all_nine': round(float((ov == 9).mean()), 4)}
        cl = {}
        for r in rows:
            cl.setdefault(r['cluster'], []).append(r)
        keys = sorted(cl)
        for a_, b_ in pairs:
            C = np.array([[len(cl[k]), sum(r[a_][0] - r[b_][0] for r in cl[k]), sum(r[a_][1] - r[b_][1] for r in cl[k])] for k in keys], float)
            idx = rng.integers(0, len(C), size=(4000, len(C)))
            bo = C[idx, 1].sum(1) / C[idx, 0].sum(1); bs = C[idx, 2].sum(1) / C[idx, 0].sum(1)
            out[f'{a_}_minus_{b_}'] = {'overlap': round(float(C[:, 1].sum() / C[:, 0].sum()), 4), 'overlap_ci95': [round(float(np.percentile(bo, 2.5)), 4), round(float(np.percentile(bo, 97.5)), 4)],
                                      'spots': round(float(C[:, 2].sum() / C[:, 0].sum()), 4), 'spots_ci95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                                      'clusters': len(keys)}
        return out

    res = {'rules': list(LU.RULES), 'settings': {'hand_days': LU.HAND_DAYS, 'freq_games': LU.FREQ_GAMES, 'freq_days': LU.FREQ_DAYS, 'freq_active': LU.FREQ_ACTIVE,
                                                 'swap_min_games': LU.SWAP_MIN_GAMES, 'swap_fill': LU.SWAP_FILL, 'swap_regular': LU.SWAP_REGULAR},
           'cluster': 'team and calendar month (bootstrap, 4,000 draws)', 'seasons': {}}
    for season in evaluate:
        rows = []
        for tid, games in by_team.items():
            games = sorted(games, key=lambda x: (x[0], x[1]))
            for i, gm in enumerate(games):
                if gm[6] != season or len(gm[2]) != 9:
                    continue
                prior = [x for x in games[max(0, i - 90):i] if x[0] < gm[0]]
                if not prior:
                    continue
                aset = set(gm[2]); last_hand = max(prior, key=lambda x: (x[0], x[1]))[3]
                r = {'cluster': (tid, date.fromordinal(gm[0]).month), 'post': gm[5] not in ('R', None), 'changed': last_hand != gm[3]}
                for rule in LU.RULES:
                    order, _ = LU.project(prior, gm[0], gm[3], rule)
                    order = order or []
                    r[rule] = (len(aset & set(order)), sum(1 for a_, q_ in zip(gm[2], order) if a_ == q_))
                rows.append(r)
        res['seasons'][str(season)] = {'all': summarize(rows), 'opposing_hand_changed': summarize([r for r in rows if r['changed']]),
                                       'opposing_hand_same': summarize([r for r in rows if not r['changed']]),
                                       'regular_season': summarize([r for r in rows if not r['post']]), 'postseason': summarize([r for r in rows if r['post']])}
        stage(f'lineup study {season}')
    return res


def names_for(ids: set, teams: bool = False) -> dict:
    """Names (and, when asked, current team abbreviations) from the public MLB people endpoint, a hundred ids a call."""
    out = {}
    ids = sorted(int(i) for i in ids if i)
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        try:
            doc = mlb('https://statsapi.mlb.com/api/v1/people?personIds=' + ','.join(str(v) for v in chunk) + ('&hydrate=currentTeam' if teams else ''))
            for pers in doc.get('people', []):
                nm = pers.get('fullName') or str(pers['id'])
                if teams:
                    ct = pers.get('currentTeam') or {}
                    out[int(pers['id'])] = {'name': nm, 'team': ct.get('abbreviation') or ct.get('name') or ''}
                else:
                    out[int(pers['id'])] = nm
        except Exception:
            pass
    return out


PLAYER_SHARDS = 16


def build_players(fit: Fitted, asof: str, stage) -> tuple[dict, dict]:
    """League-wide player cards as of a date (every hitter with a map, every pitcher with 150 pitches): an index for search
    and sharded documents (public/reports/players/h-<n>.json, p-<n>.json), the same cards the game reports carry."""
    hitters = {}; pitchers = {}
    for h in sorted(fit.maps_s):
        card = fit.hitter_card(int(h))
        if card:
            hitters[int(h)] = card
    for p, r in fit.gp.items():
        if len(r) >= 150:
            card = fit.pitcher_card(int(p))
            if card:
                pitchers[int(p)] = card
    stage(f'player cards {len(hitters)} hitters, {len(pitchers)} pitchers')
    who = names_for(set(hitters) | set(pitchers), teams=True)
    grid = {'side_ft': GU.tolist(), 'height_ft': GZ.tolist()}
    index = {'schema': 'brl.players.v1', 'asof': asof, 'built_at': datetime.now(timezone.utc).isoformat(), 'shards': PLAYER_SHARDS, 'grid': grid, 'league': fit.league,
             'hitters': {}, 'pitchers': {}}
    shards = {}
    for h, card in hitters.items():
        w = who.get(h, {}); oc = card.get('own_cost') or {}
        index['hitters'][str(h)] = {'name': w.get('name') or str(h), 'team': w.get('team') or '', 'side': card['side'], 'pitches': card['pitches'],
                                    'chase_rate': card.get('chase_rate'), 'whiff_rate': card.get('whiff_rate'), 'own_cost': oc.get('runs_per_600_pa'),
                                    'own_cost_outside': oc.get('outside_runs_per_600_pa'), 'own_cost_inside': oc.get('inside_runs_per_600_pa')}
        doc = dict(card); doc.update({'id': h, 'name': w.get('name') or str(h), 'team': w.get('team') or '', 'kind': 'hitter'})
        shards.setdefault(f'h-{h % PLAYER_SHARDS}', {})[str(h)] = doc
    for p, card in pitchers.items():
        w = who.get(p, {})
        index['pitchers'][str(p)] = {'name': w.get('name') or str(p), 'team': w.get('team') or '', 'throws': card['throws'], 'pitches': card['pitches'],
                                     'chase_rate_against': card.get('chase_rate_against'), 'looks_in_ends_out': card.get('looks_in_ends_out'),
                                     'mix': {f_: v_['share'] for f_, v_ in (card.get('mix') or {}).items()}}
        doc = dict(card); doc.update({'id': p, 'name': w.get('name') or str(p), 'team': w.get('team') or '', 'kind': 'pitcher'})
        shards.setdefault(f'p-{p % PLAYER_SHARDS}', {})[str(p)] = doc
    for k in shards:
        shards[k] = {'schema': 'brl.players.v1', 'asof': asof, 'grid': grid, 'league': fit.league, 'players': shards[k]}
    return index, shards


def sim_relievers(repo: str, token: str, branch: str) -> dict:
    """game_pk -> {side: relief arms, most likely to pitch first}, from the simulator's latest saved box for each game on
    the public prediction file (each pitcher's chance to appear in its 10,000 games), so a plan's bullpen is the one
    the simulator and the Matchups tab expect. Empty when the file cannot be read."""
    try:
        raw = D.read_blob(repo, token, 'public/predictions.json', branch)
        doc = json.loads(raw) if raw else {}
    except Exception as exc:
        print('no simulator bullpens:', type(exc).__name__, flush=True)
        return {}
    latest = {}
    for fid, f in (doc.get('forecasts') or {}).items():
        pk = f.get('game_pk')
        if pk is None:
            continue
        key = (int(f.get('version') or 0), str(f.get('saved_at') or ''))
        if pk not in latest or key > latest[pk][0]:
            latest[pk] = (key, fid)
    boxes = doc.get('box_scores') or {}
    out = {}
    for pk, (_, fid) in latest.items():
        b = boxes.get(fid)
        if not b:
            continue
        sides = {}
        for side in ('away', 'home'):
            rows = ((b.get('teams') or {}).get(side) or {}).get('pitching') or []
            pen = [r for r in rows if r.get('role') != 'starter' and r.get('player_id')]
            pen.sort(key=lambda r: -float(r.get('appearance_probability') or 0))
            sides[side] = [int(r['player_id']) for r in pen]
        out[int(pk)] = sides
    return out


def build_report(fit: Fitted, T_all: dict, day: str, asof: str, stage, max_relievers=6, sim_pens=None) -> dict:
    """The date's report: a summary document and one document per game (each with the players it needs)."""
    games = schedule(day)
    rep = {'schema': SCHEMA, 'date': day, 'asof': asof, 'built_at': datetime.now(timezone.utc).isoformat(), 'training_pitches': fit.n_train,
           'grid': {'side_ft': GU.tolist(), 'height_ft': GZ.tolist()}, 'games': {}, 'players': {}, 'pricing': 'structural' if fit.PM is not None else 'regression'}
    if not games:
        return rep
    # recent games per team from the public schedule: the last 16 days for the bench and the bullpen, the last 46 for
    # the lineup before one is posted (brl_live/lineups.py, the rule the simulator uses)
    from brl_live import lineups as LU
    d0 = date.fromisoformat(day)
    sched = mlb(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={(d0 - timedelta(days=46)).isoformat()}&endDate={(d0 - timedelta(days=1)).isoformat()}&gameType=R,F,D,L,W')
    by_team = {}; team_games = {}; sides_46 = {}
    for d in sched.get('dates', []):
        for g in d.get('games', []):
            if (g.get('status') or {}).get('abstractGameState') != 'Final':
                continue
            pk = int(g['gamePk']); ids = {}
            for side in ('away', 'home'):
                tid = int((((g.get('teams') or {}).get(side) or {}).get('team') or {}).get('id') or 0)
                ids[side] = tid
                if tid and d.get('date', '') >= (d0 - timedelta(days=16)).isoformat():
                    by_team.setdefault(tid, []).append(pk); team_games[(pk, tid)] = side
            if ids['away'] and ids['home']:
                sides_46[pk] = (ids, g.get('gameType'))
    rec_h, rec_p, rec_o = recent_players(fit.T, by_team, team_games)
    prior_games = team_game_rows(T_all, sides_46, pks=set(sides_46)) if sides_46 else {}
    stage('recent players')

    def hand_of(pid):
        rows_ = fit.gp.get(pid) if pid else None
        return None if rows_ is None or not len(rows_) else ('R' if int(fit.T['throw_r'][rows_[0]]) == 1 else 'L')
    def recent_load(pid):
        """Pitches he threw on each of the five days before the game (the bullpen card's rest columns), from the training
        pitches, so nothing from the game day itself."""
        rows_ = fit.gp.get(pid)
        if rows_ is None or not len(rows_):
            return [0, 0, 0, 0, 0]
        dd = fit.T['day'][rows_]
        return [int((dd == d0.toordinal() - k).sum()) for k in range(1, 6)]
    need_names = set()
    for g in games:
        pk = g['game_pk']
        entry = {'teams': g['teams'], 'status': g['status'], 'final': g['final'], 'start': g['start'], 'game_type': g['game_type'], 'sides': {}}
        box = None
        if g['final']:
            try:
                box = boxscore_players(pk)
            except Exception:
                box = None
        for bat_side in ('away', 'home'):
            fld_side = 'home' if bat_side == 'away' else 'away'
            tid = g['teams'][bat_side]['id']; fid = g['teams'][fld_side]['id']
            if box:
                order = box[bat_side]['battingOrder'] or box[bat_side]['batters']
                hitters = [x for x in order if x in fit.maps_s][:12]
                spots = {x: i + 1 for i, x in enumerate((box[bat_side]['battingOrder'] or [])[:9])}
                pitchers = [x for x in box[fld_side]['pitchers']]
                starter = pitchers[0] if pitchers else None
                pens = pitchers[1:]
                source = 'box score'
            else:
                lineup = g['lineups'].get(bat_side) or []
                starter = g['teams'][fld_side]['probable']
                # no posted lineup: the lineup the simulator projects (brl_live/lineups.py), then the rest of the team's
                # recent hitters off the bench
                last, src_ = LU.project(prior_games.get(tid, []), d0.toordinal(), hand_of(starter))
                last = last or []
                first9 = lineup or last              # then the bench: the team's other recent hitters (pinch hitters too)
                base = first9 + [x for x in rec_h.get(tid, []) if x not in first9]
                hitters = [x for x in base if x in fit.maps_s][:12]
                spots = {x: i + 1 for i, x in enumerate((lineup or last)[:9])}
                # the simulator's likeliest relievers when it has a saved box for the game, else the team's most used
                pens = ((sim_pens or {}).get(int(pk)) or {}).get(fld_side) or rec_p.get(fid, [])
                source = 'posted lineup' if lineup else (src_ if last else 'recent games')
            staff = ([starter] if starter else []) + [x for x in pens if x != starter]
            staff = [x for x in staff if x in fit.gp and len(fit.gp[x]) >= 150]
            if not box:                         # a game to come: the starter and the likeliest relievers with enough pitches
                staff = [x for x in staff if x == starter] + [x for x in staff if x != starter][:max_relievers]
            side_entry = {'lineup_source': source, 'hitters': hitters, 'spots': {str(x): v for x, v in spots.items() if x in hitters},
                          'pitchers': [{'id': x, 'role': 'starter' if x == starter else 'reliever', 'recent': recent_load(x)} for x in staff], 'pairs': {}}
            need_names.update(hitters); need_names.update(staff)
            h2h = fit.head_to_head(hitters, staff) if hasattr(fit, 'head_to_head') else {}
            for h in hitters:
                for p in staff:
                    pr = fit.pair(h, p)
                    if pr is None:
                        continue
                    if (int(h), int(p)) in h2h:
                        pr['h2h'] = h2h[(int(h), int(p))]
                    if g['final']:
                        gr = fit.grade(pk, h, p, (pr.get('aim') or {}).get('cells'))
                        if gr:
                            pr['grade'] = gr
                    side_entry['pairs'][f'{h}:{p}'] = pr
            entry['sides'][bat_side] = side_entry
        # game grade summary, with the chases binned by what the map predicted for the pair (the forward calibration record)
        if g['final']:
            tot = {'pairs': 0, 'outside_pitches': 0, 'in_recommended': 0, 'usual_expected': 0.0, 'chases': 0, 'chases_expected_league': 0.0, 'chases_expected_map': 0.0,
                   'log_loss_map': 0.0, 'log_loss_league': 0.0, 'bins': {k: {'pairs': 0, 'outside': 0, 'chases': 0, 'league': 0.0, 'map': 0.0} for k in BINS}}
            for se in entry['sides'].values():
                for pr in se['pairs'].values():
                    gr = pr.get('grade')
                    if not gr or 'chases' not in gr:
                        continue
                    tot['pairs'] += 1; tot['outside_pitches'] += gr['outside']; tot['chases'] += gr['chases']
                    tot['chases_expected_league'] += gr['chases_expected_league']; tot['chases_expected_map'] += gr['chases_expected_map']
                    tot['log_loss_map'] += gr.get('log_loss_map', 0.0); tot['log_loss_league'] += gr.get('log_loss_league', 0.0)
                    if 'in_recommended_cells' in gr and 'usual_share_in_cells' in gr:
                        tot['in_recommended'] += gr['in_recommended_cells']; tot['usual_expected'] += gr['usual_share_in_cells'] * gr['outside']
                    bn = tot['bins'][bin_of(pr['chase_points'])]
                    bn['pairs'] += 1; bn['outside'] += gr['outside']; bn['chases'] += gr['chases']; bn['league'] += gr['chases_expected_league']; bn['map'] += gr['chases_expected_map']
                    groups_ = dict(gr.get('by_group') or {})
                    lvl = (pr.get('evidence') or {}).get('level')
                    if lvl and 'chases' in gr:          # coverage: how much the hitter's map and the pitcher's spots rest on
                        groups_[f'coverage:{lvl}'] = {'outside': gr['outside'], 'chases': gr['chases'], 'league': gr['chases_expected_league'], 'map': gr['chases_expected_map']}
                    for gk, gv in groups_.items():
                        g_ = tot.setdefault('by_group', {}).setdefault(gk, {'outside': 0, 'chases': 0, 'league': 0.0, 'map': 0.0})
                        for kk in ('outside', 'chases', 'league', 'map'):
                            g_[kk] += gv[kk]
            for k in ('usual_expected', 'chases_expected_league', 'chases_expected_map', 'log_loss_map', 'log_loss_league'):
                tot[k] = round(tot[k], 3)
            for bn in tot['bins'].values():
                bn['league'] = round(bn['league'], 2); bn['map'] = round(bn['map'], 2)
            for g_ in (tot.get('by_group') or {}).values():
                g_['league'] = round(g_['league'], 2); g_['map'] = round(g_['map'], 2)
            entry['grade'] = tot
        entry['players'] = {}
        ids = set()
        for se in entry['sides'].values():
            ids.update(se['hitters']); ids.update(x['id'] for x in se['pitchers'])
        for pid in sorted(ids):
            card = fit.hitter_card(pid) if pid in fit.maps_s else None
            pc = fit.pitcher_card(pid)
            if card or pc:
                entry['players'][str(pid)] = {'hitter': card, 'pitcher': pc}
        rep['games'][str(pk)] = entry
        stage(f'game {pk}')
    names = names_for(need_names)
    for entry in rep['games'].values():
        for pid in list(entry['players']):
            entry['players'][pid]['name'] = names.get(int(pid), pid)
        for side in ('away', 'home'):
            entry['teams'][side]['probable_name'] = names.get(entry['teams'][side].get('probable') or -1)
    return rep


def fit_n(fit):
    return fit.n_train


def calibration(fit) -> dict:
    """CAL-01 (diagnostic): the league swing model's expected swings against actual swings, on the training rows and on
    the rows after the as-of date, split by where the pitch crossed (outside or inside the zone), pitch family, how far
    outside, and whether it looked like a strike at the decision moment; the hitter maps' expectation alongside on the
    rows of hitters with a map. Sums only (counts and expected counts); no player rows."""
    T = fit.T; tr = fit.train; te = ~tr
    p = sig(fit.off_s)
    g = T['group']
    fam = np.where(np.isin(g, (0, 1, 2, 6)), 0, np.where(np.isin(g, (3, 4)), 1, 2))
    names = ('fastball', 'breaking', 'offspeed')
    u_t = np.where(T['stand_r'] == 1, fit.xt, -fit.xt)
    dist = np.maximum.reduce([np.abs(u_t) - D.ZONE_HALF, fit.zt - D.ZONE_TOP, D.ZONE_BOT - fit.zt])
    dbin = np.digitize(dist, (0.1, 0.25, 0.5, 1.0))              # outside rows only: 0 = within 0.1 ft of the edge ... 4 = a foot or more
    u_p = np.where(T['stand_r'] == 1, fit.xp, -fit.xp)
    looks_in = ~((np.abs(u_p) > D.ZONE_HALF) | (fit.zp > D.ZONE_TOP) | (fit.zp < D.ZONE_BOT))
    ph = np.full(len(p), np.nan)
    for h, m in fit.maps_s.items():
        r = np.flatnonzero(T['batter'] == h)
        if len(r):
            ph[r] = sig(fit.off_s[r] + fit.Bs[r] @ m)
    has_map = np.isfinite(ph)
    out = {}

    y = fit.swing
    ll_l = D.logloss_vec(p, y); ll_m = np.where(has_map, D.logloss_vec(np.where(has_map, ph, 0.5), y), 0.0)

    def add(key, m):
        mm = m & has_map
        out[key] = {'n': int(m.sum()), 'swings': int(fit.swing[m].sum()), 'league': round(float(p[m].sum()), 1),
                    'n_map': int(mm.sum()), 'swings_map_rows': int(fit.swing[mm].sum()), 'league_map_rows': round(float(p[mm].sum()), 1), 'map': round(float(ph[mm].sum()), 1),
                    'll_league': round(float(ll_l[m].sum()), 2), 'll_league_map_rows': round(float(ll_l[mm].sum()), 2), 'll_map': round(float(ll_m[mm].sum()), 2)}
    for split, m0 in (('train', tr), ('test', te)):
        add(f'{split}|all|all', m0)
        for zone, mz in (('outside', fit.outside), ('inside', ~fit.outside)):
            add(f'{split}|{zone}|all', m0 & mz)
            for k, nm in enumerate(names):
                add(f'{split}|{zone}|{nm}', m0 & mz & (fam == k))
        for b in range(5):
            add(f'{split}|outside|distance{b}', m0 & fit.outside & (dbin == b))
        add(f'{split}|outside|looks_in', m0 & fit.outside & looks_in)
        add(f'{split}|outside|looks_out', m0 & fit.outside & ~looks_in)
        add(f'{split}|inside|looks_out', m0 & ~fit.outside & ~looks_in)
        for c3, cn in ((0, 'even_or_ahead'), (1, 'behind'), (2, 'two_strikes')):
            add(f'{split}|outside|count_{cn}', m0 & fit.outside & (fit.cgrp == c3))
    return out



def calibration_components(fit) -> dict:
    """CAL-02 (diagnostic): the other pieces the plans are priced with, expected against actual on the training rows and
    the rows after the as-of date: misses on swings (whiff model), called strikes on takes, fouls on contact. Split by
    season, by family, and by the signed distance from the true crossing to the nearest zone edge (negative inside).
    'next30' is the later rows within 30 days of the as-of date (what a model refit every day is asked to price).
    Sums only; no player rows."""
    T = fit.T; tr = fit.train; te = ~tr
    n30 = te & (T['day'] < getattr(fit, 'asof_day', 10 ** 9) + 30)
    g = T['group']
    fam = np.where(np.isin(g, (0, 1, 2, 6)), 0, np.where(np.isin(g, (3, 4)), 1, 2))
    names = ('fastball', 'breaking', 'offspeed')
    u_t = np.where(T['stand_r'] == 1, fit.xt, -fit.xt)
    e = np.maximum.reduce([np.abs(u_t) - D.ZONE_HALF, fit.zt - D.ZONE_TOP, D.ZONE_BOT - fit.zt])
    edges = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)
    ebin = np.digitize(e, edges)          # 0: deeper than half a foot inside ... 7: more than half a foot outside
    elab = ('in_0.5+', 'in_0.25-0.5', 'in_0.1-0.25', 'in_0-0.1', 'out_0-0.1', 'out_0.1-0.25', 'out_0.25-0.5', 'out_0.5+')
    call = T['call']; swing = (call == 1) | (call == 2); take = call == 0; contact = call == 1
    pieces = {'whiff': (swing, (call == 2).astype(float), sig(fit.off_w)),
              'called_strike': (take, (T['cs'] == 1).astype(float), fit.p_cs_own),
              'foul': (contact, ((call == 1) & (T['last_in_pa'] == 0)).astype(float), fit.p_fo)}
    season = T['season'] if 'season' in T else np.zeros(len(call), np.int64)
    out = {}
    for name, (base, y, pr) in pieces.items():
        pr = np.asarray(pr, float); ll = D.logloss_vec(pr, y)
        d = out.setdefault(name, {})

        def add(key, m):
            m = m & base
            d[key] = {'n': int(m.sum()), 'actual': int(y[m].sum()), 'expected': round(float(pr[m].sum()), 1), 'll': round(float(ll[m].sum()), 2)}
        for split, m0 in (('train', tr), ('test', te), ('next30', n30)):
            add(f'{split}|all', m0)
            for sv in np.unique(season[m0 & base]) if (m0 & base).any() else []:
                add(f'{split}|season{int(sv)}', m0 & (season == sv))
            for k, nm in enumerate(names):
                add(f'{split}|{nm}', m0 & (fam == k))
            for b, lab in enumerate(elab):
                add(f'{split}|{lab}', m0 & (ebin == b))
                if name == 'called_strike':
                    for sv in np.unique(season[m0 & base]) if (m0 & base).any() else []:
                        add(f'{split}|{lab}|season{int(sv)}', m0 & (ebin == b) & (season == sv))
    return out

def calibration_values(fit, chunk: int = 100000) -> dict:
    """CAL-03 (diagnostic): the engine's value of a swing and of a take, the chain every plan is priced with (whiff, foul,
    the run value of a ball in play, called strike, the count values), against what the swings and takes on the same
    pitches produced: the value of the next state (the count after the pitch, or the plate appearance's result when it
    ended). League level with each row's own hitter and pitcher scalars, no maps. Also the ball-in-play value alone on
    balls in play. Splits: training rows (a sample), later rows, the 30 days after the as-of date; outside or inside;
    family; signed distance to the zone edge; count group. Sums of run values; no player rows."""
    PM = fit.PM
    if PM is None:
        return {}
    T = fit.T; tr = fit.train; LW7 = D.LW7; cv = fit.cv
    rng = np.random.default_rng(5)
    tr_idx = np.flatnonzero(tr); tr_idx = np.sort(rng.choice(tr_idx, min(len(tr_idx), 200000), replace=False))
    rows = np.concatenate([tr_idx, np.flatnonzero(~tr)])
    n = len(rows); v_sw = np.empty(n); v_tk = np.empty(n); v_bip = np.empty(n); pw_all = np.empty(n); pf_all = np.empty(n)
    for i in range(0, n, chunk):
        r = rows[i:i + chunk]
        blk = PM.blocks_at(PM.xp[r], PM.zp[r], PM.xt[r], PM.zt[r], T['stand_r'][r], T['throw_r'][r], T['group'][r], T['v0'][r].astype(np.float64), T['balls'][r], T['strikes'][r])
        _, p_w, p_c, p_f, v, _ = PM.probs(blk, None, (PM.prop_p[r], PM.bip_p[r]), hs=(PM.prop_s[r], PM.prop_w[r], PM.bip_b[r]))
        b_ = T['balls'][r].astype(np.int64); k_ = T['strikes'][r].astype(np.int64)
        v_strike = np.where(k_ == 2, LW7[1], cv[np.clip(b_ * 3 + k_ + 1, 0, 11)])
        v_foul = PM.foul_value(b_, k_, cv)
        v_ball = np.where(b_ == 3, LW7[2], cv[np.clip((b_ + 1) * 3 + k_, 0, 11)])
        v_sw[i:i + len(r)] = p_w * v_strike + (1 - p_w) * (p_f * v_foul + (1 - p_f) * v)
        v_tk[i:i + len(r)] = p_c * v_strike + (1 - p_c) * v_ball
        v_bip[i:i + len(r)] = v; pw_all[i:i + len(r)] = p_w; pf_all[i:i + len(r)] = p_f
        del blk
    b = T['balls'][rows].astype(np.int64); k = T['strikes'][rows].astype(np.int64); call = T['call'][rows]; lip = T['last_in_pa'][rows] == 1
    out7 = T['out7'][rows]; fin = np.where(out7 >= 0, LW7[np.clip(out7, 0, 6)], np.nan)
    nxt_strike = cv[np.clip(b * 3 + k + 1, 0, 11)]; nxt_ball = cv[np.clip((b + 1) * 3 + k, 0, 11)]; same = cv[np.clip(b * 3 + 2, 0, 11)]
    realized = np.where(lip, fin, np.where(call == 0, np.where(T['cs'][rows] == 1, nxt_strike, nxt_ball),
                                           np.where(call == 2, nxt_strike, np.where(k == 2, same, nxt_strike))))
    swing = (call == 1) | (call == 2); take = call == 0; inplay = PM.inplay[rows]; bipv = PM.bip_value[rows]
    ok = np.isfinite(realized)
    u_t = np.where(T['stand_r'][rows] == 1, PM.xt[rows], -PM.xt[rows])
    e = np.maximum.reduce([np.abs(u_t) - D.ZONE_HALF, PM.zt[rows] - D.ZONE_TOP, D.ZONE_BOT - PM.zt[rows]])
    ebin = np.digitize(e, (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5))
    elab = ('in_0.5+', 'in_0.25-0.5', 'in_0.1-0.25', 'in_0-0.1', 'out_0-0.1', 'out_0.1-0.25', 'out_0.25-0.5', 'out_0.5+')
    g = T['group'][rows]; fam = np.where(np.isin(g, (0, 1, 2, 6)), 0, np.where(np.isin(g, (3, 4)), 1, 2)); names = ('fastball', 'breaking', 'offspeed')
    cg = np.where(k == 2, 2, np.where(b > k, 1, 0)); cgn = ('even_or_ahead', 'behind', 'two_strikes')
    is_tr = np.zeros(n, bool); is_tr[:len(tr_idx)] = True
    day = T['day'][rows]
    splits = (('train', is_tr), ('test', ~is_tr), ('next30', ~is_tr & (day < fit.asof_day + 30)))
    out = {'note': 'engine = the expected value of the next state given the decision (count value, or the result when the plate appearance ends); realized = the value of the state the pitch actually led to; run values summed',
           'count_values': [round(float(x), 4) for x in cv]}
    for nm_, dec, val in (('swing', swing, v_sw), ('take', take, v_tk), ('in_play', inplay, v_bip)):
        y = bipv if nm_ == 'in_play' else realized
        d = out.setdefault(nm_, {})

        def add(key, m):
            m = m & dec & (ok | (nm_ == 'in_play'))
            d[key] = {'n': int(m.sum()), 'engine': round(float(val[m].sum()), 2), 'realized': round(float(y[m].sum()), 2)}
            if nm_ == 'swing':
                # CAL-03b: the pieces of a swing, expected against actual (miss, foul, ball in play and its value)
                pin = (1 - pw_all[m]) * (1 - pf_all[m])
                d[key].update({'whiff_exp': round(float(pw_all[m].sum()), 1), 'whiff': int((call[m] == 2).sum()),
                               'foul_exp': round(float(((1 - pw_all[m]) * pf_all[m]).sum()), 1), 'foul': int(((call[m] == 1) & (~lip[m] if not PM.foul_tip else ~inplay[m])).sum()),
                               'bip_exp': round(float(pin.sum()), 1), 'bip': int(inplay[m].sum()),
                               'bipv_exp': round(float((pin * v_bip[m]).sum()), 2), 'bipv': round(float(bipv[m].sum()), 2)})
        for sp, m0 in splits:
            add(f'{sp}|all', m0)
            for zn, mz in (('outside', e > 0), ('inside', e <= 0)):
                add(f'{sp}|{zn}', m0 & mz)
                for j, fn_ in enumerate(names):
                    add(f'{sp}|{zn}|{fn_}', m0 & mz & (fam == j))
                for c3, cn in enumerate(cgn):
                    add(f'{sp}|{zn}|{cn}', m0 & mz & (cg == c3))
            for j, lab in enumerate(elab):
                add(f'{sp}|{lab}', m0 & (ebin == j))
    return out


def calibration_maps(fit) -> dict:
    """CAL-04 (diagnostic): where the engine's price of an extra chase parts from what chases produce. On the later rows'
    pitches out of the zone, for hitters with maps, the engine's swing and take values with the hitter's own swing and
    whiff maps, against what his swings and takes produced, split by the hitter's own part at the pitch (his map's swing
    chance minus an average hitter's with his same levels, in points). Sums only; no player rows."""
    PM = fit.PM
    if PM is None:
        return {}
    T = fit.T; tr = fit.train; LW7 = D.LW7; cv = fit.cv
    rows_all = np.flatnonzero(~tr)
    bat = T['batter'][rows_all]
    edges = (-0.10, -0.05, 0.0, 0.05, 0.10)
    base_labels = ('own_under_-10', 'own_-10_to_-5', 'own_-5_to_0', 'own_0_to_5', 'own_5_to_10', 'own_over_10')
    labels = base_labels + tuple('inside|' + x for x in base_labels)      # outside pitches keep CAL-04's names; inside pitches are prefixed
    keys = ('n', 'own', 'swings', 'swings_map', 'swings_league', 'whiffs', 'whiffs_map', 'whiffs_league', 'whiff_ll_map', 'whiff_ll_league', 'swing_value_map', 'swing_value_league', 'swing_realized',
            'takes', 'take_value', 'take_realized', 'bip', 'bip_value_engine', 'bip_value_realized')
    acc = {lab: dict.fromkeys(keys, 0.0) for lab in labels}
    hitters = 0
    for h in np.unique(bat):
        h = int(h)
        if h not in PM.maps_s:
            continue
        r = rows_all[bat == h]; hitters += 1
        b_ = T['balls'][r].astype(np.int64); k_ = T['strikes'][r].astype(np.int64)
        blk = PM.blocks_at(PM.xp[r], PM.zp[r], PM.xt[r], PM.zt[r], T['stand_r'][r], T['throw_r'][r], T['group'][r], T['v0'][r].astype(np.float64), b_, k_)
        hs = (PM.prop_s[r], PM.prop_w[r], PM.bip_b[r]); psc = (PM.prop_p[r], PM.bip_p[r])
        s_h, w_h, c_, f_h, v_h, _ = PM.probs(blk, h, psc, hs=hs)
        s_l, w_l, _, f_l, v_l, _ = PM.probs(blk, None, psc, hs=hs)
        v_strike = np.where(k_ == 2, LW7[1], cv[np.clip(b_ * 3 + k_ + 1, 0, 11)])
        v_foul = PM.foul_value(b_, k_, cv)
        v_ball = np.where(b_ == 3, LW7[2], cv[np.clip((b_ + 1) * 3 + k_, 0, 11)])
        vs_h = w_h * v_strike + (1 - w_h) * (f_h * v_foul + (1 - f_h) * v_h)
        vs_l = w_l * v_strike + (1 - w_l) * (f_l * v_foul + (1 - f_l) * v_l)
        vt = c_ * v_strike + (1 - c_) * v_ball
        call = T['call'][r]; lip = T['last_in_pa'][r] == 1; out7 = T['out7'][r]
        fin = np.where(out7 >= 0, LW7[np.clip(out7, 0, 6)], np.nan)
        nxt_s = cv[np.clip(b_ * 3 + k_ + 1, 0, 11)]; nxt_b = cv[np.clip((b_ + 1) * 3 + k_, 0, 11)]; same = cv[np.clip(b_ * 3 + 2, 0, 11)]
        real = np.where(lip, fin, np.where(call == 0, np.where(T['cs'][r] == 1, nxt_s, nxt_b), np.where(call == 2, nxt_s, np.where(k_ == 2, same, nxt_s))))
        ok = np.isfinite(real); sw = ((call == 1) | (call == 2)) & ok; tk = (call == 0) & ok
        inp = PM.inplay[r]; bipv = PM.bip_value[r]
        own = s_h - s_l; bins = np.digitize(own, edges) + np.where(fit.outside[r], 0, len(base_labels))
        for j, lab in enumerate(labels):
            m = bins == j
            if not m.any():
                continue
            a = acc[lab]; ms = m & sw; mt = m & tk; mi = m & inp
            a['n'] += int(m.sum()); a['own'] += float(own[m].sum())
            a['swings'] += int(ms.sum()); a['swings_map'] += float(s_h[m].sum()); a['swings_league'] += float(s_l[m].sum())
            a['whiffs'] += int((ms & (call == 2)).sum()); a['whiffs_map'] += float(w_h[ms].sum()); a['whiffs_league'] += float(w_l[ms].sum())
            yw = (call[ms] == 2).astype(float); a['whiff_ll_map'] += float(D.logloss_vec(w_h[ms], yw).sum()); a['whiff_ll_league'] += float(D.logloss_vec(w_l[ms], yw).sum())
            a['swing_value_map'] += float(vs_h[ms].sum()); a['swing_value_league'] += float(vs_l[ms].sum()); a['swing_realized'] += float(real[ms].sum())
            a['takes'] += int(mt.sum()); a['take_value'] += float(vt[mt].sum()); a['take_realized'] += float(real[mt].sum())
            a['bip'] += int(mi.sum()); a['bip_value_engine'] += float(v_h[mi].sum()); a['bip_value_realized'] += float(bipv[mi].sum())
    out = {lab: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()} for lab, a in acc.items()}
    out['hitters'] = hitters
    return out


def environment_record() -> dict:
    """ENG-01: the versions a run used, kept in its receipt so a result can be reproduced (python, numpy, scipy, scikit-learn, pandas, the commit)."""
    import platform
    from importlib import metadata
    out = {'python': platform.python_version(), 'commit': os.environ.get('GITHUB_SHA')}
    for pkg in ('numpy', 'scipy', 'scikit-learn', 'pandas', 'cryptography'):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out

def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']
    from cloud.security import unseal, key_bytes
    from brl_live.bookkeeping_season import STUDY_SCHEMA, study_path, study_purpose
    key = key_bytes(os.environ['BRL_PA_PACKAGE_KEY'])
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    params = json.loads((ROOT / 'tools' / 'report_params.json').read_text()) if (ROOT / 'tools' / 'report_params.json').exists() else {}
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York'))
    if os.environ.get('BRL_REPORT_DAILY'):
        # the scheduled runs: yesterday (now graded) and today, each as of its own date, and tomorrow's early plan as of today
        # (the same pitches: today's games are not in the pitch table yet), so a staff can prep the night before
        today_et = now_et.date()
        params = {'publish': True, 'dates': [(today_et - timedelta(days=1)).isoformat(), today_et.isoformat(), (today_et + timedelta(days=1)).isoformat()],
                  'asof_for': {(today_et + timedelta(days=1)).isoformat(): today_et.isoformat()}}
    global SWING_CROSS, CS_SEASON, CS_RECENT, FOUL_FIX, LAM_W, WHIFF_OWN
    SWING_CROSS = bool(params.get('swing_cross', SWING_CROSS))
    CS_SEASON = bool(params.get('cs_season', CS_SEASON))
    CS_RECENT = params.get('cs_recent', CS_RECENT) or None
    FOUL_FIX = bool(params.get('foul_fix', FOUL_FIX))
    LAM_W = float(params.get('lam_w', LAM_W))
    WHIFF_OWN = int(params.get('whiff_own', WHIFF_OWN) or 0)
    dates = params.get('dates') or [params.get('date') or now_et.date().isoformat()]
    asof = params.get('asof')                      # one as-of date for every report in the run (a backfilled month); default: each report's own date
    asof_for = params.get('asof_for') or {}        # a date's own as-of when it differs (tomorrow's early plan)
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    receipt = {'schema': 'brl.report-receipt.v1', 'run_id': run_id, 'params': params, 'started_at': datetime.now(timezone.utc).isoformat(), 'stages': [], 'reports': {}}
    receipt['environment'] = environment_record()
    t0 = time.time()

    def stage(name):
        try:
            import resource
            rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)        # peak resident set so far, MB (Linux reports KB)
        except Exception:
            rss = None
        receipt['stages'].append({'stage': name, 'at_seconds': round(time.time() - t0, 1), 'peak_rss_mb': rss}); print(name, round(time.time() - t0), 's', rss, 'MB', flush=True)
    try:
        years = {int(d[:4]) for d in dates}
        seasons = sorted(years | {y - 1 for y in years} | set(int(s) for s in params.get('seasons', ())))
        tables = []
        for year in seasons:
            raw = D.read_blob(repo, token, study_path(int(year)), branch)
            if raw is None:
                continue
            doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(int(year)))))
            if doc.get('schema') != STUDY_SCHEMA:
                raise ValueError('study schema mismatch')
            tables.append(D.pitch_table(doc, int(year), ('R', 'F', 'D', 'L', 'W'))); del doc, raw
            stage(f'load {year}')
        T = D.concat(tables); del tables
        if params.get('lineup_study'):
            receipt['lineup_study'] = lineup_study(T, stage, params['lineup_study'])
            dates = []
        if params.get('matchup_read'):
            receipt['matchup_read'] = matchup_read_study(T, stage, params['matchup_read'])
            dates = []
        if params.get('tag_study'):
            receipt['tag_study'] = tag_study(T, stage, params['tag_study'])
            dates = []
        if params.get('pitcher_tag_study'):
            receipt['pitcher_tag_study'] = pitcher_tag_study(T, stage, params['pitcher_tag_study'])
            dates = []
        if params.get('zone_study'):
            zs = params['zone_study']
            receipt['zone_study'] = zone_shrink_study(T, stage, zs)
            for name, subset in (zs.get('subsets') or {}).items():          # ZONES-02: the same study on each subset
                receipt.setdefault('zone_study_subsets', {})[name] = zone_shrink_study(T, stage, dict(zs, subset=subset))
            dates = []
        fits = {}
        sim_pens = None
        for day in sorted(dates):
            a = asof_for.get(day) or asof or day
            if a > day:
                raise ValueError(f'report for {day} cannot be as of a later date ({a})')
            if a not in fits:
                fits.clear()
                fits[a] = Fitted(T, date.fromisoformat(a).toordinal(), stage, int(params.get('min_pitches', 300)), int(params.get('min_swings', 200)))
                if params.get('calibration_diag'):
                    receipt.setdefault('calibration', {})[a] = calibration(fits[a]); stage('calibration')
                    receipt.setdefault('calibration_components', {})[a] = calibration_components(fits[a]); stage('calibration of the other pieces')
                    if params.get('calibration_values'):
                        receipt.setdefault('calibration_values', {})[a] = calibration_values(fits[a]); stage('calibration of the swing and take values')
                    if params.get('calibration_maps'):
                        receipt.setdefault('calibration_maps', {})[a] = calibration_maps(fits[a]); stage('calibration of the chase price by own part')
            if params.get('calibration_only'):
                continue
            if sim_pens is None:
                sim_pens = sim_relievers(repo, token, branch); stage('simulator bullpens')
            rep = build_report(fits[a], T, day, a, stage, int(params.get("max_relievers", 6)), sim_pens=sim_pens)
            total = 0
            summary = {k: v for k, v in rep.items() if k != 'games'}
            summary['games'] = {}
            day_files = {}
            for pk, entry in rep['games'].items():
                doc = dict(entry); doc.update({'schema': SCHEMA, 'date': day, 'asof': a, 'game_pk': int(pk), 'grid': rep['grid'], 'training_pitches': fit_n(fits[a]), 'league': fits[a].league,
                                               'built_at': rep['built_at']})
                text = json.dumps(clean(doc), separators=(',', ':')); total += len(text)
                day_files[f'public/reports/{day}/{pk}.json'] = text
                summary['games'][pk] = {'teams': {sd: {'id': entry['teams'][sd]['id'], 'name': entry['teams'][sd]['name'], 'abbr': entry['teams'][sd]['abbr']} for sd in ('away', 'home')},
                                        'status': entry['status'], 'final': entry['final'], 'start': entry['start'], 'grade': entry.get('grade'),
                                        'pairs': sum(len(se['pairs']) for se in entry['sides'].values())}
            if params.get('publish', True):
                # the day's games and its summary in one commit
                day_files[f'public/reports/{day}/index.json'] = json.dumps(clean(summary), separators=(',', ':'))
                put_many(repo, token, day_files, branch, f'BRL report {day}: {len(rep["games"])} games and the summary (as of {a})')
            del day_files
            receipt['reports'][day] = {'games': len(rep['games']), 'bytes': total, 'asof': a, 'graded': sum(1 for g in rep['games'].values() if g.get('grade'))}
            stage(f'report {day}')
            del rep, summary
            if params.get('publish', True):
                # progress so far (a run the runner kills leaves its stages and memory behind; the final receipt replaces this)
                receipt['status'] = 'in progress'
                put(repo, token, f'research/report-{run_id}.json', json.dumps(receipt, indent=1), branch, f'BRL report progress {run_id}')
        if params.get('players') or os.environ.get('BRL_REPORT_DAILY'):
            # league-wide player cards as of the latest report date (the same fitted pieces the day's reports used)
            a_last = sorted(fits)[-1] if fits else None
            if a_last is not None:
                pidx, pshards = build_players(fits[a_last], a_last, stage)
                if params.get('publish', True):
                    pfiles = {f'public/reports/players/{k}.json': json.dumps(clean(shard), separators=(',', ':')) for k, shard in sorted(pshards.items())}
                    pfiles['public/reports/players/index.json'] = json.dumps(clean(pidx), separators=(',', ':'))
                    put_many(repo, token, pfiles, branch, f'BRL player cards and index (as of {a_last})')
                receipt['players'] = {'hitters': len(pidx['hitters']), 'pitchers': len(pidx['pitchers']), 'asof': a_last, 'shards': len(pshards)}
        if params.get('publish', True):
            stage('index')
            idx, days = rebuild_index(repo, token, branch)
            rec = record_from(days)
            put_many(repo, token, {'public/reports/index.json': json.dumps(idx, separators=(',', ':'), sort_keys=True),
                                   'public/reports/record.json': json.dumps(clean(rec), separators=(',', ':'))}, branch, 'BRL reports index and record')
            receipt['record'] = {k: rec[k] for k in ('games', 'dates', 'first_date', 'last_date', 'chases', 'chases_expected_league', 'chases_expected_map', 'in_recommended', 'usual_expected', 'outside_pitches')}
        receipt['status'] = 'completed'
    except Exception as exc:
        import traceback
        receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        receipt['trace'] = traceback.format_exc()[-2500:]
        print(receipt['trace'], flush=True)
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat(); receipt['seconds'] = round(time.time() - t0, 1)
    put(repo, token, f'research/report-{run_id}.json', json.dumps(receipt, indent=1), branch, f'BRL report receipt {run_id}')
    if receipt['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
