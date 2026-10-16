"""STAB-01: how many plate appearances, swings, pitches or balls in play it takes before a player's rate means
something, and how much of it carries into the next season, from every regular season 2015 to 2026.

Input: one row per regular-season plate appearance with the season, the batter and pitcher, the outcome label and the
per-PA pitch sums of the physics table (pitches n, swings sw, whiffs wh, out-of-zone pitches oz, chases ch, in-zone
swings izs and contact izc, fastballs fb and the sum of their speeds velo) and the batted ball's ev and la.

Two measures per stat, each turned into a shrinkage constant k (the amount of league-average sample to add, so a
player's estimate is (his total + k x league rate) / (his sample + k)):

  split half    within a season, a player's first 2N units split alternately into two halves of about N; the
                correlation r between the halves across players gives k = N (1 - r) / r (how noisy the rate is);
  next season   players with at least M units in two consecutive seasons; the correlation between the seasons, at
                their mean sample n, gives k = n (1 - r) / r (how much to shrink to predict the next season, which
                also covers real change).

Both are reported for 2015-2019 (Trackman) and 2021-2026 (Hawk-Eye) separately and pooled; 2020's 60 games give
too few qualified players and are only used as a split-half season. Metrics only; no player rows leave the runner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# name: (role b/p, unit, N per half, M per season, numerator, denominator); units are what the denominator counts.
STATS = {
    'K per PA':             ('b', 'PA', 150, 300, lambda d: d['outcome'] == 'K', None),
    'BB+HBP per PA':        ('b', 'PA', 150, 300, lambda d: d['outcome'] == 'BB_HBP', None),
    'HR per PA':            ('b', 'PA', 150, 300, lambda d: d['outcome'] == 'HR', None),
    '1B per PA':            ('b', 'PA', 150, 300, lambda d: d['outcome'] == '1B', None),
    'whiff per swing':      ('b', 'swings', 150, 600, lambda d: d['wh'], lambda d: d['sw']),
    'chase per pitch out of zone': ('b', 'pitches out of zone', 150, 600, lambda d: d['ch'], lambda d: d['oz']),
    'swing per pitch':      ('b', 'pitches', 300, 1000, lambda d: d['sw'], lambda d: d['n']),
    'contact per zone swing': ('b', 'zone swings', 150, 400, lambda d: d['izc'], lambda d: d['izs']),
    'exit velocity':        ('b', 'balls in play', 75, 200, lambda d: d['ev'].fillna(0), lambda d: d['ev'].notna().astype(float)),
    'hard-hit per ball in play': ('b', 'balls in play', 75, 200, lambda d: (d['ev'] >= 95).astype(float), lambda d: d['ev'].notna().astype(float)),
    'launch angle':         ('b', 'balls in play', 75, 200, lambda d: d['la'].fillna(0), lambda d: d['la'].notna().astype(float)),
    'P: K per PA':          ('p', 'PA', 150, 300, lambda d: d['outcome'] == 'K', None),
    'P: BB+HBP per PA':     ('p', 'PA', 150, 300, lambda d: d['outcome'] == 'BB_HBP', None),
    'P: HR per PA':         ('p', 'PA', 150, 300, lambda d: d['outcome'] == 'HR', None),
    'P: whiff per swing':   ('p', 'swings', 150, 600, lambda d: d['wh'], lambda d: d['sw']),
    'P: chase per pitch out of zone': ('p', 'pitches out of zone', 150, 600, lambda d: d['ch'], lambda d: d['oz']),
    'P: zone per pitch':    ('p', 'pitches', 300, 1000, lambda d: d['n'] - d['oz'], lambda d: d['n']),
    'P: fastball speed':    ('p', 'fastballs', 50, 300, lambda d: d['velo'], lambda d: d['fb']),
    'P: exit velocity allowed': ('p', 'balls in play', 75, 200, lambda d: d['ev'].fillna(0), lambda d: d['ev'].notna().astype(float)),
    'P: hard-hit allowed':  ('p', 'balls in play', 75, 200, lambda d: (d['ev'] >= 95).astype(float), lambda d: d['ev'].notna().astype(float)),
}
ERAS = {'2015-2019': range(2015, 2020), '2021-2026': range(2021, 2027)}


def _k(n: float, r: float):
    return None if not np.isfinite(r) or r <= 0.02 else float(n * (1 - r) / r)


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 30 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def unit_frame(pa: pd.DataFrame, stat: str) -> pd.DataFrame:
    role, unit, n_half, m_season, num, den = STATS[stat]
    who = 'batter' if role == 'b' else 'pitcher'
    out = pd.DataFrame({'season': pa['season'].to_numpy(), 'player': pa[who].to_numpy(), 'order': np.arange(len(pa)),
                        'num': np.asarray(num(pa), float), 'den': np.ones(len(pa)) if den is None else np.asarray(den(pa), float)})
    return out[out['den'] > 0]


def split_half(u: pd.DataFrame, n_half: int) -> list[dict]:
    """Per season: each player's first 2N units, split alternately by plate appearance."""
    rows = []
    for season, g in u.groupby('season'):
        a_rates, b_rates, ns = [], [], []
        for _, p in g.sort_values('order').groupby('player'):
            cum = p['den'].cumsum().to_numpy()
            if cum[-1] < 2 * n_half:
                continue
            p = p.iloc[: int(np.searchsorted(cum, 2 * n_half) + 1)]
            half = np.arange(len(p)) % 2
            na, nb = p['den'].to_numpy()[half == 0].sum(), p['den'].to_numpy()[half == 1].sum()
            if na <= 0 or nb <= 0:
                continue
            a_rates.append(p['num'].to_numpy()[half == 0].sum() / na); b_rates.append(p['num'].to_numpy()[half == 1].sum() / nb); ns.append((na + nb) / 2)
        rows.append({'season': int(season), 'players': len(ns), 'n': float(np.mean(ns)) if ns else None, 'r': _corr(a_rates, b_rates)})
    return rows


def next_season(u: pd.DataFrame, m_season: int) -> list[dict]:
    tot = u.groupby(['player', 'season'])[['num', 'den']].sum().reset_index()
    tot = tot[tot['den'] >= m_season]
    tot['rate'] = tot['num'] / tot['den']
    nxt = tot.assign(season=tot['season'] - 1)
    j = tot.merge(nxt, on=['player', 'season'], suffixes=('', '_next'))
    rows = []
    for season, g in j.groupby('season'):
        rows.append({'season': int(season), 'pairs': int(len(g)), 'n': float(g['den'].mean()), 'r': _corr(g['rate'], g['rate_next'])})
    return rows


def _pool(rows: list[dict], seasons, count: str) -> dict:
    rs = [x for x in rows if x['season'] in seasons and x['n'] and np.isfinite(x['r']) and x[count] >= 30]
    if not rs:
        return {'seasons': 0}
    w = np.array([x[count] for x in rs], float)
    r = float(np.average([x['r'] for x in rs], weights=w)); n = float(np.average([x['n'] for x in rs], weights=w))
    return {'seasons': len(rs), count: int(w.sum()), 'n': round(n, 1), 'r': round(r, 3), 'k': None if _k(n, r) is None else round(_k(n, r), 1)}


def run(pa: pd.DataFrame) -> dict:
    """pa: one row per regular-season plate appearance, in date order within each season."""
    pa = pa.reset_index(drop=True)
    out = {'plate_appearances_by_season': {int(s): int(c) for s, c in pa['season'].value_counts().sort_index().items()}, 'stats': {}}
    for stat, (role, unit, n_half, m_season, _, _) in STATS.items():
        u = unit_frame(pa, stat)
        sh = split_half(u, n_half); ns = next_season(u, m_season)
        res = {'unit': unit, 'split_half_n': n_half, 'next_season_min': m_season, 'split_half': {}, 'next_season': {}}
        for era, seasons in list(ERAS.items()) + [('all', range(2015, 2027))]:
            res['split_half'][era] = _pool(sh, seasons, 'players')
            res['next_season'][era] = _pool(ns, seasons, 'pairs')
        res['by_season'] = {'split_half': [{**x, 'r': round(x['r'], 3) if np.isfinite(x['r']) else None} for x in sh],
                            'next_season': [{**x, 'r': round(x['r'], 3) if np.isfinite(x['r']) else None} for x in ns]}
        out['stats'][stat] = res
    return out
