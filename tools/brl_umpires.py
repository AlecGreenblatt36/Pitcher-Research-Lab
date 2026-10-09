"""UMP-01: the plate umpire's zone. Which umpire worked the plate comes from the public box score (officials), cached per season on the
data branch (research/umpires-<season>.json: game, umpire id and name; public facts). On the sealed pitch table, taken pitches in the
band around the zone's edges are scored against a league called-strike rate by location cell and hitter side, and each umpire's
offset by edge region (low, high, inside, outside) is the shrunk log-odds surprise of his calls. The diagnostic: do those offsets
persist, between halves of a season (odd and even dates) and across seasons, and how large is the true spread between umpires?
Metrics and per-umpire offsets leave the runner (umpires are public figures; the offsets are aggregates over thousands of calls).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

EDGE = 0.35            # ft on either side of the zone edge that counts as the edge band
HALF_PLATE = 0.83      # ft, the rule-book half width plus the ball
ZONE_BOTTOM, ZONE_TOP = 1.5, 3.5
K_SHRINK = 400.0       # taken pitches of prior weight behind each umpire's regional offset


def mlb(url: str):
    import urllib.request
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'BRL-umpires/1.0', 'Accept': 'application/json'}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 + 3 * attempt)


def plate_umpire(game_pk: int) -> dict | None:
    doc = mlb(f'https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/boxscore')
    for o in doc.get('officials') or []:
        if str(o.get('officialType') or '').lower().startswith('home plate'):
            off = o.get('official') or {}
            if off.get('id') is not None:
                return {'id': int(off['id']), 'name': str(off.get('fullName') or off['id'])}
    return None


def officials_for_games(game_pks, workers: int = 6, log=None) -> dict:
    """game_pk -> {'id', 'name'} of the plate umpire, from the public box scores."""
    out = {}
    pks = [int(p) for p in game_pks]

    def one(pk):
        try:
            return pk, plate_umpire(pk)
        except Exception as exc:
            return pk, {'error': type(exc).__name__}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (pk, rec) in enumerate(ex.map(one, pks)):
            if rec and 'error' not in rec:
                out[str(pk)] = rec
            if log and i % 500 == 0:
                log(f'officials {i} of {len(pks)}')
    return out


def load_or_build(repo, token, branch, season: int, game_pks, stage, read_blob, put_bytes, workers: int = 6) -> dict:
    """The season's cached table, extended with any games it lacks."""
    path = f'research/umpires-{season}.json'
    raw = read_blob(repo, token, path, branch)
    table = json.loads(raw.decode()) if raw else {'schema': 'brl.umpires.v1', 'season': season, 'games': {}}
    missing = [pk for pk in sorted(set(int(p) for p in game_pks)) if str(pk) not in table['games']]
    if missing:
        stage(f'umpires {season}: {len(missing)} games to fetch')
        table['games'].update(officials_for_games(missing, workers, log=lambda m: print(m, flush=True)))
        put_bytes(repo, token, path, json.dumps(table, separators=(',', ':')).encode(), branch, f'BRL: plate umpires {season}')
    stage(f'umpires {season}: {len(table["games"])} games with a plate umpire')
    return table


def region_of(px, pz, stand_r):
    """Edge region of a taken pitch, or -1 when it is not in the band: 0 low, 1 high, 2 inside, 3 outside (inside is toward the hitter)."""
    ax = np.abs(px)
    near_side = (ax >= HALF_PLATE - EDGE) & (ax <= HALF_PLATE + EDGE) & (pz >= ZONE_BOTTOM - EDGE) & (pz <= ZONE_TOP + EDGE)
    near_low = (pz >= ZONE_BOTTOM - EDGE) & (pz <= ZONE_BOTTOM + EDGE) & (ax <= HALF_PLATE + EDGE)
    near_high = (pz >= ZONE_TOP - EDGE) & (pz <= ZONE_TOP + EDGE) & (ax <= HALF_PLATE + EDGE)
    r = np.full(len(px), -1, dtype=np.int64)
    inside = np.where(stand_r == 1, px < 0, px > 0)          # a right-handed hitter stands on the catcher's left (negative px is toward him)
    r[near_side & inside] = 2; r[near_side & ~inside] = 3
    r[near_low] = 0; r[near_high] = 1                            # a corner counts as low or high (the vertical edge is where umpires differ most)
    return r


def league_rates(px, pz, stand_r, cs, cell=0.25):
    """Called-strike rate on taken pitches by location cell and hitter side (additive smoothing toward 0.5 with a weight of 2)."""
    ix = np.floor((px + 2.5) / cell).astype(np.int64); iz = np.floor((pz - 0.0) / cell).astype(np.int64)
    key = (stand_r.astype(np.int64) * 1000 + np.clip(ix, 0, 39)) * 100 + np.clip(iz, 0, 59)
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv, minlength=len(uniq)).astype(float); s = np.bincount(inv, weights=cs.astype(float), minlength=len(uniq))
    rate = (s + 1.0) / (n + 2.0)
    return rate[inv]


def offsets(ump, region, cs, p, k=K_SHRINK):
    """Per umpire and region: the shrunk log-odds surprise (observed minus expected strikes over the expected variance), with counts."""
    out = {}
    for u in np.unique(ump):
        m = ump == u
        rec = {}
        for r, name in enumerate(('low', 'high', 'inside', 'outside')):
            mm = m & (region == r)
            n = int(mm.sum())
            if n == 0:
                continue
            var = float(np.sum(p[mm] * (1 - p[mm]))); surprise = float(np.sum(cs[mm] - p[mm]))
            raw = surprise / max(var, 1e-9)
            rec[name] = {'n': n, 'offset': raw * n / (n + k), 'strikes_per_100': 100.0 * surprise / n * n / (n + k)}
        mm = m & (region >= 0)
        if mm.sum():
            var = float(np.sum(p[mm] * (1 - p[mm]))); surprise = float(np.sum(cs[mm] - p[mm])); n = int(mm.sum())
            rec['all_edges'] = {'n': n, 'offset': surprise / max(var, 1e-9) * n / (n + k), 'strikes_per_100': 100.0 * surprise / n * n / (n + k)}
        out[int(u)] = rec
    return out


def persistence(a: dict, b: dict, region: str, min_n: int = 150) -> dict:
    """Correlation across umpires of a region's offsets between two halves, with the noise-corrected spread."""
    xs, ys, ns = [], [], []
    for u in a:
        if u in b and region in a[u] and region in b[u] and a[u][region]['n'] >= min_n and b[u][region]['n'] >= min_n:
            xs.append(a[u][region]['strikes_per_100']); ys.append(b[u][region]['strikes_per_100']); ns.append(min(a[u][region]['n'], b[u][region]['n']))
    if len(xs) < 8:
        return {'umpires': len(xs)}
    xs = np.array(xs); ys = np.array(ys)
    corr = float(np.corrcoef(xs, ys)[0, 1])
    sd = float(np.std(np.concatenate([xs, ys]))); true_sd = sd * np.sqrt(max(corr, 0.0))      # the reliability of one half's figure is its correlation with the other
    return {'umpires': int(len(xs)), 'corr': round(corr, 3), 'sd_half_figures': round(sd, 3), 'true_sd_strikes_per_100': round(float(true_sd), 3), 'median_n_per_half': int(np.median(ns))}


def study(T: dict, tables: dict, params: dict, stage) -> dict:
    """tables: season -> the cached umpire table. Only taken pitches with a known plate umpire enter."""
    taken = (T['call'] == 0) & (T['group'] >= 0) & np.isfinite(T['px']) & np.isfinite(T['pz']) & (T['bunt_pa'] == 0)
    ump = np.full(len(T['game']), -1, dtype=np.int64); names = {}
    by_game = {}
    for season, tab in tables.items():
        for pk, rec in (tab.get('games') or {}).items():
            by_game[int(pk)] = int(rec['id']); names[int(rec['id'])] = rec.get('name')
    games = T['game']
    uniq_g, inv_g = np.unique(games, return_inverse=True)
    lookup = np.array([by_game.get(int(g), -1) for g in uniq_g], dtype=np.int64)
    ump = lookup[inv_g]
    keep = taken & (ump >= 0)
    stage(f'taken pitches with a plate umpire: {int(keep.sum())} of {int(taken.sum())}')
    px = T['px'][keep].astype(float); pz = T['pz'][keep].astype(float); sr = T['stand_r'][keep]; cs = T['cs'][keep].astype(float); u = ump[keep]
    season = T['season'][keep]; day = T['day'][keep]
    p = league_rates(px, pz, sr, cs)
    region = region_of(px, pz, sr)
    out = {'pitches': int(len(px)), 'edge_pitches': int((region >= 0).sum()), 'umpires': int(len(np.unique(u))), 'league_strike_rate_edges': round(float(cs[region >= 0].mean()), 4),
           'edge_shares': {name: round(float((region == r).mean()), 4) for r, name in enumerate(('low', 'high', 'inside', 'outside'))}}
    seasons = sorted(int(s) for s in np.unique(season))
    out['by_season'] = {}
    full = {}
    for s in seasons:
        ms = season == s
        a = offsets(u[ms & (day % 2 == 1)], region[ms & (day % 2 == 1)], cs[ms & (day % 2 == 1)], p[ms & (day % 2 == 1)])
        b = offsets(u[ms & (day % 2 == 0)], region[ms & (day % 2 == 0)], cs[ms & (day % 2 == 0)], p[ms & (day % 2 == 0)])
        full[s] = offsets(u[ms], region[ms], cs[ms], p[ms])
        out['by_season'][s] = {'games': int(len(np.unique(games[keep][ms]))), 'umpires': int(len(np.unique(u[ms]))),
                               'split_half': {r: persistence(a, b, r, 300 if r == 'all_edges' else 150) for r in ('low', 'high', 'inside', 'outside', 'all_edges')}}
        stage(f'season {s} offsets')
    if len(seasons) >= 2:
        s0, s1 = seasons[-2], seasons[-1]
        out['across_seasons'] = {f'{s0}_vs_{s1}': {r: persistence(full[s0], full[s1], r, 300 if r == 'all_edges' else 150) for r in ('low', 'high', 'inside', 'outside', 'all_edges')}}
    # the umpires' table for the product: the latest season's offsets with names, games and edge calls (public figures, aggregate behavior)
    last = seasons[-1]
    table = {}
    for uid, rec in full[last].items():
        if 'all_edges' in rec and rec['all_edges']['n'] >= 300:
            table[str(uid)] = {'name': names.get(uid), 'edge_calls': rec['all_edges']['n'],
                               'strikes_per_100': {r: round(rec[r]['strikes_per_100'], 2) for r in rec}}
    out['umpires_table'] = {'season': last, 'count': len(table), 'rows': table}
    ranked = sorted(table.items(), key=lambda kv: kv[1]['strikes_per_100'].get('all_edges', 0.0))
    out['widest_and_tightest'] = {'tightest': [(v['name'], v['strikes_per_100'].get('all_edges')) for _, v in ranked[:5]], 'widest': [(v['name'], v['strikes_per_100'].get('all_edges')) for _, v in ranked[-5:]]}
    return out
