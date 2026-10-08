"""Discovery experiments on the sealed pitch-by-pitch seasons, inside Actions (discovery/ has the protocols).

Experiment 'horizon' (discovery/DECISION_HORIZON_PROTOCOL.md): when does a hitter commit? Each pitch's flight is
rebuilt from the feed's measurements; the crossing a hitter would have projected if he committed tau seconds before
the plate is computed under several expectations; swing decisions are modeled on the projected crossing and the
held-out fit is profiled over tau. Only metrics, coefficients and per-hitter summaries leave the runner:
research/discovery-<experiment>-<run>.json on the ledger branch.

Settings: tools/discovery_params.json on the trigger branch.
"""
from __future__ import annotations

import base64, csv, gzip, io, json, os, sys, time, traceback
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))

G = 32.174                      # ft/s^2
FT_PER_MPH = 5280.0 / 3600.0
Y0, Y40, YPLATE = 50.0, 40.0, 17.0 / 12.0
ZONE_HALF, ZONE_TOP, ZONE_BOT = 0.83, 3.5, 1.5
TYPE_GROUPS = {'FF': 0, 'FA': 0, 'SI': 1, 'FC': 2, 'SL': 3, 'ST': 3, 'SV': 3, 'CU': 4, 'KC': 4, 'CS': 4, 'CH': 5, 'FS': 5, 'FO': 5, 'SC': 5}
GROUP_NAMES = ('four-seam', 'sinker', 'cutter', 'slider/sweeper', 'curveball', 'changeup/splitter', 'other')
SWING_CODES = {'S', 'W', 'F', 'T', 'O', 'X', 'D', 'E'}
WHIFF_CODES = {'S', 'W'}
TAKE_CODES = {'B', '*B', 'C'}


# ---------------------------------------------------------------- data access (same pattern as tools/brl_research.py)
def api(url, token, method='GET', payload=None):
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-discovery/1.0'}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode(); headers['Content-Type'] = 'application/json'
    with urlopen(Request(url, headers=headers, data=body, method=method), timeout=120) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def read_blob(repo, token, path, branch):
    try:
        value = api(f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}', token)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if value.get('encoding') == 'base64' and value.get('content'):
        return base64.b64decode(''.join(value['content'].split()))
    blob = api(f'https://api.github.com/repos/{repo}/git/blobs/{value["sha"]}', token)
    return base64.b64decode(''.join(blob['content'].split()))


def put_text(repo, token, path, text, branch, message):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    for attempt in range(6):
        payload = {'message': message, 'content': base64.b64encode(text.encode()).decode(), 'branch': branch}
        try:
            payload['sha'] = api(url + '?ref=' + branch, token)['sha']
        except HTTPError as exc:
            if exc.code != 404:
                raise
        try:
            return api(url, token, 'PUT', payload)
        except HTTPError as exc:
            if exc.code != 409 or attempt == 5:
                raise
            time.sleep(3 + 3 * attempt)


# ---------------------------------------------------------------- pitch table
FIELDS = ('season', 'day', 'game', 'pitcher', 'batter', 'stand_r', 'throw_r', 'inning', 'group', 'balls', 'strikes', 'call',
          'v0', 'v1', 'spin', 'pfx_x', 'pfx_z', 'px', 'pz', 'x0', 'z0', 'ext', 'last_in_pa', 'bunt_pa')
CALLS = {'take': 0, 'swing_contact': 1, 'whiff': 2, 'other': 3}


def pitch_table(doc: dict, season: int) -> dict:
    """One row per pitch of the regular season, as numpy arrays (pitch-level data stays on the runner)."""
    cols = {k: [] for k in FIELDS}
    for gpk, game in (doc.get('games') or {}).items():
        if game.get('game_type') != 'R':
            continue
        day = date.fromisoformat(str(game['date'])[:10]).toordinal()
        for row in game.get('rows') or []:
            if row.get('b') is None or row.get('p') is None:
                continue
            pitches = row.get('pitches') or []
            bunt = 1 if 'bunt' in str(row.get('e') or '') else 0
            for j, pt in enumerate(pitches):
                typ, code, balls, strikes, v0, v1, spin, pfx_x, pfx_z, px, pz, x0, z0, ext, _zone = (list(pt) + [None] * 15)[:15]
                code = str(code or '')
                call = 0 if code in TAKE_CODES else 2 if code in WHIFF_CODES else 1 if code in SWING_CODES else 3
                cols['season'].append(season); cols['day'].append(day); cols['game'].append(int(gpk))
                cols['pitcher'].append(int(row['p'])); cols['batter'].append(int(row['b']))
                cols['stand_r'].append(1 if str(row.get('s') or 'R') == 'R' else 0); cols['throw_r'].append(1 if str(row.get('t') or 'R') == 'R' else 0)
                cols['inning'].append(int(row.get('inning') or 0))
                cols['group'].append(TYPE_GROUPS.get(str(typ or '').upper(), 6) if typ not in ('PO', 'IN', 'AB') else -1)
                cols['balls'].append(int(balls if balls is not None else -1)); cols['strikes'].append(int(strikes if strikes is not None else -1))
                cols['call'].append(call)
                for k, v in (('v0', v0), ('v1', v1), ('spin', spin), ('pfx_x', pfx_x), ('pfx_z', pfx_z), ('px', px), ('pz', pz), ('x0', x0), ('z0', z0), ('ext', ext)):
                    cols[k].append(np.nan if v is None else float(v))
                cols['last_in_pa'].append(1 if j == len(pitches) - 1 else 0); cols['bunt_pa'].append(bunt)
    out = {}
    for k, v in cols.items():
        out[k] = np.asarray(v, dtype=np.float32 if k in ('v0', 'v1', 'spin', 'pfx_x', 'pfx_z', 'px', 'pz', 'x0', 'z0', 'ext') else np.int64)
    return out


def concat(tables: list) -> dict:
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def take(t: dict, mask) -> dict:
    return {k: v[mask] for k, v in t.items()}


# ---------------------------------------------------------------- rebuilt flight (reconstructed quantities)
def rebuild(t: dict) -> dict:
    """Constant-acceleration flight from y = 50 ft to the front of the plate. Adds flight time (s), time from 40 ft,
    accelerations ax, az (ft/s^2, gravity included in az), initial velocities and the spin accelerations."""
    v0 = t['v0'].astype(np.float64) * FT_PER_MPH; v1 = t['v1'].astype(np.float64) * FT_PER_MPH
    dy = Y0 - YPLATE
    ay = (v0 ** 2 - v1 ** 2) / (2 * dy)
    with np.errstate(invalid='ignore', divide='ignore'):
        tf = (v0 - v1) / ay
        disc = np.maximum(v0 ** 2 - 2 * ay * (Y0 - Y40), 0.0)
        t40 = (v0 - np.sqrt(disc)) / ay
        T = tf - t40
        asx = 2 * (t['pfx_x'].astype(np.float64) / 12.0) / T ** 2
        asz = 2 * (t['pfx_z'].astype(np.float64) / 12.0) / T ** 2
    ax = asx; az = asz - G
    vx0 = (t['px'] - t['x0'] - 0.5 * ax * tf ** 2) / tf
    vz0 = (t['pz'] - t['z0'] - 0.5 * az * tf ** 2) / tf
    ok = np.isfinite(tf) & (ay > 8) & (ay < 50) & (tf > 0.3) & (tf < 0.7) & np.isfinite(ax) & np.isfinite(az) & np.isfinite(t['px']) & np.isfinite(t['pz'])
    return {'ay': ay, 'tf': tf, 'T40': T, 'ax': ax, 'az': az, 'asx': asx, 'asz': asz, 'vx0': vx0, 'vz0': vz0, 'ok': ok}


def group_means(keys: np.ndarray, values: np.ndarray, mask: np.ndarray):
    """Mean of values per key (rows where mask); returns (unique keys, means, counts)."""
    k = keys[mask]; v = values[mask]
    uk, inv = np.unique(k, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uk)); cnt = np.bincount(inv, minlength=len(uk))
    return uk, sums / np.maximum(cnt, 1), cnt


def lookup(uk, means, keys, default):
    idx = np.searchsorted(uk, keys); idx = np.clip(idx, 0, len(uk) - 1)
    hit = uk[idx] == keys
    return np.where(hit, means[idx], default)


def observer_accels(t: dict, f: dict) -> dict:
    """Expected accelerations (ax, az) for each pitch under observers O2 (fastball) and O3 (own type), from the
    pitcher's season averages. O1 (straight) is (0, -g)."""
    ok = f['ok'] & (t['group'] >= 0)
    ps = t['pitcher'] * 10 + (t['season'] - 2020)
    key_type = ps * 10 + t['group']
    out = {}
    for comp in ('ax', 'az'):
        uk, m, _ = group_means(key_type, f[comp], ok)
        out['own_' + comp] = lookup(uk, m, key_type, np.nan)
    # primary fastball: most-thrown of four-seam, sinker, cutter in the pitcher-season
    fb = ok & np.isin(t['group'], (0, 1, 2))
    uk, _, cnt = group_means(key_type, np.ones(len(ps)), fb)
    best = {}
    for k, c in zip(uk, cnt):
        p = k // 10
        if p not in best or c > best[p][1]:
            best[p] = (k, c)
    fb_key = np.array([best.get(p, (-1, 0))[0] for p in ps]) if len(ps) else np.zeros(0, int)
    league = {comp: float(np.nanmean(f[comp][ok & (t['group'] == 0)])) for comp in ('ax', 'az')}
    for comp in ('ax', 'az'):
        uk2, m2, _ = group_means(key_type, f[comp], ok)
        out['fb_' + comp] = lookup(uk2, m2, fb_key, league[comp])
        out['own_' + comp] = np.where(np.isfinite(out['own_' + comp]), out['own_' + comp], out['fb_' + comp])
    return out


def mixture_projection(t: dict, f: dict, tau: float, kappa: float, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observer O4: posterior over the pitcher's types from what is visible at commit (speed, vertical and horizontal
    velocity, position), each type a Gaussian with its own spread inflated by kappa times the league spread
    (perceptual noise), prior = the pitcher's usage of each type in that strike count to that batter hand.
    Returns projected crossing (x, z) and the posterior entropy in bits."""
    n = len(t['px']); ok = f['ok'] & (t['group'] >= 0)
    tc = f['tf'] - tau
    vy0 = -t['v0'].astype(np.float64) * FT_PER_MPH
    cues = np.column_stack([
        -(vy0 + f['ay'] * tc),                                   # speed toward the plate at commit
        f['vz0'] + f['az'] * tc,                                  # vertical velocity at commit
        f['vx0'] + f['ax'] * tc,                                  # horizontal velocity at commit
        t['z0'] + f['vz0'] * tc + 0.5 * f['az'] * tc ** 2,        # height at commit
        t['x0'] + f['vx0'] * tc + 0.5 * f['ax'] * tc ** 2,        # side at commit
    ])
    league_sd = np.nanstd(cues[ok], axis=0)
    psh = (t['pitcher'] * 10 + (t['season'] - 2020)) * 2 + t['stand_r']
    xs = np.full(n, np.nan); zs = np.full(n, np.nan); ent = np.full(n, np.nan)
    order = np.argsort(psh, kind='stable'); bounds = np.flatnonzero(np.diff(psh[order])) + 1
    for idx in np.split(order, bounds):
        idx = idx[ok[idx]]
        if len(idx) < 30:
            continue
        groups = np.unique(t['group'][idx])
        logp = np.full((len(idx), len(groups)), -np.inf); proj_x = np.zeros((len(idx), len(groups))); proj_z = np.zeros((len(idx), len(groups)))
        for gi, gname in enumerate(groups):
            sel = idx[t['group'][idx] == gname]
            if len(sel) < 5:
                continue
            mu = cues[sel].mean(axis=0); sd = np.sqrt(cues[sel].var(axis=0) + (kappa * league_sd) ** 2) + 1e-6
            ll = -0.5 * (((cues[idx] - mu) / sd) ** 2).sum(axis=1) - np.log(sd).sum()
            # prior: usage by strikes (0, 1, 2) for this pitcher-season-hand, smoothed
            s = t['strikes'][idx]
            prior = np.empty(len(idx))
            for k in (0, 1, 2):
                in_k = t['strikes'][idx] == k
                use = (np.sum(t['group'][idx][in_k] == gname) + 1.0) / (np.sum(in_k) + len(groups))
                prior[s == k] = use
            logp[:, gi] = ll + np.log(prior)
            ax_k, az_k = f['ax'][sel].mean(), f['az'][sel].mean()
            proj_x[:, gi] = t['px'][idx] - 0.5 * (f['ax'][idx] - ax_k) * tau ** 2
            proj_z[:, gi] = t['pz'][idx] - 0.5 * (f['az'][idx] - az_k) * tau ** 2
        m = logp.max(axis=1, keepdims=True); w = np.exp(logp - m); w /= w.sum(axis=1, keepdims=True)
        xs[idx] = (w * proj_x).sum(axis=1); zs[idx] = (w * proj_z).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ent[idx] = -np.nansum(np.where(w > 0, w * np.log2(w), 0.0), axis=1)
    return xs, zs, ent


def projected(t: dict, f: dict, acc: dict, observer: str, tau: float, shuffle=None):
    """Projected plate crossing (x, z) for a commit tau seconds before the plate."""
    ax, az = f['ax'], f['az']
    if shuffle is not None:
        ax, az = ax[shuffle], az[shuffle]
    if observer == 'straight':
        ox, oz = 0.0, -G
    elif observer == 'fastball':
        ox, oz = acc['fb_ax'], acc['fb_az']
    elif observer == 'own_type':
        ox, oz = acc['own_ax'], acc['own_az']
    else:
        raise ValueError(observer)
    return t['px'] - 0.5 * (ax - ox) * tau ** 2, t['pz'] - 0.5 * (az - oz) * tau ** 2


# ---------------------------------------------------------------- swing model
def hats(x: np.ndarray, knots) -> np.ndarray:
    """Piecewise-linear (hat) basis; values outside the knots are clipped to the end knots."""
    k = np.asarray(knots, float); x = np.clip(x, k[0], k[-1])
    out = np.zeros((len(x), len(k)), dtype=np.float32)
    j = np.clip(np.searchsorted(k, x, side='right') - 1, 0, len(k) - 2)
    w = (x - k[j]) / (k[j + 1] - k[j])
    out[np.arange(len(x)), j] = 1 - w; out[np.arange(len(x)), j + 1] = w
    return out


E_KNOTS = (-0.8, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.8, 1.2, 2.0)
U_KNOTS = (-2.0, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 2.0)
Z_KNOTS = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
V_KNOTS = (70.0, 80.0, 86.0, 90.0, 94.0, 99.0)


def location_block(x: np.ndarray, z: np.ndarray, stand_r: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    u = np.where(stand_r == 1, x, -x)                      # positive = away from the batter
    e = np.maximum(np.maximum(np.abs(u) - ZONE_HALF, z - ZONE_TOP), ZONE_BOT - z)
    base = np.hstack([hats(e, E_KNOTS), hats(u, U_KNOTS), hats(z, Z_KNOTS)])
    blocks = [base * (strikes == k)[:, None] for k in (0, 1, 2)]
    return np.hstack(blocks).astype(np.float32)


def control_block(t: dict, prop: np.ndarray) -> np.ndarray:
    n = len(t['balls'])
    cnt = np.zeros((n, 12), np.float32); cnt[np.arange(n), np.clip(t['balls'], 0, 3) * 3 + np.clip(t['strikes'], 0, 2)] = 1
    grp = np.zeros((n, 7), np.float32); grp[np.arange(n), np.clip(t['group'], 0, 6)] = 1
    grp2 = grp * (t['strikes'] == 2)[:, None]
    vel = hats(t['v0'].astype(np.float64), V_KNOTS)
    return np.hstack([cnt, grp, grp2, vel, prop[:, None].astype(np.float32), t['stand_r'][:, None].astype(np.float32),
                      (t['stand_r'] == t['throw_r'])[:, None].astype(np.float32)])


def swing_propensity(t: dict, k: float = 300.0) -> np.ndarray:
    """Logit of each batter's swing rate on earlier dates (shrunk toward the league by k pitches)."""
    swing = (t['call'] == 1) | (t['call'] == 2)
    valid = t['call'] <= 2
    order = np.lexsort((t['day'], t['batter']))
    out = np.zeros(len(swing))
    b, d = t['batter'][order], t['day'][order]
    s = swing[order].astype(float); v = valid[order].astype(float)
    league = s.sum() / max(v.sum(), 1)
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or b[i] != b[start]:
            # cumulative counts strictly before each date
            cs = np.cumsum(s[start:i]); cv = np.cumsum(v[start:i]); dd = d[start:i]
            first = np.searchsorted(dd, dd, side='left')            # index of the first pitch of the same date
            prev_s = np.where(first > 0, cs[np.maximum(first - 1, 0)], 0.0); prev_v = np.where(first > 0, cv[np.maximum(first - 1, 0)], 0.0)
            rate = (prev_s + k * league) / (prev_v + k)
            out[order[start:i]] = np.log(rate / (1 - rate))
            start = i
    return out


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0, warm=None):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=300, tol=1e-6, warm_start=warm is not None)
    if warm is not None:
        m.coef_ = warm[0].copy(); m.intercept_ = warm[1].copy(); m.classes_ = np.array([0, 1])
    m.fit(X, y)
    return m


def logloss_vec(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clustered_ci(diff: np.ndarray, clusters: np.ndarray, reps: int = 400, seed: int = 7):
    uc, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uc)); cnt = np.bincount(inv, minlength=len(uc))
    rng = np.random.default_rng(seed); draws = np.empty(reps)
    for r in range(reps):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        draws[r] = (w * sums).sum() / max((w * cnt).sum(), 1)
    return float(diff.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------- bat tracking (public leaderboard, fetched on the runner)
BAT_URLS = (
    'https://baseballsavant.mlb.com/leaderboard/bat-tracking?attackZone=&batSide=&contactType=&count=&dateStart={y}-03-01&dateEnd={y}-11-30&gameType=Regular&isHardHit=&minSwings=100&minGroupSwings=1&pitchHand=&pitchType=&seasonStart=&seasonEnd=&team=&type=batter&csv=true',
    'https://baseballsavant.mlb.com/leaderboard/bat-tracking?type=batter&seasonStart={y}&seasonEnd={y}&minSwings=100&csv=true',
    'https://baseballsavant.mlb.com/leaderboard/bat-tracking?type=batter&season={y}&csv=true',
)


def bat_tracking(seasons=(2024, 2025)) -> tuple[dict, dict]:
    out, notes = {}, {}
    for y in seasons:
        for url in BAT_URLS:
            try:
                with urlopen(Request(url.format(y=y), headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'}), timeout=60) as r:
                    text = r.read().decode('utf-8-sig', 'replace')
                rows = list(csv.DictReader(io.StringIO(text)))
                if not rows:
                    continue
                cols = list(rows[0].keys())
                idc = next((c for c in cols if c.strip('"﻿') in ('id', 'player_id', 'batter')), None)
                bs = next((c for c in cols if 'bat_speed' in c and 'avg' in c), None) or next((c for c in cols if 'bat_speed' in c), None)
                sl = next((c for c in cols if 'swing_length' in c and 'avg' in c), None) or next((c for c in cols if 'swing_length' in c), None)
                sw = next((c for c in cols if c in ('swings_competitive', 'competitive_swings', 'swings')), None)
                n = 0
                for row in rows:
                    try:
                        pid = int(float(row[idc]))
                        out.setdefault(pid, {})[y] = {'bat_speed': float(row[bs]) if bs and row.get(bs) not in (None, '') else None,
                                                      'swing_length': float(row[sl]) if sl and row.get(sl) not in (None, '') else None,
                                                      'swings': float(row[sw]) if sw and row.get(sw) not in (None, '') else None}
                        n += 1
                    except (TypeError, ValueError, KeyError):
                        continue
                notes[y] = {'url': url.format(y=y)[:80], 'rows': n, 'columns': cols[:30]}
                if n:
                    break
            except Exception as exc:
                notes.setdefault(y, {})['error_' + str(BAT_URLS.index(url))] = type(exc).__name__ + ': ' + str(exc)[:120]
    return out, notes


# ---------------------------------------------------------------- the experiment
def horizon(T: dict, params: dict, stage) -> dict:
    res = {}
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    res['pitches'] = {'all': int(len(keep)), 'kept': int(keep.sum())}
    # reconstruction checks (reconstructed, not measured)
    g = T['group']
    res['rebuild'] = {
        'flight_time_50ft_s': {k: round(float(np.nanpercentile(F['tf'][F['ok']], q)), 4) for k, q in (('p05', 5), ('p50', 50), ('p95', 95))},
        'time_last_40ft_s_p50': round(float(np.nanmedian(F['T40'][F['ok']])), 4),
        'spin_accel_by_group': {GROUP_NAMES[i]: {'asx_mean': round(float(np.nanmean(F['asx'][F['ok'] & (g == i)])), 2),
                                                 'asz_mean': round(float(np.nanmean(F['asz'][F['ok'] & (g == i)])), 2),
                                                 'n': int((F['ok'] & (g == i)).sum())} for i in range(7)},
    }
    stage('observer accelerations')
    acc = observer_accels(T, F)
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}; acc = {k: v[keep] for k, v in acc.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.int64)
    prop = swing_propensity(T)
    C = control_block(T, prop)
    rng = np.random.default_rng(int(params.get('seed', 11)))
    train_idx = []
    for s in (2023, 2024, 2025):
        ii = np.flatnonzero(T['season'] == s)
        train_idx.append(rng.choice(ii, min(len(ii), int(params.get('train_per_season', 200000))), replace=False))
    train_idx = np.sort(np.concatenate(train_idx))
    test_all = np.flatnonzero(T['season'] == 2026)
    test_idx = np.sort(rng.choice(test_all, min(len(test_all), int(params.get('test_n', 500000))), replace=False))
    res['split'] = {'train': int(len(train_idx)), 'test': int(len(test_idx)), 'swing_rate_test': round(float(swing[test_idx].mean()), 4)}
    taus = [float(x) for x in params.get('taus', (0.0, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30))]
    observers = list(params.get('observers', ('straight', 'fastball', 'own_type', 'mix')))
    kappa = float(params.get('kappa', 1.0))

    def design(x, z, idx):
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), C[idx]])

    def score(x, z, label, warm=None):
        Xtr = design(x, z, train_idx); m = fit_logistic(Xtr, swing[train_idx], warm=warm)
        Xte = design(x, z, test_idx)
        p = m.predict_proba(Xte)[:, 1]
        ll = logloss_vec(p, swing[test_idx])
        ptr = m.predict_proba(Xtr)[:, 1]
        return {'test_logloss': float(ll.mean()), 'train_logloss': float(logloss_vec(ptr, swing[train_idx]).mean())}, ll, m

    stage('tau 0 (true crossing)')
    base, ll0, m0 = score(T['px'], T['pz'], 'actual')
    res['actual'] = base
    games_test = T['game'][test_idx]
    profiles = {}
    mix_cache = {}
    for obs in observers:
        prof = []
        for tau in taus:
            if tau == 0.0:
                prof.append({'tau': 0.0, **base, 'delta_vs_actual': [0.0, 0.0, 0.0]}); continue
            if obs == 'mix':
                x, z, ent = mixture_projection(T, F, tau, kappa)
                mix_cache[tau] = ent
                bad = ~np.isfinite(x); x = np.where(bad, T['px'], x); z = np.where(bad, T['pz'], z)
            else:
                x, z = projected(T, F, acc, obs, tau)
            out, ll, _ = score(x, z, f'{obs}-{tau}')
            d = clustered_ci(ll - ll0, games_test)
            prof.append({'tau': tau, **out, 'delta_vs_actual': [round(v, 6) for v in d]})
            stage(f'{obs} tau {tau}: {out["test_logloss"] - base["test_logloss"]:+.5f}')
        profiles[obs] = prof
    res['profiles'] = profiles
    # best observer and tau (lowest test log loss)
    best = min(((o, p['tau'], p['test_logloss']) for o, pr in profiles.items() for p in pr), key=lambda r: r[2])
    res['best'] = {'observer': best[0], 'tau': best[1], 'test_logloss': best[2], 'gain_vs_actual': best[2] - base['test_logloss']}
    # The checks and the per-hitter horizons use the best observer with a closed-form projection (not the mixture).
    best = min(((o, p['tau'], p['test_logloss']) for o, pr in profiles.items() if o != 'mix' for p in pr), key=lambda r: r[2])
    res['best_closed_form'] = {'observer': best[0], 'tau': best[1], 'test_logloss': best[2], 'gain_vs_actual': best[2] - base['test_logloss']}
    if best[1] > 0:
        stage('checks at the best observer and tau')
        obs, tau = best[0], best[1]
        # 1. shuffle the spin accelerations within pitcher-season-type
        key = (T['pitcher'] * 10 + (T['season'] - 2020)) * 10 + T['group']
        order = np.argsort(key, kind='stable'); bounds = np.flatnonzero(np.diff(key[order])) + 1
        perm = np.arange(len(key))
        for grp in np.split(order, bounds):
            perm[grp] = rng.permutation(grp)
        x, z = projected(T, F, acc, obs, tau, shuffle=perm)
        out, ll, _ = score(x, z, 'shuffle')
        res['check_shuffle'] = {**out, 'delta_vs_actual': [round(v, 6) for v in clustered_ci(ll - ll0, games_test)]}
        # 2. both: actual and projected crossing together
        x, z = projected(T, F, acc, obs, tau)
        def design_both(idx):
            return np.hstack([location_block(T['px'][idx], T['pz'][idx], T['stand_r'][idx], T['strikes'][idx]),
                              location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), C[idx]])
        mb = fit_logistic(design_both(train_idx), swing[train_idx])
        llb = logloss_vec(mb.predict_proba(design_both(test_idx))[:, 1], swing[test_idx])
        res['check_both'] = {'test_logloss': float(llb.mean()), 'delta_vs_actual': [round(v, 6) for v in clustered_ci(llb - ll0, games_test)]}
        # 3. split by true location: chases (outside) and takes of strikes (inside)
        u = np.where(T['stand_r'] == 1, T['px'], -T['px'])
        inside = (np.abs(u) <= ZONE_HALF) & (T['pz'] <= ZONE_TOP) & (T['pz'] >= ZONE_BOT)
        split = {}
        for name, msk in (('outside', ~inside), ('inside', inside)):
            tr = train_idx[msk[train_idx]]; te = test_idx[msk[test_idx]]
            rows = []
            for tau2 in taus:
                xx, zz = (T['px'], T['pz']) if tau2 == 0 else projected(T, F, acc, obs, tau2)
                m = fit_logistic(np.hstack([location_block(xx[tr], zz[tr], T['stand_r'][tr], T['strikes'][tr]), C[tr]]), swing[tr])
                p = m.predict_proba(np.hstack([location_block(xx[te], zz[te], T['stand_r'][te], T['strikes'][te]), C[te]]))[:, 1]
                rows.append({'tau': tau2, 'test_logloss': float(logloss_vec(p, swing[te]).mean())})
            split[name] = {'test_pitches': int(len(te)), 'profile': rows}
        res['check_split'] = split
        # 4. per-hitter horizon with the global coefficients fixed
        stage('per-hitter horizons')
        res['hitters'] = hitter_horizons(T, F, acc, obs, C, swing, params)
        stage('bat tracking')
        bt, notes = bat_tracking()
        res['bat_tracking_fetch'] = notes
        res['hitters']['validation'] = validate_hitters(res['hitters'], bt)
    return res


def hitter_horizons(T, F, acc, obs, C, swing, params) -> dict:
    """Each hitter's own profile over a fine tau grid, with the swing model fixed at its global fit on 2023-2025.
    The model is refitted at each grid point on everyone (so the location map matches that tau); a hitter's horizon
    is where his own pitches are best explained, from a quadratic through the best point and its neighbors."""
    grid = [round(x, 3) for x in np.arange(0.0, 0.3001, 0.025)]
    dev = np.flatnonzero(T['season'] <= 2025)
    rng = np.random.default_rng(5)
    fit_idx = np.sort(rng.choice(dev, min(len(dev), int(params.get('hitter_fit_n', 600000))), replace=False))
    batters = T['batter'][dev]
    ub, inv = np.unique(batters, return_inverse=True)
    counts = np.bincount(inv)
    min_p = int(params.get('hitter_min_pitches', 2500))
    keep_b = counts >= min_p
    odd = (T['day'][dev] % 2).astype(int)
    LL = np.zeros((len(grid), len(ub))); LLh = np.zeros((2, len(grid), len(ub)))
    for gi, tau in enumerate(grid):
        x, z = (T['px'], T['pz']) if tau == 0 else projected(T, F, acc, obs, tau)
        X = np.hstack([location_block(x[fit_idx], z[fit_idx], T['stand_r'][fit_idx], T['strikes'][fit_idx]), C[fit_idx]])
        m = fit_logistic(X, swing[fit_idx])
        p = np.empty(len(dev))
        for s in range(0, len(dev), 400000):
            sl = dev[s:s + 400000]
            p[s:s + 400000] = m.predict_proba(np.hstack([location_block(x[sl], z[sl], T['stand_r'][sl], T['strikes'][sl]), C[sl]]))[:, 1]
        ll = -logloss_vec(p, swing[dev])
        LL[gi] = np.bincount(inv, weights=ll, minlength=len(ub))
        for h in (0, 1):
            LLh[h, gi] = np.bincount(inv, weights=ll * (odd == h), minlength=len(ub))

    def peak(curve):
        i = int(np.argmax(curve))
        if 0 < i < len(grid) - 1:
            y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            return float(grid[i] + np.clip(off, -1, 1) * 0.025)
        return float(grid[i])

    rows = []
    for j in np.flatnonzero(keep_b):
        rows.append({'batter': int(ub[j]), 'pitches': int(counts[j]), 'tau': round(peak(LL[:, j]), 4),
                     'tau_odd': round(peak(LLh[1, :, j]), 4), 'tau_even': round(peak(LLh[0, :, j]), 4),
                     'gain_at_best_vs_0_per_1000': round(float((LL[:, j].max() - LL[0, j]) / counts[j] * 1000), 4)})
    taus = np.array([r['tau'] for r in rows]); a = np.array([r['tau_odd'] for r in rows]); b = np.array([r['tau_even'] for r in rows])
    rel = float(np.corrcoef(a, b)[0, 1]) if len(rows) > 3 else None
    total = LL[:, keep_b].sum(axis=1)
    return {'grid': grid, 'observer': obs, 'min_pitches': min_p, 'n_hitters': len(rows),
            'pooled_profile_per_1000': [round(float((v - total[0]) / counts[keep_b].sum() * 1000), 4) for v in total],
            'tau_quantiles': {q: round(float(np.percentile(taus, q)), 4) for q in (10, 25, 50, 75, 90)} if len(rows) else None,
            'split_half_correlation': None if rel is None else round(rel, 4),
            'rows': rows}


def validate_hitters(h: dict, bt: dict) -> dict:
    rows = h.get('rows') or []
    pairs = []
    for r in rows:
        seasons = bt.get(r['batter']) or {}
        speeds = [v['bat_speed'] for v in seasons.values() if v.get('bat_speed')]
        lengths = [v['swing_length'] for v in seasons.values() if v.get('swing_length')]
        if speeds:
            pairs.append((r['tau'], float(np.mean(speeds)), float(np.mean(lengths)) if lengths else np.nan, r['pitches']))
    if len(pairs) < 20:
        return {'n': len(pairs), 'note': 'too few hitters with bat tracking'}
    a = np.array(pairs)
    from scipy import stats
    out = {'n': len(pairs)}
    for name, col in (('bat_speed', 1), ('swing_length', 2)):
        ok = np.isfinite(a[:, col])
        r, p = stats.spearmanr(a[ok, 0], a[ok, col])
        out[name] = {'spearman': round(float(r), 4), 'p': float(p), 'n': int(ok.sum())}
    return out


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']
    from cloud.security import unseal, key_bytes
    from brl_live.bookkeeping_season import STUDY_SCHEMA, study_path, study_purpose
    key = key_bytes(os.environ['BRL_PA_PACKAGE_KEY'])
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    params = json.loads((ROOT / 'tools' / 'discovery_params.json').read_text())
    experiment = str(params.get('experiment') or 'horizon')
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    receipt = {'schema': 'brl.discovery-receipt.v1', 'experiment': experiment, 'run_id': run_id, 'params': params,
               'started_at': datetime.now(timezone.utc).isoformat(), 'stages': []}
    t0 = time.time()

    def stage(name):
        receipt['stages'].append({'stage': name, 'at_seconds': round(time.time() - t0, 1)}); print(name, round(time.time() - t0), 's', flush=True)
    try:
        tables = []
        for year in params.get('seasons', (2023, 2024, 2025, 2026)):
            stage(f'load {year}')
            raw = read_blob(repo, token, study_path(int(year)), branch)
            if raw is None:
                continue
            doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(int(year)))))
            if doc.get('schema') != STUDY_SCHEMA:
                raise ValueError('study schema mismatch')
            tables.append(pitch_table(doc, int(year))); del doc, raw
        T = concat(tables); del tables
        receipt['seasons_rows'] = {int(s): int((T['season'] == s).sum()) for s in np.unique(T['season'])}
        if experiment == 'horizon':
            receipt['results'] = horizon(T, params, stage)
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:400]
        receipt['where'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in traceback.extract_tb(exc.__traceback__)[-6:]]
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat(); receipt['seconds'] = round(time.time() - t0, 1)
    text = json.dumps(receipt, indent=1, default=float)
    put_text(repo, token, f'research/discovery-{experiment}-{run_id}.json', text, branch, 'BRL: discovery ' + experiment)
    print(json.dumps({k: receipt.get(k) for k in ('status', 'error', 'where', 'seconds')}, default=str))


if __name__ == '__main__':
    main()
