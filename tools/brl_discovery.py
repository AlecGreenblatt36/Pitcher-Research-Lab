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
SUBTYPES = ('FF', 'FA', 'SI', 'FC', 'SL', 'ST', 'SV', 'CU', 'KC', 'CS', 'CH', 'FS', 'FO', 'SC', 'KN', 'EP')
FIELDS = ('season', 'day', 'game', 'pitcher', 'batter', 'stand_r', 'throw_r', 'inning', 'group', 'sub', 'balls', 'strikes', 'call',
          'v0', 'v1', 'spin', 'pfx_x', 'pfx_z', 'px', 'pz', 'x0', 'z0', 'ext', 'last_in_pa', 'bunt_pa', 'ab', 'pitch_no', 'la', 'ls')
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
                st = str(typ or '').upper(); cols['sub'].append(SUBTYPES.index(st) if st in SUBTYPES else len(SUBTYPES))
                cols['balls'].append(int(balls if balls is not None else -1)); cols['strikes'].append(int(strikes if strikes is not None else -1))
                cols['call'].append(call)
                for k, v in (('v0', v0), ('v1', v1), ('spin', spin), ('pfx_x', pfx_x), ('pfx_z', pfx_z), ('px', px), ('pz', pz), ('x0', x0), ('z0', z0), ('ext', ext)):
                    cols[k].append(np.nan if v is None else float(v))
                cols['last_in_pa'].append(1 if j == len(pitches) - 1 else 0); cols['bunt_pa'].append(bunt)
                cols['ab'].append(int(row.get('i') or 0)); cols['pitch_no'].append(j)
                hit = row.get('hit') if j == len(pitches) - 1 and code in ('X', 'D', 'E') else None
                cols['la'].append(np.nan if not hit or hit[1] is None else float(hit[1])); cols['ls'].append(np.nan if not hit or hit[0] is None else float(hit[0]))
    out = {}
    for k, v in cols.items():
        out[k] = np.asarray(v, dtype=np.float32 if k in ('v0', 'v1', 'spin', 'pfx_x', 'pfx_z', 'px', 'pz', 'x0', 'z0', 'ext', 'la', 'ls') else np.int64)
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


def loo_means(keys: np.ndarray, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Leave-one-out mean of values within each key (rows outside mask get the full mean of their key or nan)."""
    k = keys; uk, inv = np.unique(k, return_inverse=True)
    s = np.bincount(inv, weights=np.where(mask, values, 0.0), minlength=len(uk)); c = np.bincount(inv, weights=mask.astype(float), minlength=len(uk))
    own = np.where(mask, values, 0.0); n = c[inv] - mask
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, (s[inv] - own) / n, np.nan)


def offset_logit(y, off, g, w=None, iters=30):
    """Logistic fit of y on an offset plus intercept a and slope b on g. Returns (a, b, se_b)."""
    w = np.ones(len(y)) if w is None else w
    a = b = 0.0
    for _ in range(iters):
        eta = off + a + b * g; p = 1 / (1 + np.exp(-eta)); r = w * (y - p); q = w * p * (1 - p)
        H = np.array([[q.sum(), (q * g).sum()], [(q * g).sum(), (q * g * g).sum()]]); grad = np.array([r.sum(), (r * g).sum()])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), grad); a += step[0]; b += step[1]
        if np.abs(step).max() < 1e-10:
            break
    cov = np.linalg.inv(H + 1e-9 * np.eye(2))
    return float(a), float(b), float(np.sqrt(max(cov[1, 1], 0.0)))


def boot_slope(y, off, g, games, reps=200, seed=3):
    uc, inv = np.unique(games, return_inverse=True); rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        cnt = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc)).astype(float)
        out.append(offset_logit(y, off, g, w=cnt[inv], iters=12)[1])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def horizon2(T: dict, params: dict, stage) -> dict:
    """The reviewer's tests (discovery/DECISION_HORIZON_PROTOCOL.md, addendum): selection on 2025 and confirmation
    on 2026; pitch-type-specific location maps as the baseline; the profile over the within-type surprise only (how
    much this pitch broke more or less than the pitcher's usual for that pitch type), which a type-by-location
    interaction cannot produce; the primary-fastball subset; and a direct estimate of tau squared from the
    displacement along the swing-decision gradient, for the within-type surprise and for the type-level shift."""
    res = {}
    F = rebuild(T)
    keep = (F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
            & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1)))
    g = T['group']
    res['rebuild'] = {'flight_time_50ft_s_p50': round(float(np.nanmedian(F['tf'][F['ok']])), 4),
                      'spin_accel_by_group': {GROUP_NAMES[i]: {'asx': round(float(np.nanmean(F['asx'][F['ok'] & (g == i)])), 2),
                                                               'asz': round(float(np.nanmean(F['asz'][F['ok'] & (g == i)])), 2)} for i in range(7)},
                      'count_00_share': round(float(np.mean((T['balls'] == 0) & (T['strikes'] == 0))), 4)}
    acc = observer_accels(T, F)
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}; acc = {k: v[keep] for k, v in acc.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.int64)
    C = control_block(T, swing_propensity(T))
    key_sub = (T['pitcher'] * 10 + (T['season'] - 2020)) * 100 + T['sub']
    ok = np.isfinite(F['ax']) & np.isfinite(F['az'])
    mx = loo_means(key_sub, F['ax'], ok); mz = loo_means(key_sub, F['az'], ok)
    wx = np.where(np.isfinite(mx), F['ax'] - mx, 0.0); wz = np.where(np.isfinite(mz), F['az'] - mz, 0.0)
    bx = np.where(np.isfinite(mx), mx, F['ax']) - acc['fb_ax']; bz = np.where(np.isfinite(mz), mz, F['az']) - acc['fb_az']
    res['surprise_sd_ft_s2'] = {'within_x': round(float(np.std(wx)), 3), 'within_z': round(float(np.std(wz)), 3),
                                'between_x': round(float(np.std(bx)), 3), 'between_z': round(float(np.std(bz)), 3)}
    # primary fastball subtype of each pitcher-season
    ps = T['pitcher'] * 10 + (T['season'] - 2020)
    fbmask = np.isin(T['sub'], (0, 1, 2, 3))
    kk = ps * 100 + T['sub']; uk, cnt = np.unique(kk[fbmask], return_counts=True)
    best = {}
    for k, c in zip(uk, cnt):
        if k // 100 not in best or c > best[k // 100][1]:
            best[k // 100] = (k, c)
    prim = np.isin(kk, np.array([v[0] for v in best.values()]))
    rng = np.random.default_rng(int(params.get('seed', 11)))
    def sample(seasons, n):
        ii = np.flatnonzero(np.isin(T['season'], seasons)); return np.sort(rng.choice(ii, min(len(ii), n), replace=False))
    tr = sample((2023, 2024), int(params.get('train_n', 500000)))
    va = sample((2025,), int(params.get('val_n', 500000)))
    te = sample((2026,), int(params.get('test_n', 700000)))
    res['split'] = {'train': int(len(tr)), 'select_2025': int(len(va)), 'test_2026': int(len(te))}
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]

    def shared(x, z, idx):
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), C[idx]])

    def typed(x, z, idx):
        u = np.where(T['stand_r'][idx] == 1, x[idx], -x[idx]); zz = z[idx]
        e = np.maximum(np.maximum(np.abs(u) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
        base = np.hstack([hats(e, E_KNOTS), hats(u, U_KNOTS), hats(zz, Z_KNOTS)])
        inter = (base[:, :, None] * oh[idx, None, :]).reshape(len(idx), -1)
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), inter, C[idx]])

    def fit_score(designf, x, z, rows=None):
        trr = tr if rows is None else tr[rows[tr]]; var = va if rows is None else va[rows[va]]; ter = te if rows is None else te[rows[te]]
        m = fit_logistic(designf(x, z, trr), swing[trr])
        out = {}
        for name, idx in (('select_2025', var), ('test_2026', ter)):
            p = m.predict_proba(designf(x, z, idx))[:, 1]; out[name] = logloss_vec(p, swing[idx])
        return m, out

    taus = [float(v) for v in params.get('taus', (0.0, 0.10, 0.15, 0.175, 0.20, 0.25))]
    summary = {}
    # P1: shared maps, type-level observers (the original test, now selected on 2025)
    for obs in ('fastball', 'own_type'):
        rows = []
        for tau in taus:
            x, z = (T['px'], T['pz']) if tau == 0 else projected(T, F, acc, obs, tau)
            _, ll = fit_score(shared, x, z)
            rows.append({'tau': tau, 'select_2025': float(ll['select_2025'].mean()), 'test_2026': float(ll['test_2026'].mean())})
        summary['shared_' + obs] = rows; stage('shared ' + obs)
    # baseline with pitch-type location maps at tau 0
    m_typed, ll_t0 = fit_score(typed, T['px'], T['pz'])
    summary['typed_actual'] = {'select_2025': float(ll_t0['select_2025'].mean()), 'test_2026': float(ll_t0['test_2026'].mean())}
    stage('typed baseline')
    # P2: type maps, within-type surprise only
    rows = []
    for tau in taus:
        x, z = T['px'] - 0.5 * wx * tau ** 2, T['pz'] - 0.5 * wz * tau ** 2
        _, ll = fit_score(typed, x, z)
        r = {'tau': tau, 'select_2025': float(ll['select_2025'].mean()), 'test_2026': float(ll['test_2026'].mean())}
        if tau > 0:
            r['delta_2026'] = [round(v, 7) for v in clustered_ci(ll['test_2026'] - ll_t0['test_2026'], T['game'][te])]
        rows.append(r)
    summary['typed_within'] = rows; stage('typed within-type surprise')
    # P3: primary fastballs only, shared maps, within-type surprise (fastball observer = own-type observer here)
    rows = []
    _, ll_f0 = fit_score(shared, T['px'], T['pz'], rows=prim)
    for tau in taus:
        x, z = T['px'] - 0.5 * wx * tau ** 2, T['pz'] - 0.5 * wz * tau ** 2
        _, ll = (None, ll_f0) if tau == 0 else fit_score(shared, x, z, rows=prim)
        r = {'tau': tau, 'select_2025': float(ll['select_2025'].mean()), 'test_2026': float(ll['test_2026'].mean())}
        if tau > 0:
            r['delta_2026'] = [round(v, 7) for v in clustered_ci(ll['test_2026'] - ll_f0['test_2026'], T['game'][te[prim[te]]])]
        rows.append(r)
    summary['fastball_within'] = rows; stage('primary fastballs')
    res['profiles'] = summary
    # L: direct estimate of tau^2 from displacement along the decision gradient (typed model fixed at tau 0)
    h = 0.02
    def grad_terms(idx, dx, dz):
        f0 = np.empty(len(idx)); gt = np.empty(len(idx))
        for s in range(0, len(idx), 250000):
            ii = idx[s:s + 250000]
            df = lambda x, z: m_typed.decision_function(typed(x, z, ii))
            f0[s:s + 250000] = df(T['px'], T['pz'])
            fx = (df(T['px'] + h, T['pz']) - df(T['px'] - h, T['pz'])) / (2 * h)
            fz = (df(T['px'], T['pz'] + h) - df(T['px'], T['pz'] - h)) / (2 * h)
            gt[s:s + 250000] = -0.5 * (fx * dx[ii] + fz * dz[ii])
        return f0, gt
    lin = {}
    for name, idx in (('select_2025', va), ('test_2026', te)):
        for part, dx, dz in (('within', wx, wz), ('between', bx, bz)):
            for subset, rows in (('all', None), ('primary_fastball', prim)):
                if part == 'between' and subset == 'primary_fastball':
                    continue
                ii = idx if rows is None else idx[rows[idx]]
                off, gterm = grad_terms(ii, dx, dz)
                a, b, se = offset_logit(swing[ii].astype(float), off, gterm)
                lo, hi = boot_slope(swing[ii].astype(float), off, gterm, T['game'][ii], reps=int(params.get('boot', 150)))
                lin[f'{name}/{part}/{subset}'] = {'tau_squared': round(b, 6), 'se': round(se, 6), 'ci': [round(lo, 6), round(hi, 6)],
                                                  'tau_s': round(float(np.sign(b) * np.sqrt(abs(b))), 4), 'pitches': int(len(ii))}
    res['gradient_estimate'] = lin; stage('gradient estimates')
    # per-hitter: within-type tau^2 on 2023-2025 with the typed model, shrunk, split by odd and even days
    dev = np.flatnonzero(T['season'] <= 2025)
    off, gterm = grad_terms(dev, wx, wz)
    yb = swing[dev].astype(float); bat = T['batter'][dev]; odd = (T['day'][dev] % 2).astype(int)
    ub, inv = np.unique(bat, return_inverse=True); cnt = np.bincount(inv)
    sel = np.flatnonzero(cnt >= int(params.get('hitter_min_pitches', 3000)))
    order = np.argsort(inv, kind='stable'); bounds = np.flatnonzero(np.diff(inv[order])) + 1; groups = np.split(order, bounds)
    est = []
    for j in sel:
        ii = groups[j]
        _, b_all, se_all = offset_logit(yb[ii], off[ii], gterm[ii])
        b_o = offset_logit(yb[ii][odd[ii] == 1], off[ii][odd[ii] == 1], gterm[ii][odd[ii] == 1])[1]
        b_e = offset_logit(yb[ii][odd[ii] == 0], off[ii][odd[ii] == 0], gterm[ii][odd[ii] == 0])[1]
        est.append((int(ub[j]), int(cnt[j]), b_all, se_all, b_o, b_e))
    E = np.array(est, float) if est else np.zeros((0, 6))
    hit = {'n_hitters': int(len(E)), 'min_pitches': int(params.get('hitter_min_pitches', 3000))}
    if len(E) > 10:
        mean = float(np.average(E[:, 2], weights=1 / E[:, 3] ** 2)); var_obs = float(np.var(E[:, 2])); var_noise = float(np.mean(E[:, 3] ** 2))
        var_true = max(var_obs - var_noise, 0.0)
        shrink = var_true / (var_true + E[:, 3] ** 2)
        post = mean + shrink * (E[:, 2] - mean)
        hit.update(pooled_tau_squared=round(mean, 6), sd_observed=round(float(np.sqrt(var_obs)), 6), sd_noise=round(float(np.sqrt(var_noise)), 6),
                   sd_true=round(float(np.sqrt(var_true)), 6), split_half_correlation=round(float(np.corrcoef(E[:, 4], E[:, 5])[0, 1]), 4),
                   rows=[{'batter': int(r[0]), 'pitches': int(r[1]), 'tau_squared': round(r[2], 6), 'se': round(r[3], 6), 'shrunk': round(float(p), 6)}
                         for r, p in zip(E, post)])
    res['hitters'] = hit; stage('per-hitter')
    bt, notes = bat_tracking()
    res['bat_tracking_fetch'] = notes
    if hit.get('rows') and bt:
        pairs = []
        for r in hit['rows']:
            sp = [v['bat_speed'] for v in (bt.get(r['batter']) or {}).values() if v.get('bat_speed')]
            ln = [v['swing_length'] for v in (bt.get(r['batter']) or {}).values() if v.get('swing_length')]
            if sp:
                pairs.append((r['shrunk'], np.mean(sp), np.mean(ln) if ln else np.nan))
        if len(pairs) > 20:
            from scipy import stats
            A = np.array(pairs)
            res['hitters']['validation'] = {'n': len(pairs), 'bat_speed_spearman': round(float(stats.spearmanr(A[:, 0], A[:, 1])[0]), 4),
                                            'swing_length_spearman': round(float(stats.spearmanr(A[np.isfinite(A[:, 2]), 0], A[np.isfinite(A[:, 2]), 2])[0]), 4)}
    return res


def horizon3(T: dict, params: dict, stage) -> dict:
    """Is a hitter's horizon new information or a re-expression of his chase rate? Horizons are measured on
    2023-2024 only (shared location map with the straight observer, as in the first run, and with pitch-type maps
    so the league's type-level differences are absorbed), then tested against what the same hitters did in
    2025-2026: strikeout rate, walk rate, chase rate and whiffs on breaking balls, controlling for the plate-
    discipline statistics a scout already has from 2023-2024."""
    res = {}
    F = rebuild(T)
    keep = (F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
            & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1)))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.int64); whiff = (T['call'] == 2)
    C = control_block(T, swing_propensity(T))
    u = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    inzone = (np.abs(u) <= ZONE_HALF) & (T['pz'] <= ZONE_TOP) & (T['pz'] >= ZONE_BOT)
    brk = np.isin(T['group'], (3, 4))
    dev = np.flatnonzero(T['season'] <= 2024); fut = np.flatnonzero(T['season'] >= 2025)
    rng = np.random.default_rng(int(params.get('seed', 11)))
    fit_idx = np.sort(rng.choice(dev, min(len(dev), int(params.get('hitter_fit_n', 500000))), replace=False))
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]

    def shared(x, z, idx):
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), C[idx]])

    def typed(x, z, idx):
        uu = np.where(T['stand_r'][idx] == 1, x[idx], -x[idx]); zz = z[idx]
        e = np.maximum(np.maximum(np.abs(uu) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
        b = np.hstack([hats(e, E_KNOTS), hats(uu, U_KNOTS), hats(zz, Z_KNOTS)])
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), (b[:, :, None] * oh[idx, None, :]).reshape(len(idx), -1), C[idx]])

    bat = T['batter']; ub, inv = np.unique(bat, return_inverse=True)
    n_dev = np.bincount(inv[dev], minlength=len(ub)); n_fut = np.bincount(inv[fut], minlength=len(ub))
    sel = (n_dev >= int(params.get('min_dev', 2000))) & (n_fut >= int(params.get('min_fut', 1500)))
    odd = (T['day'] % 2).astype(int)
    out_h = {}
    for name, designf, grid in (('shared', shared, [round(x, 3) for x in np.arange(0.0, 0.3001, 0.025)]),
                                ('typed', typed, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])):
        LL = np.zeros((len(grid), len(ub))); LLh = np.zeros((2, len(grid), len(ub)))
        for gi, tau in enumerate(grid):
            x = T['px'] - 0.5 * F['asx'] * tau ** 2; z = T['pz'] - 0.5 * F['asz'] * tau ** 2      # straight observer
            m = fit_logistic(designf(x, z, fit_idx), swing[fit_idx])
            ll = np.zeros(len(dev))
            for s in range(0, len(dev), 250000):
                ii = dev[s:s + 250000]
                ll[s:s + 250000] = -logloss_vec(m.predict_proba(designf(x, z, ii))[:, 1], swing[ii])
            LL[gi] = np.bincount(inv[dev], weights=ll, minlength=len(ub))
            for h in (0, 1):
                LLh[h, gi] = np.bincount(inv[dev], weights=ll * (odd[dev] == h), minlength=len(ub))
            stage(f'{name} tau {tau}')

        def peak(curve):
            i = int(np.argmax(curve)); step = grid[1] - grid[0]
            if 0 < i < len(grid) - 1:
                y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]; den = y0 - 2 * y1 + y2
                return float(grid[i] + (np.clip(0.5 * (y0 - y2) / den, -1, 1) * step if den < 0 else 0.0))
            return float(grid[i])
        out_h[name] = {'grid': grid, 'tau': np.array([peak(LL[:, j]) for j in range(len(ub))]),
                       'odd': np.array([peak(LLh[1, :, j]) for j in range(len(ub))]), 'even': np.array([peak(LLh[0, :, j]) for j in range(len(ub))]),
                       'pooled': [round(float(v), 2) for v in (LL[:, sel].sum(axis=1) - LL[0, sel].sum()) / n_dev[sel].sum() * 1000]}

    def rate(num_mask, den_mask, idx):
        num = np.bincount(inv[idx], weights=num_mask[idx].astype(float), minlength=len(ub)); den = np.bincount(inv[idx], weights=den_mask[idx].astype(float), minlength=len(ub))
        with np.errstate(invalid='ignore', divide='ignore'):
            return num / den
    pa_end = T['last_in_pa'] == 1
    k_pa = pa_end & (T['call'] == 2) & (T['strikes'] == 2)
    bb_pa = pa_end & (T['call'] == 0) & (T['balls'] == 3)
    stats_dev = {'o_swing': rate(swing.astype(bool) & ~inzone, ~inzone, dev), 'z_swing': rate(swing.astype(bool) & inzone, inzone, dev),
                 'whiff_per_swing': rate(whiff, swing.astype(bool), dev), 'k_rate': rate(k_pa, pa_end, dev), 'bb_rate': rate(bb_pa, pa_end, dev),
                 'brk_chase': rate(swing.astype(bool) & ~inzone & brk, ~inzone & brk, dev)}
    stats_fut = {'k_rate': rate(k_pa, pa_end, fut), 'bb_rate': rate(bb_pa, pa_end, fut), 'o_swing': rate(swing.astype(bool) & ~inzone, ~inzone, fut),
                 'brk_chase': rate(swing.astype(bool) & ~inzone & brk, ~inzone & brk, fut), 'brk_whiff': rate(whiff & brk, swing.astype(bool) & brk, fut),
                 'whiff_per_swing': rate(whiff, swing.astype(bool), fut)}
    idx = np.flatnonzero(sel)
    res['n_hitters'] = int(len(idx))
    rows = []
    for j in idx:
        rows.append({'batter': int(ub[j]), 'n_dev': int(n_dev[j]), 'n_fut': int(n_fut[j]),
                     **{f'tau_{k}': round(float(out_h[k]['tau'][j]), 4) for k in out_h},
                     **{f'dev_{k}': round(float(v[j]), 4) for k, v in stats_dev.items()}, **{f'fut_{k}': round(float(v[j]), 4) for k, v in stats_fut.items()}})
    res['rows'] = rows
    res['pooled_profiles'] = {k: {'grid': v['grid'], 'per_1000': v['pooled']} for k, v in out_h.items()}
    from scipy import stats
    tests = {}
    for hk in out_h:
        t = out_h[hk]['tau'][idx]
        tests[hk] = {'split_half': round(float(np.corrcoef(out_h[hk]['odd'][idx], out_h[hk]['even'][idx])[0, 1]), 4),
                     'corr_with_dev': {k: round(float(stats.spearmanr(t, v[idx], nan_policy='omit')[0]), 4) for k, v in stats_dev.items()}}
        # incremental validity: future outcome on the 2023-2024 discipline statistics, with and without the horizon
        X0 = np.column_stack([stats_dev[k][idx] for k in ('o_swing', 'z_swing', 'whiff_per_swing', 'k_rate', 'bb_rate', 'brk_chase')])
        inc = {}
        for fk in ('k_rate', 'bb_rate', 'o_swing', 'brk_chase', 'brk_whiff', 'whiff_per_swing'):
            y = stats_fut[fk][idx]; okr = np.isfinite(y) & np.isfinite(X0).all(axis=1) & np.isfinite(t)
            Xa = np.column_stack([np.ones(okr.sum()), X0[okr]]); Xb = np.column_stack([Xa, t[okr]])
            def r2(X):
                beta, *_ = np.linalg.lstsq(X, y[okr], rcond=None); e = y[okr] - X @ beta; return 1 - e.var() / y[okr].var(), beta
            ra, _ = r2(Xa); rb, beta = r2(Xb)
            # leave-one-out prediction error with and without the horizon
            def loo(X):
                H = X @ np.linalg.pinv(X.T @ X) @ X.T; beta, *_ = np.linalg.lstsq(X, y[okr], rcond=None); e = y[okr] - X @ beta
                return float(np.mean((e / (1 - np.diag(H))) ** 2))
            boots = []
            rr = np.random.default_rng(1)
            for _ in range(500):
                b = rr.integers(0, okr.sum(), okr.sum())
                bb, *_ = np.linalg.lstsq(Xb[b], y[okr][b], rcond=None); boots.append(bb[-1])
            sd_t = float(np.std(t[okr]))
            inc[fk] = {'r2_without': round(float(ra), 4), 'r2_with': round(float(rb), 4), 'loo_mse_without': loo(Xa), 'loo_mse_with': loo(Xb),
                       'coef_per_sd_horizon': round(float(beta[-1] * sd_t), 5), 'ci_per_sd': [round(float(np.percentile(boots, 2.5) * sd_t), 5), round(float(np.percentile(boots, 97.5) * sd_t), 5)],
                       'outcome_sd': round(float(np.std(y[okr])), 5), 'n': int(okr.sum())}
        tests[hk]['incremental'] = inc
    res['tests'] = tests
    return res


def offset_logit_k(y, off, G, w=None, iters=30):
    """Logistic fit of y on an offset plus an intercept and k slopes (columns of G). Returns (coef, se) for the slopes."""
    w = np.ones(len(y)) if w is None else w
    X = np.column_stack([np.ones(len(y)), G]); beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + X @ beta))); r = w * (y - p); q = w * p * (1 - p)
        H = X.T @ (X * q[:, None]) + 1e-9 * np.eye(X.shape[1]); step = np.linalg.solve(H, X.T @ r); beta += step
        if np.abs(step).max() < 1e-10:
            break
    cov = np.linalg.inv(H)
    return beta[1:], np.sqrt(np.clip(np.diag(cov)[1:], 0, None))


def horizon4(T: dict, params: dict, stage) -> dict:
    """Mechanism checks on the within-type surprise (protocol addendum 2): the horizontal and vertical surprises
    should give the same commit time if hitters extrapolate the flight; commit time by pitch speed separates a
    fixed time before arrival from a fixed distance; by pitch type, count, batter hand and season; and the
    within-type profile carried past 0.25 s to find where it turns."""
    res = {}
    F = rebuild(T)
    keep = (F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
            & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1)))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.int64)
    C = control_block(T, swing_propensity(T))
    key_sub = (T['pitcher'] * 10 + (T['season'] - 2020)) * 100 + T['sub']
    ok = np.isfinite(F['ax']) & np.isfinite(F['az'])
    mx = loo_means(key_sub, F['ax'], ok); mz = loo_means(key_sub, F['az'], ok); mv = loo_means(key_sub, T['v0'].astype(float), np.isfinite(T['v0']))
    wx = np.where(np.isfinite(mx), F['ax'] - mx, 0.0); wz = np.where(np.isfinite(mz), F['az'] - mz, 0.0)
    wv = np.where(np.isfinite(mv), T['v0'] - mv, 0.0)
    rng = np.random.default_rng(int(params.get('seed', 11)))
    def sample(seasons, n):
        ii = np.flatnonzero(np.isin(T['season'], seasons)); return np.sort(rng.choice(ii, min(len(ii), n), replace=False))
    tr = sample((2023, 2024), int(params.get('train_n', 500000)))
    ev = np.flatnonzero(T['season'] >= 2025)
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]

    def typed(x, z, idx):
        uu = np.where(T['stand_r'][idx] == 1, x[idx], -x[idx]); zz = z[idx]
        e = np.maximum(np.maximum(np.abs(uu) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
        b = np.hstack([hats(e, E_KNOTS), hats(uu, U_KNOTS), hats(zz, Z_KNOTS)])
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), (b[:, :, None] * oh[idx, None, :]).reshape(len(idx), -1), C[idx]])

    m = fit_logistic(typed(T['px'], T['pz'], tr), swing[tr]); stage('typed model')
    h = 0.02
    f0 = np.empty(len(ev)); fx = np.empty(len(ev)); fz = np.empty(len(ev))
    for s in range(0, len(ev), 250000):
        ii = ev[s:s + 250000]; df = lambda x, z: m.decision_function(typed(x, z, ii))
        f0[s:s + 250000] = df(T['px'], T['pz'])
        fx[s:s + 250000] = (df(T['px'] + h, T['pz']) - df(T['px'] - h, T['pz'])) / (2 * h)
        fz[s:s + 250000] = (df(T['px'], T['pz'] + h) - df(T['px'], T['pz'] - h)) / (2 * h)
    stage('gradients')
    y = swing[ev].astype(float); gx = -0.5 * fx * wx[ev]; gz = -0.5 * fz * wz[ev]; gb = gx + gz
    games = T['game'][ev]

    def est(mask, two=False):
        if mask.sum() < 20000:
            return None
        if two:
            coef, se = offset_logit_k(y[mask], f0[mask], np.column_stack([gx[mask], gz[mask]]))
            return {'tau2_x': round(float(coef[0]), 5), 'se_x': round(float(se[0]), 5), 'tau2_z': round(float(coef[1]), 5), 'se_z': round(float(se[1]), 5),
                    'tau_x': round(float(np.sign(coef[0]) * np.sqrt(abs(coef[0]))), 4), 'tau_z': round(float(np.sign(coef[1]) * np.sqrt(abs(coef[1]))), 4), 'n': int(mask.sum())}
        a, b, se = offset_logit(y[mask], f0[mask], gb[mask])
        return {'tau2': round(b, 5), 'se': round(se, 5), 'tau': round(float(np.sign(b) * np.sqrt(abs(b))), 4), 'n': int(mask.sum())}

    allm = np.ones(len(ev), bool)
    out = {'all': est(allm), 'x_and_z': est(allm, two=True)}
    v0 = T['v0'][ev]
    out['by_speed'] = {f'{lo}-{hi}': {**(est((v0 >= lo) & (v0 < hi)) or {}), 'mean_mph': round(float(v0[(v0 >= lo) & (v0 < hi)].mean()), 1) if ((v0 >= lo) & (v0 < hi)).any() else None}
                       for lo, hi in ((60, 82), (82, 86), (86, 90), (90, 93), (93, 95), (95, 105))}
    out['by_group'] = {GROUP_NAMES[gi]: est(T['group'][ev] == gi) for gi in range(6)}
    out['by_strikes'] = {str(k): est(T['strikes'][ev] == k) for k in (0, 1, 2)}
    out['by_batter_hand'] = {'right': est(T['stand_r'][ev] == 1), 'left': est(T['stand_r'][ev] == 0)}
    out['same_hand'] = {'same': est(T['stand_r'][ev] == T['throw_r'][ev]), 'opposite': est(T['stand_r'][ev] != T['throw_r'][ev])}
    out['by_season'] = {str(s): est(T['season'][ev] == s) for s in (2025, 2026)}
    out['by_inning'] = {'1-3': est(T['inning'][ev] <= 3), '4-6': est((T['inning'][ev] >= 4) & (T['inning'][ev] <= 6)), '7+': est(T['inning'][ev] >= 7)}
    # speed surprise as a placebo direction: a faster-than-usual pitch at the same place should not move the decision map
    coef, se2 = offset_logit_k(y, f0, np.column_stack([gb, wv[ev]]))
    out['with_speed_surprise'] = {'tau2': round(float(coef[0]), 5), 'se': round(float(se2[0]), 5), 'speed_coef_per_mph': round(float(coef[1]), 5), 'speed_se': round(float(se2[1]), 5)}
    res['gradient'] = out; stage('splits')
    # profile past 0.25 s on the within-type surprise (2026 log loss, type maps)
    te = np.sort(rng.choice(np.flatnonzero(T['season'] == 2026), min(int((T['season'] == 2026).sum()), 600000), replace=False))
    base = logloss_vec(m.predict_proba(typed(T['px'], T['pz'], te))[:, 1], swing[te])
    prof = [{'tau': 0.0, 'd_per_1000': 0.0}]
    for tau in [float(t) for t in params.get('taus_long', (0.2, 0.25, 0.3, 0.35, 0.4))]:
        x = T['px'] - 0.5 * wx * tau ** 2; z = T['pz'] - 0.5 * wz * tau ** 2
        mt = fit_logistic(typed(x, z, tr), swing[tr])
        ll = logloss_vec(mt.predict_proba(typed(x, z, te))[:, 1], swing[te])
        d = clustered_ci(ll - base, T['game'][te])
        prof.append({'tau': tau, 'd_per_1000': round(d[0] * 1000, 3), 'ci': [round(d[1] * 1000, 3), round(d[2] * 1000, 3)]})
        stage(f'long tau {tau}')
    res['profile_long'] = prof
    return res


def familiarity(T: dict) -> tuple[np.ndarray, np.ndarray]:
    """For each pitch: how many earlier plate appearances this batter had against this pitcher in this game, and how
    many earlier pitches of this exact pitch type he had seen from him in this game (current PA included)."""
    order = np.lexsort((T['pitch_no'], T['ab'], T['batter'], T['pitcher'], T['game']))
    g, p, b, ab, sub = T['game'][order], T['pitcher'][order], T['batter'][order], T['ab'][order], T['sub'][order]
    tto = np.zeros(len(order), np.int64); expo = np.zeros(len(order), np.int64)
    start = 0
    n = len(order)
    while start < n:
        end = start
        while end < n and g[end] == g[start] and p[end] == p[start] and b[end] == b[start]:
            end += 1
        abs_ = ab[start:end]; uniq = np.unique(abs_)
        tto[start:end] = np.searchsorted(uniq, abs_)
        seen = {}
        for k in range(start, end):
            s = int(sub[k]); expo[k] = seen.get(s, 0); seen[s] = seen.get(s, 0) + 1
        start = end
    out_t = np.empty(n, np.int64); out_e = np.empty(n, np.int64)
    out_t[order] = tto; out_e[order] = expo
    return out_t, out_e


def horizon5(T: dict, params: dict, stage) -> dict:
    """Does seeing a pitcher again move a hitter's horizon later? Within-type commit time (type maps, as in the
    decisive test) and the type-level projection reliance (one shared map, gravity-only projection) by how many
    times the batter has faced this pitcher today and by how many pitches of this type he has already seen today."""
    res = {}
    F = rebuild(T)
    keep = (F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
            & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1)))
    tto, expo = familiarity(T)
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}; tto = tto[keep]; expo = expo[keep]
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.int64)
    C = control_block(T, swing_propensity(T))
    key_sub = (T['pitcher'] * 10 + (T['season'] - 2020)) * 100 + T['sub']
    ok = np.isfinite(F['ax']) & np.isfinite(F['az'])
    mx = loo_means(key_sub, F['ax'], ok); mz = loo_means(key_sub, F['az'], ok)
    wx = np.where(np.isfinite(mx), F['ax'] - mx, 0.0); wz = np.where(np.isfinite(mz), F['az'] - mz, 0.0)
    rng = np.random.default_rng(int(params.get('seed', 11)))
    dev_all = np.flatnonzero(T['season'] <= 2024)
    tr = np.sort(rng.choice(dev_all, min(len(dev_all), int(params.get('train_n', 500000))), replace=False))
    ev = np.flatnonzero(T['season'] >= 2025)
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]

    def typed(x, z, idx):
        uu = np.where(T['stand_r'][idx] == 1, x[idx], -x[idx]); zz = z[idx]
        e = np.maximum(np.maximum(np.abs(uu) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
        b = np.hstack([hats(e, E_KNOTS), hats(uu, U_KNOTS), hats(zz, Z_KNOTS)])
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), (b[:, :, None] * oh[idx, None, :]).reshape(len(idx), -1), C[idx]])

    def shared(x, z, idx):
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), C[idx]])
    h = 0.02
    terms = {}
    for name, designf, dx, dz in (('within', typed, wx, wz), ('type_level', shared, F['asx'], F['asz'])):
        m = fit_logistic(designf(T['px'], T['pz'], tr), swing[tr])
        f0 = np.empty(len(ev)); gt = np.empty(len(ev))
        for s in range(0, len(ev), 250000):
            ii = ev[s:s + 250000]; df = lambda x, z: m.decision_function(designf(x, z, ii))
            f0[s:s + 250000] = df(T['px'], T['pz'])
            fx = (df(T['px'] + h, T['pz']) - df(T['px'] - h, T['pz'])) / (2 * h); fz = (df(T['px'], T['pz'] + h) - df(T['px'], T['pz'] - h)) / (2 * h)
            gt[s:s + 250000] = -0.5 * (fx * dx[ii] + fz * dz[ii])
        terms[name] = (f0, gt); stage('gradient ' + name)
    y = swing[ev].astype(float)
    tt = tto[ev]; ee = expo[ev]; starter_like = np.ones(len(ev), bool)

    def est(name, mask):
        if mask.sum() < 15000:
            return None
        f0, gt = terms[name]
        a, b, se = offset_logit(y[mask], f0[mask], gt[mask])
        return {'tau2': round(b, 5), 'se': round(se, 5), 'tau': round(float(np.sign(b) * np.sqrt(abs(b))), 4), 'n': int(mask.sum())}
    out = {}
    for name in terms:
        out[name] = {'all': est(name, np.ones(len(ev), bool)),
                     'times_faced': {str(k): est(name, (tt == k) if k < 3 else (tt >= 3)) for k in (0, 1, 2, 3)},
                     'type_seen_today': {lab: est(name, msk) for lab, msk in (('0', ee == 0), ('1-2', (ee >= 1) & (ee <= 2)), ('3-5', (ee >= 3) & (ee <= 5)), ('6+', ee >= 6))},
                     'type_seen_first_time_faced': {lab: est(name, msk & (tt == 0)) for lab, msk in (('0', ee == 0), ('1-2', (ee >= 1) & (ee <= 2)), ('3+', ee >= 3))},
                     'times_faced_first_pitch_of_type': {str(k): est(name, ((tt == k) if k < 2 else (tt >= 2)) & (ee == 0)) for k in (0, 1, 2)}}
    res['by_familiarity'] = out
    res['counts'] = {'times_faced': {str(k): int(((tt == k) if k < 3 else (tt >= 3)).sum()) for k in (0, 1, 2, 3)}}
    return res


def horizon6(T: dict, params: dict, stage) -> dict:
    """Pitcher side: hitters cannot use a pitch's movement surprise after the horizon, so a pitcher whose pitches
    vary more around their own average shape might win more decisions. Per pitcher, the usage-weighted spread of
    within-type spin acceleration (as inches of surprise at the 260 ms horizon), measured on 2023-2024, tested
    against his 2025-2026 strikeout, walk, chase and whiff rates with his 2023-2024 rates, velocity and movement as
    controls."""
    res = {}
    F = rebuild(T)
    keep = (F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0) & (T['strikes'] <= 2))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    tau = float(params.get('tau', 0.26))
    key_sub = (T['pitcher'] * 10 + (T['season'] - 2020)) * 100 + T['sub']
    ok = np.isfinite(F['ax']) & np.isfinite(F['az'])
    mx = loo_means(key_sub, F['ax'], ok); mz = loo_means(key_sub, F['az'], ok)
    wx = np.where(np.isfinite(mx), F['ax'] - mx, np.nan); wz = np.where(np.isfinite(mz), F['az'] - mz, np.nan)
    surprise_in = 0.5 * np.sqrt(wx ** 2 + wz ** 2) * tau ** 2 * 12          # inches of unexpected movement after the horizon
    swing = (T['call'] == 1) | (T['call'] == 2); whiff = T['call'] == 2
    u = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    inzone = (np.abs(u) <= ZONE_HALF) & (T['pz'] <= ZONE_TOP) & (T['pz'] >= ZONE_BOT)
    pa_end = T['last_in_pa'] == 1
    k_pa = pa_end & whiff & (T['strikes'] == 2); bb_pa = pa_end & (T['call'] == 0) & (T['balls'] == 3)
    pit = T['pitcher']; up, inv = np.unique(pit, return_inverse=True)
    dev = T['season'] <= 2024; fut = T['season'] >= 2025
    def rate(num, den, period):
        n = np.bincount(inv, weights=(num & period).astype(float), minlength=len(up)); d = np.bincount(inv, weights=(den & period).astype(float), minlength=len(up))
        with np.errstate(invalid='ignore', divide='ignore'):
            return n / d
    def mean_of(v, period, mask=None):
        m = period & np.isfinite(v) & (True if mask is None else mask)
        n = np.bincount(inv, weights=np.where(m, v, 0.0), minlength=len(up)); d = np.bincount(inv, weights=m.astype(float), minlength=len(up))
        with np.errstate(invalid='ignore', divide='ignore'):
            return n / d
    fb = np.isin(T['group'], (0, 1))
    move = np.sqrt(F['asx'] ** 2 + F['asz'] ** 2)
    X = {'k_rate': rate(k_pa, pa_end, dev), 'bb_rate': rate(bb_pa, pa_end, dev), 'o_swing': rate(swing & ~inzone, ~inzone, dev),
         'whiff_per_swing': rate(whiff, swing, dev), 'zone_rate': rate(inzone, np.ones(len(inzone), bool), dev),
         'fb_velo': mean_of(T['v0'].astype(float), dev, fb), 'fb_ride': mean_of(F['asz'], dev, fb), 'movement': mean_of(move, dev)}
    S = {'surprise_in': mean_of(surprise_in, dev), 'surprise_fb_in': mean_of(surprise_in, dev, fb), 'surprise_offspeed_in': mean_of(surprise_in, dev, ~fb)}
    Y = {'k_rate': rate(k_pa, pa_end, fut), 'bb_rate': rate(bb_pa, pa_end, fut), 'o_swing': rate(swing & ~inzone, ~inzone, fut),
         'whiff_per_swing': rate(whiff, swing, fut), 'zone_rate': rate(inzone, np.ones(len(inzone), bool), fut)}
    n_dev = np.bincount(inv, weights=dev.astype(float), minlength=len(up)); n_fut = np.bincount(inv, weights=fut.astype(float), minlength=len(up))
    sel = np.flatnonzero((n_dev >= int(params.get('min_dev', 1500))) & (n_fut >= int(params.get('min_fut', 1000))))
    res['n_pitchers'] = int(len(sel))
    from scipy import stats
    res['surprise_quantiles_in'] = {q: round(float(np.nanpercentile(S['surprise_in'][sel], q)), 3) for q in (10, 50, 90)}
    res['corr_with_dev'] = {sk: {k: round(float(stats.spearmanr(S[sk][sel], v[sel], nan_policy='omit')[0]), 4) for k, v in X.items()} for sk in S}
    # stability of the surprise itself: 2023-2024 against 2025-2026
    S_fut = mean_of(surprise_in, fut)
    res['surprise_year_to_year'] = round(float(stats.spearmanr(S['surprise_in'][sel], S_fut[sel], nan_policy='omit')[0]), 4)
    X0 = np.column_stack([X[k][sel] for k in X])
    inc = {}
    for sk in S:
        s = S[sk][sel]
        for yk, yv in Y.items():
            yy = yv[sel]; okr = np.isfinite(yy) & np.isfinite(X0).all(axis=1) & np.isfinite(s)
            Xa = np.column_stack([np.ones(okr.sum()), X0[okr]]); Xb = np.column_stack([Xa, s[okr]])
            def fit(Xm):
                beta, *_ = np.linalg.lstsq(Xm, yy[okr], rcond=None); e = yy[okr] - Xm @ beta
                H = Xm @ np.linalg.pinv(Xm.T @ Xm) @ Xm.T
                return 1 - e.var() / yy[okr].var(), float(np.mean((e / (1 - np.diag(H))) ** 2)), beta
            ra, la, _ = fit(Xa); rb, lb, beta = fit(Xb)
            rr = np.random.default_rng(2); boots = []
            for _ in range(500):
                b = rr.integers(0, okr.sum(), okr.sum()); bb, *_ = np.linalg.lstsq(Xb[b], yy[okr][b], rcond=None); boots.append(bb[-1])
            sd = float(np.std(s[okr]))
            inc[f'{sk}->{yk}'] = {'r2': [round(float(ra), 4), round(float(rb), 4)], 'loo_mse_x1e5': [round(la * 1e5, 3), round(lb * 1e5, 3)],
                                  'per_sd': round(float(beta[-1] * sd), 5), 'ci': [round(float(np.percentile(boots, 2.5) * sd), 5), round(float(np.percentile(boots, 97.5) * sd), 5)],
                                  'outcome_sd': round(float(np.std(yy[okr])), 5), 'n': int(okr.sum())}
    res['incremental'] = inc
    return res


def horizon7(T: dict, params: dict, stage) -> dict:
    """Contact horizon: after deciding to swing, how late can the hitter still steer the bat? For balls in play, the
    launch angle against the pitch's vertical movement surprise. If the bat is aimed at the crossing projected from
    tau_c before the plate, the ball arrives 0.5 * wz * tau_c^2 above the aim, and contact below the ball's center
    lifts the launch angle by about k degrees per inch (bat-ball geometry; Brantley and Kording 2022 use a ball radius
    of 1.45 in). The slope b of launch angle on 0.5 * wz (inches per s^2) gives tau_c = sqrt(b / k); k is reported
    as a range because it is assumed, not measured. Also: vertical surprise against whiffs on swings."""
    res = {}
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    key_sub = (T['pitcher'] * 10 + (T['season'] - 2020)) * 100 + T['sub']
    ok = np.isfinite(F['az']) & np.isfinite(F['ax'])
    mz = loo_means(key_sub, F['az'], ok); mx = loo_means(key_sub, F['ax'], ok)
    wz = np.where(np.isfinite(mz), F['az'] - mz, np.nan); wx = np.where(np.isfinite(mx), F['ax'] - mx, np.nan)
    s_in = 0.5 * wz * 12.0                       # inches of vertical surprise per s^2 of horizon
    bip = np.isfinite(T['la']) & np.isfinite(s_in) & (T['ls'] > 40)
    n = int(bip.sum()); res['balls_in_play'] = n
    def design(idx):
        oh = np.eye(7)[np.clip(T['group'][idx], 0, 6)]
        cnt = np.zeros((len(idx), 12)); cnt[np.arange(len(idx)), np.clip(T['balls'][idx], 0, 3) * 3 + np.clip(T['strikes'][idx], 0, 2)] = 1
        u = np.where(T['stand_r'][idx] == 1, T['px'][idx], -T['px'][idx])
        return np.hstack([oh, cnt, hats(T['pz'][idx].astype(float), Z_KNOTS), hats(u.astype(float), U_KNOTS), hats(T['v0'][idx].astype(float), V_KNOTS),
                          (T['stand_r'][idx] == T['throw_r'][idx])[:, None].astype(float)])
    out = {}
    for name, mask in (('all', bip), ('fastballs', bip & np.isin(T['group'], (0, 1, 2))), ('breaking', bip & np.isin(T['group'], (3, 4))), ('offspeed', bip & (T['group'] == 5)),
                       ('2025-2026', bip & (T['season'] >= 2025)), ('2023-2024', bip & (T['season'] <= 2024))):
        idx = np.flatnonzero(mask)
        X = np.column_stack([design(idx), s_in[idx]]); y = T['la'][idx].astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ beta
        # game-clustered standard error for the last coefficient
        XtX_inv = np.linalg.pinv(X.T @ X); g = T['game'][idx]; ug, gi = np.unique(g, return_inverse=True)
        S = np.zeros((len(ug), X.shape[1])); np.add.at(S, gi, X * e[:, None]); meat = S.T @ S
        se = float(np.sqrt((XtX_inv @ meat @ XtX_inv)[-1, -1]))
        b = float(beta[-1])
        out[name] = {'n': int(len(idx)), 'slope_deg_per_in_s2': round(b, 3), 'se': round(se, 3),
                     'tau_c_for_k': {str(k): (round(float(np.sqrt(b / k)), 4) if b > 0 else None) for k in (12, 16, 20, 25)}}
    res['launch_angle'] = out; stage('launch angle')
    # placebo: horizontal surprise should not move launch angle the same way
    idx = np.flatnonzero(bip & np.isfinite(wx))
    X = np.column_stack([design(idx), s_in[idx], 0.5 * wx[idx] * 12.0]); y = T['la'][idx].astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res['placebo_horizontal_slope'] = round(float(beta[-1]), 3); res['vertical_slope_same_fit'] = round(float(beta[-2]), 3)
    # decision horizon reference from the same pitches would be ~0.26 s; implied launch-angle change if the bat
    # were aimed from the decision horizon: k * 0.26^2 per inch-per-s^2
    res['reference_slope_if_tau_c_equals_decision_horizon'] = {str(k): round(k * 0.26 ** 2, 3) for k in (12, 16, 20, 25)}
    return res


def position_at(T, F, tau):
    """Rebuilt position (x, y, z, ft) of each pitch tau seconds before it reaches the front of the plate."""
    t = F['tf'] - tau
    v0 = T['v0'].astype(np.float64) * FT_PER_MPH
    y = Y0 - v0 * t + 0.5 * F['ay'] * t ** 2
    x = T['x0'] + F['vx0'] * t + 0.5 * F['ax'] * t ** 2
    z = T['z0'] + F['vz0'] * t + 0.5 * F['az'] * t ** 2
    return x, y, z


def horizon8(T: dict, params: dict, stage) -> dict:
    """Tunneling at the measured horizon. For each pitch that follows another in the same plate appearance, the
    visual-angle separation between the two flights (as seen from the batter's eye) at the same time before each
    reaches the plate. Profile over that time: which moment's separation best predicts a chase (swing at a pitch
    outside the zone) and a whiff (miss on a swing) on the second pitch, beyond its own type, speed, movement and
    location, the first pitch's type and location, the count and the batter. Prediction written before the run:
    chases are best predicted near 260 ms and whiffs closer to the 100-125 ms steering limit; public metrics use
    150-175 ms."""
    res = {}
    F = rebuild(T)
    base_ok = F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
    order = np.lexsort((T['pitch_no'], T['ab'], T['game']))
    prev = np.full(len(order), -1)
    same = (T['game'][order][1:] == T['game'][order][:-1]) & (T['ab'][order][1:] == T['ab'][order][:-1]) & (T['pitch_no'][order][1:] == T['pitch_no'][order][:-1] + 1)
    prev[order[1:][same]] = order[:-1][same]
    has_prev = (prev >= 0) & base_ok & np.where(prev >= 0, base_ok[np.maximum(prev, 0)], False) & (T['call'] <= 2)
    idx = np.flatnonzero(has_prev); pidx = prev[idx]
    res['pairs'] = int(len(idx))
    swing = ((T['call'] == 1) | (T['call'] == 2)); whiff = T['call'] == 2
    C = control_block(T, swing_propensity(T))
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]
    u_all = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    inzone = (np.abs(u_all) <= ZONE_HALF) & (T['pz'] <= ZONE_TOP) & (T['pz'] >= ZONE_BOT)
    eye_x = np.where(T['stand_r'] == 1, -2.4, 2.4); eye_y, eye_z = 1.0, 5.1

    def angles(i, tau):
        x, y, z = position_at(take(T, i), {k: v[i] for k, v in F.items()}, tau) if tau > 0 else (T['px'][i].astype(float), np.full(len(i), YPLATE), T['pz'][i].astype(float))
        d = np.maximum(y - eye_y, 0.5)
        return np.degrees(np.arctan2(x - eye_x[i], d)), np.degrees(np.arctan2(z - eye_z, d))

    def base_design(i, j):
        uu = u_all[i]; zz = T['pz'][i]
        e = np.maximum(np.maximum(np.abs(uu) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
        b = np.hstack([hats(e, E_KNOTS), hats(uu, U_KNOTS), hats(zz, Z_KNOTS)])
        prev_loc = np.hstack([hats(u_all[j], U_KNOTS), hats(T['pz'][j].astype(float), Z_KNOTS)])
        prev_type = oh[j]
        mv = np.column_stack([F['asx'][i] * np.where(T['throw_r'][i] == 1, 1, -1), F['asz'][i]]) / 10.0
        return np.hstack([location_block(T['px'][i], T['pz'][i], T['stand_r'][i], T['strikes'][i]), (b[:, :, None] * oh[i, None, :]).reshape(len(i), -1),
                          C[i], prev_loc, prev_type, prev_type * oh[i].sum(axis=1, keepdims=True) * (T['group'][i] == T['group'][j])[:, None], mv]).astype(np.float32)

    taus = [float(t) for t in params.get('taus', (0.0, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.26, 0.30, 0.35))]
    D_KNOTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    dev = T['season'][idx] <= 2024; fut = ~dev
    out = {}
    for name, sel in (('chase', ~inzone[idx]), ('whiff', swing[idx]), ('swing_in_zone', inzone[idx])):
        ii = idx[sel]; jj = pidx[sel]
        y = (whiff[ii] if name == 'whiff' else swing[ii]).astype(np.int64)
        dv, fu = dev[sel], fut[sel]
        rng = np.random.default_rng(3)
        trn = np.flatnonzero(dv); trn = np.sort(rng.choice(trn, min(len(trn), int(params.get('train_n', 400000))), replace=False))
        tst = np.flatnonzero(fu); tst = np.sort(rng.choice(tst, min(len(tst), int(params.get('test_n', 400000))), replace=False))
        Xb = base_design(ii, jj)
        mb = fit_logistic(Xb[trn], y[trn]); llb = logloss_vec(mb.predict_proba(Xb[tst])[:, 1], y[tst])
        prof = []
        for tau in taus:
            ax_i, az_i = angles(ii, tau); ax_j, az_j = angles(jj, tau)
            sep = np.sqrt((ax_i - ax_j) ** 2 + (az_i - az_j) ** 2)
            X = np.hstack([Xb, hats(sep, D_KNOTS)])
            m = fit_logistic(X[trn], y[trn]); ll = logloss_vec(m.predict_proba(X[tst])[:, 1], y[tst])
            d = clustered_ci(ll - llb, T['game'][ii][tst])
            prof.append({'tau': tau, 'gain_per_1000': round(-d[0] * 1000, 4), 'ci': [round(-d[2] * 1000, 4), round(-d[1] * 1000, 4)],
                         'sep_median_deg': round(float(np.median(sep)), 3)})
        out[name] = {'pitches_train': int(len(trn)), 'pitches_test': int(len(tst)), 'rate_test': round(float(y[tst].mean()), 4), 'base_logloss': float(llb.mean()), 'profile': prof}
        stage('tunnel ' + name)
    res['outcomes'] = out
    return res


def horizon9(T: dict, params: dict, stage) -> dict:
    """Blind window or expectation pull? (reviewer's strongest remaining alternative.) Each pitch's miss from where
    this pitcher usually puts this pitch type in this kind of count to this batter side is split into the part from
    its movement surprise over the flight from 50 ft (0.5 * w * t_f^2) and the rest, the line it left the hand on.
    Both enter as displacements along the type-map swing gradient, with pitcher-season intercepts. A blind window
    discounts the movement part by about (tau / t_f)^2 and the line part not at all (the hitter sees the line long
    before he commits). A pull toward the expected spot discounts both parts the same."""
    res = {}
    F = rebuild(T)
    keep = (F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
            & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1)))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.int64)
    C = control_block(T, swing_propensity(T))
    ps = T['pitcher'] * 10 + (T['season'] - 2020)
    key_sub = ps * 100 + T['sub']
    ok = np.isfinite(F['ax']) & np.isfinite(F['az'])
    mx = loo_means(key_sub, F['ax'], ok); mz = loo_means(key_sub, F['az'], ok)
    wx = np.where(np.isfinite(mx), F['ax'] - mx, 0.0); wz = np.where(np.isfinite(mz), F['az'] - mz, 0.0)
    bucket = np.clip(T['strikes'], 0, 2) * 2 + (T['balls'] >= 2)
    key_loc = (key_sub * 2 + T['stand_r']) * 6 + bucket
    key_loc2 = key_sub * 2 + T['stand_r']
    allok = np.isfinite(T['px']) & np.isfinite(T['pz'])
    ex = loo_means(key_loc, T['px'].astype(float), allok); ez = loo_means(key_loc, T['pz'].astype(float), allok)
    uk, cnt = np.unique(key_loc, return_counts=True); n_key = cnt[np.searchsorted(uk, key_loc)]
    ex2 = loo_means(key_loc2, T['px'].astype(float), allok); ez2 = loo_means(key_loc2, T['pz'].astype(float), allok)
    ex = np.where((n_key >= 12) & np.isfinite(ex), ex, ex2); ez = np.where((n_key >= 12) & np.isfinite(ez), ez, ez2)
    good = np.isfinite(ex) & np.isfinite(ez)
    mvx, mvz = 0.5 * wx * F['tf'] ** 2, 0.5 * wz * F['tf'] ** 2
    rlx, rlz = (T['px'] - ex) - mvx, (T['pz'] - ez) - mvz
    res['miss_sd_in'] = {'movement_x': round(float(np.std(mvx[good]) * 12), 2), 'movement_z': round(float(np.std(mvz[good]) * 12), 2),
                         'line_x': round(float(np.std(rlx[good]) * 12), 2), 'line_z': round(float(np.std(rlz[good]) * 12), 2)}
    rng = np.random.default_rng(int(params.get('seed', 11)))
    dev_all = np.flatnonzero(T['season'] <= 2024)
    tr = np.sort(rng.choice(dev_all, min(len(dev_all), int(params.get('train_n', 500000))), replace=False))
    ev = np.flatnonzero((T['season'] >= 2025) & good)
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]

    def typed(x, z, idx):
        uu = np.where(T['stand_r'][idx] == 1, x[idx], -x[idx]); zz = z[idx]
        e = np.maximum(np.maximum(np.abs(uu) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
        b = np.hstack([hats(e, E_KNOTS), hats(uu, U_KNOTS), hats(zz, Z_KNOTS)])
        return np.hstack([location_block(x[idx], z[idx], T['stand_r'][idx], T['strikes'][idx]), (b[:, :, None] * oh[idx, None, :]).reshape(len(idx), -1), C[idx]])
    m = fit_logistic(typed(T['px'], T['pz'], tr), swing[tr]); stage('typed model')
    h = 0.02
    f0 = np.empty(len(ev)); fx = np.empty(len(ev)); fz = np.empty(len(ev))
    for s in range(0, len(ev), 250000):
        ii = ev[s:s + 250000]; df = lambda x, z: m.decision_function(typed(x, z, ii))
        f0[s:s + 250000] = df(T['px'], T['pz'])
        fx[s:s + 250000] = (df(T['px'] + h, T['pz']) - df(T['px'] - h, T['pz'])) / (2 * h)
        fz[s:s + 250000] = (df(T['px'], T['pz'] + h) - df(T['px'], T['pz'] - h)) / (2 * h)
    stage('gradients')
    y = swing[ev].astype(float)
    # pitcher-season intercepts on top of the typed model (ridge toward zero)
    up, inv = np.unique(ps[ev], return_inverse=True); c = np.zeros(len(up))
    for _ in range(8):
        p = 1 / (1 + np.exp(-(f0 + c[inv])))
        num = np.bincount(inv, weights=y - p, minlength=len(up)); den = np.bincount(inv, weights=p * (1 - p), minlength=len(up)) + 50.0
        c += num / den - 50.0 * c / den
    off = f0 + c[inv]
    g_mv = -(fx * mvx[ev] + fz * mvz[ev]); g_rl = -(fx * rlx[ev] + fz * rlz[ev])

    def two(mask):
        coef, se = offset_logit_k(y[mask], off[mask], np.column_stack([g_mv[mask], g_rl[mask]]))
        tf = float(np.median(F['tf'][ev][mask]))
        return {'k_movement': round(float(coef[0]), 4), 'se_movement': round(float(se[0]), 4), 'k_line': round(float(coef[1]), 4), 'se_line': round(float(se[1]), 4),
                'blind_window_expects_movement': round((0.261 / tf) ** 2, 4), 'median_flight_s': round(tf, 4), 'n': int(mask.sum())}
    out = {'all': two(np.ones(len(ev), bool))}
    for gi in (0, 1, 3, 4, 5):
        out[GROUP_NAMES[gi]] = two(T['group'][ev] == gi)
    out['two_strikes'] = two(T['strikes'][ev] == 2); out['fewer_strikes'] = two(T['strikes'][ev] < 2)
    # bootstrap by game for the difference k_movement - k_line
    games = T['game'][ev]; ug, gi_ = np.unique(games, return_inverse=True); rr = np.random.default_rng(9); diffs = []
    for _ in range(int(params.get('boot', 60))):
        w = np.bincount(rr.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)[gi_]
        coef, _ = offset_logit_k(y, off, np.column_stack([g_mv, g_rl]), w=w, iters=10)
        diffs.append(coef[0] - coef[1])
    out['difference_ci'] = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    res['decomposition'] = out
    return res


def horizon10(T: dict, params: dict, stage) -> dict:
    """Arsenal flights for the pitch-pair explorer, and the paired tunneling comparison of 260 against 175 ms.
    Exports per pitcher and pitch type in 2026 (at least 60 pitches): count, average start and end speed, average
    position and velocity at 50 ft, average accelerations, average crossing, chase and whiff rates. Aggregates
    only, like a public arsenal table; no pitch rows."""
    res = {}
    F = rebuild(T)
    ok = F['ok'] & (T['group'] >= 0)
    s26 = ok & (T['season'] == int(params.get('season', 2026)))
    key = T['pitcher'] * 100 + T['sub']
    swing = (T['call'] == 1) | (T['call'] == 2); whiff = T['call'] == 2
    u = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    inzone = (np.abs(u) <= ZONE_HALF) & (T['pz'] <= ZONE_TOP) & (T['pz'] >= ZONE_BOT)
    uk, inv = np.unique(key[s26], return_inverse=True)
    cnt = np.bincount(inv)
    def m(v):
        return np.bincount(inv, weights=v[s26].astype(float), minlength=len(uk)) / np.maximum(cnt, 1)
    fields = {'v0': T['v0'], 'v1': T['v1'], 'x0': T['x0'], 'z0': T['z0'], 'vx0': F['vx0'], 'vz0': F['vz0'], 'ax': F['ax'], 'az': F['az'], 'ay': F['ay'],
              'px': T['px'], 'pz': T['pz'], 'ext': np.where(np.isfinite(T['ext']), T['ext'], 6.3), 'throw_r': T['throw_r']}
    means = {k: m(v) for k, v in fields.items()}
    out_sw = np.bincount(inv, weights=(swing & ~inzone)[s26].astype(float), minlength=len(uk)); out_n = np.bincount(inv, weights=(~inzone)[s26].astype(float), minlength=len(uk))
    sw_n = np.bincount(inv, weights=swing[s26].astype(float), minlength=len(uk)); wh = np.bincount(inv, weights=whiff[s26].astype(float), minlength=len(uk))
    rows = []
    for j in np.flatnonzero(cnt >= int(params.get('min_pitches', 60))):
        rows.append({'p': int(uk[j] // 100), 't': SUBTYPES[int(uk[j] % 100)] if int(uk[j] % 100) < len(SUBTYPES) else 'OT', 'n': int(cnt[j]),
                     **{k: round(float(v[j]), 4) for k, v in means.items()},
                     'chase': round(float(out_sw[j] / out_n[j]), 4) if out_n[j] > 0 else None, 'whiff': round(float(wh[j] / sw_n[j]), 4) if sw_n[j] > 0 else None})
    res['arsenal'] = rows; res['arsenal_season'] = int(params.get('season', 2026)); stage('arsenal')
    # paired: chase model with the separation at 260 ms against the same model with it at 175 ms
    base_ok = F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
    order = np.lexsort((T['pitch_no'], T['ab'], T['game']))
    prev = np.full(len(order), -1)
    same = (T['game'][order][1:] == T['game'][order][:-1]) & (T['ab'][order][1:] == T['ab'][order][:-1]) & (T['pitch_no'][order][1:] == T['pitch_no'][order][:-1] + 1)
    prev[order[1:][same]] = order[:-1][same]
    has_prev = (prev >= 0) & base_ok & np.where(prev >= 0, base_ok[np.maximum(prev, 0)], False) & (T['call'] <= 2) & ~inzone
    idx = np.flatnonzero(has_prev); pidx = prev[idx]
    C = control_block(T, swing_propensity(T))
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]
    eye_x = np.where(T['stand_r'] == 1, -2.4, 2.4)
    def angles(i, tau):
        x, y, z = position_at(take(T, i), {k: v[i] for k, v in F.items()}, tau)
        d = np.maximum(y - 1.0, 0.5)
        return np.degrees(np.arctan2(x - eye_x[i], d)), np.degrees(np.arctan2(z - 5.1, d))
    uu = u[idx]; zz = T['pz'][idx]
    e = np.maximum(np.maximum(np.abs(uu) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
    b = np.hstack([hats(e, E_KNOTS), hats(uu, U_KNOTS), hats(zz, Z_KNOTS)])
    mv = np.column_stack([F['asx'][idx] * np.where(T['throw_r'][idx] == 1, 1, -1), F['asz'][idx]]) / 10.0
    Xb = np.hstack([location_block(T['px'][idx], T['pz'][idx], T['stand_r'][idx], T['strikes'][idx]), (b[:, :, None] * oh[idx, None, :]).reshape(len(idx), -1), C[idx],
                    hats(u[pidx], U_KNOTS), hats(T['pz'][pidx].astype(float), Z_KNOTS), oh[pidx], mv]).astype(np.float32)
    y = swing[idx].astype(np.int64)
    rng = np.random.default_rng(3)
    dv = np.flatnonzero(T['season'][idx] <= 2024); fu = np.flatnonzero(T['season'][idx] >= 2025)
    trn = np.sort(rng.choice(dv, min(len(dv), 500000), replace=False)); tst = fu
    D_KNOTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    lls = {}
    for tau in (0.175, 0.26):
        a1, b1 = angles(idx, tau); a2, b2 = angles(pidx, tau)
        X = np.hstack([Xb, hats(np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2), D_KNOTS)])
        mdl = fit_logistic(X[trn], y[trn]); lls[tau] = logloss_vec(mdl.predict_proba(X[tst])[:, 1], y[tst])
        stage(f'paired {tau}')
    d = clustered_ci(lls[0.26] - lls[0.175], T['game'][idx][tst], reps=500)
    res['paired_260_minus_175_per_1000'] = [round(v * 1000, 4) for v in d]
    res['paired_test_pitches'] = int(len(tst))
    return res


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
        elif experiment == 'horizon2':
            receipt['results'] = horizon2(T, params, stage)
        elif experiment == 'horizon3':
            receipt['results'] = horizon3(T, params, stage)
        elif experiment == 'horizon4':
            receipt['results'] = horizon4(T, params, stage)
        elif experiment == 'horizon5':
            receipt['results'] = horizon5(T, params, stage)
        elif experiment == 'horizon6':
            receipt['results'] = horizon6(T, params, stage)
        elif experiment == 'horizon7':
            receipt['results'] = horizon7(T, params, stage)
        elif experiment == 'horizon8':
            receipt['results'] = horizon8(T, params, stage)
        elif experiment == 'horizon9':
            receipt['results'] = horizon9(T, params, stage)
        elif experiment == 'horizon10':
            receipt['results'] = horizon10(T, params, stage)
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
