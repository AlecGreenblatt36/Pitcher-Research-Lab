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


def put_bytes(repo, token, path, raw, branch, message):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    for attempt in range(6):
        payload = {'message': message, 'content': base64.b64encode(raw).decode(), 'branch': branch}
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


def savant_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('brl_savant', ROOT / 'tools' / 'brl_savant.py')
    sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)
    return sv


def load_savant(repo, token, branch, key, year: int, parts=('a', 'b')) -> list:
    """The sealed Savant pitches of a season, one column set per stored part (merge them with brl_matchup.merge)."""
    from cloud.security import unseal
    sv = savant_module(); got = []
    for part in parts:
        raw = read_blob(repo, token, sv.path(year, part), branch)
        if raw is not None:
            got.append(sv.from_bytes(unseal(raw, key, sv.purpose(year, part))))
    return got


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
          'v0', 'v1', 'spin', 'pfx_x', 'pfx_z', 'px', 'pz', 'x0', 'z0', 'ext', 'last_in_pa', 'bunt_pa', 'ab', 'pitch_no', 'la', 'ls', 'cs', 'zone', 'out7', 'half', 'post')
OUT7 = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')
CALLS = {'take': 0, 'swing_contact': 1, 'whiff': 2, 'other': 3}


def pitch_table(doc: dict, season: int, game_types=('R',)) -> dict:
    """One row per pitch of the regular season (and the postseason rounds when asked), as numpy arrays (pitch-level
    data stays on the runner)."""
    cols = {k: [] for k in FIELDS}
    for gpk, game in (doc.get('games') or {}).items():
        if game.get('game_type') not in game_types:
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
                cols['cs'].append(1 if code == 'C' else 0)
                try:
                    cols['zone'].append(int(_zone) if _zone is not None else -1)
                except (TypeError, ValueError):
                    cols['zone'].append(-1)
                o_ = str(row.get('o') or '')
                cols['out7'].append(OUT7.index(o_) if o_ in OUT7 else -1)
                cols['half'].append(0 if str(row.get('half') or 'top') == 'top' else 1)
                cols['post'].append(0 if game.get('game_type') == 'R' else 1)
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


def horizon11(T: dict, params: dict, stage) -> dict:
    """From prediction to a lever: pitchers change grips, slots and shapes between seasons. For each pitcher's pitch
    type in both 2025 and 2026, how far its average flight is from the flights of the pitches that usually come before
    it (weighted by how often each one does), at the 260 ms decision moment, at 175 ms and at the plate. Does the
    change in separation at the decision moment predict the change in chase rate on that pitch type, beyond the change
    in its own speed, movement, location and usage? Pitcher-level changes difference out everything fixed about him."""
    res = {}
    F = rebuild(T)
    ok = F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
    pairs_list = [tuple(int(v) for v in pr) for pr in params.get('pairs', [[int(params.get('from_season', 2025)), int(params.get('to_season', 2026))]])]
    seasons = tuple(sorted({s for pr in pairs_list for s in pr}))
    swing = (T['call'] == 1) | (T['call'] == 2); whiff = T['call'] == 2
    u = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    inzone = (np.abs(u) <= ZONE_HALF) & (T['pz'] <= ZONE_TOP) & (T['pz'] >= ZONE_BOT)
    order = np.lexsort((T['pitch_no'], T['ab'], T['game']))
    prev = np.full(len(order), -1)
    same = (T['game'][order][1:] == T['game'][order][:-1]) & (T['ab'][order][1:] == T['ab'][order][:-1]) & (T['pitch_no'][order][1:] == T['pitch_no'][order][:-1] + 1)
    prev[order[1:][same]] = order[:-1][same]
    unit = {}
    for s in seasons:
        msk = ok & (T['season'] == s)
        key = T['pitcher'] * 100 + T['sub']
        uk, inv = np.unique(key[msk], return_inverse=True); cnt = np.bincount(inv)
        mean = lambda v: np.bincount(inv, weights=v[msk].astype(float), minlength=len(uk)) / np.maximum(cnt, 1)
        fl = {k: mean(v) for k, v in (('v0', T['v0']), ('v1', T['v1']), ('x0', T['x0']), ('z0', T['z0']), ('vx0', F['vx0']), ('vz0', F['vz0']),
                                       ('ax', F['ax']), ('az', F['az']), ('ay', F['ay']), ('px', T['px']), ('pz', T['pz']), ('asx', F['asx']), ('asz', F['asz']))}
        sums = lambda m: np.bincount(inv, weights=m[msk].astype(float), minlength=len(uk))
        chase = sums(swing & ~inzone) / np.maximum(sums(~inzone), 1); wh = sums(whiff) / np.maximum(sums(swing), 1); zone = sums(inzone) / np.maximum(cnt, 1)
        # transitions: previous pitch type for each pitch of this type
        pm = msk & (prev >= 0)
        pidx = prev[pm]; cur_key = key[pm]; prev_key = T['pitcher'][pidx] * 100 + T['sub'][pidx]
        trans = {}
        for c, p_ in zip(cur_key, prev_key):
            if c != p_ and c // 100 == p_ // 100:
                d = trans.setdefault(int(c), {}); d[int(p_)] = d.get(int(p_), 0) + 1
        tot = np.bincount(np.unique(T['pitcher'][msk], return_inverse=True)[1])
        for j, k in enumerate(uk):
            unit[(int(k), s)] = {'n': int(cnt[j]), 'chase': float(chase[j]), 'whiff': float(wh[j]), 'zone': float(zone[j]),
                                 **{f: float(v[j]) for f, v in fl.items()}, 'trans': trans.get(int(k), {})}
    def pos(r, tau, eye_x):
        V0, V1 = r['v0'] * FT_PER_MPH, r['v1'] * FT_PER_MPH; tf = (V0 - V1) / r['ay']; t = tf - tau
        y = Y0 - V0 * t + 0.5 * r['ay'] * t * t; x = r['x0'] + r['vx0'] * t + 0.5 * r['ax'] * t * t; z = r['z0'] + r['vz0'] * t + 0.5 * r['az'] * t * t
        d = max(y - 1.0, 0.5)
        return np.degrees(np.arctan2(x - eye_x, d)), np.degrees(np.arctan2(z - 5.1, d)), x, z
    def sep(a, b, tau):
        out = []
        for ex in (-2.4, 2.4):
            a1, b1, xa, za = pos(a, tau, ex); a2, b2, xb, zb = pos(b, tau, ex)
            out.append(np.hypot(a1 - a2, b1 - b2))
        return float(np.mean(out))
    def plate(a, b):
        return float(np.hypot(a['px'] - b['px'], a['pz'] - b['pz']) * 12)
    rows = []
    min_n = int(params.get('min_n', 150))
    for (k, s), r in list(unit.items()):
      for (s_from, s_to) in pairs_list:
        if s != s_from or (k, s_to) not in unit:
            continue
        r2 = unit[(k, s_to)]
        if r['n'] < min_n or r2['n'] < min_n:
            continue
        feats = {}
        okk = True
        for tag, rr, ss in (('a', r, s_from), ('b', r2, s_to)):
            tr = {p_: c for p_, c in rr['trans'].items() if (p_, ss) in unit and unit[(p_, ss)]['n'] >= 60}
            w = sum(tr.values())
            if w < 40:
                okk = False; break
            feats[tag] = {'s260': sum(c * sep(rr, unit[(p_, ss)], 0.26) for p_, c in tr.items()) / w,
                          's175': sum(c * sep(rr, unit[(p_, ss)], 0.175) for p_, c in tr.items()) / w,
                          'plate': sum(c * plate(rr, unit[(p_, ss)]) for p_, c in tr.items()) / w}
        if not okk:
            continue
        rows.append({'pitcher': k // 100, 'type': SUBTYPES[k % 100] if k % 100 < len(SUBTYPES) else 'OT', 'n': min(r['n'], r2['n']), 'from': s_from,
                     'd_chase': r2['chase'] - r['chase'], 'd_whiff': r2['whiff'] - r['whiff'], 'd_s260': feats['b']['s260'] - feats['a']['s260'],
                     'd_s175': feats['b']['s175'] - feats['a']['s175'], 'd_plate': feats['b']['plate'] - feats['a']['plate'],
                     'd_v0': r2['v0'] - r['v0'], 'd_asx': abs(r2['asx']) - abs(r['asx']), 'd_asz': r2['asz'] - r['asz'], 'd_zone': r2['zone'] - r['zone'],
                     'd_pz': r2['pz'] - r['pz'], 's260_a': feats['a']['s260']})
    res['units'] = len(rows)
    stage('units')
    if len(rows) < 50:
        return res
    import numpy.linalg as la
    A = {k: np.array([x[k] for x in rows], float) for k in rows[0] if k not in ('type',)}
    w = np.sqrt(A['n'])
    piv = A['pitcher']; up, pinv = np.unique(piv, return_inverse=True)
    def fit(ycol, xcols):
        X = np.column_stack([np.ones(len(w))] + [A[c] for c in xcols]); y = A[ycol]
        Xw, yw = X * w[:, None], y * w
        beta = la.lstsq(Xw, yw, rcond=None)[0]
        rng = np.random.default_rng(4); boots = []
        for _ in range(800):
            pick = rng.integers(0, len(up), len(up)); idx = np.concatenate([np.flatnonzero(pinv == p) for p in pick])
            boots.append(la.lstsq(Xw[idx], yw[idx], rcond=None)[0])
        B = np.array(boots)
        return {c: {'coef': round(float(beta[i + 1]), 5), 'ci': [round(float(np.percentile(B[:, i + 1], 2.5)), 5), round(float(np.percentile(B[:, i + 1], 97.5)), 5)],
                    'sd_x': round(float(np.std(A[c])), 4)} for i, c in enumerate(xcols)}
    controls = ['d_v0', 'd_asx', 'd_asz', 'd_zone', 'd_pz']
    res['sd_change'] = {k: round(float(np.std(A[k])), 4) for k in ('d_s260', 'd_s175', 'd_plate', 'd_chase', 'd_whiff')}
    res['chase'] = {'s260': fit('d_chase', ['d_s260', 'd_plate'] + controls), 's175': fit('d_chase', ['d_s175', 'd_plate'] + controls),
                    'both': fit('d_chase', ['d_s260', 'd_s175', 'd_plate'] + controls)}
    res['whiff'] = {'s260': fit('d_whiff', ['d_s260', 'd_plate'] + controls)}
    big = np.abs(A['d_s260']) > np.percentile(np.abs(A['d_s260']), 80)
    res['largest_changes'] = sorted([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x, b in zip(rows, big) if b], key=lambda x: -abs(x['d_s260']))[:40]
    return res


def horizon12(T: dict, params: dict, stage) -> dict:
    """The second clock per hitter: how much a hitter's launch angle follows the pitch's vertical movement surprise
    (a hitter who keeps steering later is thrown off less, a smaller slope). Measured on 2023-2024, checked for
    repeatability (odd and even days), against public bat speed and swing length, and against 2025-2026 contact
    outcomes with the 2023-2024 values of the same outcomes as controls."""
    res = {}
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    key_sub = (T['pitcher'] * 10 + (T['season'] - 2020)) * 100 + T['sub']
    ok = np.isfinite(F['az'])
    mz = loo_means(key_sub, F['az'], ok)
    s_in = 0.5 * np.where(np.isfinite(mz), F['az'] - mz, np.nan) * 12.0
    bip = np.isfinite(T['la']) & np.isfinite(s_in) & (T['ls'] > 40)
    idx = np.flatnonzero(bip)
    oh = np.eye(7)[np.clip(T['group'][idx], 0, 6)]
    cnt = np.zeros((len(idx), 12)); cnt[np.arange(len(idx)), np.clip(T['balls'][idx], 0, 3) * 3 + np.clip(T['strikes'][idx], 0, 2)] = 1
    uu = np.where(T['stand_r'][idx] == 1, T['px'][idx], -T['px'][idx])
    X = np.hstack([oh, cnt, hats(T['pz'][idx].astype(float), Z_KNOTS), hats(uu.astype(float), U_KNOTS), hats(T['v0'][idx].astype(float), V_KNOTS)])
    y = T['la'][idx].astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ beta
    s = s_in[idx]; bat = T['batter'][idx]; day = T['day'][idx]; season = T['season'][idx]
    dev = season <= 2024
    ub, inv = np.unique(bat, return_inverse=True)
    def slopes(mask):
        n = np.bincount(inv[mask], minlength=len(ub)).astype(float)
        sx = np.bincount(inv[mask], weights=s[mask], minlength=len(ub)); sy = np.bincount(inv[mask], weights=r[mask], minlength=len(ub))
        sxx = np.bincount(inv[mask], weights=s[mask] ** 2, minlength=len(ub)); sxy = np.bincount(inv[mask], weights=s[mask] * r[mask], minlength=len(ub))
        syy = np.bincount(inv[mask], weights=r[mask] ** 2, minlength=len(ub))
        with np.errstate(invalid='ignore', divide='ignore'):
            vx = sxx / n - (sx / n) ** 2; cxy = sxy / n - (sx / n) * (sy / n); b = cxy / vx
            resid_var = (syy / n - (sy / n) ** 2) - b * cxy
            se = np.sqrt(resid_var / (n * vx))
        return b, se, n
    b_all, se_all, n_all = slopes(dev)
    b_odd, _, _ = slopes(dev & (day % 2 == 1)); b_even, _, _ = slopes(dev & (day % 2 == 0))
    sel = n_all >= int(params.get('min_bip', 500))
    mean = float(np.average(b_all[sel], weights=1 / se_all[sel] ** 2)); var_obs = float(np.var(b_all[sel])); var_noise = float(np.mean(se_all[sel] ** 2))
    var_true = max(var_obs - var_noise, 0.0)
    shrunk = mean + (var_true / (var_true + se_all ** 2)) * (b_all - mean)
    res['hitters'] = int(sel.sum())
    res['slope'] = {'pooled': round(mean, 4), 'sd_observed': round(float(np.sqrt(var_obs)), 4), 'sd_noise': round(float(np.sqrt(var_noise)), 4),
                    'sd_true': round(float(np.sqrt(var_true)), 4), 'split_half': round(float(np.corrcoef(b_odd[sel], b_even[sel])[0, 1]), 4)}
    # outcomes per hitter (all pitches, not just balls in play)
    swing = (T['call'] == 1) | (T['call'] == 2); whiff = T['call'] == 2
    allb = T['batter']; ia = np.searchsorted(ub, allb); okb = (ia < len(ub)) & (ub[np.clip(ia, 0, len(ub) - 1)] == allb)
    def rate(num, den, period):
        m = okb & period
        nn = np.bincount(ia[m], weights=num[m].astype(float), minlength=len(ub)); dd = np.bincount(ia[m], weights=den[m].astype(float), minlength=len(ub))
        with np.errstate(invalid='ignore', divide='ignore'):
            return nn / dd
    la_all = T['la']; ls_all = T['ls']; inplay = np.isfinite(la_all) & (ls_all > 40)
    sweet = inplay & (la_all >= 8) & (la_all <= 32); hard = inplay & (ls_all >= 95)
    outs = {}
    for tag, period in (('dev', T['season'] <= 2024), ('fut', T['season'] >= 2025)):
        outs[tag] = {'whiff': rate(whiff, swing, period), 'sweet': rate(sweet, inplay, period), 'hard': rate(hard, inplay, period),
                     'la_sd': None}
        m = okb & period & inplay
        n_ = np.bincount(ia[m], minlength=len(ub)).astype(float); s1 = np.bincount(ia[m], weights=la_all[m], minlength=len(ub)); s2 = np.bincount(ia[m], weights=la_all[m] ** 2, minlength=len(ub))
        with np.errstate(invalid='ignore', divide='ignore'):
            outs[tag]['la_sd'] = np.sqrt(s2 / n_ - (s1 / n_) ** 2)
    from scipy import stats
    res['corr_with_dev'] = {k: round(float(stats.spearmanr(shrunk[sel], v[sel], nan_policy='omit')[0]), 4) for k, v in outs['dev'].items()}
    inc = {}
    X0 = np.column_stack([outs['dev'][k][sel] for k in ('whiff', 'sweet', 'hard', 'la_sd')])
    for fk in ('whiff', 'sweet', 'hard', 'la_sd'):
        yy = outs['fut'][fk][sel]; okr = np.isfinite(yy) & np.isfinite(X0).all(axis=1)
        Xa = np.column_stack([np.ones(okr.sum()), X0[okr]]); Xb = np.column_stack([Xa, shrunk[sel][okr]])
        def fit(Xm):
            bb, *_ = np.linalg.lstsq(Xm, yy[okr], rcond=None); e = yy[okr] - Xm @ bb
            H = Xm @ np.linalg.pinv(Xm.T @ Xm) @ Xm.T
            return 1 - e.var() / yy[okr].var(), float(np.mean((e / (1 - np.diag(H))) ** 2)), bb
        ra, la_, _ = fit(Xa); rb, lb, bb = fit(Xb)
        rr = np.random.default_rng(5); boots = []
        for _ in range(500):
            pick = rr.integers(0, okr.sum(), okr.sum()); c, *_ = np.linalg.lstsq(Xb[pick], yy[okr][pick], rcond=None); boots.append(c[-1])
        sd = float(np.std(shrunk[sel][okr]))
        inc[fk] = {'r2': [round(float(ra), 4), round(float(rb), 4)], 'loo_x1e5': [round(la_ * 1e5, 3), round(lb * 1e5, 3)], 'per_sd': round(float(bb[-1] * sd), 5),
                   'ci': [round(float(np.percentile(boots, 2.5) * sd), 5), round(float(np.percentile(boots, 97.5) * sd), 5)], 'n': int(okr.sum())}
    res['incremental'] = inc
    bt, notes = bat_tracking()
    res['bat_tracking_fetch'] = {k: {kk: vv for kk, vv in v.items() if kk != 'columns'} for k, v in notes.items()}
    pairs = []
    for j in np.flatnonzero(sel):
        seasons_bt = bt.get(int(ub[j])) or {}
        sp = [v['bat_speed'] for v in seasons_bt.values() if v.get('bat_speed')]; ln = [v['swing_length'] for v in seasons_bt.values() if v.get('swing_length')]
        if sp:
            pairs.append((shrunk[j], np.mean(sp), np.mean(ln) if ln else np.nan))
    if len(pairs) > 20:
        A = np.array(pairs)
        res['bat_tracking'] = {'n': len(pairs), 'bat_speed_spearman': round(float(stats.spearmanr(A[:, 0], A[:, 1])[0]), 4),
                               'swing_length_spearman': round(float(stats.spearmanr(A[np.isfinite(A[:, 2]), 0], A[np.isfinite(A[:, 2]), 2])[0]), 4)}
    order = np.argsort(shrunk[sel])
    ids = ub[sel]
    res['examples'] = {'latest_steering': [{'batter': int(ids[i]), 'slope': round(float(shrunk[sel][i]), 4)} for i in order[:12]],
                       'earliest_steering': [{'batter': int(ids[i]), 'slope': round(float(shrunk[sel][i]), 4)} for i in order[-12:][::-1]]}
    return res


EU_KNOTS = (-0.8, -0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5)


def batter_zones(T: dict, ok: np.ndarray, prior_n: float = 100.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Each batter's strike-zone top and bottom (ft) from the official zone numbers, which the feed assigns with the
    batter's own zone (1-9 inside, top row 1-3, bottom row 7-9; 11-14 outside quadrants): near the middle of the
    plate, the height that best separates the top row from pitches above it, and the bottom row from pitches below
    it, pulled toward the league value by prior_n pitches. Returns per-pitch top and bottom and a summary."""
    mid = ok & (np.abs(T['px']) < 0.5) & np.isfinite(T['pz'])
    z = T['pz'].astype(np.float64); zn = T['zone']
    sets = {'top': (mid & np.isin(zn, (1, 2, 3)), mid & np.isin(zn, (11, 12)) & (z > 2.5)),
            'bot': (mid & np.isin(zn, (13, 14)) & (z < 2.5), mid & np.isin(zn, (7, 8, 9)))}   # (below, above) the boundary

    def boundary(below, above):
        if len(below) == 0 or len(above) == 0:
            return np.nan
        cand = np.unique(np.concatenate([below, above]))
        b = np.sort(below); a = np.sort(above)
        err = (len(b) - np.searchsorted(b, cand, side='right')) + np.searchsorted(a, cand, side='left')
        best = cand[err == err.min()]
        return float(np.median(best))
    out = {}
    summary = {}
    for name, (lo_m, hi_m) in sets.items():
        league = boundary(z[lo_m], z[hi_m])
        val = np.full(len(z), league)
        bat = T['batter']
        ub = np.unique(bat[lo_m | hi_m])
        lo_i = np.flatnonzero(lo_m); hi_i = np.flatnonzero(hi_m)
        lo_b = bat[lo_i]; hi_b = bat[hi_i]
        o_lo = np.argsort(lo_b, kind='stable'); o_hi = np.argsort(hi_b, kind='stable')
        lo_s, hi_s = lo_b[o_lo], hi_b[o_hi]
        est = {}
        for b_ in ub:
            a0, a1 = np.searchsorted(lo_s, b_, 'left'), np.searchsorted(lo_s, b_, 'right')
            c0, c1 = np.searchsorted(hi_s, b_, 'left'), np.searchsorted(hi_s, b_, 'right')
            zl = z[lo_i[o_lo[a0:a1]]]; zh = z[hi_i[o_hi[c0:c1]]]
            n = min(len(zl), len(zh))
            if n < 10:
                continue
            c = boundary(zl, zh)
            est[int(b_)] = (n * c + prior_n * league) / (n + prior_n)
        keys = np.array(sorted(est)); vals = np.array([est[k] for k in keys])
        if len(keys):
            val = lookup(keys, vals, bat, league)
        out[name] = val
        summary[name] = {'league_ft': round(league, 4), 'batters': int(len(keys)),
                         'batter_p10_p50_p90': [round(float(v), 3) for v in np.percentile(vals, (10, 50, 90))] if len(vals) else None}
    return out['top'], out['bot'], summary


def horizon13(T: dict, params: dict, stage) -> dict:
    """Umpires as a negative control for the decision horizon, and the umpire's own window
    (discovery/DECISION_HORIZON_PROTOCOL.md, addendum 11). The same instrument as the decisive test (pitch-type location
    maps, within-type movement surprise, displacement along the decision gradient) applied to called strikes on taken
    pitches. If the hitters' horizon came from tracking error (the measured crossing and the measured movement erring
    together), the umpires' calls would show it too, more sharply. Heights are standardized with each batter's own
    zone, recovered from the official zone numbers. A second term moves the crossing along the ball's velocity at the
    plate (positive: the call follows the ball past the front of the plate toward the glove). Hitters' swings are run
    through the identical code for comparison."""
    res = {}
    F = rebuild(T)
    ok = (F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
          & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1)) & (T['call'] <= 2))
    top, bot, zsum = batter_zones(T, ok)
    res['batter_zones'] = zsum
    T = take(T, ok); F = {k: v[ok] for k, v in F.items()}; top, bot = top[ok], bot[ok]
    stage('batter zones')
    ps = T['pitcher'] * 10 + (T['season'] - 2020)
    key_sub = ps * 100 + T['sub']
    okx = np.isfinite(F['ax']) & np.isfinite(F['az'])
    mx = loo_means(key_sub, F['ax'], okx); mz = loo_means(key_sub, F['az'], okx)
    wx = np.where(np.isfinite(mx), F['ax'] - mx, 0.0); wz = np.where(np.isfinite(mz), F['az'] - mz, 0.0)
    vxp = F['vx0'] + F['ax'] * F['tf']; vzp = F['vz0'] + F['az'] * F['tf']
    res['plate_velocity_ft_s'] = {'vx_sd': round(float(np.std(vxp)), 3), 'vz_mean': round(float(np.mean(vzp)), 3), 'vz_sd': round(float(np.std(vzp)), 3)}
    take_m = T['call'] == 0
    res['taken'] = {'pitches': int(take_m.sum()), 'called_strike_share': round(float(T['cs'][take_m].mean()), 4),
                    'zone_missing_share': round(float(np.mean(T['zone'] < 0)), 4)}
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]
    n = len(T['balls'])
    cnt = np.zeros((n, 12), np.float32); cnt[np.arange(n), np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)] = 1
    C_ump = np.hstack([cnt, oh, hats(T['v0'].astype(np.float64), V_KNOTS), T['stand_r'][:, None].astype(np.float32),
                       (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)])
    C_hit = control_block(T, swing_propensity(T))
    span = np.clip(top - bot, 1.2, 3.0)

    def design(x, z, idx, Cm):
        u = np.where(T['stand_r'][idx] == 1, x[idx], -x[idx])
        zz = 1.5 + 2.0 * (z[idx] - bot[idx]) / span[idx]
        e = np.maximum(np.maximum(np.abs(u) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
        base = np.hstack([hats(e, EU_KNOTS), hats(u, U_KNOTS), hats(zz, Z_KNOTS)])
        blocks = [base * (T['strikes'][idx] == k)[:, None] for k in (0, 1, 2)] + [base * (T['balls'][idx] == 3)[:, None]]
        inter = (base[:, :, None] * oh[idx, None, :]).reshape(len(idx), -1)
        return np.hstack(blocks + [inter, Cm[idx]]).astype(np.float32)

    rng = np.random.default_rng(int(params.get('seed', 13)))
    pops = {'umpire': (take_m, T['cs'].astype(np.int64), C_ump), 'hitter': (np.ones(n, bool), ((T['call'] == 1) | (T['call'] == 2)).astype(np.int64), C_hit)}
    grid = [float(v) for v in params.get('tau2_grid', (-0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.0684))]
    h = 0.02
    for pname, (pm, y, Cm) in pops.items():
        if pname not in params.get('populations', ('umpire', 'hitter')):
            continue
        def sample(seasons, k):
            ii = np.flatnonzero(pm & np.isin(T['season'], seasons)); return np.sort(rng.choice(ii, min(len(ii), k), replace=False))
        tr = sample((2023, 2024), int(params.get('train_n', 500000)))
        va = sample((2025,), int(params.get('val_n', 400000)))
        te = sample((2026,), int(params.get('test_n', 400000)))
        out = {'split': {'train': int(len(tr)), 'select_2025': int(len(va)), 'test_2026': int(len(te))}}
        D = lambda x, z, idx: design(x, z, idx, Cm)
        m0 = fit_logistic(D(T['px'], T['pz'], tr), y[tr])
        base_ll = {nm: logloss_vec(m0.predict_proba(D(T['px'], T['pz'], idx))[:, 1], y[idx]) for nm, idx in (('select_2025', va), ('test_2026', te))}
        prof = []
        for t2 in grid:
            if t2 == 0.0:
                prof.append({'tau2': 0.0, 'select_2025': float(base_ll['select_2025'].mean()), 'test_2026': float(base_ll['test_2026'].mean())})
                continue
            x, z = T['px'] - 0.5 * wx * t2, T['pz'] - 0.5 * wz * t2
            m = fit_logistic(D(x, z, tr), y[tr])
            r = {'tau2': t2}
            for nm, idx in (('select_2025', va), ('test_2026', te)):
                ll = logloss_vec(m.predict_proba(D(x, z, idx))[:, 1], y[idx]); r[nm] = float(ll.mean())
                if nm == 'test_2026':
                    r['delta_2026_per_1000'] = [round(v * 1000, 4) for v in clustered_ci(ll - base_ll['test_2026'], T['game'][idx], reps=200)]
            prof.append(r)
        out['profile'] = prof
        stage(pname + ' profile')
        # gradient estimates: within-type surprise (tau squared) and plate velocity (seconds past the front of the plate)
        est = {}
        for nm, idx in (('select_2025', va), ('test_2026', te)):
            f0 = np.empty(len(idx)); fx = np.empty(len(idx)); fz = np.empty(len(idx))
            for s0 in range(0, len(idx), 200000):
                ii = idx[s0:s0 + 200000]; df = lambda x, z: m0.decision_function(D(x, z, ii))
                f0[s0:s0 + 200000] = df(T['px'], T['pz'])
                fx[s0:s0 + 200000] = (df(T['px'] + h, T['pz']) - df(T['px'] - h, T['pz'])) / (2 * h)
                fz[s0:s0 + 200000] = (df(T['px'], T['pz'] + h) - df(T['px'], T['pz'] - h)) / (2 * h)
            yy = y[idx].astype(float)
            g_w = -0.5 * (fx * wx[idx] + fz * wz[idx]); g_v = fx * vxp[idx] + fz * vzp[idx]
            perm = rng.permutation(len(idx)); g_shuf = -0.5 * (fx * wx[idx][perm] + fz * wz[idx][perm])
            a, b, se = offset_logit(yy, f0, g_w)
            lo, hi = boot_slope(yy, f0, g_w, T['game'][idx], reps=int(params.get('boot', 100)))
            coef, se2 = offset_logit_k(yy, f0, np.column_stack([g_w, g_v]))
            _, bs, ses = offset_logit(yy, f0, g_shuf)
            row = {'tau2_within': round(b, 6), 'se': round(se, 6), 'ci': [round(lo, 6), round(hi, 6)],
                   'tau_ms': round(float(np.sign(b) * np.sqrt(abs(b)) * 1000), 1),
                   'joint': {'tau2_within': round(float(coef[0]), 6), 'se_tau2': round(float(se2[0]), 6),
                             'delta_s': round(float(coef[1]), 5), 'se_delta': round(float(se2[1]), 5)},
                   'shuffled_surprise': {'tau2': round(bs, 6), 'se': round(ses, 6)}, 'pitches': int(len(idx)),
                   'mean_abs_gradient_per_ft': round(float(np.mean(np.sqrt(fx ** 2 + fz ** 2))), 3)}
            if pname == 'umpire':
                by = {}
                for gi in (0, 1, 3, 4, 5):
                    mk = T['group'][idx] == gi
                    if mk.sum() > 2000:
                        _, bg, sg = offset_logit(yy[mk], f0[mk], g_w[mk]); by[GROUP_NAMES[gi]] = {'tau2': round(bg, 6), 'se': round(sg, 6), 'n': int(mk.sum())}
                row['by_type'] = by
                edge = np.abs(f0) < 2.0
                _, be, sge = offset_logit(yy[edge], f0[edge], g_w[edge]); row['near_the_edge'] = {'tau2': round(be, 6), 'se': round(sge, 6), 'n': int(edge.sum())}
            est[nm] = row
        out['gradient'] = est
        stage(pname + ' gradient')
        res[pname] = out
    return res


def _wls(y, X, w):
    """Weighted least squares with an intercept; returns coefficients (intercept first) and R squared."""
    A = np.column_stack([np.ones(len(y)), X]); sw = np.sqrt(w)
    beta = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    r = y - A @ beta; ybar = np.average(y, weights=w)
    return beta, float(1 - np.sum(w * r ** 2) / np.sum(w * (y - ybar) ** 2))


def horizon14(T: dict, params: dict, stage) -> dict:
    """Pitcher-level decision-moment tunneling (discovery/DECISION_HORIZON_PROTOCOL.md, addendum 12). For every
    consecutive pair of pitches of different types in a plate appearance, the visual-angle separation of the two
    flights from the batter's eye 260 ms and 175 ms before each reaches the plate, and at the plate. A pitcher-season's
    tunneling is the average early separation of his different-type pairs. The outcome is chase above expected: his
    swing rate on pitches outside the zone minus a league model's expectation for those pitches (location, count,
    pitch type, speed, batter's prior swing rate). Tests: reliability (odd against even days, season to season);
    cross-sample validity (tunneling on odd days against chase above expected on even days, with plate separation
    and stuff held fixed); 260 against 175 ms; next season's chase above expected beyond this season's."""
    res = {}
    F = rebuild(T)
    ok = (F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
          & (T['call'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1)))
    order = np.lexsort((T['pitch_no'], T['ab'], T['game']))
    prev = np.full(len(order), -1)
    same = (T['game'][order][1:] == T['game'][order][:-1]) & (T['ab'][order][1:] == T['ab'][order][:-1]) & (T['pitch_no'][order][1:] == T['pitch_no'][order][:-1] + 1)
    prev[order[1:][same]] = order[:-1][same]
    has_prev = (prev >= 0) & ok & np.where(prev >= 0, ok[np.maximum(prev, 0)], False)
    pi = np.flatnonzero(has_prev); pj = prev[pi]
    eye_x = np.where(T['stand_r'] == 1, -2.4, 2.4); eye_y, eye_z = 1.0, 5.1

    def angles(i, tau):
        if tau > 0:
            x, y, z = position_at(take(T, i), {k: v[i] for k, v in F.items()}, tau)
        else:
            x, y, z = T['px'][i].astype(float), np.full(len(i), YPLATE), T['pz'][i].astype(float)
        d = np.maximum(y - eye_y, 0.5)
        return np.degrees(np.arctan2(x - eye_x[i], d)), np.degrees(np.arctan2(z - eye_z, d))
    sep = {}
    for name, tau in (('s260', 0.26), ('s175', 0.175), ('plate', 0.0)):
        a1, b1 = angles(pi, tau); a2, b2 = angles(pj, tau)
        sep[name] = np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)
    diff = T['group'][pi] != T['group'][pj]
    res['pairs'] = {'all': int(len(pi)), 'different_types': int(diff.sum()),
                    'median_deg_different': {k: round(float(np.median(v[diff])), 3) for k, v in sep.items()},
                    'median_deg_same': {k: round(float(np.median(v[~diff])), 3) for k, v in sep.items()}}
    stage('pairs')
    # league expectation of a chase for every pitch outside the zone
    u_all = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    inzone = (np.abs(u_all) <= ZONE_HALF) & (T['pz'] <= ZONE_TOP) & (T['pz'] >= ZONE_BOT)
    swing = (T['call'] == 1) | (T['call'] == 2)
    ooz = np.flatnonzero(ok & ~inzone)
    C = control_block(T, swing_propensity(T))
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]

    def design(i):
        uu = u_all[i]; zz = T['pz'][i]
        e = np.maximum(np.maximum(np.abs(uu) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
        b = np.hstack([hats(e, E_KNOTS), hats(uu, U_KNOTS), hats(zz, Z_KNOTS)])
        return np.hstack([location_block(T['px'][i], T['pz'][i], T['stand_r'][i], T['strikes'][i]), (b[:, :, None] * oh[i, None, :]).reshape(len(i), -1), C[i]]).astype(np.float32)
    rng = np.random.default_rng(int(params.get('seed', 14)))
    trn = ooz[np.isin(T['season'][ooz], (2023, 2024))]
    trn = np.sort(rng.choice(trn, min(len(trn), int(params.get('train_n', 500000))), replace=False))
    m = fit_logistic(design(trn), swing[trn].astype(np.int64))
    p_exp = np.full(len(T['px']), np.nan)
    for s0 in range(0, len(ooz), 300000):
        ii = ooz[s0:s0 + 300000]; p_exp[ii] = m.predict_proba(design(ii))[:, 1]
    stage('expected chase')
    # pitcher-season (and odd or even day) aggregates
    key = T['pitcher'] * 10 + (T['season'] - 2020)
    half = (T['day'] % 2).astype(int)
    fb = ok & np.isin(T['group'], (0, 1))

    def agg(mask_pitch, mask_pair):
        k_p = key[mask_pitch]; uk = np.unique(k_p)
        out = {int(k): {} for k in uk}
        def mean_by(keys, vals, name, w=None):
            u, inv = np.unique(keys, return_inverse=True)
            sm = np.bincount(inv, weights=vals if w is None else vals * w); cn = np.bincount(inv, weights=None if w is None else w)
            for kk, s_, c_ in zip(u, sm, cn):
                if int(kk) in out:
                    out[int(kk)][name] = float(s_ / c_) if c_ > 0 else np.nan; out[int(kk)]['n_' + name] = float(c_)
        oo = mask_pitch & ~inzone
        mean_by(key[oo], swing[oo].astype(float), 'chase'); mean_by(key[oo], p_exp[oo], 'chase_exp')
        mean_by(key[mask_pitch], inzone[mask_pitch].astype(float), 'zone')
        sw = mask_pitch & swing; mean_by(key[sw], (T['call'][sw] == 2).astype(float), 'whiff')
        fbm = mask_pitch & fb; mean_by(key[fbm], T['v0'][fbm].astype(float), 'fb_velo'); mean_by(key[fbm], F['asz'][fbm], 'fb_rise')
        brk = mask_pitch & np.isin(T['group'], (3, 4)); mean_by(key[brk], np.abs(F['asx'][brk]), 'brk_sweep')
        pm = mask_pair & diff
        for nm in ('s260', 's175', 'plate'):
            mean_by(key[pi][pm], sep[nm][pm], nm)
        mean_by(key[pi][mask_pair], diff[mask_pair].astype(float), 'share_different')
        return out
    full = agg(ok, np.ones(len(pi), bool))
    halves = [agg(ok & (half == h), half[pi] == h) for h in (0, 1)]
    stage('aggregates')
    min_ooz, min_pairs = int(params.get('min_ooz', 400)), int(params.get('min_pairs', 300))
    cols = ['s260', 's175', 'plate', 'share_different', 'fb_velo', 'fb_rise', 'brk_sweep', 'zone', 'chase', 'chase_exp']

    def table(dct, scale=1.0):
        rows = []
        for k, v in dct.items():
            if v.get('n_chase', 0) >= min_ooz * scale and v.get('n_s260', 0) >= min_pairs * scale and all(np.isfinite(v.get(c, np.nan)) for c in cols if c != 'brk_sweep'):
                rows.append([k] + [v.get(c, np.nan) for c in cols] + [v['n_chase'], v['n_s260']])
        return np.array(rows, float)
    A = table(full); H0 = table(halves[0], 0.5); H1 = table(halves[1], 0.5)
    idx = {c: i + 1 for i, c in enumerate(cols)}; idx['n_chase'] = len(cols) + 1; idx['n_pairs'] = len(cols) + 2
    res['pitcher_seasons'] = int(len(A))

    def corr_join(X, Y, c):
        kx = {int(r[0]): r for r in X}; common = [k for k in kx if k in {int(r[0]) for r in Y}]
        ky = {int(r[0]): r for r in Y}
        a = np.array([kx[k][idx[c]] for k in common]); b = np.array([ky[k][idx[c]] for k in common])
        ok_ = np.isfinite(a) & np.isfinite(b)
        return round(float(np.corrcoef(a[ok_], b[ok_])[0, 1]), 4), int(ok_.sum())
    rel = {}
    for c in ('s260', 's175', 'plate', 'chase', 'chase_exp'):
        rel[c] = corr_join(H0, H1, c)
    cae = lambda R: R[:, idx['chase']] - R[:, idx['chase_exp']]
    k0 = {int(r[0]): r for r in H0}; k1 = {int(r[0]): r for r in H1}; common = sorted(set(k0) & set(k1))
    a = np.array([cae(k0[k][None, :])[0] for k in common]); b = np.array([cae(k1[k][None, :])[0] for k in common])
    rel['chase_above_expected'] = (round(float(np.corrcoef(a, b)[0, 1]), 4), len(common))
    # season to season
    kA = {int(r[0]): r for r in A}
    nxt = [(k, k + 1) for k in kA if (k + 1) in kA and (k % 10) + 2020 < 2026]
    yy = {}
    for c in ('s260', 's175', 'plate'):
        x1 = np.array([kA[k][idx[c]] for k, _ in nxt]); x2 = np.array([kA[k2][idx[c]] for _, k2 in nxt]); yy[c] = round(float(np.corrcoef(x1, x2)[0, 1]), 4)
    x1 = np.array([cae(kA[k][None, :])[0] for k, _ in nxt]); x2 = np.array([cae(kA[k2][None, :])[0] for _, k2 in nxt]); yy['chase_above_expected'] = round(float(np.corrcoef(x1, x2)[0, 1]), 4)
    res['reliability'] = {'odd_vs_even_days': rel, 'season_to_season': yy, 'pairs_of_seasons': len(nxt)}
    stage('reliability')
    # cross-sample validity: tunneling on one half, chase above expected on the other (both directions pooled)
    def zs(v, w):
        m_ = np.average(v, weights=w); s_ = np.sqrt(np.average((v - m_) ** 2, weights=w)); return (v - m_) / s_
    rows = []
    for src, dst in ((k0, k1), (k1, k0)):
        for k in common:
            r_s, r_d = src[k], dst[k]
            if not np.isfinite(r_s[idx['brk_sweep']]):
                r_s = r_s.copy(); r_s[idx['brk_sweep']] = 0.0
            rows.append([k, cae(r_d[None, :])[0], r_s[idx['s260']], r_s[idx['s175']], r_s[idx['plate']], r_s[idx['share_different']], r_s[idx['fb_velo']], r_s[idx['fb_rise']],
                         r_s[idx['brk_sweep']], r_s[idx['zone']], min(r_s[idx['n_chase']], r_d[idx['n_chase']])])
    V = np.array(rows, float); w = V[:, -1]
    names = ['s260', 's175', 'plate', 'share_different', 'fb_velo', 'fb_rise', 'brk_sweep', 'zone']
    Z = np.column_stack([zs(V[:, 2 + j], w) for j in range(len(names))]); y = V[:, 1] * 100     # points of chase rate
    specs = {'s260': ['s260', 'plate', 'share_different', 'fb_velo', 'fb_rise', 'brk_sweep', 'zone'],
             's175': ['s175', 'plate', 'share_different', 'fb_velo', 'fb_rise', 'brk_sweep', 'zone'],
             'both': ['s260', 's175', 'plate', 'share_different', 'fb_velo', 'fb_rise', 'brk_sweep', 'zone'],
             'controls_only': ['plate', 'share_different', 'fb_velo', 'fb_rise', 'brk_sweep', 'zone']}
    pk = V[:, 0].astype(np.int64) // 10; up, inv = np.unique(pk, return_inverse=True)
    boot_rng = np.random.default_rng(15)
    cross = {}
    for nm, sp_ in specs.items():
        X = Z[:, [names.index(c) for c in sp_]]
        beta, r2 = _wls(y, X, w)
        bs = []
        for _ in range(int(params.get('boot', 300))):
            cnt = np.bincount(boot_rng.integers(0, len(up), len(up)), minlength=len(up)).astype(float)[inv]
            bs.append(_wls(y, X, w * cnt)[0][1:])
        bs = np.array(bs)
        cross[nm] = {'r2': round(r2, 4), 'coef_points_per_sd': {c: [round(float(beta[1 + j]), 3), round(float(np.percentile(bs[:, j], 2.5)), 3), round(float(np.percentile(bs[:, j], 97.5)), 3)] for j, c in enumerate(sp_)}}
    cross['n_pitcher_seasons'] = int(len(common)); cross['corr_s260_s175'] = round(float(np.corrcoef(V[:, 2], V[:, 3])[0, 1]), 4)
    cross['corr_s260_plate'] = round(float(np.corrcoef(V[:, 2], V[:, 4])[0, 1]), 4)
    res['cross_sample'] = cross
    stage('cross-sample validity')
    # next season: chase above expected in t+1 on this season's chase above expected plus tunneling
    rows = []
    for k, k2 in nxt:
        r1, r2_ = kA[k], kA[k2]
        rows.append([cae(r2_[None, :])[0] * 100, cae(r1[None, :])[0] * 100, r1[idx['s260']], r1[idx['s175']], r1[idx['plate']], r1[idx['share_different']], r1[idx['fb_velo']], r1[idx['fb_rise']],
                     (r1[idx['brk_sweep']] if np.isfinite(r1[idx['brk_sweep']]) else 0.0), r1[idx['zone']], min(r1[idx['n_chase']], r2_[idx['n_chase']]), k // 10])
    N_ = np.array(rows, float); w2 = N_[:, -2]; y2 = N_[:, 0]
    feats = ['cae_now', 's260', 's175', 'plate', 'share_different', 'fb_velo', 'fb_rise', 'brk_sweep', 'zone']
    Z2 = np.column_stack([zs(N_[:, 1 + j], w2) for j in range(len(feats))])
    up2, inv2 = np.unique(N_[:, -1].astype(np.int64), return_inverse=True)
    nxt_out = {}
    base_cols = ['cae_now', 'plate', 'share_different', 'fb_velo', 'fb_rise', 'brk_sweep', 'zone']
    for nm, extra in (('base', []), ('plus_s260', ['s260']), ('plus_s175', ['s175'])):
        cs = base_cols + extra
        X = Z2[:, [feats.index(c) for c in cs]]
        beta, r2 = _wls(y2, X, w2)
        bs, gains = [], []
        Xb_ = Z2[:, [feats.index(c) for c in base_cols]]
        for _ in range(int(params.get('boot', 300))):
            cnt = np.bincount(boot_rng.integers(0, len(up2), len(up2)), minlength=len(up2)).astype(float)[inv2]
            b_ = _wls(y2, X, w2 * cnt); bs.append(b_[0][1:])
            if extra:
                gains.append(b_[1] - _wls(y2, Xb_, w2 * cnt)[1])
        bs = np.array(bs)
        row = {'r2': round(r2, 4), 'coef_points_per_sd': {c: [round(float(beta[1 + j]), 3), round(float(np.percentile(bs[:, j], 2.5)), 3), round(float(np.percentile(bs[:, j], 97.5)), 3)] for j, c in enumerate(cs)}}
        if extra:
            row['r2_gain'] = [round(float(np.mean(gains)), 4), round(float(np.percentile(gains, 2.5)), 4), round(float(np.percentile(gains, 97.5)), 4)]
        nxt_out[nm] = row
    nxt_out['n_pairs_of_seasons'] = int(len(N_))
    res['next_season'] = nxt_out
    stage('next season')
    # 2026 leaderboard (per-pitcher summaries only)
    lb = A[(A[:, 0] % 10) == 6]
    lb = lb[lb[:, idx['n_pairs']] >= 600]
    rows = [{'p': int(r[0]) // 10, 's260': round(float(r[idx['s260']]), 3), 's175': round(float(r[idx['s175']]), 3), 'plate': round(float(r[idx['plate']]), 3),
             'chase': round(float(r[idx['chase']]), 4), 'chase_exp': round(float(r[idx['chase_exp']]), 4), 'pairs': int(r[idx['n_pairs']])} for r in lb]
    res['leaderboard_2026'] = sorted(rows, key=lambda r: r['s260'])
    return res


# ---------------------------------------------------------------- ball-strike challenges (2026, from the official play-by-play)
ROLE = {'none': 0, 'batter': 1, 'catcher': 2, 'pitcher': 3}


def challenge_rows(doc: dict) -> list:
    """Study-shaped rows of one game's play-by-play with five extra fields on every pitch: the challenger's role (0 none,
    1 batter, 2 catcher, 3 pitcher), overturned (0/1), the original call (1 strike, 0 ball, -1 not a called pitch), and the
    batting and the fielding team's failed challenges before this pitch in the game (every pitch, challenged or not)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('brl_backfill_for_challenges', ROOT / 'tools' / 'brl_bookkeeping_backfill.py')
    bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
    rows, failed = [], {}
    for play in (doc or {}).get('allPlays') or []:
        about = play.get('about') or {}
        if not about.get('isComplete'):
            continue
        m = play.get('matchup') or {}
        batter = (m.get('batter') or {}).get('id'); pitcher = (m.get('pitcher') or {}).get('id')
        if batter is None or pitcher is None:
            continue
        ph = bf.physics(play)
        pitch_events = [e for e in play.get('playEvents') or [] if e.get('isPitch') is True]
        if len(pitch_events) != len(ph['pitches']):
            continue
        extra = []
        bat_side = 'away' if about.get('isTopInning') else 'home'; fld_side = 'home' if bat_side == 'away' else 'away'
        for e, pt in zip(pitch_events, ph['pitches']):
            code = str((e.get('details') or {}).get('code') or '')
            balls0, strikes0 = pt[2], pt[3]
            c = e.get('count') or {}
            balls1, strikes1 = int(c.get('balls', balls0) or 0), int(c.get('strikes', strikes0) or 0)
            called = code in ('B', '*B', 'C')
            final = 1 if strikes1 > strikes0 else 0 if balls1 > balls0 else (1 if code == 'C' else 0)
            role, over, orig = 0, 0, (final if called else -1)
            bat_f, fld_f = failed.get(bat_side, 0), failed.get(fld_side, 0)
            rd = e.get('reviewDetails') or {}
            if called and rd and rd.get('reviewType') == 'MJ' and rd.get('player'):
                pid = (rd.get('player') or {}).get('id')
                role = 1 if pid == batter else 3 if pid == pitcher else 2
                over = 1 if rd.get('isOverturned') else 0
                orig = (1 - final) if over else final
                if not over:
                    side = bat_side if role == 1 else fld_side
                    failed[side] = failed.get(side, 0) + 1
            extra.append([role, over, orig, bat_f, fld_f])
        pitches = [list(pt) + ex for pt, ex in zip(ph['pitches'], extra)]
        rows.append({'i': int(about['atBatIndex']), 'inning': about.get('inning'), 'half': 'top' if about.get('isTopInning') else 'bottom',
                     'o': 'X', 'e': str((play.get('result') or {}).get('eventType') or ''), 'p': int(pitcher), 'b': int(batter),
                     's': str((m.get('batSide') or {}).get('code') or 'R'), 't': str((m.get('pitchHand') or {}).get('code') or ''),
                     'pitches': pitches, 'hit': ph['hit']})
    return rows


def challenge_study(season: int, workers: int = 6) -> dict:
    """The season's regular-season play-by-play as a study document whose pitches carry the challenge fields."""
    import importlib.util
    from concurrent.futures import ThreadPoolExecutor
    spec = importlib.util.spec_from_file_location('brl_backfill_for_challenges2', ROOT / 'tools' / 'brl_bookkeeping_backfill.py')
    bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
    games = [g for g in bf.completed_games(season, f'{season}-11-30') if g['game_type'] == 'R']

    def fetch(g):
        for attempt in range(3):
            try:
                return g, bf.get_json(f'{bf.API}/game/{g["game_pk"]}/playByPlay')
            except Exception:
                time.sleep(3 * (attempt + 1))
        return g, None
    out = {'schema': 'challenge-study', 'year': season, 'games': {}}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for g, doc in pool.map(fetch, games):
            if doc:
                out['games'][str(g['game_pk'])] = {'date': g['date'], 'game_type': 'R', 'rows': challenge_rows(doc)}
    return out


def challenge_columns(doc: dict) -> dict:
    """The five challenge fields in pitch_table's row order (same games, rows and skips)."""
    cols = {'role': [], 'over': [], 'orig': [], 'bat_failed': [], 'fld_failed': []}
    for gpk, game in (doc.get('games') or {}).items():
        if game.get('game_type') != 'R':
            continue
        for row in game.get('rows') or []:
            if row.get('b') is None or row.get('p') is None:
                continue
            for pt in row.get('pitches') or []:
                ex = (list(pt) + [None] * 20)[15:20]
                cols['role'].append(int(ex[0] or 0)); cols['over'].append(int(ex[1] or 0))
                cols['orig'].append(-1 if ex[2] is None else int(ex[2]))
                cols['bat_failed'].append(int(ex[3] or 0)); cols['fld_failed'].append(int(ex[4] or 0))
    return {k: np.asarray(v, np.int64) for k, v in cols.items()}


def challenges(T: dict, X: dict, params: dict, stage) -> dict:
    """Addendum 13: who sees the end of the pitch? The decision to challenge a called pitch, by batters (called strikes),
    catchers and pitchers (called balls), through the instrument of the umpire test: a location model with pitch-type maps
    and each batter's zone, and the decision displaced along the within-type late movement (tau squared; hitters' swings
    0.066, umpires' calls 0). Cross-fitted by game halves. Then batters' overturn rate by the late movement toward the zone."""
    res = {}
    F = rebuild(T)
    ok = (F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
          & (T['call'] == 0) & (X['orig'] >= 0))
    top, bot, zsum = batter_zones(T, F['ok'] & (T['group'] >= 0) & (T['call'] <= 2))
    res['batter_zones'] = zsum
    T = take(T, ok); F = {k: v[ok] for k, v in F.items()}; X = {k: v[ok] for k, v in X.items()}; top, bot = top[ok], bot[ok]
    stage('called pitches')
    ps = T['pitcher'] * 10 + (T['season'] - 2020)
    key_sub = ps * 100 + T['sub']
    okx = np.isfinite(F['ax']) & np.isfinite(F['az'])
    mx = loo_means(key_sub, F['ax'], okx); mz = loo_means(key_sub, F['az'], okx)
    wx = np.where(np.isfinite(mx), F['ax'] - mx, 0.0); wz = np.where(np.isfinite(mz), F['az'] - mz, 0.0)
    n = len(T['balls'])
    res['called'] = {'pitches': int(n), 'original_strikes': int((X['orig'] == 1).sum()),
                     'challenges': {r: int(((X['role'] == v)).sum()) for r, v in ROLE.items() if v},
                     'overturned': {r: round(float(X['over'][X['role'] == v].mean()), 4) if (X['role'] == v).any() else None for r, v in ROLE.items() if v}}
    oh = np.eye(7, dtype=np.float32)[np.clip(T['group'], 0, 6)]
    cnt = np.zeros((n, 12), np.float32); cnt[np.arange(n), np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)] = 1
    inn = np.zeros((n, 3), np.float32); inn[np.arange(n), np.where(T['inning'] <= 6, 0, np.where(T['inning'] <= 8, 1, 2))] = 1
    def left_block(failed):
        b_ = np.zeros((n, 3), np.float32); b_[np.arange(n), np.clip(2 - failed, 0, 2)] = 1; return b_
    common = np.hstack([cnt, oh, inn, T['stand_r'][:, None].astype(np.float32), (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)])
    Cm_by_role = {'batter': np.hstack([common, left_block(X['bat_failed'])]), 'catcher': np.hstack([common, left_block(X['fld_failed'])]),
                  'pitcher': np.hstack([common, left_block(X['fld_failed'])])}
    span = np.clip(top - bot, 1.2, 3.0)

    def edge(x, z, idx):
        u = np.where(T['stand_r'][idx] == 1, x[idx], -x[idx]); zz = 1.5 + 2.0 * (z[idx] - bot[idx]) / span[idx]
        return u, zz, np.maximum(np.maximum(np.abs(u) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)

    def design(x, z, idx, Cm):
        u, zz, e = edge(x, z, idx)
        base = np.hstack([hats(e, EU_KNOTS), hats(u, U_KNOTS), hats(zz, Z_KNOTS)])
        return np.hstack([base, (base[:, :, None] * oh[idx, None, :]).reshape(len(idx), -1), Cm[idx]]).astype(np.float32)

    halves = (T['game'] % 2 == 0)
    grid = [float(v) for v in params.get('tau2_grid', (0.0, 0.01, 0.02, 0.04, 0.0684, 0.1))]
    h = 0.02
    pops = {'batter': (X['orig'] == 1, X['role'] == ROLE['batter']), 'catcher': (X['orig'] == 0, X['role'] == ROLE['catcher']),
            'pitcher': (X['orig'] == 0, X['role'] == ROLE['pitcher'])}
    for pname, (pm, ych) in pops.items():
        y = ych.astype(np.int64)
        out = {'rows': int(pm.sum()), 'challenges': int(y[pm].sum())}
        if out['challenges'] < 50:
            res[pname] = out
            continue
        prof = {t2: [0.0, 0] for t2 in grid}
        est = []
        Cm = Cm_by_role[pname]
        for a_half in (True, False):
            tr = np.flatnonzero(pm & (halves == a_half)); te = np.flatnonzero(pm & (halves != a_half))
            C = float(params.get('C', 0.3))
            m0 = fit_logistic(design(T['px'], T['pz'], tr, Cm), y[tr], C=C)
            for t2 in grid:
                x, z = T['px'] - 0.5 * wx * t2, T['pz'] - 0.5 * wz * t2
                m = m0 if t2 == 0.0 else fit_logistic(design(x, z, tr, Cm), y[tr], C=C)
                ll = logloss_vec(m.predict_proba(design(x, z, te, Cm))[:, 1], y[te])
                prof[t2][0] += ll.sum(); prof[t2][1] += len(te)
            df = lambda x, z: m0.decision_function(design(x, z, te, Cm))
            f0 = df(T['px'], T['pz'])
            fx = (df(T['px'] + h, T['pz']) - df(T['px'] - h, T['pz'])) / (2 * h)
            fz = (df(T['px'], T['pz'] + h) - df(T['px'], T['pz'] - h)) / (2 * h)
            g_w = -0.5 * (fx * wx[te] + fz * wz[te])
            a, b, se = offset_logit(y[te].astype(float), f0, g_w)
            lo, hi = boot_slope(y[te].astype(float), f0, g_w, T['game'][te], reps=int(params.get('boot', 100)))
            est.append({'tau2': b, 'se': se, 'ci': [lo, hi], 'n': int(len(te)), 'challenges': int(y[te].sum())})
        p0 = prof[0.0][0] / prof[0.0][1]
        out['profile'] = [{'tau2': t2, 'nats_per_1000_vs_crossing': round((v[0] / v[1] - p0) * 1000, 4)} for t2, v in prof.items()]
        w_ = np.array([1 / max(e['se'], 1e-9) ** 2 for e in est]); b_ = np.array([e['tau2'] for e in est])
        pooled = float((w_ * b_).sum() / w_.sum()); se_p = float(1 / np.sqrt(w_.sum()))
        out['tau2'] = {'halves': [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()} for e in est],
                       'pooled': round(pooled, 6), 'se': round(se_p, 6), 'ci': [round(pooled - 1.96 * se_p, 6), round(pooled + 1.96 * se_p, 6)],
                       'tau_ms': round(float(np.sign(pooled) * np.sqrt(abs(pooled)) * 1000), 1)}
        res[pname] = out
        stage(pname)
    # Overturn rate by the late movement toward the zone: the edge distance of the pitch as it would look from 260 ms
    # minus at the crossing (positive: it moved late toward or into the zone)
    tau_h = float(params.get('tau_h', 0.2615))
    idx_all = np.arange(n)
    _, _, e0 = edge(T['px'], T['pz'], idx_all)
    _, _, eh = edge(T['px'] - 0.5 * wx * tau_h ** 2, T['pz'] - 0.5 * wz * tau_h ** 2, idx_all)
    inward = eh - e0
    for pname, v in (('batter', ROLE['batter']), ('catcher', ROLE['catcher']), ('pitcher', ROLE['pitcher'])):
        sel = np.flatnonzero(X['role'] == v)
        if len(sel) < 60:
            continue
        q = np.quantile(inward[sel], (1 / 3, 2 / 3))
        terc = np.digitize(inward[sel], q)
        rates = [[round(float(inward[sel][terc == k].mean() * 12), 2), round(float(X['over'][sel][terc == k].mean()), 4), int((terc == k).sum())] for k in range(3)]
        d = X['over'][sel][terc == 2].astype(float); c = X['over'][sel][terc == 0].astype(float)
        se = float(np.sqrt(d.var() / max(len(d), 1) + c.var() / max(len(c), 1)))
        res[pname + '_overturn_by_late_inward_inches'] = {'terciles_[inches, overturn rate, n]': rates,
                                                          'top_minus_bottom': round(float(d.mean() - c.mean()), 4), 'se': round(se, 4)}
    return res


def _zone_frame(T: dict, top: np.ndarray, bot: np.ndarray):
    """Location against each batter's own zone: u (feet, toward his side positive), zz (zone units, 1.5 to 3.5 inside) and
    edge (feet outside the zone, negative inside), plus the location, count, inning and handedness design used by DISC-08."""
    n = len(T['balls'])
    span = np.clip(top - bot, 1.2, 3.0)
    u = np.where(T['stand_r'] == 1, T['px'], -T['px']); zz = 1.5 + 2.0 * (T['pz'] - bot) / span
    edge = np.maximum(np.maximum(np.abs(u) - ZONE_HALF, zz - ZONE_TOP), ZONE_BOT - zz)
    cnt = np.zeros((n, 12), np.float32); cnt[np.arange(n), np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)] = 1
    inn = np.zeros((n, 3), np.float32); inn[np.arange(n), np.where(T['inning'] <= 6, 0, np.where(T['inning'] <= 8, 1, 2))] = 1
    base = np.hstack([hats(edge, EU_KNOTS), hats(u, U_KNOTS), hats(zz, Z_KNOTS), cnt, inn,
                      T['stand_r'][:, None].astype(np.float32)]).astype(np.float32)
    return u, zz, edge, base


def _left_or(D: np.ndarray, y: np.ndarray, games: np.ndarray, params: dict, seed: int, k: int = 1) -> dict | list:
    """Odds ratios of the last k columns (challenges-left indicators against two left), game-level bootstrap. One dict when
    k is 1, else a list in column order."""
    C = float(params.get('C', 1.0))
    m = fit_logistic(D, y, C=C)
    b = [float(x) for x in m.coef_[0][-k:]]
    ug = np.unique(games); rng = np.random.default_rng(seed)
    order = np.argsort(games, kind='stable'); gs = games[order]
    starts = np.searchsorted(gs, ug); ends = np.searchsorted(gs, ug, side='right')
    boots = []
    for _ in range(int(params.get('boot', 100))):
        pick = rng.integers(0, len(ug), len(ug))
        ii = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        if y[ii].sum() < 20:
            continue
        boots.append([float(x) for x in fit_logistic(D[ii], y[ii], C=C, warm=(m.coef_, m.intercept_)).coef_[0][-k:]])
    boots = np.asarray(boots).reshape(-1, k)
    out = []
    for j in range(k):
        lo, hi = (float(np.percentile(boots[:, j], 2.5)), float(np.percentile(boots[:, j], 97.5))) if len(boots) else (None, None)
        out.append({'log_odds': round(b[j], 4), 'odds_ratio': round(float(np.exp(b[j])), 4),
                    'ci_odds_ratio': [round(float(np.exp(lo)), 4), round(float(np.exp(hi)), 4)] if len(boots) else None,
                    'rows': int(len(y)), 'events': int(y.sum()), 'rows_flagged': int(D[:, D.shape[1] - k + j].sum()), 'boot_reps': int(len(boots))})
    return out[0] if k == 1 else out


def scarcity(T: dict, X: dict, params: dict, stage) -> dict:
    """Addendum 14 (DISC-08): does running low on challenges change how hitters and catchers act? Innings 1-9 of 2026, with
    each side's challenges left before the pitch (two minus its failed challenges; a successful challenge is kept).
    Challenges (called pitches; batters on called strikes, catchers on called balls): the odds of a challenge with one left
    against two at the same location against the batter's zone, count, inning and handedness (logistic; game-level
    bootstrap), the same with the batter's own challenge habit from his other games, the overturn rate of the challenges
    made, and how often a clear miss stands (a called strike at least an inch outside the batter's zone, or a called ball at
    least an inch inside it), by challenges left; a placebo puts the other team's challenges left in the same model. Swings: the odds of a swing at pitches within two inches of the zone's
    edge with the batting side one or no challenge left against two, all counts and two strikes."""
    res = {}
    F = rebuild(T)
    top, bot, _ = batter_zones(T, F['ok'] & (T['group'] >= 0) & (T['call'] <= 2))
    inch = 1.0 / 12.0
    base_ok = (F['ok'] & (T['group'] >= 0) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2)
               & (T['inning'] <= 9))
    # swings near the edge, by the batting side's challenges left (every pitch with a swing decision)
    okw = base_ok & (T['call'] <= 2)
    Tw = take(T, okw); bat_left_w = np.clip(2 - X['bat_failed'][okw], 0, 2)
    _, _, ew, basew = _zone_frame(Tw, top[okw], bot[okw])
    swing = ((Tw['call'] == 1) | (Tw['call'] == 2)).astype(np.int64)
    sw = {}
    for name, sel in (('all_counts', np.ones(len(swing), bool)), ('two_strikes', Tw['strikes'] == 2)):
        near = sel & (np.abs(ew) <= 2 * inch)
        rates = {}
        for side, ss in (('inside_edge', near & (ew <= 0)), ('outside_edge', near & (ew > 0))):
            rates[side] = {f'left_{lv}': [round(float(swing[ss & (bat_left_w == lv)].mean()), 4) if (ss & (bat_left_w == lv)).any() else None,
                                          int((ss & (bat_left_w == lv)).sum())] for lv in (2, 1, 0)}
        idx = np.flatnonzero(near)
        D = np.hstack([basew[idx], (bat_left_w[idx] == 1).astype(np.float32)[:, None], (bat_left_w[idx] == 0).astype(np.float32)[:, None]])
        one, none = _left_or(D, swing[idx], Tw['game'][idx], params, 141, k=2)
        sw[name] = {'swing_rates': rates, 'one_left_vs_two': one, 'none_left_vs_two': none}
    res['swings_near_edge'] = sw
    del Tw, basew
    stage('swings near the edge')
    ok = base_ok & (T['call'] == 0) & (X['orig'] >= 0)
    T = take(T, ok); X = {k: v[ok] for k, v in X.items()}; top, bot = top[ok], bot[ok]
    stage('called pitches, innings 1-9')
    u, zz, edge, base = _zone_frame(T, top, bot)
    for pname, orig, role, failed, clear in (
            ('batter', 1, ROLE['batter'], X['bat_failed'], edge > inch),
            ('catcher', 0, ROLE['catcher'], X['fld_failed'], edge < -inch)):
        left = np.clip(2 - failed, 0, 2)
        pm = X['orig'] == orig
        y = (X['role'] == role).astype(np.int64)
        over = X['over'].astype(float)
        out = {'called': int(pm.sum()), 'challenges': int(y[pm].sum())}
        table = []
        for lv in (2, 1, 0):
            sel = pm & (left == lv); csel = sel & clear; ch = sel & (y == 1)
            stands = csel & ~((y == 1) & (over == 1))
            table.append({'left': lv, 'called': int(sel.sum()), 'clear_misses': int(csel.sum()),
                          'challenge_rate_on_clear_misses': round(float(y[csel].mean()), 4) if csel.any() else None,
                          'clear_misses_that_stand': round(float(stands[csel].mean()), 4) if csel.any() else None,
                          'challenges': int(ch.sum()), 'overturn_rate': round(float(over[ch].mean()), 4) if ch.any() else None})
        out['by_challenges_left'] = table
        by_inning = []
        for lo_, hi_, name in ((1, 6, '1-6'), (7, 8, '7-8'), (9, 9, '9')):
            row = {'innings': name}
            for lv in (2, 1):
                csel = pm & clear & (left == lv) & (T['inning'] >= lo_) & (T['inning'] <= hi_)
                row[f'left_{lv}'] = [round(float(y[csel].mean()), 4) if csel.any() else None, int(csel.sum())]
            by_inning.append(row)
        out['challenge_rate_on_clear_misses_by_inning'] = by_inning
        idx = np.flatnonzero(pm & (left >= 1))
        one = (left[idx] == 1).astype(np.float32)[:, None]
        out['one_left_vs_two'] = _left_or(np.hstack([base[idx], one]), y[idx], T['game'][idx], params, 14)
        # placebo: the other team's challenges left (same game flow, inning and umpire, no scarcity of one's own)
        other = np.clip(2 - (X['fld_failed'] if pname == 'batter' else X['bat_failed']), 0, 2)
        o_other, o_one = _left_or(np.hstack([base[idx], (other[idx] <= 1).astype(np.float32)[:, None], one]), y[idx], T['game'][idx],
                                  params, 16, k=2)
        out['placebo_other_team_one_or_none_left'] = o_other
        out['one_left_vs_two_with_placebo'] = o_one
        if pname == 'batter':
            # the batter's own habit: challenges per clear-miss called strike in his other games (shrunk to the league)
            cm = pm & clear & (left >= 1)
            bid = T['batter']; g = T['game']
            ub, binv = np.unique(bid, return_inverse=True)
            tot_c = np.bincount(binv, weights=(y * cm).astype(float), minlength=len(ub))
            tot_n = np.bincount(binv, weights=cm.astype(float), minlength=len(ub))
            key = bid.astype(np.int64) * 10_000_000 + g.astype(np.int64)
            uk, kinv = np.unique(key, return_inverse=True)
            gc = np.bincount(kinv, weights=(y * cm).astype(float), minlength=len(uk))[kinv]
            gn = np.bincount(kinv, weights=cm.astype(float), minlength=len(uk))[kinv]
            league = float(y[cm].mean()) if cm.any() else 0.05
            prior = 20.0
            habit = (tot_c[binv] - gc + prior * league) / (tot_n[binv] - gn + prior)
            hl = np.log(habit / (1 - habit)).astype(np.float32)[:, None]
            out['one_left_vs_two_with_habit'] = _left_or(np.hstack([base[idx], hl[idx], one]), y[idx], T['game'][idx], params, 15)
        res[pname] = out
        stage(pname)
    return res


# ---------------------------------------------------------------- MATCHUP-01: hitters' swing maps at the decision moment
def _ridge_offset(B, y, off, lam, iters=25):
    """Ridge logistic with an offset (Newton); B small dense design. Returns coefficients."""
    b = np.zeros(B.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(off + B @ b)))
        g = B.T @ (y - p) - lam * b
        H = (B * (p * (1 - p))[:, None]).T @ B + lam * np.eye(B.shape[1])
        step = np.linalg.solve(H, g); b += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return b


def hitter_basis(x, z, stand_r, strikes):
    u = np.where(stand_r == 1, x, -x)
    e = np.maximum(np.maximum(np.abs(u) - ZONE_HALF, z - ZONE_TOP), ZONE_BOT - z)
    return np.hstack([hats(e, (-0.8, -0.4, -0.15, 0.0, 0.15, 0.4, 0.8, 1.5)), hats(u, (-1.5, -0.8, -0.3, 0.3, 0.8, 1.5)),
                      hats(z, (0.5, 1.5, 2.2, 2.9, 3.6, 4.5)), (strikes == 2)[:, None].astype(np.float32)]).astype(np.float64)


def pitcher_propensity(t: dict, k: float = 600.0) -> np.ndarray:
    """Logit of the swing rate of hitters facing each pitcher on earlier dates (shrunk toward the league by k pitches)."""
    tt = dict(t); tt['batter'] = t['pitcher']
    return swing_propensity(tt, k)


def matchup_swing(T: dict, params: dict, stage, positions: dict | None = None) -> dict:
    """Does each hitter's own swing map, read at the decision moment, predict his swings (and his chases against a given
    pitcher) beyond the league map and additive hitter and pitcher terms? Train 2023-2024, choose the shrinkage on the
    second half of 2024, score 2025 (2026 is left out: its plate locations use a different reference)."""
    res = {}
    seasons = [int(s) for s in params.get('train', (2023, 2024))] + [int(params.get('test', 2025))]
    T = take(T, np.isin(T['season'], seasons))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    tau = float(params.get('tau', 0.26))
    xp, zp = projected(T, F, None, 'straight', tau)
    reps = {'true': (T['px'].astype(np.float64), T['pz'].astype(np.float64)), 'percept': (xp, zp)}
    zrel = bool(params.get('zone_relative'))       # MATCHUP-08: heights on each batter's own zone (training seasons' zone numbers)
    if zrel:
        top_b, bot_b, zsum = batter_zones(T, np.isin(T['season'], seasons[:-1]))
        res['batter_zones'] = zsum
        hgt = np.clip(top_b - bot_b, 1.2, 2.8)
        zn_ = lambda z_: 1.5 + 2.0 * (z_ - bot_b) / hgt          # the batter's zone mapped onto the fixed 1.5 to 3.5 ft
        reps = {'percept': (xp, zp), 'percept_zone': (xp, zn_(zp))}
    brel = positions is not None                  # MATCHUP-09: the sideways position measured partly from the batter (STANCE-01)
    if brel:
        offs = np.array([positions.get(int(h_), np.nan) for h_ in np.unique(T['batter'])])
        mean_off = float(np.nanmean(offs)) if np.isfinite(offs).any() else 28.0
        lut = {int(h_): v_ for h_, v_ in zip(np.unique(T['batter']), offs)}
        dlt = np.array([lut.get(int(h_), np.nan) for h_ in T['batter']]); has_pos = np.isfinite(dlt)
        dlt = np.where(has_pos, (dlt - mean_off) / 12.0, 0.0)          # ft farther off the plate than the average hitter
        sgn = np.where(T['stand_r'] == 1, 1.0, -1.0)                     # x grows away from a right-handed hitter, toward a left-handed one
        weights = [float(w_) for w_ in params.get('body_weights', (0.25, 0.5, 1.0))]
        reps = {'percept': (xp, zp)}
        for w_ in weights:
            reps[f'percept_body_{w_:g}'] = (xp + sgn * w_ * dlt, zp)
        res['positions'] = {'hitters_with_position': int(np.isfinite(offs).sum()), 'mean_off_plate_in': round(mean_off, 2),
                            'decisions_with_position_share': round(float(has_pos.mean()), 4), 'weights': weights}
    C = np.hstack([control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    stage('features')
    test_season = seasons[-1]
    tr = np.isin(T['season'], seasons[:-1]); te = T['season'] == test_season
    mid = date(seasons[-2], 7, 1).toordinal()
    fit_a = tr & ~((T['season'] == seasons[-2]) & (T['day'] >= mid)); val = tr & (T['season'] == seasons[-2]) & (T['day'] >= mid)
    rng = np.random.default_rng(int(params.get('seed', 11)))
    n_league = int(params.get('league_n', 600000))
    lams = [float(v) for v in params.get('lams', (3.0, 10.0, 30.0, 100.0, 300.0))]
    min_n = int(params.get('min_pitches', 300))
    games = T['game'][te]; yt = swing[te]
    out = {}
    preds = {}
    for name, (x, z) in reps.items():
        X = np.hstack([location_block(x, z, T['stand_r'], T['strikes']), C])
        def league(rows):
            idx = np.flatnonzero(rows); idx = rng.choice(idx, min(len(idx), n_league), replace=False) if len(idx) > n_league else idx
            m = fit_logistic(X[idx], swing[idx])
            return m.decision_function(X)
        off_a = league(fit_a)                     # league model without the validation half, for choosing shrinkage
        off = league(tr)                          # league model on all training pitches, for the test
        val_league_ll = float(logloss_vec(1 / (1 + np.exp(-off_a[val])), swing[val]).mean())
        Bm = hitter_basis(x, z, T['stand_r'], T['strikes'])
        bat = T['batter']
        def groups(rows):
            idx = np.flatnonzero(rows); order = np.argsort(bat[idx], kind='stable'); idx = idx[order]
            hb = bat[idx]; cut = np.flatnonzero(np.diff(hb)) + 1
            return {int(g[0]): g for g in (idx[a:b] for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)])) if len(g)} if len(idx) else {}
        def key_of(g):
            return int(bat[g[0]])
        ga = {key_of(g): g for g in groups(fit_a).values()}
        gv = {key_of(g): g for g in groups(val).values()}
        gt = {key_of(g): g for g in groups(tr).values()}
        ge = {key_of(g): g for g in groups(te).values()}
        # choose the shrinkage on the validation half
        val_ll = {}
        lo_val = {lam: off_a.copy() for lam in lams}
        for h, r in ga.items():
            if len(r) < min_n or h not in gv:
                continue
            v = gv[h]
            for lam in lams:
                b = _ridge_offset(Bm[r], swing[r], off_a[r], lam)
                lo_val[lam][v] += Bm[v] @ b
        for lam in lams:
            val_ll[lam] = float(logloss_vec(1 / (1 + np.exp(-lo_val[lam][val])), swing[val]).mean())
        best = min(val_ll, key=val_ll.get)
        # refit hitter maps on all training pitches at the chosen shrinkage, score the test season
        lo_all = off.copy(); covered = np.zeros(len(off), bool)
        for h, r in gt.items():
            if len(r) < min_n or h not in ge:
                continue
            b = _ridge_offset(Bm[r], swing[r], off[r], best)
            e = ge[h]
            lo_all[e] += Bm[e] @ b; covered[e] = True
        lo_te = lo_all[te]; covered = covered[te]
        p_league = 1 / (1 + np.exp(-off[te])); p_hit = 1 / (1 + np.exp(-lo_te))
        preds[name] = (p_league, p_hit)
        out[name] = {'shrinkage_chosen': best, 'validation_logloss': {str(k): round(v, 5) for k, v in val_ll.items()}, 'validation_league_logloss': round(val_league_ll, 6),
                     'test_logloss_league': round(float(logloss_vec(p_league, yt).mean()), 5), 'test_logloss_hitter_maps': round(float(logloss_vec(p_hit, yt).mean()), 5),
                     'test_share_with_hitter_map': round(float(covered.mean()), 4)}
        stage('representation ' + name)
    cc = lambda d: [round(v * 1000, 3) for v in clustered_ci(d, games)]
    res['representations'] = out
    lp, hp = [logloss_vec(p, yt) for p in preds['percept']]
    if brel:
        # the weight is chosen on the validation half by the league model's log loss (the plate frame is weight 0)
        body = [k_ for k_ in out if k_.startswith('percept_body_')]
        res['body_weight_validation_league_logloss'] = {k_: out[k_]['validation_league_logloss'] for k_ in body + ['percept']}
        chosen = min(body, key=lambda k_: out[k_]['validation_league_logloss'])
        res['body_weight_chosen'] = chosen
        g_ = {}
        for k_ in body:
            lb, hb_ = [logloss_vec(p, yt) for p in preds[k_]]
            g_[k_] = {'body_over_plate_league': cc(lp - lb), 'body_over_plate_hitter_maps': cc(hp - hb_)}
        res['gain_nats_per_1000_decisions'] = {'chosen': g_[chosen], 'by_weight': g_, 'hitter_maps_over_league_plate': cc(lp - hp)}
        preds['percept_zone'] = preds[chosen]
    elif zrel:
        lz, hz = [logloss_vec(p, yt) for p in preds['percept_zone']]
        res['gain_nats_per_1000_decisions'] = {'zone_over_fixed_league': cc(lp - lz), 'zone_over_fixed_hitter_maps': cc(hp - hz),
                                               'hitter_maps_over_league_fixed': cc(lp - hp), 'hitter_maps_over_league_zone': cc(lz - hz)}
    else:
        lt, ht = [logloss_vec(p, yt) for p in preds['true']]
        res['gain_nats_per_1000_decisions'] = {'percept_over_true_league': cc(lt - lp), 'percept_over_true_hitter_maps': cc(ht - hp),
                                               'hitter_maps_over_league_true': cc(lt - ht), 'hitter_maps_over_league_percept': cc(lp - hp)}
    # matchups: chases by hitter-pitcher pair on pitches outside the zone, observed against the league-plus-additive model
    xt, zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    u = np.where(T['stand_r'] == 1, xt, -xt)
    outside = ((np.abs(u) > ZONE_HALF) | (zt > ZONE_TOP) | (zt < ZONE_BOT))[te]
    pair = T['batter'][te].astype(np.int64) * 1_000_000 + T['pitcher'][te].astype(np.int64)
    pl, ph = preds['percept']
    rows = []
    for k in np.unique(pair[outside]):
        sel = outside & (pair == k)
        if sel.sum() >= int(params.get('pair_min', 10)):
            rows.append((sel.sum(), yt[sel].mean(), pl[sel].mean(), ph[sel].mean()))
    if rows:
        R = np.asarray(rows, float); w = R[:, 0]
        resid = R[:, 1] - R[:, 2]; pred = R[:, 3] - R[:, 2]
        slope = float(np.sum(w * pred * resid) / max(np.sum(w * pred * pred), 1e-12))
        corr = float(np.corrcoef(resid, pred)[0, 1]) if len(R) > 3 else None
        bs = []
        for _ in range(300):
            j = rng.integers(0, len(R), len(R))
            bs.append(np.sum(w[j] * pred[j] * resid[j]) / max(np.sum(w[j] * pred[j] ** 2), 1e-12))
        res['pairs_outside_zone'] = {'pairs': int(len(R)), 'pitches': int(w.sum()), 'sd_observed_minus_additive': round(float(np.sqrt(np.average(resid ** 2, weights=w))), 4),
                                     'sd_predicted_pair_effect': round(float(np.sqrt(np.average(pred ** 2, weights=w))), 4), 'corr': round(corr, 4) if corr is not None else None,
                                     'slope_observed_on_predicted': [round(slope, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    stage('pairs')
    return res


# ---------------------------------------------------------------- BENCH-01: the decision-moment model against a flexible learner
def bench_swing(T: dict, params: dict, stage) -> dict:
    """BENCH-01. Does the decision-moment representation beat the strongest flexible model given the same inputs? A
    gradient-boosted classifier (scikit-learn's histogram boosting) gets every measured input of the pitch (speed at
    release and plate, movement, spin, release point and extension, the true crossing), the count, pitch group, batter
    and pitcher hands and both players' earlier swing levels (the same controls the logistic models have), so it could
    learn the decision-moment projection itself, since that is a function of those inputs. G0: boosting on those inputs.
    G1: G0 plus the standard hitter heat map (league logistic and hitter maps on the true crossing, as a predicted
    log-odds). Ours: MATCHUP-01's league logistic and hitter maps at the decision moment. G2: G1 plus ours as an input.
    Maps and league models used as inputs are cross-fitted on the training rows (two folds by game), and fitted on all
    training rows for the test. Log loss on the test season's decisions, game-clustered intervals."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    res = {}
    final = bool(params.get('final_eval'))          # BENCH-01F: trained on everything before August 1, 2026, scored once on the untouched months
    if final and not params.get('frozen_commit'):
        raise ValueError('bench_swing scores the untouched months only as the registered evaluation of a frozen commit')
    test_from = date(2026, 8, 1).toordinal()
    train = [int(v) for v in params.get('train', (2023, 2024))]; test = int(params.get('test', 2025))
    if final:
        T = take(T, np.isin(T['season'], (2023, 2024, 2025, 2026)) & (T['post'] == 0))
    else:
        T = take(T, np.isin(T['season'], train + [test]))
        if test == 2026:
            T = take(T, (T['season'] != 2026) | (T['day'] < test_from))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    target = str(params.get('target', 'swing'))      # BENCH-02: 'whiff' scores misses on swings with the same design
    if target == 'whiff':
        keep = keep & ((T['call'] == 1) | (T['call'] == 2))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    n = len(T['call'])
    if target == 'whiff':
        swing = (T['call'] == 2).astype(np.float64)                 # the outcome modeled (a miss on a swing), named as in the swing design
        lgr = float(swing.mean())
        def prior_logit(key, k_):
            nn, ss = _prior_by_day(key.astype(np.int64), T['day'].astype(np.int64), swing, np.ones(n, bool))
            r_ = (ss + k_ * lgr) / (nn + k_); return np.log(r_ / (1 - r_))
        prop = prior_logit(T['batter'], 150.0); pprop = prior_logit(T['pitcher'], 300.0)
    else:
        swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
        prop = swing_propensity(T); pprop = pitcher_propensity(T)
    res['target'] = target
    xp, zp = projected(T, F, None, 'straight', float(params.get('tau', 0.26)))
    px, pz = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    C = np.hstack([control_block(T, prop), pprop[:, None].astype(np.float32)])
    if final:
        tr = T['day'] < test_from; te = T['day'] >= test_from
    else:
        tr = np.isin(T['season'], train); te = T['season'] == test
    fold = (T['game'] % 2).astype(np.int64)
    rng = np.random.default_rng(int(params.get('seed', 11)))
    lam = float(params.get('lam', 10.0)); min_n = int(params.get('min_pitches', 300)); n_league = int(params.get('league_n', 600000))
    stage('features')

    def league_and_maps(x, z, fit_rows, pred_rows):
        X = np.hstack([location_block(x, z, T['stand_r'], T['strikes']), C])
        idx = np.flatnonzero(fit_rows); idx = rng.choice(idx, min(len(idx), n_league), replace=False)
        off = fit_logistic(X[idx], swing[idx]).decision_function(X)
        Bm = hitter_basis(x, z, T['stand_r'], T['strikes'])
        lo = off.copy(); ge = _groups(T['batter'], pred_rows)
        for h, b in _hitter_maps(Bm, swing, off, _groups(T['batter'], fit_rows), lam, min_n).items():
            if h in ge:
                e = ge[h]; lo[e] += Bm[e] @ b
        return off, lo
    feats = {}
    for name, (x, z) in (('true', (px, pz)), ('percept', (xp, zp))):
        lg_ = np.zeros(n); mp_ = np.zeros(n)
        for f in (0, 1):
            off, lo = league_and_maps(x, z, tr & (fold != f), tr & (fold == f))
            r_ = tr & (fold == f); lg_[r_] = off[r_]; mp_[r_] = lo[r_]
        off, lo = league_and_maps(x, z, tr, te)
        lg_[te] = off[te]; mp_[te] = lo[te]
        feats[name] = (lg_, mp_)
        stage('maps ' + name)
    raw = np.column_stack([T[k].astype(np.float64) for k in ('v0', 'v1', 'pfx_x', 'pfx_z', 'spin', 'x0', 'z0', 'ext', 'px', 'pz', 'balls', 'strikes', 'group', 'stand_r', 'throw_r')]
                          + [prop, pprop, (T['stand_r'] == T['throw_r']).astype(np.float64)])
    yt = swing[te]; games = T['game'][te]
    n_gbm = int(params.get('gbm_n', 1500000))
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), n_gbm), replace=False)

    def gbm(Xm, label):
        m = HistGradientBoostingClassifier(max_iter=int(params.get('gbm_iter', 400)), learning_rate=float(params.get('gbm_lr', 0.08)), max_leaf_nodes=63,
                                           min_samples_leaf=200, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                           categorical_features=[12], random_state=0)
        m.fit(Xm[idx], swing[idx])
        p = m.predict_proba(Xm[te])[:, 1]
        res.setdefault('gbm_iterations', {})[label] = int(m.n_iter_)
        stage('boosting ' + label)
        return p
    P = {}
    P['G0_boosting'] = gbm(raw, 'G0')
    P['G1_boosting_plus_heat_map'] = gbm(np.column_stack([raw, feats['true'][1]]), 'G1')
    P['G2_G1_plus_decision_moment'] = gbm(np.column_stack([raw, feats['true'][1], feats['percept'][1]]), 'G2')
    sig_ = lambda v: 1 / (1 + np.exp(-v))
    P['league_true'] = sig_(feats['true'][0][te]); P['heat_map_true'] = sig_(feats['true'][1][te])
    P['league_decision_moment'] = sig_(feats['percept'][0][te]); P['ours_decision_moment'] = sig_(feats['percept'][1][te])
    L = {k: logloss_vec(v, yt) for k, v in P.items()}
    cc = lambda d: [round(v * 1000, 3) for v in clustered_ci(d, games)]
    res['test_decisions'] = int(te.sum())
    res['logloss'] = {k: round(float(v.mean()), 6) for k, v in L.items()}
    res['gain_nats_per_1000_decisions'] = {
        'ours_over_G1': cc(L['G1_boosting_plus_heat_map'] - L['ours_decision_moment']),
        'ours_over_G0': cc(L['G0_boosting'] - L['ours_decision_moment']),
        'G2_over_G1': cc(L['G1_boosting_plus_heat_map'] - L['G2_G1_plus_decision_moment']),
        'G1_over_G0': cc(L['G0_boosting'] - L['G1_boosting_plus_heat_map']),
        'ours_over_heat_map_logistic': cc(L['heat_map_true'] - L['ours_decision_moment'])}
    return res


# ---------------------------------------------------------------- MATCHUP-04: hitter maps by pitch family at the decision moment
def family_basis(B, group):
    """MATCHUP-04's map basis: the shared surface (hitter_basis, last column the level), its location bands times
    breaking (groups 3 and 4) and times offspeed (group 5), and the two family levels."""
    loc = B[:, :-1]
    brk = np.isin(group, (3, 4))[:, None].astype(np.float64); ofs = (group == 5)[:, None].astype(np.float64)
    return np.hstack([B, loc * brk, loc * ofs, brk, ofs])


FAMILY_SHRINKAGE = {'location': 10.0, 'location_by_family': 3.0}     # chosen on the validation half in MATCHUP-04 (run 37889588250)
FAMILY_SHRINKAGE_LEAGUE_FAMILY = {'location': 10.0, 'location_by_family': 10.0}   # MATCHUP-05 (league with its own family part, run 37891547233)


def matchup_family(T: dict, params: dict, stage) -> dict:
    """MATCHUP-01's hitter maps at the decision moment treat every pitch family alike: a hitter's map is one surface
    over where the pitch appears to be headed. If hitters answer the same apparent location differently for a
    fastball and a breaking ball (some chase sliders low and away and lay off fastballs there), a map with a family
    part (the location bands times breaking and times offspeed, on top of the shared surface) predicts better. League
    model and data as MATCHUP-01 (train 2023-2024, shrinkage on the second half of 2024, score 2025, paired by game).
    final_eval (MATCHUP-04F): league and both maps fitted on 2023-2025 at the shrinkage MATCHUP-04 chose, scored once
    on the 2026 pitches from August 1 (the untouched months)."""
    res = {}
    final = bool(params.get('final_eval'))
    if final and not params.get('frozen_commit'):
        raise ValueError('the family maps are scored on the untouched months only as the registered evaluation of a frozen commit')
    fwd = params.get('forward_from')                # FWD-01: everything before this date trains, everything from it on is scored
    if final and fwd:
        T = take(T, np.isin(T['season'], (2023, 2024, 2025, 2026)))
    elif final:
        T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | ((T['season'] == 2026) & (T['day'] >= date(2026, 8, 1).toordinal())))
    else:
        post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    LBk = location_block(xp, zp, T['stand_r'], T['strikes']); Ck = np.hstack([control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    lf = bool(params.get('league_family'))          # MATCHUP-05: the league model gets the same family-by-location part as the hitter maps
    extra = params.get('extra_part')                # MATCHUP-06: 'platoon' (location bands times same-hand pitcher) or 'two_strike' (times two strikes)
    if extra == 'platoon':
        xind = (T['stand_r'] == T['throw_r']).astype(np.float64)
    elif extra == 'two_strike':
        xind = (T['strikes'] == 2).astype(np.float64)
    else:
        xind = None
    if lf:
        Bh0 = hitter_basis(xp, zp, T['stand_r'], T['strikes'])[:, :-1]
        lx = [(Bh0 * xind[:, None]).astype(np.float32)] if extra == 'platoon' else []     # the league's own platoon-by-location part (it is split by count already)
        X = np.hstack([LBk, (Bh0 * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32), (Bh0 * (T['group'] == 5)[:, None]).astype(np.float32)] + lx + [Ck]); del Bh0, lx
    else:
        X = np.hstack([LBk, Ck])
    if final and fwd:
        d0 = date.fromisoformat(str(fwd)).toordinal()
        tr = T['day'] < d0; te = T['day'] >= d0
    elif final:
        tr = T['season'] <= 2025; te = T['season'] == 2026
    else:
        tr = np.isin(T['season'], (2023, 2024)); te = T['season'] == 2025
    mid = date(2024, 7, 1).toordinal()
    fit_a = tr & ~((T['season'] == 2024) & (T['day'] >= mid)); val = tr & (T['season'] == 2024) & (T['day'] >= mid)
    rng = np.random.default_rng(11)
    def league(rows):
        idx = np.flatnonzero(rows); idx = rng.choice(idx, min(len(idx), 600000), replace=False) if len(idx) > 600000 else idx
        return fit_logistic(X[idx], swing[idx]).decision_function(X)
    if final:
        off = league(tr); off_a = None
    else:
        off_a, off = league(fit_a), league(tr)
    if lf:                                            # the plain league (no family-by-location part), for the size of the league-wide pattern
        X = np.hstack([LBk, Ck]); off_plain = league(tr)
    del X, LBk, Ck
    stage('league')
    B = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    bases = {'location': B, 'location_by_family': family_basis(B, T['group'])}
    if xind is not None:
        bases = {'location_by_family': bases['location_by_family'], 'family_plus_extra': np.hstack([bases['location_by_family'], B[:, :-1] * xind[:, None], xind[:, None]])}
    bat = T['batter']
    gt, ge = _groups(bat, tr), _groups(bat, te)
    if not final:
        ga, gv = _groups(bat, fit_a), _groups(bat, val)
    yt = swing[te]; games = T['game'][te]
    preds = {}; out = {}
    for name, Bm in bases.items():
        val_ll = {}
        if final:
            best = (FAMILY_SHRINKAGE_LEAGUE_FAMILY if lf else FAMILY_SHRINKAGE)[name]
        else:
            for lam in (3.0, 10.0, 30.0, 100.0):
                lo = off_a.copy()
                for h, b in _hitter_maps(Bm, swing, off_a, ga, lam, 300).items():
                    if h in gv:
                        v = gv[h]; lo[v] += Bm[v] @ b
                val_ll[lam] = float(logloss_vec(1 / (1 + np.exp(-lo[val])), swing[val]).mean())
            best = min(val_ll, key=val_ll.get)
        lo = off.copy(); cov = np.zeros(len(swing), bool)
        for h, b in _hitter_maps(Bm, swing, off, gt, best, 300).items():
            if h in ge:
                e = ge[h]; lo[e] += Bm[e] @ b; cov[e] = True
        preds[name] = 1 / (1 + np.exp(-lo[te]))
        out[name] = {'shrinkage': best, 'validation': {str(k): round(v, 5) for k, v in val_ll.items()}, 'covered': round(float(cov[te].mean()), 4),
                     'test_logloss': round(float(logloss_vec(preds[name], yt).mean()), 5)}
        stage('maps ' + name)
    res['test_decisions'] = int(te.sum())
    p_l = 1 / (1 + np.exp(-off[te]))
    if xind is not None:
        ll_l = logloss_vec(p_l, yt); ll_f = logloss_vec(preds['location_by_family'], yt); ll_x = logloss_vec(preds['family_plus_extra'], yt)
        cc = lambda d: [round(v * 1000, 3) for v in clustered_ci(d, games)]
        res['maps'] = out
        res['gain_nats_per_1000_decisions'] = {'family_over_league': cc(ll_l - ll_f), 'extra_over_family': cc(ll_f - ll_x)}
        m2 = xind[te] == 1
        res['extra_over_family_where_it_applies'] = {'decisions': int(m2.sum()), 'gain': [round(v * 1000, 3) for v in clustered_ci((ll_f - ll_x)[m2], games[m2])]}
        res['extra_part'] = extra
        return res
    ll_l = logloss_vec(p_l, yt); ll_a = logloss_vec(preds['location'], yt); ll_b = logloss_vec(preds['location_by_family'], yt)
    cc = lambda d: [round(v * 1000, 3) for v in clustered_ci(d, games)]
    res['maps'] = out
    res['gain_nats_per_1000_decisions'] = {'location_over_league': cc(ll_l - ll_a), 'family_over_location': cc(ll_a - ll_b), 'family_over_league': cc(ll_l - ll_b)}
    if lf:
        res['gain_nats_per_1000_decisions']['league_family_over_plain_league'] = cc(logloss_vec(1 / (1 + np.exp(-off_plain[te])), yt) - ll_l)
    # where the family part helps: decisions on breaking and offspeed pitches outside the zone
    u = np.where(T['stand_r'][te] == 1, T['px'][te], -T['px'][te])
    outside = (np.abs(u) > ZONE_HALF) | (T['pz'][te] > ZONE_TOP) | (T['pz'][te] < ZONE_BOT)
    for nm, m in (('fastball_outside', outside & np.isin(T['group'][te], (0, 1, 2))), ('breaking_outside', outside & np.isin(T['group'][te], (3, 4))),
                  ('offspeed_outside', outside & (T['group'][te] == 5)), ('all_inside', ~outside)):
        res.setdefault('family_over_location_by_subset', {})[nm] = {'decisions': int(m.sum()),
                                                                   'gain': [round(v * 1000, 3) for v in clustered_ci((ll_a - ll_b)[m], games[m])] if m.sum() > 1000 else None}
    if params.get('by_release'):
        # PERCEPT-01: can hitters answer pitch families differently only when they can tell them apart early? Each
        # pitcher's separation between his fastballs (groups 0-2) and breaking balls (3-4) at release (x0, z0 at 50 ft
        # and extension), in units of his own within-family scatter (training pitches, at least 60 of each); the
        # family part's gain by thirds of that separation, and the difference between the outer thirds
        rel = {}
        trp = _groups(T['pitcher'], tr)
        fbm = np.isin(T['group'], (0, 1, 2)); brm = np.isin(T['group'], (3,) if params.get('sliders_only') else (3, 4))
        for pid, r in trp.items():
            a = r[fbm[r]]; b = r[brm[r]]
            if len(a) < 60 or len(b) < 60:
                continue
            if params.get('separation') == 'speed':        # PERCEPT-02: the speed gap between the two families (the plainest type cue in flight)
                A = T['v0'][a][:, None].astype(np.float64); Bq = T['v0'][b][:, None].astype(np.float64)
            else:
                A = np.column_stack([T['x0'][a], T['z0'][a], T['ext'][a]]).astype(np.float64); Bq = np.column_stack([T['x0'][b], T['z0'][b], T['ext'][b]]).astype(np.float64)
            ok_a = np.isfinite(A).all(1); ok_b = np.isfinite(Bq).all(1)
            if ok_a.sum() < 60 or ok_b.sum() < 60:
                continue
            A, Bq = A[ok_a], Bq[ok_b]
            sd = np.sqrt((A.var(0) * len(A) + Bq.var(0) * len(Bq)) / (len(A) + len(Bq))) + 1e-6
            rel[int(pid)] = float(np.sqrt(np.sum(((A.mean(0) - Bq.mean(0)) / sd) ** 2)))
        pt = T['pitcher'][te]
        d_ = np.array([rel.get(int(p_), np.nan) for p_ in pt])
        ok = np.isfinite(d_)
        cut = np.nanpercentile(np.array(list(rel.values())), [33.3, 66.7]) if rel else [0, 0]
        third = np.searchsorted(cut, d_)
        bres = {'pitchers_measured': len(rel), 'cuts': [round(float(c), 3) for c in cut]}
        for t3, nm in ((0, 'tight_release'), (1, 'middle'), (2, 'separated_release')):
            m = ok & (third == t3)
            if m.sum() < 1000:
                continue
            g_ = {'decisions': int(m.sum()), 'family_over_location': [round(v * 1000, 3) for v in clustered_ci((ll_a - ll_b)[m], games[m])],
                  'family_over_location_breaking_outside': [round(v * 1000, 3) for v in clustered_ci((ll_a - ll_b)[m & outside & np.isin(T['group'][te], (3, 4))], games[m & outside & np.isin(T['group'][te], (3, 4))])]}
            if lf:
                g_['league_family_over_plain'] = [round(v * 1000, 3) for v in clustered_ci((logloss_vec(1 / (1 + np.exp(-off_plain[te])), yt) - ll_l)[m], games[m])]
            bres[nm] = g_
        lo_, hi_ = ok & (third == 0), ok & (third == 2)
        dd = np.where(hi_, ll_a - ll_b, 0.0) / max(hi_.mean(), 1e-9) - np.where(lo_, ll_a - ll_b, 0.0) / max(lo_.mean(), 1e-9)
        bres['separated_minus_tight_family_gain'] = [round(v * 1000, 3) for v in clustered_ci(dd, games)]
        res['by_release_separation'] = bres
    return res


# ---------------------------------------------------------------- MAPS-01: are a hitter's own chase spots a trait?
def map_stability_study(T: dict, params: dict, stage) -> dict:
    """How much of a hitter's own part (MATCHUP-05F's family map minus the same-side mean map) carries from one season
    to the next. League with its family part fitted on 2023-2025; hitter maps (shrinkage 10, at least 300 pitches in
    each period) fitted separately on 2023-2024 and on 2025, and on the odd and even games of 2025 (split halves).
    Each own part read on a standard grid (7 by 7 cells from the scouting view, times the three families, first pitch);
    per hitter, the correlation between periods across those 147 points; the split-half correlation within 2025 bounds
    what any map from one season's pitches can repeat; their ratio is the trait share."""
    res = {}
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26); del F
    Bh = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    X = np.hstack([location_block(xp, zp, T['stand_r'], T['strikes']), (Bh[:, :-1] * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32),
                   (Bh[:, :-1] * (T['group'] == 5)[:, None]).astype(np.float32), control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    rng = np.random.default_rng(11)
    idx = rng.choice(len(swing), min(len(swing), 600000), replace=False)
    off = fit_logistic(X[idx], swing[idx]).decision_function(X); del X
    Bm = family_basis(Bh, T['group']); del Bh
    stage('league')
    gu = np.array([-1.25, -0.83, -0.42, 0.0, 0.42, 0.83, 1.25]); gz = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    UU, ZZ = np.meshgrid(gu, gz); uu, zz = UU.ravel(), ZZ.ravel()
    Bg = hitter_basis(uu, zz, np.ones(len(uu), np.int64), np.zeros(len(uu), np.int64))
    G = np.vstack([family_basis(Bg, np.full(len(uu), g, np.int64)) for g in (0, 3, 5)])
    outside_g = np.tile((np.abs(uu) > ZONE_HALF) | (zz > ZONE_TOP) | (zz < ZONE_BOT), 3)
    side_all = {}
    for h, r in _groups(T['batter'], np.ones(len(swing), bool)).items():
        side_all[h] = int(np.round(T['stand_r'][r].mean()))
    def own_grids(rows):
        maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], rows), 10.0, 300)
        Ss = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
        for h, m in maps.items():
            Ss[side_all[h]] = Ss[side_all[h]] + m; Nn[side_all[h]] += 1
        return {h: G @ (m - (Ss[side_all[h]] - m) / max(Nn[side_all[h]] - 1, 1)) for h, m in maps.items()}
    A = own_grids(np.isin(T['season'], (2023, 2024))); B = own_grids(T['season'] == 2025)
    odd = (T['season'] == 2025) & (T['game'] % 2 == 1); even = (T['season'] == 2025) & (T['game'] % 2 == 0)
    O = own_grids(odd); E = own_grids(even)
    stage('maps')
    def corr(P, Q, mask=None):
        out = []
        for h in set(P) & set(Q):
            p_, q_ = (P[h], Q[h]) if mask is None else (P[h][mask], Q[h][mask])
            if np.std(p_) > 0 and np.std(q_) > 0:
                out.append(float(np.corrcoef(p_, q_)[0, 1]))
        return np.asarray(out)
    for nm, P, Q in (('seasons_2023_24_vs_2025', A, B), ('split_halves_2025', O, E)):
        for part, mask in (('all_cells', None), ('outside_cells', outside_g)):
            c = corr(P, Q, mask)
            res.setdefault(nm, {})[part] = {'hitters': int(len(c)), 'median': round(float(np.median(c)), 3), 'quartiles': [round(float(np.percentile(c, 25)), 3), round(float(np.percentile(c, 75)), 3)],
                                            'mean': round(float(c.mean()), 3)}
    for part in ('all_cells', 'outside_cells'):
        sh = res['split_halves_2025'][part]['median']
        # split halves use half a season each; Spearman-Brown to a full season before the ratio
        full = 2 * sh / (1 + sh) if sh > -1 else float('nan')
        res.setdefault('trait_share', {})[part] = round(float(res['seasons_2023_24_vs_2025'][part]['median'] / full), 3) if full > 0 else None
    return res


# ---------------------------------------------------------------- MATCHUP-01, scored once on the untouched set
def harmonize_2026(T: dict) -> dict:
    """2026 feed locations measured at the middle of the plate, moved to the front (the earlier seasons' reference):
    each pitch's height and side minus what its flight does over the 8.5 inches between the two."""
    T = dict(T)
    F = rebuild(T)
    m = (T['season'] == 2026) & F['ok']
    vy = (T['v1'].astype(np.float64)) * FT_PER_MPH
    dt = (YPLATE - 8.5 / 12.0) / vy
    vz = F['vz0'] + F['az'] * F['tf']; vx = F['vx0'] + F['ax'] * F['tf']
    T['pz'] = np.where(m, T['pz'] - vz * dt, T['pz']).astype(np.float32)
    T['px'] = np.where(m, T['px'] - vx * dt, T['px']).astype(np.float32)
    return T


def matchup_final(T: dict, params: dict, stage) -> dict:
    """The frozen MATCHUP-01 model (league and hitter maps at the decision moment, shrinkage 10, tau 0.26) fitted on
    2023-2025 and scored once on the 2026 pitches from August 1 (the program's untouched set)."""
    if not params.get('final_eval') or not params.get('frozen_commit'):
        raise ValueError('matchup_final runs only as the registered final evaluation of a frozen commit')
    if params.get('harmonize_2026'):
        T = harmonize_2026(T)
    test_from = date(2026, 8, 1).toordinal()
    fwd = params.get('forward_from')                 # FWD-01: everything before this date trains, everything from it on is scored
    if fwd:
        T = take(T, np.isin(T['season'], (2023, 2024, 2025, 2026)))
    else:
        T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | ((T['season'] == 2026) & (T['day'] >= test_from)))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    C = np.hstack([control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    if fwd:
        d0 = date.fromisoformat(str(fwd)).toordinal(); tr = T['day'] < d0; te = T['day'] >= d0
    else:
        tr = T['season'] <= 2025; te = T['season'] == 2026
    rng = np.random.default_rng(11)
    preds = {}
    reps_f = (('true', (T['px'].astype(np.float64), T['pz'].astype(np.float64))), ('percept', (xp, zp)))
    zrel = bool(params.get('zone_relative'))              # MATCHUP-08F: batter zones (training rows' zone numbers) against the fixed zone
    if zrel:
        top_b, bot_b, _zs = batter_zones(T, tr); hgt = np.clip(top_b - bot_b, 1.2, 2.8)
        reps_f = (('percept', (xp, zp)), ('percept_zone', (xp, 1.5 + 2.0 * (zp - bot_b) / hgt)))
    for name, (x, z) in reps_f:
        X = np.hstack([location_block(x, z, T['stand_r'], T['strikes']), C])
        idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
        off = fit_logistic(X[idx], swing[idx]).decision_function(X)
        Bm = hitter_basis(x, z, T['stand_r'], T['strikes'])
        lo = off.copy(); cov = np.zeros(len(off), bool)
        ge = _groups(T['batter'], te)
        for h, b in _hitter_maps(Bm, swing, off, _groups(T['batter'], tr), 10.0, 300).items():
            if h in ge:
                e = ge[h]; lo[e] += Bm[e] @ b; cov[e] = True
        preds[name] = (1 / (1 + np.exp(-off[te])), 1 / (1 + np.exp(-lo[te])), float(cov[te].mean()))
        stage('final ' + name)
    yt = swing[te]; g = T['game'][te]
    lp, hp = logloss_vec(preds['percept'][0], yt), logloss_vec(preds['percept'][1], yt)
    cc = lambda d: [round(v * 1000, 3) for v in clustered_ci(d, g)]
    if zrel:
        lz, hz = logloss_vec(preds['percept_zone'][0], yt), logloss_vec(preds['percept_zone'][1], yt)
        res = {'test_decisions': int(te.sum()), 'covered': round(preds['percept'][2], 4),
               'gain_nats_per_1000_decisions': {'zone_over_fixed_league': cc(lp - lz), 'zone_over_fixed_hitter_maps': cc(hp - hz),
                                                'hitter_maps_over_league_fixed': cc(lp - hp), 'hitter_maps_over_league_zone': cc(lz - hz)}}
    else:
        lt, ht = logloss_vec(preds['true'][0], yt), logloss_vec(preds['true'][1], yt)
        res = {'test_decisions': int(te.sum()), 'covered': round(preds['percept'][2], 4),
               'gain_nats_per_1000_decisions': {'percept_over_true_league': cc(lt - lp), 'percept_over_true_hitter_maps': cc(ht - hp),
                                                'hitter_maps_over_league_true': cc(lt - ht), 'hitter_maps_over_league_percept': cc(lp - hp)}}
    xt, zt = T['px'][te], T['pz'][te]
    u = np.where(T['stand_r'][te] == 1, xt, -xt)
    outside = (np.abs(u) > ZONE_HALF) | (zt > ZONE_TOP) | (zt < ZONE_BOT)
    pair = T['batter'][te].astype(np.int64) * 1_000_000 + T['pitcher'][te].astype(np.int64)
    pl, ph = preds['percept'][0], preds['percept'][1]
    rows = []
    for k in np.unique(pair[outside]):
        sel = outside & (pair == k)
        if sel.sum() >= 10:
            rows.append((sel.sum(), yt[sel].mean(), pl[sel].mean(), ph[sel].mean()))
    if rows:
        R = np.asarray(rows, float); w = R[:, 0]; resid = R[:, 1] - R[:, 2]; pred = R[:, 3] - R[:, 2]
        bs = []
        for _ in range(300):
            j = rng.integers(0, len(R), len(R)); bs.append(np.sum(w[j] * pred[j] * resid[j]) / max(np.sum(w[j] * pred[j] ** 2), 1e-12))
        res['pairs_outside_zone'] = {'pairs': int(len(R)), 'slope': [round(float(np.sum(w * pred * resid) / np.sum(w * pred * pred)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    return res


# ---------------------------------------------------------------- MATCHUP-03: hitters' whiff maps at the decision moment
def _hitter_maps(Bm, y, off, groups_fit, lam, min_n):
    return {h: _ridge_offset(Bm[r], y[r], off[r], lam) for h, r in groups_fit.items() if len(r) >= min_n}


def _groups(keyarr, rows):
    idx = np.flatnonzero(rows)
    if not len(idx):
        return {}
    o = np.argsort(keyarr[idx], kind='stable'); idx = idx[o]; k = keyarr[idx]
    cut = np.flatnonzero(np.diff(k)) + 1
    return {int(keyarr[g[0]]): g for g in (idx[a:b] for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)]))}


def matchup_whiff(T: dict, params: dict, stage) -> dict:
    """MATCHUP-01 for misses: on swings, each hitter's own whiff function over where the pitch appeared to be headed at
    the decision moment and the movement he could not see after it (true crossing minus that projection), against the
    league whiff model with additive hitter and pitcher terms (earlier whiff rates, shrunk). Train 2023-2024 (shrinkage
    on the second half of 2024), score 2025, paired by game; the same on the true crossing for comparison."""
    res = {}
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & ((T['call'] == 1) | (T['call'] == 2)) & (T['strikes'] >= 0) & (T['bunt_pa'] == 0)
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    y = (T['call'] == 2).astype(np.float64)
    tau = float(params.get('tau', 0.26))
    xp, zp = projected(T, F, None, 'straight', tau)
    xt, zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    sgn = np.where(T['stand_r'] == 1, 1.0, -1.0)
    du, dz = (xt - xp) * sgn, zt - zp                                    # movement after the decision moment (ft)
    tt = dict(T); tt['call'] = np.where(y == 1, 1, 0)                    # whiff habit: earlier whiff rate on swings
    prop_b = swing_propensity(tt, 200.0)
    tp = dict(tt); tp['batter'] = T['pitcher']; prop_p = swing_propensity(tp, 400.0)
    grp = np.zeros((len(y), 7), np.float32); grp[np.arange(len(y)), np.clip(T['group'], 0, 6)] = 1
    vel = hats(T['v0'].astype(np.float64), V_KNOTS)
    C = np.hstack([grp, vel, (T['strikes'] == 2)[:, None], prop_b[:, None], prop_p[:, None], (T['stand_r'] == T['throw_r'])[:, None]]).astype(np.float32)
    late = np.hstack([hats(dz, (-1.5, -1.0, -0.6, -0.3, 0.0, 0.3)), hats(du, (-1.0, -0.5, 0.0, 0.5, 1.0))]).astype(np.float32)
    seasons = (2023, 2024, 2025)
    tr = np.isin(T['season'], seasons[:-1]); te = T['season'] == seasons[-1]
    mid = date(seasons[-2], 7, 1).toordinal()
    fit_a = tr & ~((T['season'] == seasons[-2]) & (T['day'] >= mid)); val = tr & (T['season'] == seasons[-2]) & (T['day'] >= mid)
    rng = np.random.default_rng(int(params.get('seed', 11)))
    lams = [float(v) for v in params.get('lams', (3.0, 10.0, 30.0, 100.0))]
    min_n = int(params.get('min_swings', 250))
    out, preds = {}, {}
    fam = np.column_stack([np.isin(T['group'], (0, 1, 2)), np.isin(T['group'], (3, 4)), np.isin(T['group'], (5,))]).astype(np.float64)
    for name, (x, z) in (('true', (xt, zt)), ('percept', (xp, zp))):
        X = np.hstack([location_block(x, z, T['stand_r'], T['strikes']), C] + ([late] if name == 'percept' else []))
        def league(rows):
            idx = np.flatnonzero(rows); idx = rng.choice(idx, min(len(idx), int(params.get('league_n', 600000))), replace=False) if len(idx) > int(params.get('league_n', 600000)) else idx
            return fit_logistic(X[idx], y[idx]).decision_function(X)
        off_a, off = league(fit_a), league(tr)
        hb = hitter_basis(x, z, T['stand_r'], T['strikes'])
        Bm = np.hstack([hb, fam] + ([hats(dz, (-1.5, -0.8, -0.3, 0.2)).astype(np.float64)] if name == 'percept' else [hats(zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)]))
        bat = T['batter']
        ga, gv, gt, ge = _groups(bat, fit_a), _groups(bat, val), _groups(bat, tr), _groups(bat, te)
        val_ll = {}
        for lam in lams:
            lo = off_a.copy()
            for h, b in _hitter_maps(Bm, y, off_a, ga, lam, min_n).items():
                if h in gv:
                    v = gv[h]; lo[v] += Bm[v] @ b
            val_ll[lam] = float(logloss_vec(1 / (1 + np.exp(-lo[val])), y[val]).mean())
        best = min(val_ll, key=val_ll.get)
        lo = off.copy(); cov = np.zeros(len(y), bool)
        for h, b in _hitter_maps(Bm, y, off, gt, best, min_n).items():
            if h in ge:
                e = ge[h]; lo[e] += Bm[e] @ b; cov[e] = True
        pl, ph = 1 / (1 + np.exp(-off[te])), 1 / (1 + np.exp(-lo[te]))
        preds[name] = (pl, ph)
        out[name] = {'shrinkage': best, 'validation': {str(k): round(v, 5) for k, v in val_ll.items()}, 'covered': round(float(cov[te].mean()), 4),
                     'test_logloss_league': round(float(logloss_vec(pl, y[te]).mean()), 5), 'test_logloss_hitter': round(float(logloss_vec(ph, y[te]).mean()), 5)}
        stage('whiff ' + name)
    yt = y[te]; g = T['game'][te]
    lt, ht = [logloss_vec(p, yt) for p in preds['true']]; lp, hp = [logloss_vec(p, yt) for p in preds['percept']]
    cc = lambda d: [round(v * 1000, 3) for v in clustered_ci(d, g)]
    res['representations'] = out
    res['test_swings'] = int(te.sum()); res['test_whiff_rate'] = round(float(yt.mean()), 4)
    res['gain_nats_per_1000_swings'] = {'percept_over_true_league': cc(lt - lp), 'percept_over_true_hitter': cc(ht - hp),
                                        'hitter_over_league_true': cc(lt - ht), 'hitter_over_league_percept': cc(lp - hp)}
    return res


# ---------------------------------------------------------------- STEER-01: where the bat aims, read from misses
def steer_profile(T: dict, params: dict, stage) -> dict:
    """Swing decisions follow the flight as it looked about 260 ms out (Decision Horizon) while misses follow the true
    crossing far better than that (MATCHUP-03). If the bat keeps being steered until a later moment and no further, misses
    are best explained by the flight as it looked at that moment: the gravity-only projection from tau seconds before
    the plate, with one location map shared by all pitch types (as in the horizon's first run). Held-out whiff log loss
    on 2025 swings for tau on a grid from 0 (the true crossing) to 0.26 s; models fitted on 2023-2024."""
    res = {}
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & ((T['call'] == 1) | (T['call'] == 2)) & (T['strikes'] >= 0) & (T['bunt_pa'] == 0)
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    y = (T['call'] == 2).astype(np.float64)
    tt = dict(T); tt['call'] = np.where(y == 1, 1, 0)
    prop_b = swing_propensity(tt, 200.0)
    grp = np.zeros((len(y), 7), np.float32); grp[np.arange(len(y)), np.clip(T['group'], 0, 6)] = 1
    C = np.hstack([grp, hats(T['v0'].astype(np.float64), V_KNOTS), (T['strikes'] == 2)[:, None], prop_b[:, None], (T['stand_r'] == T['throw_r'])[:, None]]).astype(np.float32)
    tr = np.isin(T['season'], (2023, 2024)); te = T['season'] == 2025
    rng = np.random.default_rng(11)
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), int(params.get('train_n', 700000))), replace=False)
    yt = y[te]; g = T['game'][te]
    taus = [float(v) for v in params.get('taus', (0.0, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.22, 0.26))]
    lls = {}
    for tau in taus:
        x, z = projected(T, F, None, 'straight', tau) if tau > 0 else (T['px'].astype(np.float64), T['pz'].astype(np.float64))
        X = np.hstack([location_block(x, z, T['stand_r'], T['strikes']), C])
        m = fit_logistic(X[idx], y[idx])
        lls[tau] = logloss_vec(m.predict_proba(X[te])[:, 1], yt)
        stage(f'tau {tau}')
    base = lls[0.0]
    res['test_swings'] = int(te.sum())
    res['profile_nats_per_1000_vs_true_crossing'] = {str(t): [round(v * 1000, 3) for v in clustered_ci(base - l, g)] for t, l in lls.items()}
    best = max(lls, key=lambda t: float((base - lls[t]).mean()))
    res['best_tau_s'] = best
    # bootstrap of the best tau over games
    ug, gi = np.unique(g, return_inverse=True)
    sums = {t: np.bincount(gi, weights=base - l, minlength=len(ug)) for t, l in lls.items()}
    cnt = np.bincount(gi, minlength=len(ug)); picks = []
    for _ in range(300):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        picks.append(max(taus, key=lambda t: (w * sums[t]).sum() / max((w * cnt).sum(), 1)))
    res['best_tau_bootstrap'] = {str(t): round(float(np.mean(np.asarray(picks) == t)), 3) for t in taus}
    return res


# ---------------------------------------------------------------- MATCHUP-02: do decision-moment matchups move plate-appearance outcomes
def matchup_pa(T: dict, params: dict, stage) -> dict:
    """Hitter maps and the league map at the decision moment as in MATCHUP-01 (fitted on 2023-2024, shrinkage 10). For
    every 2025 hitter-pitcher pair, the pitcher's 2024 pitches (his arsenal as the hitter could have known it) are run
    through the hitter's map and the league map: the matchup's expected extra swing rate on pitches outside the zone
    (chase deviation) and inside it (zone-swing deviation), beyond the additive terms. Do these predict the pair's 2025
    strikeouts and walks beyond both players' earlier strikeout and walk rates? Coefficients and out-of-sample log loss
    by game-parity cross-fitting within 2025."""
    res = {}
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    Tk = take(T, keep); Fk = {k: v[keep] for k, v in F.items()}
    swing = ((Tk['call'] == 1) | (Tk['call'] == 2)).astype(np.float64)
    xp, zp = projected(Tk, Fk, None, 'straight', float(params.get('tau', 0.26)))
    C = np.hstack([control_block(Tk, swing_propensity(Tk)), pitcher_propensity(Tk)[:, None].astype(np.float32)])
    X = np.hstack([location_block(xp, zp, Tk['stand_r'], Tk['strikes']), C])
    tr = np.isin(Tk['season'], (2023, 2024))
    rng = np.random.default_rng(int(params.get('seed', 11)))
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), int(params.get('league_n', 600000))), replace=False)
    league = fit_logistic(X[idx], swing[idx])
    off = league.decision_function(X)
    Bm = hitter_basis(xp, zp, Tk['stand_r'], Tk['strikes'])
    bat = Tk['batter']
    maps = {}
    ordr = np.argsort(bat[tr], kind='stable'); tri = np.flatnonzero(tr)[ordr]; hb = bat[tri]
    cut = np.flatnonzero(np.diff(hb)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(tri)]):
        r = tri[a:b]
        if len(r) >= int(params.get('min_pitches', 300)):
            maps[int(bat[r[0]])] = _ridge_offset(Bm[r], swing[r], off[r], float(params.get('lam', 10.0)))
    stage(f'hitter maps {len(maps)}')
    # each pitcher's 2024 pitches, as the arsenal sample (decision-moment crossing, count, outside or inside the zone)
    u_true = np.where(Tk['stand_r'] == 1, Tk['px'], -Tk['px'])
    outside = (np.abs(u_true) > ZONE_HALF) | (Tk['pz'] > ZONE_TOP) | (Tk['pz'] < ZONE_BOT)
    arsenal_rows = np.flatnonzero(Tk['season'] == 2024)
    pit = Tk['pitcher']
    o2 = np.argsort(pit[arsenal_rows], kind='stable'); ar = arsenal_rows[o2]; pa_ = pit[ar]
    cut2 = np.flatnonzero(np.diff(pa_)) + 1
    arsenal = {}
    for a, b in zip(np.r_[0, cut2], np.r_[cut2, len(ar)]):
        r = ar[a:b]
        if len(r) >= int(params.get('min_arsenal', 300)):
            arsenal[int(pit[r[0]])] = r if len(r) <= 1500 else rng.choice(r, 1500, replace=False)
    stage(f'arsenals {len(arsenal)}')
    # 2025 plate appearances (first pitch of each), their outcome and both players' earlier strikeout and walk rates
    P = take(T, (T['season'] == 2025) & (T['pitch_no'] == 0) & (T['out7'] >= 0))
    y7 = P['out7'].astype(int)
    K = (y7 == 1).astype(float); BB = (y7 == 2).astype(float)
    # earlier rates from 2023-2024 plate appearances plus earlier 2025 days
    PA_all = take(T, (T['pitch_no'] == 0) & (T['out7'] >= 0))
    def rate(key_all, key_p, cls, k):
        yy = (PA_all['out7'] == cls).astype(float)
        nn, ss = _prior_by_day(np.r_[key_all, key_p].astype(np.int64), np.r_[PA_all['day'], P['day'] + 0].astype(np.int64),
                               np.r_[yy, np.zeros(len(key_p))], np.r_[np.ones(len(key_all), bool), np.zeros(len(key_p), bool)])
        nn, ss = nn[len(key_all):], ss[len(key_all):]
        lg = yy.mean()
        rr = (ss + k * lg) / (nn + k)
        return np.log(rr / (1 - rr))
    kb, kp = rate(PA_all['batter'], P['batter'], 1, 150.0), rate(PA_all['pitcher'], P['pitcher'], 1, 300.0)
    bb_b, bb_p = rate(PA_all['batter'], P['batter'], 2, 150.0), rate(PA_all['pitcher'], P['pitcher'], 2, 300.0)
    stage('rates')
    # pair features: hitter map minus league on the pitcher's arsenal (the arsenal's hand split to the hitter's side)
    pairs = {}
    chase = np.full(len(y7), np.nan); zsw = np.full(len(y7), np.nan)
    for i, (h, p_) in enumerate(zip(P['batter'], P['pitcher'])):
        h, p_ = int(h), int(p_)
        if h not in maps or p_ not in arsenal:
            continue
        key = (h, p_, int(P['stand_r'][i]))
        if key not in pairs:
            r = arsenal[p_]
            r = r[Tk['stand_r'][r] == P['stand_r'][i]] if np.sum(Tk['stand_r'][r] == P['stand_r'][i]) >= 100 else r
            dev_logit = Bm[r] @ maps[h]
            p_l = 1 / (1 + np.exp(-off[r])); p_h = 1 / (1 + np.exp(-(off[r] + dev_logit)))
            o = outside[r]
            pairs[key] = (float((p_h - p_l)[o].mean()) if o.any() else 0.0, float((p_h - p_l)[~o].mean()) if (~o).any() else 0.0)
        chase[i], zsw[i] = pairs[key]
    stage(f'pairs {len(pairs)}')
    have = np.isfinite(chase)
    res['rows'] = {'plate_appearances_2025': int(len(y7)), 'with_pair_features': int(have.sum()), 'pairs': len(pairs),
                   'chase_dev_sd_points': round(float(np.nanstd(chase) * 100), 3), 'zone_swing_dev_sd_points': round(float(np.nanstd(zsw) * 100), 3)}
    plat = (P['stand_r'] == P['throw_r']).astype(float)
    games = P['game'][have]
    out = {}
    for name, y, base in (('strikeout', K, np.column_stack([kb, kp, plat])), ('walk', BB, np.column_stack([bb_b, bb_p, plat]))):
        Xb = base[have]; Xm = np.column_stack([Xb, chase[have] * 100, zsw[have] * 100]); yy = y[have]
        par = games % 2 == 0
        ll0 = np.zeros(len(yy)); ll1 = np.zeros(len(yy)); coefs = []
        from sklearn.linear_model import LogisticRegression
        for side in (True, False):
            fr, pr = par == side, par != side
            m0 = LogisticRegression(C=1e4, max_iter=500).fit(Xb[fr], yy[fr]); m1 = LogisticRegression(C=1e4, max_iter=500).fit(Xm[fr], yy[fr])
            ll0[pr] = logloss_vec(m0.predict_proba(Xb[pr])[:, 1], yy[pr]); ll1[pr] = logloss_vec(m1.predict_proba(Xm[pr])[:, 1], yy[pr])
            coefs.append(m1.coef_[0][-2:])
        full = LogisticRegression(C=1e4, max_iter=500).fit(Xm, yy)
        bs = []
        ug = np.unique(games); gi = np.searchsorted(ug, games)
        for _ in range(int(params.get('reps', 100))):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi]
            sel = np.repeat(np.arange(len(yy)), w)
            bs.append(LogisticRegression(C=1e4, max_iter=300).fit(Xm[sel], yy[sel]).coef_[0][-2:])
        bs = np.asarray(bs)
        out[name] = {'rate': round(float(yy.mean()), 4), 'gain_nats_per_1000_pa': [round(v * 1000, 3) for v in clustered_ci(ll0 - ll1, games)],
                     'coef_chase_dev_per_point': [round(float(full.coef_[0][-2]), 4), round(float(np.percentile(bs[:, 0], 2.5)), 4), round(float(np.percentile(bs[:, 0], 97.5)), 4)],
                     'coef_zone_swing_dev_per_point': [round(float(full.coef_[0][-1]), 4), round(float(np.percentile(bs[:, 1], 2.5)), 4), round(float(np.percentile(bs[:, 1], 97.5)), 4)]}
        # observed rate by quintile of the chase deviation, against the base model
        q = np.nanpercentile(chase[have], [20, 40, 60, 80]); qi = np.searchsorted(q, chase[have])
        m0f = LogisticRegression(C=1e4, max_iter=500).fit(Xb, yy); p0 = m0f.predict_proba(Xb)[:, 1]
        out[name]['by_chase_quintile'] = [{'chase_dev_points': round(float(chase[have][qi == j].mean() * 100), 2), 'observed': round(float(yy[qi == j].mean()), 4),
                                           'base_model': round(float(p0[qi == j].mean()), 4), 'pa': int((qi == j).sum())} for j in range(5)]
    res['outcomes'] = out
    stage('plate appearances')
    return res


# ---------------------------------------------------------------- ENGINE-01: a pitch-by-pitch plate appearance from the two clocks
def _count_chain(sw, wh, cs, fo):
    """P(strikeout), P(walk), P(ball in play) of a plate appearance from 0-0, given per-count chances (dicts keyed by
    (balls, strikes)): swing, whiff on a swing, called strike on a take, foul on contact."""
    from functools import lru_cache
    @lru_cache(None)
    def go(b, k):
        if k >= 3:
            return (1.0, 0.0, 0.0)
        if b >= 4:
            return (0.0, 1.0, 0.0)
        s, w, c, f = sw[(b, k)], wh[(b, k)], cs[(b, k)], fo[(b, k)]
        p_strike = s * w + (1 - s) * c + (s * (1 - w) * f if k < 2 else 0.0)
        p_ball = (1 - s) * (1 - c)
        p_stay = s * (1 - w) * f if k == 2 else 0.0
        p_bip = s * (1 - w) * (1 - f)
        ks, bs, ips = go(b, k + 1); kb, bb, ipb = go(b + 1, k)
        tot = 1 - p_stay
        return ((p_strike * ks + p_ball * kb) / tot, (p_strike * bs + p_ball * bb) / tot, (p_bip + p_strike * ips + p_ball * ipb) / tot)
    return go(0, 0)


def engine_pa(T: dict, params: dict, stage) -> dict:
    """Each 2025 hitter-pitcher pair played pitch by pitch from earlier seasons only: the pitcher's 2024 pitches in each
    count (his arsenal), the hitter's swing map at the decision moment (MATCHUP-01) and his whiff map on the true crossing
    (MATCHUP-03), league models for called strikes and fouls; a count-by-count chain gives the pair's strikeout and walk
    chances, with the hitter's maps and with the league maps plus his additive terms. Does the difference (the matchup
    the engine sees) predict the pair's 2025 strikeouts and walks beyond both players' rates?"""
    res = {}
    post_mode = bool(params.get('postseason'))          # EXPLOIT-03: maps from the 2023-2025 regular seasons, targeting in those postseasons
    if 'post' not in T:
        T = dict(T); T['post'] = np.zeros(len(T['season']), np.int64)
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    Tk = take(T, keep); Fk = {k: v[keep] for k, v in F.items()}
    swing = ((Tk['call'] == 1) | (Tk['call'] == 2)).astype(np.float64); whiff = (Tk['call'] == 2).astype(np.float64)
    take_ = Tk['call'] == 0; cs = (Tk['cs'] == 1).astype(np.float64)
    contact = Tk['call'] == 1
    foul = (contact & (Tk['last_in_pa'] == 0)).astype(np.float64)        # contact that did not end the plate appearance
    xp, zp = projected(Tk, Fk, None, 'straight', 0.26)
    xt, zt = Tk['px'].astype(np.float64), Tk['pz'].astype(np.float64)
    tr = np.isin(Tk['season'], (2023, 2024))
    rng = np.random.default_rng(11)
    def sample(rows, n=600000):
        idx = np.flatnonzero(rows); return rng.choice(idx, min(len(idx), n), replace=False) if len(idx) > n else idx
    # swing: league + hitter maps at the decision moment
    Cs = np.hstack([control_block(Tk, swing_propensity(Tk)), pitcher_propensity(Tk)[:, None].astype(np.float32)])
    famv = bool(params.get('family'))       # ENGINE-02: MATCHUP-05F's swing representation (league family part, hitter family maps)
    Bs = hitter_basis(xp, zp, Tk['stand_r'], Tk['strikes'])
    lf_parts = [(Bs[:, :-1] * np.isin(Tk['group'], (3, 4))[:, None]).astype(np.float32), (Bs[:, :-1] * (Tk['group'] == 5)[:, None]).astype(np.float32)] if famv else []
    Xs = np.hstack([location_block(xp, zp, Tk['stand_r'], Tk['strikes'])] + lf_parts + [Cs]); del lf_parts
    i = sample(tr); off_s = fit_logistic(Xs[i], swing[i]).decision_function(Xs); del Xs
    if famv:
        Bs = family_basis(Bs, Tk['group'])
    maps_s = _hitter_maps(Bs, swing, off_s, _groups(Tk['batter'], tr), 10.0, 300)
    stage(f'swing maps {len(maps_s)}')
    # whiff: league + hitter maps on the true crossing, on swings
    tw = dict(Tk); tw['call'] = np.where(whiff == 1, 1, np.where(swing == 1, 0, 3))
    prop_w = swing_propensity(tw, 200.0)
    grp = np.zeros((len(swing), 7), np.float32); grp[np.arange(len(swing)), np.clip(Tk['group'], 0, 6)] = 1
    Cw = np.hstack([grp, hats(Tk['v0'].astype(np.float64), V_KNOTS), (Tk['strikes'] == 2)[:, None], prop_w[:, None], (Tk['stand_r'] == Tk['throw_r'])[:, None]]).astype(np.float32)
    Xw = np.hstack([location_block(xt, zt, Tk['stand_r'], Tk['strikes']), Cw])
    sw_rows = tr & (swing == 1)
    i = sample(sw_rows); off_w = fit_logistic(Xw[i], whiff[i]).decision_function(Xw)
    fam = np.column_stack([np.isin(Tk['group'], (0, 1, 2)), np.isin(Tk['group'], (3, 4)), np.isin(Tk['group'], (5,))]).astype(np.float64)
    Bw = np.hstack([hitter_basis(xt, zt, Tk['stand_r'], Tk['strikes']), fam, hats(zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
    maps_w = _hitter_maps(Bw, whiff, off_w, _groups(Tk['batter'], sw_rows), 30.0, 250)
    stage(f'whiff maps {len(maps_w)}')
    # each hitter's map as a constant: its average log-odds shift over his own training pitches (swings for whiffs)
    g_s, g_w = _groups(Tk['batter'], tr), _groups(Tk['batter'], sw_rows)
    lev_s = {h: float((Bs[g_s[h]] @ b).mean()) for h, b in maps_s.items()}
    lev_w = {h: float((Bw[g_w[h]] @ b).mean()) for h, b in maps_w.items()}
    # league called strikes on takes and fouls on contact
    Xc = np.hstack([location_block(xt, zt, Tk['stand_r'], Tk['strikes']), (Tk['stand_r'] == Tk['throw_r'])[:, None].astype(np.float32)])
    i = sample(tr & take_); p_cs = fit_logistic(Xc[i], cs[i]).predict_proba(Xc)[:, 1]
    i = sample(tr & contact); p_fo = fit_logistic(Xw[i], foul[i]).predict_proba(Xw)[:, 1]
    stage('called strikes and fouls')
    # arsenals: each pitcher's 2024 pitches
    ars = {pid: r for pid, r in _groups(Tk['pitcher'], Tk['season'] == 2024).items() if len(r) >= 400}
    P = take(T, (T['season'] == 2025) & (T['pitch_no'] == 0) & (T['out7'] >= 0))
    pairs = {}
    for b_, p_, st in set(zip(P['batter'].tolist(), P['pitcher'].tolist(), P['stand_r'].tolist())):
        if b_ in maps_s and b_ in maps_w and p_ in ars:
            pairs[(b_, p_, st)] = None
    stage(f'pairs {len(pairs)}')
    sig = lambda v: 1 / (1 + np.exp(-v))
    def chain_from(ps, pw, pcs, pfo, ci):
        # per-count chances over the arsenal's pitches in that count (the strike count when fewer than 15 pitches)
        use_c = np.bincount(ci, minlength=12) >= 15; ki = ci % 3
        def agg(w):
            return np.where(use_c, np.bincount(ci, weights=w, minlength=12), np.bincount(ki, weights=w, minlength=3)[np.arange(12) % 3])
        n_ = agg(np.ones(len(ps))); s_ = agg(ps); sw_ = agg(ps * pw); t_ = agg(1 - ps); tc_ = agg((1 - ps) * pcs)
        c_ = agg(ps * (1 - pw)); cf_ = agg(ps * (1 - pw) * pfo)
        d = lambda v: {(b, k): float(v[b * 3 + k]) for b in range(4) for k in range(3)}
        return _count_chain(d(s_ / np.maximum(n_, 1e-9)), d(sw_ / np.maximum(s_, 1e-9)), d(tc_ / np.maximum(t_, 1e-9)), d(cf_ / np.maximum(c_, 1e-9)))
    cidx = (Tk['balls'].astype(np.int64) * 3 + Tk['strikes'].astype(np.int64))
    arsenal_cache = {}
    for key in list(pairs):
        b_, p_, st = key
        if (p_, st) not in arsenal_cache:
            r = ars[p_]
            side = Tk['stand_r'][r] == st
            if side.sum() >= 150:
                r = r[side]
            A = {'r': r, 'ci': cidx[r], 'pcs': p_cs[r], 'pfo': p_fo[r], 'os': off_s[r], 'ow': off_w[r], 'Bs': Bs[r], 'Bw': Bw[r]}
            A['league'] = chain_from(sig(A['os']), sig(A['ow']), A['pcs'], A['pfo'], A['ci'])
            arsenal_cache[(p_, st)] = A
        A = arsenal_cache[(p_, st)]
        hit = chain_from(sig(A['os'] + A['Bs'] @ maps_s[b_]), sig(A['ow'] + A['Bw'] @ maps_w[b_]), A['pcs'], A['pfo'], A['ci'])
        lev = chain_from(sig(A['os'] + lev_s[b_]), sig(A['ow'] + lev_w[b_]), A['pcs'], A['pfo'], A['ci'])
        lg = A['league']
        pairs[key] = (hit[0] - lg[0], hit[1] - lg[1], lg[0], lg[1], lev[0] - lg[0], lev[1] - lg[1])
    stage('chains')
    # 2025 plate appearances: outcome on rates plus the engine's matchup deviation
    y7 = P['out7'].astype(int); K = (y7 == 1).astype(float); BB = (y7 == 2).astype(float)
    PA_all = take(T, (T['pitch_no'] == 0) & (T['out7'] >= 0))
    def rate(key_all, key_p, cls, k):
        yy = (PA_all['out7'] == cls).astype(float)
        nn, ss = _prior_by_day(np.r_[key_all, key_p].astype(np.int64), np.r_[PA_all['day'], P['day']].astype(np.int64),
                               np.r_[yy, np.zeros(len(key_p))], np.r_[np.ones(len(key_all), bool), np.zeros(len(key_p), bool)])
        nn, ss = nn[len(key_all):], ss[len(key_all):]
        lg = yy.mean(); rr = (ss + k * lg) / (nn + k)
        return np.log(rr / (1 - rr))
    kb, kp = rate(PA_all['batter'], P['batter'], 1, 150.0), rate(PA_all['pitcher'], P['pitcher'], 1, 300.0)
    bb_b, bb_p = rate(PA_all['batter'], P['batter'], 2, 150.0), rate(PA_all['pitcher'], P['pitcher'], 2, 300.0)
    dk, db, lk, lb, ck, cb = (np.full(len(y7), np.nan) for _ in range(6))
    for j, key in enumerate(zip(P['batter'].tolist(), P['pitcher'].tolist(), P['stand_r'].tolist())):
        v = pairs.get(key)
        if v is not None:
            dk[j], db[j], lk[j], lb[j], ck[j], cb[j] = v
    have = np.isfinite(dk)
    res['rows'] = {'plate_appearances': int(len(y7)), 'with_engine': int(have.sum()), 'pairs': len(pairs),
                   'engine_k_dev_sd_points': round(float(np.nanstd(dk) * 100), 3), 'engine_bb_dev_sd_points': round(float(np.nanstd(db) * 100), 3),
                   'chain_level_check': {'league_chain_k': round(float(np.nanmean(lk)), 4), 'observed_k': round(float(K[have].mean()), 4),
                                         'league_chain_bb': round(float(np.nanmean(lb)), 4), 'observed_bb': round(float(BB[have].mean()), 4)}}
    from sklearn.linear_model import LogisticRegression
    games = P['game'][have]; plat = (P['stand_r'] == P['throw_r']).astype(float)
    hit_id, pit_id = P['batter'][have], P['pitcher'][have]

    def two_way(v, a, b, iters=12):
        # v = grand mean + hitter part + pitcher part + pair remainder (backfitting over the 2025 plate appearances)
        ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
        na, nb = np.bincount(ia), np.bincount(ib); mu = float(v.mean()); ea = np.zeros(len(ua)); eb = np.zeros(len(ub))
        for _ in range(iters):
            ea = np.bincount(ia, weights=v - mu - eb[ib]) / na
            eb = np.bincount(ib, weights=v - mu - ea[ia]) / nb
        return ea[ia], eb[ib], v - mu - ea[ia] - eb[ib]

    def crossfit(Xa, Xb_, yy):
        par = games % 2 == 0; la = np.zeros(len(yy)); lb_ = np.zeros(len(yy))
        for side in (True, False):
            fr, pr = par == side, par != side
            ma = LogisticRegression(C=1e4, max_iter=500).fit(Xa[fr], yy[fr]); mb = LogisticRegression(C=1e4, max_iter=500).fit(Xb_[fr], yy[fr])
            la[pr] = logloss_vec(ma.predict_proba(Xa[pr])[:, 1], yy[pr]); lb_[pr] = logloss_vec(mb.predict_proba(Xb_[pr])[:, 1], yy[pr])
        return [round(v * 1000, 3) for v in clustered_ci(la - lb_, games)]

    def boot_coefs(X, yy, cols):
        ug = np.unique(games); gi = np.searchsorted(ug, games); bs = []
        for _ in range(int(params.get('reps', 100))):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi]; sel = np.repeat(np.arange(len(yy)), w)
            bs.append(LogisticRegression(C=1e4, max_iter=300).fit(X[sel], yy[sel]).coef_[0][cols])
        full = LogisticRegression(C=1e4, max_iter=500).fit(X, yy).coef_[0][cols]
        bs = np.asarray(bs)
        return [[round(float(full[i]), 4), round(float(np.percentile(bs[:, i], 2.5)), 4), round(float(np.percentile(bs[:, i], 97.5)), 4)] for i in range(len(cols))]

    out = {}
    for name, y, base, dev, devc in (('strikeout', K, np.column_stack([kb, kp, plat]), dk, ck), ('walk', BB, np.column_stack([bb_b, bb_p, plat]), db, cb)):
        Xb = base[have]; yy = y[have]
        d = dev[have] * 100                                   # engine deviation in points of probability
        h_part, p_part, pair = two_way(d, hit_id, pit_id)
        # the pair part split: what the count chain makes of the hitter's tendencies as constants (no map shape), and
        # what the shape of his maps adds against this pitcher's arsenal
        pair_chain = two_way(devc[have] * 100, hit_id, pit_id)[2]; pair_shape = pair - pair_chain
        X_raw = np.column_stack([Xb, d]); X_lv = np.column_stack([Xb, h_part, p_part]); X_all = np.column_stack([X_lv, pair])
        X_ch = np.column_stack([X_lv, pair_chain]); X_split = np.column_stack([X_lv, pair_chain, pair_shape])
        q = np.nanpercentile(pair, [20, 40, 60, 80]); qi = np.searchsorted(q, pair)
        p_lv = LogisticRegression(C=1e4, max_iter=500).fit(X_lv, yy).predict_proba(X_lv)[:, 1]
        out[name] = {'sd_points': {'raw': round(float(d.std()), 3), 'hitter_part': round(float(h_part.std()), 3), 'pitcher_part': round(float(p_part.std()), 3),
                                   'pair_part': round(float(pair.std()), 3), 'pair_chain': round(float(pair_chain.std()), 3), 'pair_shape': round(float(pair_shape.std()), 3)},
                     'gain_nats_per_1000_pa': {'pair_over_levels': crossfit(X_lv, X_all, yy), 'shape_over_levels_and_chain': crossfit(X_ch, X_split, yy),
                                               'levels_over_rates': crossfit(Xb, X_lv, yy), 'raw_over_rates': crossfit(Xb, X_raw, yy)},
                     'coef_per_point': dict(zip(('hitter_part', 'pitcher_part', 'pair_part'), boot_coefs(X_all, yy, [-3, -2, -1]))),
                     'coef_split_per_point': dict(zip(('pair_chain', 'pair_shape'), boot_coefs(X_split, yy, [-2, -1]))),
                     'coef_raw_per_point': boot_coefs(X_raw, yy, [-1])[0],
                     'by_pair_quintile': [{'pair_points': round(float(pair[qi == j].mean()), 3), 'observed': round(float(yy[qi == j].mean()), 4),
                                           'levels_model': round(float(p_lv[qi == j].mean()), 4), 'pa': int((qi == j).sum())} for j in range(5)]}
        stage('outcome ' + name)
    res['outcomes'] = out
    return res


# ---------------------------------------------------------------- DISCIPLINE-01: decision-moment discipline as a projection input
def discipline_study(T: dict, params: dict, stage) -> dict:
    """Do a hitter's decision-moment swing map and true-crossing whiff map, read on one standard set of pitches (so the
    pitches he happened to be thrown drop out), project his next season's strikeouts and walks better than his own
    rates and raw plate-discipline numbers (chase rate, zone-swing rate, whiffs per swing)? Each source season's maps
    and league models are fitted on that season only; targets are the next season's plate appearances. Trained on
    2023 to 2024, scored on 2024 to 2025 and 2025 to 2026 (through July); intervals clustered by hitter."""
    res = {}
    seasons = sorted(int(v) for v in np.unique(T['season']))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    Tk = take(T, keep); Fk = {k: v[keep] for k, v in F.items()}
    swing = ((Tk['call'] == 1) | (Tk['call'] == 2)).astype(np.float64); whiff = (Tk['call'] == 2).astype(np.float64)
    xp, zp = projected(Tk, Fk, None, 'straight', 0.26)
    xt, zt = Tk['px'].astype(np.float64), Tk['pz'].astype(np.float64)
    u_t = np.where(Tk['stand_r'] == 1, xt, -xt)
    outside = (np.abs(u_t) > ZONE_HALF) | (zt > ZONE_TOP) | (zt < ZONE_BOT)
    prop_s = swing_propensity(Tk); prop_p = pitcher_propensity(Tk)
    Ls = location_block(xp, zp, Tk['stand_r'], Tk['strikes']); Cs = control_block(Tk, prop_s)
    Xs = np.hstack([Ls, Cs, prop_p[:, None].astype(np.float32)]); i_prop_s = Ls.shape[1] + 32
    Bs = hitter_basis(xp, zp, Tk['stand_r'], Tk['strikes'])
    tw = dict(Tk); tw['call'] = np.where(whiff == 1, 1, np.where(swing == 1, 0, 3)); prop_w = swing_propensity(tw, 200.0)
    grp = np.zeros((len(swing), 7), np.float32); grp[np.arange(len(swing)), np.clip(Tk['group'], 0, 6)] = 1
    Lw = location_block(xt, zt, Tk['stand_r'], Tk['strikes'])
    Xw = np.hstack([Lw, grp, hats(Tk['v0'].astype(np.float64), V_KNOTS), (Tk['strikes'] == 2)[:, None].astype(np.float32), prop_w[:, None].astype(np.float32),
                    (Tk['stand_r'] == Tk['throw_r'])[:, None].astype(np.float32)]); i_prop_w = Lw.shape[1] + 7 + len(V_KNOTS) + 1
    fam = np.column_stack([np.isin(Tk['group'], (0, 1, 2)), np.isin(Tk['group'], (3, 4)), np.isin(Tk['group'], (5,))]).astype(np.float64)
    Bw = np.hstack([hitter_basis(xt, zt, Tk['stand_r'], Tk['strikes']), fam, hats(zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
    rng = np.random.default_rng(int(params.get('seed', 11)))
    stage('features')
    lg = lambda r: np.log(r / (1 - r))
    feats = {}
    for s_ in seasons[:-1]:
        rows = Tk['season'] == s_
        if rows.sum() < 100000:
            continue
        idx = np.flatnonzero(rows); idx = rng.choice(idx, min(len(idx), int(params.get('league_n', 400000))), replace=False)
        m_s = fit_logistic(Xs[idx], swing[idx])
        maps_s = _hitter_maps(Bs, swing, m_s.decision_function(Xs), _groups(Tk['batter'], rows), 10.0, int(params.get('min_pitches', 600)))
        sw_rows = rows & (swing == 1)
        idx = np.flatnonzero(sw_rows); idx = rng.choice(idx, min(len(idx), int(params.get('league_n', 400000))), replace=False)
        m_w = fit_logistic(Xw[idx], whiff[idx])
        maps_w = _hitter_maps(Bw, whiff, m_w.decision_function(Xw), _groups(Tk['batter'], sw_rows), 30.0, 250)
        ref = rng.choice(np.flatnonzero(rows), min(int(rows.sum()), int(params.get('reference_n', 40000))), replace=False)
        ref_side = {side: ref[Tk['stand_r'][ref] == side] for side in (0, 1)}
        # league logits on the reference pitches with the propensity column at zero; a hitter's own propensity enters
        # through its coefficient (the models are linear in it)
        base_s, base_w = {}, {}
        for side, R in ref_side.items():
            Xr = Xs[R].astype(np.float64).copy(); Xr[:, i_prop_s] = 0.0; base_s[side] = m_s.decision_function(Xr)
            Xq = Xw[R].astype(np.float64).copy(); Xq[:, i_prop_w] = 0.0; base_w[side] = m_w.decision_function(Xq)
        c_s, c_w = float(m_s.coef_[0][i_prop_s]), float(m_w.coef_[0][i_prop_w])
        lg_sw = float(swing[rows].mean()); lg_wh = float(whiff[rows & (swing == 1)].mean())
        gr = _groups(Tk['batter'], rows)
        out = {}
        for h, r in gr.items():
            if h not in maps_s or h not in maps_w:
                continue
            side = int(np.round(Tk['stand_r'][r].mean()))
            R = ref_side[side]
            n_ = len(r); ps = (swing[r].sum() + 300 * lg_sw) / (n_ + 300)
            pw = (whiff[r].sum() + 200 * lg_wh) / (swing[r].sum() + 200)
            p_sw = 1 / (1 + np.exp(-(base_s[side] + c_s * lg(ps) + Bs[R] @ maps_s[h])))
            p_wh = 1 / (1 + np.exp(-(base_w[side] + c_w * lg(pw) + Bw[R] @ maps_w[h])))
            o = outside[R]
            out[h] = {'map_chase': float(p_sw[o].mean()), 'map_zswing': float(p_sw[~o].mean()), 'map_whiff': float((p_sw * p_wh).sum() / p_sw.sum()),
                      'raw_chase': float((swing[r][outside[r]].sum() + 30 * 0.29) / (outside[r].sum() + 30)), 'raw_zswing': float((swing[r][~outside[r]].sum() + 30 * 0.66) / ((~outside[r]).sum() + 30)),
                      'raw_whiff': float((whiff[r].sum() + 50 * 0.24) / (swing[r].sum() + 50)), 'pitches': n_}
        feats[s_] = out
        stage(f'season {s_}: {len(out)} hitters')
    # plate appearances of the next season, with the source season's rates
    P = take(T, (T['pitch_no'] == 0) & (T['out7'] >= 0))
    K = (P['out7'] == 1).astype(float); BB = (P['out7'] == 2).astype(float)
    rate = {}
    for s_ in seasons:
        m = P['season'] == s_
        hb = P['batter'][m]; uh, ih = np.unique(hb, return_inverse=True)
        nk = np.bincount(ih, weights=K[m]); nb = np.bincount(ih, weights=BB[m]); npa = np.bincount(ih)
        rate[s_] = {int(h): ((nk[i] + 100 * K[m].mean()) / (npa[i] + 100), (nb[i] + 100 * BB[m].mean()) / (npa[i] + 100), int(npa[i])) for i, h in enumerate(uh)}
    names_base = ['k_rate', 'bb_rate', 'raw_chase', 'raw_zswing', 'raw_whiff', 'log_pa']
    names_map = ['map_chase', 'map_zswing', 'map_whiff']
    blocks = {}
    for s_ in sorted(feats):
        t_ = s_ + 1
        if t_ not in seasons:
            continue
        m = P['season'] == t_
        rows = []
        for j in np.flatnonzero(m):
            h = int(P['batter'][j]); f = feats[s_].get(h); rr = rate[s_].get(h)
            if f is None or rr is None:
                continue
            rows.append((j, [lg(rr[0]), lg(rr[1]), lg(f['raw_chase']), lg(f['raw_zswing']), lg(f['raw_whiff']), np.log(rr[2])],
                         [lg(f['map_chase']), lg(f['map_zswing']), lg(f['map_whiff'])]))
        if rows:
            J = np.asarray([r_[0] for r_ in rows]); XB = np.asarray([r_[1] for r_ in rows]); XM = np.asarray([r_[2] for r_ in rows])
            blocks[(s_, t_)] = (J, XB, XM)
    res['pairs'] = {f'{a}-{b}': int(len(v[0])) for (a, b), v in blocks.items()}
    keys = sorted(blocks)
    if len(keys) < 2:
        res['error'] = 'need a training pair and a test pair'; return res
    train = keys[0]; tests = keys[1:]
    from sklearn.linear_model import LogisticRegression
    out = {}
    for name, y in (('strikeout', K), ('walk', BB)):
        J, XB, XM = blocks[train]
        m0 = LogisticRegression(C=1e4, max_iter=500).fit(XB, y[J]); m1 = LogisticRegression(C=1e4, max_iter=500).fit(np.hstack([XB, XM]), y[J])
        o = {'coefs_map_terms': dict(zip(names_map, [round(float(c), 4) for c in m1.coef_[0][-3:]]))}
        for tk in tests:
            Jt, XBt, XMt = blocks[tk]; yt = y[Jt]
            l0 = logloss_vec(m0.predict_proba(XBt)[:, 1], yt); l1 = logloss_vec(m1.predict_proba(np.hstack([XBt, XMt]))[:, 1], yt)
            o[f'gain_nats_per_1000_pa_{tk[0]}_{tk[1]}'] = [round(v * 1000, 3) for v in clustered_ci(l0 - l1, P['batter'][Jt])]
        out[name] = o
    res['outcomes'] = out
    # how the map measures relate to the raw ones (training pair)
    J, XB, XM = blocks[train]
    res['corr_map_raw'] = {'chase': round(float(np.corrcoef(XB[:, 2], XM[:, 0])[0, 1]), 3), 'zswing': round(float(np.corrcoef(XB[:, 3], XM[:, 1])[0, 1]), 3),
                           'whiff': round(float(np.corrcoef(XB[:, 4], XM[:, 2])[0, 1]), 3)}
    stage('projections')
    return res


# ---------------------------------------------------------------- EXPLOIT-01: do pitchers already aim at a hitter's own swing map?
def exploit_study(T: dict, params: dict, stage) -> dict:
    """Hitter swing maps at the decision moment fitted on 2023-2024 (MATCHUP-01, shrinkage 10), read on 2025 pitches.
    For each 2025 hitter-pitcher pair, the hitter's map deviation from the league (extra swing chance) on the pitches
    the pitcher actually threw him, against the same pitcher's 2025 pitches to other hitters of the same side read
    through this hitter's map (what he would have seen if pitched like everyone else), in the same count group. Outside
    the zone a positive difference means pitchers put more pitches where this hitter chases; inside the zone a negative
    one means more pitches where he takes. Headroom: the same pitcher's other pitches in the same count group, the
    best of them for this hitter (as many as were thrown to him), gives what aiming fully at the map would reach.
    Also: whether the targeting grows with how distinct the hitter's map is, and whether targeted pitches got the extra
    chases the map predicts."""
    res = {}
    post_mode = bool(params.get('postseason'))          # EXPLOIT-03: maps from the 2023-2025 regular seasons, targeting in those postseasons
    if 'post' not in T:
        T = dict(T); T['post'] = np.zeros(len(T['season']), np.int64)
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    Lb = location_block(xp, zp, T['stand_r'], T['strikes']); prop = swing_propensity(T)
    X = np.hstack([Lb, control_block(T, prop), pitcher_propensity(T)[:, None].astype(np.float32)]); i_prop = Lb.shape[1] + 32
    if post_mode:
        tr = T['post'] == 0; te = (T['post'] == 1) & np.isin(T['season'], post_seasons)
        res['postseason_pitches'] = {str(y): int((te & (T['season'] == y)).sum()) for y in post_seasons}
        res['regular_season_pitches_for_maps'] = int(tr.sum())
    else:
        tr = np.isin(T['season'], (2023, 2024)) & (T['post'] == 0); te = (T['season'] == 2025) & (T['post'] == 0)
    rng = np.random.default_rng(11)
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
    league = fit_logistic(X[idx], swing[idx]); off = league.decision_function(X); c_prop = float(league.coef_[0][i_prop])
    Bm = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], tr), 10.0, 300)
    stage(f'maps {len(maps)}')
    u_t = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    outside = (np.abs(u_t) > ZONE_HALF) | (T['pz'] > ZONE_TOP) | (T['pz'] < ZONE_BOT)
    cgrp = np.where(T['strikes'] == 2, 2, np.where(T['balls'] > T['strikes'], 1, 0))          # two strikes, behind, other
    p_l = 1 / (1 + np.exp(-off))
    # each 2025 pitcher's pitches by side and count group (sampled), for the counterfactual
    pool = {}
    pool_src = te
    if post_mode and params.get('pool') == 'regular':          # his regular-season pitches of that year to other hitters
        pool_src = (T['post'] == 0)
    pool_season = bool(post_mode)
    for pid, r in _groups(T['pitcher'], pool_src).items():
        for ssn in (post_seasons if pool_season else (None,)):
            r_s = r if ssn is None else r[T['season'][r] == ssn]
            for side in (0, 1):
                for cg in (0, 1, 2):
                    rr = r_s[(T['stand_r'][r_s] == side) & (cgrp[r_s] == cg)]
                    if len(rr) >= 30:
                        pool[(pid, side, cg) if ssn is None else (pid, ssn, side, cg)] = rr if len(rr) <= 600 else rng.choice(rr, 600, replace=False)
    rows = []
    bat_te = _groups(T['batter'], te)
    for h, r in bat_te.items():
        if h not in maps:
            continue
        mh = maps[h]
        pr = T['pitcher'][r]
        for pid in np.unique(pr):
          rp_all = r[pr == pid]
          for ssn in (np.unique(T['season'][rp_all]) if pool_season else (None,)):
            rp = rp_all if ssn is None else rp_all[T['season'][rp_all] == ssn]
            side = int(T['stand_r'][rp[0]])
            for zone_name, zmask in (('outside', outside), ('inside', ~outside)):
                for cg in (0, 1, 2):
                    a = rp[zmask[rp] & (cgrp[rp] == cg)]
                    key = (int(pid), side, cg) if ssn is None else (int(pid), int(ssn), side, cg)
                    if len(a) == 0 or key not in pool:
                        continue
                    b = pool[key]; b = b[zmask[b] & (T['batter'][b] != h)]
                    if len(b) < 10:
                        continue
                    da = 1 / (1 + np.exp(-(off[a] + Bm[a] @ mh))) - p_l[a]
                    ob = off[b] + c_prop * (float(prop[a].mean()) - prop[b])        # the other hitters' pitches with this hitter's own swing level
                    db = 1 / (1 + np.exp(-(ob + Bm[b] @ mh))) - 1 / (1 + np.exp(-ob))
                    best = np.sort(db)[::-1][:len(a)] if zone_name == 'outside' else np.sort(db)[:len(a)]
                    rows.append((h, int(pid), zone_name == 'outside', cg, len(a), float(da.mean()), float(db.mean()), float(best.mean()),
                                 float(swing[a].mean()), float(p_l[a].mean())))
    stage(f'pair cells {len(rows)}')
    if not rows:
        res['error'] = 'no pairs'; return res
    R = np.asarray([r_[2:] for r_ in rows], float)          # outside, cg, n, actual, generic, best, observed swing, league
    H = np.asarray([r_[0] for r_ in rows]); out = {}
    for zone_name, zo in (('outside', 1.0), ('inside', 0.0)):
        m = R[:, 0] == zo
        sg = 1.0 if zo else -1.0                     # inside the zone, aiming at the map means fewer swings (more takes)
        n = R[m, 2]; act, gen, best = sg * R[m, 3], sg * R[m, 4], sg * R[m, 5]
        tgt = np.average(act - gen, weights=n); room = np.average(best - gen, weights=n)
        # bootstrap over hitters
        uh, ih = np.unique(H[m], return_inverse=True)
        sa = np.bincount(ih, weights=n * (act - gen)); sr = np.bincount(ih, weights=n * (best - gen)); sn = np.bincount(ih, weights=n)
        bs = []
        for _ in range(300):
            w = np.bincount(rng.integers(0, len(uh), len(uh)), minlength=len(uh))
            bs.append(((w * sa).sum() / (w * sn).sum(), (w * sa).sum() / max((w * sr).sum(), 1e-12)))
        bs = np.asarray(bs)
        # does targeting grow with how distinct the hitter's map is (his generic deviation's size)
        hs = {}
        for i_, h in enumerate(H[m]):
            hs.setdefault(h, []).append((n[i_], act[i_] - gen[i_], gen[i_]))
        hv = np.asarray([[sum(a_[0] for a_ in v), sum(a_[0] * a_[1] for a_ in v) / sum(a_[0] for a_ in v), sum(a_[0] * a_[2] for a_ in v) / sum(a_[0] for a_ in v)] for v in hs.values()])
        slope = float(np.polyfit(hv[:, 2], hv[:, 1], 1, w=np.sqrt(hv[:, 0]))[0]) if len(hv) > 10 else None
        out[zone_name] = {'pitches': int(n.sum()), 'cells': int(m.sum()), 'hitters': int(len(uh)),
                          'targeting_points': [round(float(tgt) * 100, 3), round(float(np.percentile(bs[:, 0], 2.5)) * 100, 3), round(float(np.percentile(bs[:, 0], 97.5)) * 100, 3)],
                          'headroom_points': round(float(room) * 100, 3),
                          'share_of_headroom_used': [round(float(tgt / room), 3) if room != 0 else None, round(float(np.percentile(bs[:, 1], 2.5)), 3), round(float(np.percentile(bs[:, 1], 97.5)), 3)],
                          'targeting_on_generic_deviation_slope': round(slope, 3) if slope is not None else None,
                          'generic_deviation_sd_points': round(float(np.sqrt(np.average((gen - np.average(gen, weights=n)) ** 2, weights=n))) * 100, 3)}
    res['by_zone'] = out
    stage('targeting')
    return res


# ---------------------------------------------------------------- EXPLOIT-02: do pitchers aim at a hitter's own whiff holes?
def exploit2_study(T: dict, params: dict, stage) -> dict:
    """EXPLOIT-01 for both of a hitter's maps, each as his own part (his map minus the same-side mean map of the other
    hitters with maps, VALUE-08): the whiff map on the true crossing (MATCHUP-03's representation: league whiff model on
    swings, hitter maps on location, family and height bands, shrinkage 30, at least 250 swings) and the swing map at
    the decision moment (MATCHUP-05F's: league with its family part, family maps, shrinkage 10). Maps from 2023-2024;
    for each 2025 hitter-pitcher pair, zone side and count group, the own-part deviation at the pitches thrown to him
    against the same pitcher's pitches to other hitters of that side read through his maps at his own levels. Whiffs:
    positive means more pitches where he misses more than the average hitter (on a swing). Chases: outside positive
    means more pitches where he chases more; inside negative means more where he takes more. Headroom: the best of the
    same pool. Bootstrap over hitters."""
    res = {}
    post_mode = bool(params.get('postseason'))          # EXPLOIT-03: maps from the 2023-2025 regular seasons, targeting in those postseasons
    if 'post' not in T:
        T = dict(T); T['post'] = np.zeros(len(T['season']), np.int64)
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64); whiff = (T['call'] == 2).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    xt, zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    tr = np.isin(T['season'], (2023, 2024)); te = T['season'] == 2025
    rng = np.random.default_rng(11)
    def sample(rows, n=600000):
        idx = np.flatnonzero(rows); return rng.choice(idx, min(len(idx), n), replace=False) if len(idx) > n else idx
    def own_maps(Bm, y, off, rows, lam, min_n):
        maps = _hitter_maps(Bm, y, off, _groups(T['batter'], rows), lam, min_n)
        ub_, bi_ = np.unique(T['batter'][tr], return_inverse=True)
        side_of = dict(zip(ub_.tolist(), np.round(np.bincount(bi_, weights=T['stand_r'][tr]) / np.bincount(bi_)).astype(int).tolist()))
        Ss = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
        for h, m in maps.items():
            Ss[side_of[h]] = Ss[side_of[h]] + m; Nn[side_of[h]] += 1
        return maps, {h: (Ss[side_of[h]] - m) / max(Nn[side_of[h]] - 1, 1) for h, m in maps.items()}
    kinds = {}
    # whiffs: league on swings at the true crossing, every pitch read as if swung at
    tw = dict(T); tw['call'] = np.where(whiff == 1, 1, np.where(swing == 1, 0, 3)); prop_w = swing_propensity(tw, 200.0)
    grp = np.zeros((len(swing), 7), np.float32); grp[np.arange(len(swing)), np.clip(T['group'], 0, 6)] = 1
    Lw = location_block(xt, zt, T['stand_r'], T['strikes'])
    Xw = np.hstack([Lw, grp, hats(T['v0'].astype(np.float64), V_KNOTS), (T['strikes'] == 2)[:, None].astype(np.float32), prop_w[:, None].astype(np.float32),
                    (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)]); i_pw = Lw.shape[1] + 7 + len(V_KNOTS) + 1
    i = sample(tr & (swing == 1)); m_w = fit_logistic(Xw[i], whiff[i]); off_w = m_w.decision_function(Xw); c_w = float(m_w.coef_[0][i_pw]); del Xw, Lw
    famc = np.column_stack([np.isin(T['group'], (0, 1, 2)), np.isin(T['group'], (3, 4)), np.isin(T['group'], (5,))]).astype(np.float64)
    Bw = np.hstack([hitter_basis(xt, zt, T['stand_r'], T['strikes']), famc, hats(zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
    maps_w, mbar_w = own_maps(Bw, whiff, off_w, tr & (swing == 1), 30.0, 250)
    kinds['whiff'] = (Bw, off_w, prop_w, c_w, maps_w, mbar_w)
    stage(f'whiff maps {len(maps_w)}')
    # chases: MATCHUP-05F's swing representation
    prop_s = swing_propensity(T)
    Ls = location_block(xp, zp, T['stand_r'], T['strikes']); Bh = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    Xs = np.hstack([Ls, (Bh[:, :-1] * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32), (Bh[:, :-1] * (T['group'] == 5)[:, None]).astype(np.float32),
                    control_block(T, prop_s), pitcher_propensity(T)[:, None].astype(np.float32)]); i_ps = Ls.shape[1] + 2 * (Bh.shape[1] - 1) + 32
    i = sample(tr); m_s = fit_logistic(Xs[i], swing[i]); off_s = m_s.decision_function(Xs); c_s = float(m_s.coef_[0][i_ps]); del Xs, Ls
    Bs = family_basis(Bh, T['group']); del Bh
    maps_s, mbar_s = own_maps(Bs, swing, off_s, tr, 10.0, 300)
    kinds['chase'] = (Bs, off_s, prop_s, c_s, maps_s, mbar_s)
    stage(f'swing maps {len(maps_s)}')
    u_t = np.where(T['stand_r'] == 1, xt, -xt)
    outside = (np.abs(u_t) > ZONE_HALF) | (zt > ZONE_TOP) | (zt < ZONE_BOT)
    cgrp = np.where(T['strikes'] == 2, 2, np.where(T['balls'] > T['strikes'], 1, 0))
    pool = {}
    pool_src = te
    if post_mode and params.get('pool') == 'regular':          # his regular-season pitches of that year to other hitters
        pool_src = (T['post'] == 0)
    pool_season = bool(post_mode)
    for pid, r in _groups(T['pitcher'], pool_src).items():
        for ssn in (post_seasons if pool_season else (None,)):
            r_s = r if ssn is None else r[T['season'][r] == ssn]
            for side in (0, 1):
                for cg in (0, 1, 2):
                    rr = r_s[(T['stand_r'][r_s] == side) & (cgrp[r_s] == cg)]
                    if len(rr) >= 30:
                        pool[(pid, side, cg) if ssn is None else (pid, ssn, side, cg)] = rr if len(rr) <= 600 else rng.choice(rr, 600, replace=False)
    sig = lambda v: 1 / (1 + np.exp(-v))
    bat_te = _groups(T['batter'], te)
    for kind, (Bm, off, prop, c_prop, maps, mbar) in kinds.items():
        rows = []
        for h, r in bat_te.items():
            if h not in maps:
                continue
            mh, mb = maps[h], mbar[h]
            pr = T['pitcher'][r]
            for pid in np.unique(pr):
                rp = r[pr == pid]; side = int(T['stand_r'][rp[0]])
                for zo in (True, False):
                    for cg in (0, 1, 2):
                        a = rp[(outside[rp] == zo) & (cgrp[rp] == cg)]
                        key = (int(pid), side, cg)
                        if len(a) == 0 or key not in pool:
                            continue
                        b = pool[key]; b = b[(outside[b] == zo) & (T['batter'][b] != h)]
                        if len(b) < 10:
                            continue
                        da = sig(off[a] + Bm[a] @ mh) - sig(off[a] + Bm[a] @ mb)
                        ob = off[b] + c_prop * (float(prop[a].mean()) - prop[b])
                        db = sig(ob + Bm[b] @ mh) - sig(ob + Bm[b] @ mb)
                        sg = 1.0 if (kind == 'whiff' or zo) else -1.0
                        best = np.sort(sg * db)[::-1][:len(a)]
                        rows.append((h, zo, len(a), sg * float(da.mean()), sg * float(db.mean()), float(best.mean())))
        out = {}
        R = np.asarray([r_[1:] for r_ in rows], float); H = np.asarray([r_[0] for r_ in rows])
        for zone_name, zo in (('outside', 1.0), ('inside', 0.0)):
            m = R[:, 0] == zo
            n = R[m, 1]; act, gen, best = R[m, 2], R[m, 3], R[m, 4]
            tgt = np.average(act - gen, weights=n); room = np.average(best - gen, weights=n)
            uh, ih = np.unique(H[m], return_inverse=True)
            sa = np.bincount(ih, weights=n * (act - gen)); sr = np.bincount(ih, weights=n * (best - gen)); sn = np.bincount(ih, weights=n)
            bs = []
            for _ in range(int(params.get('reps', 300))):
                w = np.bincount(rng.integers(0, len(uh), len(uh)), minlength=len(uh))
                bs.append(((w * sa).sum() / (w * sn).sum(), (w * sa).sum() / max((w * sr).sum(), 1e-12)))
            bs = np.asarray(bs)
            out[zone_name] = {'pitches': int(n.sum()), 'hitters': int(len(uh)),
                              'targeting_points': [round(float(tgt) * 100, 3), round(float(np.percentile(bs[:, 0], 2.5)) * 100, 3), round(float(np.percentile(bs[:, 0], 97.5)) * 100, 3)],
                              'headroom_points': round(float(room) * 100, 3),
                              'share_of_headroom_used': [round(float(tgt / room), 3) if room != 0 else None, round(float(np.percentile(bs[:, 1], 2.5)), 3), round(float(np.percentile(bs[:, 1], 97.5)), 3)]}
        res[kind] = out
        stage('targeting ' + kind)
    return res


# ---------------------------------------------------------------- VALUE-01: what a hitter's own map is worth to the pitcher
def value_study(T: dict, params: dict, stage) -> dict:
    """Pitchers do not aim at a hitter's own decision-moment swing map (EXPLOIT-01), so where a pitch falls on that map,
    beyond the league's map at the same spot, is as good as chance with respect to everything else about the plate
    appearance. Each 2025 pitch: d = the hitter's extra swing chance there (map fitted on 2023-2024, MATCHUP-01). The
    plate appearance's run value (linear weights of its final outcome) on d outside and inside the zone, with the count
    (twelve categories), the league's swing chance at that spot, zone distance bands, pitch group, both players' earlier
    run values and platoon; game-clustered intervals. Placebo: d from another hitter's map (same side, shuffled) must
    show nothing. Worth of aiming: within each pitcher's own outside pitches to a side in a count group, the best third
    for this hitter against the average, times the outside coefficient and the outside pitches per plate appearance."""
    res = {}
    final = bool(params.get('final_eval'))
    if final and not params.get('frozen_commit'):
        raise ValueError('the final value scoring runs only as the registered evaluation of a frozen commit')
    if final:      # VALUE-01F: maps from 2023-2025, scored on the 2026 pitches from August 1 (the untouched set)
        T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | ((T['season'] == 2026) & (T['day'] >= date(2026, 8, 1).toordinal())))
    else:
        post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    X = np.hstack([location_block(xp, zp, T['stand_r'], T['strikes']), control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    tr = np.isin(T['season'], (2023, 2024, 2025)) if final else np.isin(T['season'], (2023, 2024))
    test_season = 2026 if final else 2025
    rng = np.random.default_rng(11)
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
    off = fit_logistic(X[idx], swing[idx]).decision_function(X)
    Bm = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], tr), 10.0, 300)
    stage(f'maps {len(maps)}')
    te = (T['season'] == test_season) & (T['out7'] >= 0) & np.isin(T['batter'], np.asarray(list(maps), dtype=np.int64))
    ix = np.flatnonzero(te)
    sig = lambda v: 1 / (1 + np.exp(-v))
    p_l = sig(off[ix])
    bat = T['batter'][ix]
    D = np.zeros(len(ix)); Dp = np.zeros(len(ix))
    # placebo: each hitter's pitches read through another hitter's map of the same side
    sides = {}
    for h in maps:
        rows_h = T['stand_r'][T['batter'] == h]
        sides.setdefault(int(np.round(rows_h.mean())) if len(rows_h) else 1, []).append(h)
    perm = {}
    for sd, hs in sides.items():
        sh = list(hs); rng.shuffle(sh)
        perm.update({h: sh[(k + 1) % len(sh)] for k, h in enumerate(hs)})
    gb = _groups(bat, np.ones(len(ix), bool))
    for h, rr in gb.items():
        B = Bm[ix[rr]]
        D[rr] = (sig(off[ix[rr]] + B @ maps[h]) - p_l[rr]) * 100
        Dp[rr] = (sig(off[ix[rr]] + B @ maps[perm[h]]) - p_l[rr]) * 100
    xt, zt = T['px'][ix].astype(np.float64), T['pz'][ix].astype(np.float64)
    u_t = np.where(T['stand_r'][ix] == 1, xt, -xt)
    e = np.maximum(np.maximum(np.abs(u_t) - ZONE_HALF, zt - ZONE_TOP), ZONE_BOT - zt)
    outside = e > 0
    y = LW7[T['out7'][ix].astype(int)]
    # earlier run values per plate appearance of both players (shrunk), from first pitches of plate appearances
    first = (T['pitch_no'] == 0) & (T['out7'] >= 0)
    rv_all = LW7[np.clip(T['out7'], 0, 6).astype(int)]
    def prior_rv(key):
        nn, ss = _prior_by_day(np.r_[key[first], key[ix]].astype(np.int64), np.r_[T['day'][first], T['day'][ix]].astype(np.int64),
                               np.r_[rv_all[first], np.zeros(len(ix))], np.r_[np.ones(int(first.sum()), bool), np.zeros(len(ix), bool)])
        nn, ss = nn[int(first.sum()):], ss[int(first.sum()):]
        lg_ = float(rv_all[first].mean())
        return (ss + 200 * lg_) / (nn + 200)
    rv_b, rv_p = prior_rv(T['batter']), prior_rv(T['pitcher'])
    cnt = np.zeros((len(ix), 11)); cc = np.clip(T['balls'][ix], 0, 3) * 3 + np.clip(T['strikes'][ix], 0, 2)
    for k in range(1, 12):
        cnt[:, k - 1] = cc == k
    grp = np.zeros((len(ix), 6)); g_ = np.clip(T['group'][ix], 0, 6)
    for k in range(1, 7):
        grp[:, k - 1] = g_ == k
    base = np.column_stack([np.ones(len(ix)), cnt, grp, p_l, np.log(p_l / (1 - p_l)), outside, hats(e, (-0.8, -0.4, -0.15, 0.0, 0.15, 0.4, 0.8, 1.5)), rv_b, rv_p,
                            (T['stand_r'][ix] == T['throw_r'][ix]).astype(float)])
    games = T['game'][ix]
    ug, gi = np.unique(games, return_inverse=True)
    def ols_ci(Xd, yy, reps=200):
        b = np.linalg.lstsq(Xd, yy, rcond=None)[0]
        p_ = Xd.shape[1]; XtX = np.zeros((len(ug), p_, p_))
        for a_ in range(p_):
            for c_ in range(a_, p_):
                v = np.bincount(gi, weights=Xd[:, a_] * Xd[:, c_], minlength=len(ug)); XtX[:, a_, c_] = v; XtX[:, c_, a_] = v
        Xty = np.column_stack([np.bincount(gi, weights=Xd[:, a_] * yy, minlength=len(ug)) for a_ in range(p_)])
        dr = []
        for _ in range(reps):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
            dr.append(np.linalg.solve(np.tensordot(w, XtX, 1) + 1e-9 * np.eye(p_), w @ Xty))
        dr = np.asarray(dr)
        return b, np.percentile(dr, 2.5, axis=0), np.percentile(dr, 97.5, axis=0)
    out = {}
    for name, dd in (('own_map', D), ('placebo_other_hitter', Dp)):
        Xd = np.column_stack([base, dd * outside, dd * ~outside])
        b, lo, hi = ols_ci(Xd, y)
        out[name] = {'runs_per_point_outside': [round(float(b[-2]), 6), round(float(lo[-2]), 6), round(float(hi[-2]), 6)],
                     'runs_per_point_inside': [round(float(b[-1]), 6), round(float(lo[-1]), 6), round(float(hi[-1]), 6)]}
        stage('regression ' + name)
    res['plate_appearance_run_value'] = out
    res['rows'] = {'test_pitches': int(len(ix)), 'test_season': test_season, 'outside_share': round(float(outside.mean()), 4), 'd_sd_points_outside': round(float(D[outside].std()), 3),
                   'd_sd_points_inside': round(float(D[~outside].std()), 3), 'outside_pitches_per_pa': round(float(outside.sum() / max(int((T['pitch_no'][ix] == 0).sum()), 1)), 3)}
    # worth of aiming: best third of each pitcher's own outside pitches for this hitter against their average
    cg = np.where(T['strikes'][ix] == 2, 2, np.where(T['balls'][ix] > T['strikes'][ix], 1, 0))
    pit = T['pitcher'][ix]
    gains = []; ns = []
    pool = {}
    for pid, rr in _groups(pit, np.ones(len(ix), bool)).items():
        for sd in (0, 1):
            for c3 in (0, 1, 2):
                m_ = rr[(T['stand_r'][ix][rr] == sd) & (cg[rr] == c3) & outside[rr]]
                if len(m_) >= 30:
                    pool[(pid, sd, c3)] = m_ if len(m_) <= 400 else rng.choice(m_, 400, replace=False)
    hit_side = {h: int(np.round(T['stand_r'][ix][rr].mean())) for h, rr in gb.items()}
    for h, rr in gb.items():
        sd = hit_side[h]
        for pid in np.unique(pit[rr]):
            for c3 in (0, 1, 2):
                key = (int(pid), sd, c3)
                if key not in pool:
                    continue
                n_here = int(np.sum((pit[rr] == pid) & (cg[rr] == c3) & outside[rr]))
                if n_here == 0:
                    continue
                q = pool[key]
                dq = (sig(off[ix[q]] + Bm[ix[q]] @ maps[h]) - p_l[q]) * 100
                # within thirds of the league's swing chance, so aiming does not trade away the location's general quality
                tq = np.searchsorted(np.percentile(p_l[q], [33.3, 66.7]), p_l[q])
                parts = []
                for t3 in range(3):
                    dd_ = dq[tq == t3]
                    if len(dd_) >= 3:
                        k3 = max(1, len(dd_) // 3); parts.append(float(np.sort(dd_)[::-1][:k3].mean() - dd_.mean()))
                if parts:
                    gains.append(float(np.mean(parts))); ns.append(n_here)
    if gains:
        g_w = float(np.average(gains, weights=ns))
        b_out = out['own_map']['runs_per_point_outside']
        per_pa = g_w * res['rows']['outside_pitches_per_pa']
        res['aiming'] = {'best_third_minus_average_points': round(g_w, 3),
                         'runs_per_plate_appearance': [round(b_out[0] * per_pa, 5), round(b_out[1] * per_pa, 5), round(b_out[2] * per_pa, 5)],
                         'runs_per_6200_plate_appearances': [round(b_out[0] * per_pa * 6200, 1), round(b_out[1] * per_pa * 6200, 1), round(b_out[2] * per_pa * 6200, 1)],
                         'note': 'hitter run values (negative = runs saved by the pitcher); aiming every outside pitch at the best third of the pitcher\'s own outside locations for that hitter, within thirds of the league swing chance so the location stays as good in general, the effect of each pitch on its plate appearance as measured above'}
    stage('aiming')
    return res


# ---------------------------------------------------------------- VALUE-02: the same edge with real command
def value2_study(T: dict, params: dict, stage) -> dict:
    """VALUE-01 assumed every pitch lands where it is aimed. Here each aim point (one of the pitcher's own 2025 spots to
    that side and count group) is scattered by command error: the pitch crosses at the aim plus a two-dimensional
    normal offset with SD sigma per axis (the decision-moment picture shifts with it), and the hitter's own deviation
    is averaged over the scatter. The pitcher picks the best third of his aim points for this hitter by that expected
    deviation (outside the zone the highest, inside the lowest, since inside a higher deviation helps the hitter),
    within thirds of the league's swing chance at the aim; the gain against all his aim points, times the run value
    per point (VALUE-01's regression, re-estimated here on the same 2025 pitches) and the pitches per plate
    appearance. Sigma 0, 0.3, 0.6 and 0.9 ft. A pitch that scatters across the zone edge keeps the coefficient of where
    it was aimed (a simplification). Development data only (2025)."""
    res = {}
    final = bool(params.get('final_eval'))
    if final and not params.get('frozen_commit'):
        raise ValueError('the final value scoring runs only as the registered evaluation of a frozen commit')
    dev_test = int(params.get('test_season', 2025))   # VALUE-11: 2026 through July as a development test season
    if final:      # VALUE-02F: maps from 2023-2025, scored on the 2026 pitches from August 1 (second look at the untouched months)
        T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | ((T['season'] == 2026) & (T['day'] >= date(2026, 8, 1).toordinal())))
    else:
        T = take(T, np.isin(T['season'], (2023, 2024, 2025) + ((2026,) if dev_test == 2026 else ())))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    if params.get('zone_relative'):                       # VALUE-15: every height on each batter's own zone (MATCHUP-08)
        tr_z = np.isin(T['season'], (2023, 2024, 2025)) if final else np.isin(T['season'], tuple(int(v) for v in params.get('map_seasons', (2023, 2024))))
        top_b, bot_b, zsum = batter_zones(T, tr_z); hgt = np.clip(top_b - bot_b, 1.2, 2.8)
        zn_ = lambda z_: 1.5 + 2.0 * (z_ - bot_b) / hgt
        zp = zn_(zp); T = dict(T); T['pz'] = zn_(T['pz'].astype(np.float64)).astype(np.float32)
        res['batter_zones'] = zsum
    LB = location_block(xp, zp, T['stand_r'], T['strikes']); nL = LB.shape[1]
    Bh = hitter_basis(xp, zp, T['stand_r'], T['strikes']); nh = Bh.shape[1] - 1
    lf = bool(params.get('league_family'))                # VALUE-07: the league model gets the maps' family-by-location part
    fam_parts = [(Bh[:, :-1] * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32), (Bh[:, :-1] * (T['group'] == 5)[:, None]).astype(np.float32)] if lf else []
    X = np.hstack([LB] + fam_parts + [control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)]); del fam_parts
    tr = np.isin(T['season'], (2023, 2024, 2025)) if final else np.isin(T['season'], tuple(int(v) for v in params.get('map_seasons', (2023, 2024))))   # VALUE-10: map_seasons
    test_season = 2026 if final else dev_test
    rng = np.random.default_rng(11)
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
    league = fit_logistic(X[idx], swing[idx]); off = league.decision_function(X); del X
    cf_ = league.coef_[0].astype(np.float64); wL = cf_[:nL]
    wB, wO = (cf_[nL:nL + nh], cf_[nL + nh:nL + 2 * nh]) if lf else (None, None)

    def league_loc(LBm, Bh_m, g):
        # the league's location terms at these points (with the family-by-location part when the league has one)
        v = LBm.astype(np.float64) @ wL
        if lf:
            v = v + np.isin(g, (3, 4)) * (Bh_m[:, :-1] @ wB) + (g == 5) * (Bh_m[:, :-1] @ wO)
        return v
    fam = bool(params.get('family_maps'))                 # VALUE-06: MATCHUP-04's maps, with a part by pitch family
    Bm = family_basis(Bh, T['group']) if fam else Bh
    maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], tr), float(params.get('shrinkage', FAMILY_SHRINKAGE['location_by_family'] if fam else 10.0)), 300)
    stage(f'maps {len(maps)}')
    te = (T['season'] == test_season) & (T['out7'] >= 0) & np.isin(T['batter'], np.asarray(list(maps), dtype=np.int64))
    ix = np.flatnonzero(te)
    sig = lambda v: 1 / (1 + np.exp(-v))
    p_l = sig(off[ix])
    D = np.zeros(len(ix))
    gb = _groups(T['batter'][ix], np.ones(len(ix), bool))
    for h, rr in gb.items():
        D[rr] = (sig(off[ix[rr]] + Bm[ix[rr]] @ maps[h]) - p_l[rr]) * 100
    specific = bool(params.get('specific'))           # VALUE-08: the hitter's own part, net of the shape all hitters' maps share
    mbar = {}; Dmean = np.zeros(len(ix))
    if specific:
        # each hitter's same-side mean map over the other hitters with maps (equal weights; side from training pitches)
        ub_, bi_ = np.unique(T['batter'][tr], return_inverse=True)
        side_of = dict(zip(ub_.tolist(), np.round(np.bincount(bi_, weights=T['stand_r'][tr]) / np.bincount(bi_)).astype(int).tolist()))
        Ssum = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
        for h, m in maps.items():
            Ssum[side_of[h]] = Ssum[side_of[h]] + m; Nn[side_of[h]] += 1
        mbar = {h: (Ssum[side_of[h]] - m) / max(Nn[side_of[h]] - 1, 1) for h, m in maps.items()}
        for h, rr in gb.items():
            Dmean[rr] = (sig(off[ix[rr]] + Bm[ix[rr]] @ mbar[h]) - p_l[rr]) * 100
        D = D - Dmean
    xt, zt = T['px'][ix].astype(np.float64), T['pz'][ix].astype(np.float64)
    u_t = np.where(T['stand_r'][ix] == 1, xt, -xt)
    e = np.maximum(np.maximum(np.abs(u_t) - ZONE_HALF, zt - ZONE_TOP), ZONE_BOT - zt)
    outside = e > 0
    y = LW7[T['out7'][ix].astype(int)]
    first = (T['pitch_no'] == 0) & (T['out7'] >= 0)
    pitch_estimand = params.get('estimand') == 'pitch'   # VALUE-17: each pitch's realized value (the next count state's value or the terminal
    ci_all = np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)   # outcome, minus the current state's), with only pre-action history as a control
    fin_all = np.where(T['out7'] >= 0, LW7[np.clip(T['out7'], 0, 6)], np.nan)
    cv = np.zeros(12)                                       # count values from the training rows (mean plate-appearance value by count)
    for c_ in range(12):
        mc = tr & (ci_all == c_) & np.isfinite(fin_all); cv[c_] = float(fin_all[mc].mean()) if mc.any() else 0.0
    res['count_values'] = {f'{c_ // 3}-{c_ % 3}': round(float(cv[c_]), 4) for c_ in range(12)}
    if pitch_estimand:
        pak_all = T['game'].astype(np.int64) * 1000 + T['ab'].astype(np.int64)
        ordr = np.lexsort((T['pitch_no'], pak_all)); pak_o = pak_all[ordr]
        same_next = np.r_[pak_o[1:] == pak_o[:-1], False]
        nxt_val = np.where(same_next, cv[np.r_[ci_all[ordr][1:], 0]], fin_all[ordr]) - cv[ci_all[ordr]]
        realized_all = np.empty(len(T['game'])); realized_all[ordr] = nxt_val
        y = realized_all[ix]
        res['estimand'] = {'kind': 'pitch (telescoping realized value)'}
    rv_all = LW7[np.clip(T['out7'], 0, 6).astype(int)]
    def prior_rv(key):
        nn, ss = _prior_by_day(np.r_[key[first], key[ix]].astype(np.int64), np.r_[T['day'][first], T['day'][ix]].astype(np.int64),
                               np.r_[rv_all[first], np.zeros(len(ix))], np.r_[np.ones(int(first.sum()), bool), np.zeros(len(ix), bool)])
        nn, ss = nn[int(first.sum()):], ss[int(first.sum()):]
        lg_ = float(rv_all[first].mean()); return (ss + 200 * lg_) / (nn + 200)
    rv_b, rv_p = prior_rv(T['batter']), prior_rv(T['pitcher'])
    cnt = np.zeros((len(ix), 11)); cc = np.clip(T['balls'][ix], 0, 3) * 3 + np.clip(T['strikes'][ix], 0, 2)
    for k in range(1, 12):
        cnt[:, k - 1] = cc == k
    grp = np.zeros((len(ix), 6)); g_ = np.clip(T['group'][ix], 0, 6)
    for k in range(1, 7):
        grp[:, k - 1] = g_ == k
    Xd = np.column_stack([np.ones(len(ix)), cnt, grp, p_l, np.log(p_l / (1 - p_l)), outside, hats(e, (-0.8, -0.4, -0.15, 0.0, 0.15, 0.4, 0.8, 1.5)), rv_b, rv_p,
                          (T['stand_r'][ix] == T['throw_r'][ix]).astype(float)] + ([Dmean * outside, Dmean * ~outside] if specific else []) + [D * outside, D * ~outside])
    b = np.linalg.lstsq(Xd, y, rcond=None)[0]; b_out, b_in = float(b[-2]), float(b[-1])
    if params.get('diag_hook') is not None and not params.get('reprice_structural'):   # VALUE-17 (synthetic worlds): the regression's pieces, for pricing against the generator's truth
        params['diag_hook'](ix=ix, D=D, outside=outside, y=y, Xd=Xd, b=b)
    # game bootstrap of the two coefficients (the gains below are nearly fixed, so the value's interval follows these)
    ug, gi = np.unique(T['game'][ix], return_inverse=True); p_ = Xd.shape[1]
    XtX = np.zeros((len(ug), p_, p_))
    for a_ in range(p_):
        for c_ in range(a_, p_):
            v_ = np.bincount(gi, weights=Xd[:, a_] * Xd[:, c_], minlength=len(ug)); XtX[:, a_, c_] = v_; XtX[:, c_, a_] = v_
    Xty = np.column_stack([np.bincount(gi, weights=Xd[:, a_] * y, minlength=len(ug)) for a_ in range(p_)])
    draws = []
    for _ in range(int(params.get('reps', 200))):
        w_ = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
        draws.append(np.linalg.solve(np.tensordot(w_, XtX, 1) + 1e-9 * np.eye(p_), w_ @ Xty)[-2:])
    draws = np.asarray(draws)
    n_pa = max(int((T['pitch_no'][ix] == 0).sum()), 1)
    res['coefficients'] = {'test_season': test_season, 'test_pitches': int(len(ix)), 'runs_per_point_outside': round(b_out, 6), 'runs_per_point_inside': round(b_in, 6),
                           'outside_pitches_per_pa': round(float(outside.sum() / n_pa), 3), 'inside_pitches_per_pa': round(float((~outside).sum() / n_pa), 3)}
    if specific:
        res['coefficients'].update({'deviation': 'own map minus the same-side mean map', 'shared_part_outside': round(float(b[-4]), 6), 'shared_part_inside': round(float(b[-3]), 6),
                                    'sd_points_shared_outside': round(float(Dmean[outside].std()), 3), 'sd_points_own_part_outside': round(float(D[outside].std()), 3)})
    if params.get('bands'):
        # VALUE-16: the outside run value per point by how far outside the zone the pitch crossed (inches), same regression
        e_in = e * 12.0; bands_ = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99.0)]
        Bc = [D * outside * ((e_in >= lo_) & (e_in < hi_)) for lo_, hi_ in bands_]
        Xb = np.column_stack([Xd[:, :-2], D * ~outside] + Bc); pb = Xb.shape[1]; nb = len(bands_)
        bb = np.linalg.lstsq(Xb, y, rcond=None)[0][-nb:]
        XtXb = np.zeros((len(ug), pb, pb))
        for a_ in range(pb):
            for c_ in range(a_, pb):
                v_ = np.bincount(gi, weights=Xb[:, a_] * Xb[:, c_], minlength=len(ug)); XtXb[:, a_, c_] = v_; XtXb[:, c_, a_] = v_
        Xtyb = np.column_stack([np.bincount(gi, weights=Xb[:, a_] * y, minlength=len(ug)) for a_ in range(pb)])
        rb_ = np.random.default_rng(17); dr_ = []
        for _ in range(int(params.get('reps', 200))):
            w_ = np.bincount(rb_.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
            dr_.append(np.linalg.solve(np.tensordot(w_, XtXb, 1) + 1e-9 * np.eye(pb), w_ @ Xtyb)[-nb:])
        dr_ = np.asarray(dr_)
        res['bands'] = {f'{lo_:g}_to_{hi_:g}_in': {'pitches': int((outside & (e_in >= lo_) & (e_in < hi_)).sum()),
                                                   'own_part_sd_points': round(float(D[outside & (e_in >= lo_) & (e_in < hi_)].std()), 3),
                                                   'runs_per_point': [round(float(bb[k_]), 6), round(float(np.percentile(dr_[:, k_], 2.5)), 6), round(float(np.percentile(dr_[:, k_], 97.5)), 6)]}
                        for k_, (lo_, hi_) in enumerate(bands_)}
    if params.get('checks'):
        # attempts to break the coefficient (own random stream, so the aiming below is unchanged): (1) placebo, each
        # hitter's pitches read through another hitter's map of the same side, must show nothing; (2) only the variation
        # the aiming uses, the deviation minus its mean for this hitter, pitch group and side of the zone edge, must keep it
        rc = np.random.default_rng(5)
        def boot(Xm, reps=200, clus=None):
            cid = gi if clus is None else clus; ncl = int(cid.max()) + 1
            pk = Xm.shape[1]; XtX_ = np.zeros((ncl, pk, pk))
            for a_ in range(pk):
                for c_ in range(a_, pk):
                    v_ = np.bincount(cid, weights=Xm[:, a_] * Xm[:, c_], minlength=ncl); XtX_[:, a_, c_] = v_; XtX_[:, c_, a_] = v_
            Xty_ = np.column_stack([np.bincount(cid, weights=Xm[:, a_] * y, minlength=ncl) for a_ in range(pk)])
            dr = []
            for _ in range(reps):
                w_ = np.bincount(rc.integers(0, ncl, ncl), minlength=ncl).astype(float)
                dr.append(np.linalg.solve(np.tensordot(w_, XtX_, 1) + 1e-9 * np.eye(pk), w_ @ Xty_)[-2:])
            dr = np.asarray(dr)
            return [[round(float(np.percentile(dr[:, j], q)), 6) for q in (2.5, 97.5)] for j in (0, 1)]
        by_side = {}
        for h, rr in gb.items():
            by_side.setdefault(int(np.round(T['stand_r'][ix[rr]].mean())), []).append(h)
        perm = {}
        for hs in by_side.values():
            sh = list(hs); rc.shuffle(sh); perm.update({h: sh[(k + 1) % len(sh)] for k, h in enumerate(hs)})
        Dp = np.zeros(len(ix))
        for h, rr in gb.items():
            Dp[rr] = (sig(off[ix[rr]] + Bm[ix[rr]] @ maps[perm[h]]) - p_l[rr]) * 100
        if specific:
            Dp = Dp - Dmean
        # the placebo is one random reassignment; five more give its spread across reassignments (point estimates)
        extra = []
        for _ in range(int(params.get('placebo_draws', 5))):
            pm = {}
            for hs in by_side.values():
                sh = list(hs); rc.shuffle(sh); pm.update({h: sh[(k + 1) % len(sh)] for k, h in enumerate(hs)})
            Dq = np.zeros(len(ix))
            for h, rr in gb.items():
                Dq[rr] = (sig(off[ix[rr]] + Bm[ix[rr]] @ maps[pm[h]]) - p_l[rr]) * 100
            if specific:
                Dq = Dq - Dmean
            bq = np.linalg.lstsq(np.column_stack([Xd[:, :-2], Dq * outside, Dq * ~outside]), y, rcond=None)[0]
            extra.append(round(float(bq[-2]), 6))
        _, ki = np.unique((T['batter'][ix].astype(np.int64) * 10 + np.clip(T['group'][ix], 0, 6)) * 2 + outside, return_inverse=True)
        Dw = D - (np.bincount(ki, weights=D) / np.bincount(ki))[ki]
        chk = {'own_map': {'outside': round(b_out, 6), 'inside': round(b_in, 6),
                           'outside_interval': [round(float(np.percentile(draws[:, 0], q)), 6) for q in (2.5, 97.5)],
                           'inside_interval': [round(float(np.percentile(draws[:, 1], q)), 6) for q in (2.5, 97.5)]}}
        for nm, dd in (('placebo_other_hitter', Dp), ('within_hitter_group_and_zone_side', Dw)):
            Xk = np.column_stack([Xd[:, :-2], dd * outside, dd * ~outside])
            bk_ = np.linalg.lstsq(Xk, y, rcond=None)[0]
            ci_ = boot(Xk)
            chk[nm] = {'outside': round(float(bk_[-2]), 6), 'inside': round(float(bk_[-1]), 6), 'outside_interval': ci_[0], 'inside_interval': ci_[1],
                       'sd_points_outside': round(float(dd[outside].std()), 3)}
        chk['own_map']['sd_points_outside'] = round(float(D[outside].std()), 3)
        chk['placebo_other_hitter']['outside_more_reassignments'] = extra
        # VALUE-17: the game interval of one reassignment is too narrow for the pitch estimand; the spread across reassignments is the placebo's interval
        allp = [chk['placebo_other_hitter']['outside']] + list(extra)
        chk['placebo_other_hitter']['outside_interval_reassignments'] = [round(float(min(allp)), 6), round(float(max(allp)), 6)]
        chk['placebo_other_hitter']['reassignments'] = int(len(allp))
        # (3) one pitch's coefficient may count its neighbors: pitches in one plate appearance share the hitter's map and
        # the pitcher's spots, so their deviations are correlated; holding the plate appearance's other pitches fixed
        # gives the per-pitch effect the aiming multiplies by pitches per plate appearance
        pa_i = np.unique(T['game'][ix].astype(np.int64) * 1000 + T['ab'][ix].astype(np.int64), return_inverse=True)[1]
        if pitch_estimand:
            # pre-action history only: the sum over the plate appearance's earlier pitches (never a later one)
            o_ix = np.lexsort((T['pitch_no'][ix], pa_i)); pa_s = pa_i[o_ix]
            def loo(v):
                vs = v[o_ix]; cs_ = np.cumsum(vs); startv = np.r_[0.0, cs_[:-1]]
                firstpos = np.flatnonzero(np.r_[True, pa_s[1:] != pa_s[:-1]]); base_ = np.repeat(startv[firstpos], np.diff(np.r_[firstpos, len(pa_s)]))
                outv = np.empty(len(v)); outv[o_ix] = (startv - base_); return outv
        else:
            loo = lambda v: np.bincount(pa_i, weights=v)[pa_i] - v
        vo = D * outside; s1 = np.bincount(pa_i, weights=vo); s2 = np.bincount(pa_i, weights=vo * vo); no = np.bincount(pa_i, weights=outside.astype(float))
        mu_o, var_o = float(D[outside].mean()), float(D[outside].var())
        chk['within_pa_correlation_outside'] = round(float(((s1 ** 2 - s2).sum() / max((no * (no - 1)).sum(), 1.0) - mu_o ** 2) / var_o), 4)
        for nm, dd in (('own_map_holding_pa_others', D), ('within_hitter_holding_pa_others', Dw)):
            Xk = np.column_stack([Xd[:, :-2], loo(dd * outside), loo(dd * ~outside), dd * outside, dd * ~outside])
            bk_ = np.linalg.lstsq(Xk, y, rcond=None)[0]
            ci_ = boot(Xk)
            chk[nm] = {'outside': round(float(bk_[-2]), 6), 'inside': round(float(bk_[-1]), 6), 'outside_interval': ci_[0], 'inside_interval': ci_[1],
                       'others_outside': round(float(bk_[-4]), 6), 'others_inside': round(float(bk_[-3]), 6)}
            if params.get('cluster_checks'):
                # intervals resampling hitters and pitchers instead of games (a hitter's map and a pitcher's habits span many games)
                for cl in ('batter', 'pitcher'):
                    cidx = np.unique(T[cl][ix], return_inverse=True)[1]
                    cc_ = boot(Xk, clus=cidx)
                    chk[nm][f'outside_interval_by_{cl}'] = cc_[0]; chk[nm][f'inside_interval_by_{cl}'] = cc_[1]
        if params.get('within_spot'):
            # (4) different hitters at the same spot: every column demeaned within cells of where the pitch crossed (0.2 ft
            # squares, by batter side, count group and pitch group), so no location effect shared by hitters can carry it
            cgp = np.where(T['strikes'][ix] == 2, 2, np.where(T['balls'][ix] > T['strikes'][ix], 1, 0))
            cx = np.clip(np.floor((u_t + 2.5) / 0.2), 0, 25).astype(np.int64); cz = np.clip(np.floor(zt / 0.2), 0, 30).astype(np.int64)
            cell = ((((T['stand_r'][ix].astype(np.int64) * 3 + cgp) * 7 + np.clip(T['group'][ix], 0, 6)) * 26 + cx) * 31 + cz)
            _, ci_ = np.unique(cell, return_inverse=True); nc = np.bincount(ci_).astype(float)
            dm = lambda v: v - (np.bincount(ci_, weights=v) / nc)[ci_]
            keep_c = nc[ci_] >= 2
            Xs_ = np.column_stack([dm(Xd[:, j]) for j in range(1, Xd.shape[1])]); ys_ = dm(y)
            Xs_, ys_ = Xs_[keep_c], ys_[keep_c]
            bs_ = np.linalg.lstsq(Xs_, ys_, rcond=None)[0]
            ug2, gi2 = np.unique(T['game'][ix][keep_c], return_inverse=True); pk = Xs_.shape[1]
            XtX2 = np.zeros((len(ug2), pk, pk))
            for a_ in range(pk):
                for c_ in range(a_, pk):
                    v_ = np.bincount(gi2, weights=Xs_[:, a_] * Xs_[:, c_], minlength=len(ug2)); XtX2[:, a_, c_] = v_; XtX2[:, c_, a_] = v_
            Xty2 = np.column_stack([np.bincount(gi2, weights=Xs_[:, a_] * ys_, minlength=len(ug2)) for a_ in range(pk)])
            dr = []
            for _ in range(200):
                w_ = np.bincount(rc.integers(0, len(ug2), len(ug2)), minlength=len(ug2)).astype(float)
                dr.append(np.linalg.solve(np.tensordot(w_, XtX2, 1) + 1e-9 * np.eye(pk), w_ @ Xty2)[-2:])
            dr = np.asarray(dr)
            chk['within_spot'] = {'outside': round(float(bs_[-2]), 6), 'inside': round(float(bs_[-1]), 6),
                                  'outside_interval': [round(float(np.percentile(dr[:, 0], q)), 6) for q in (2.5, 97.5)],
                                  'inside_interval': [round(float(np.percentile(dr[:, 1], q)), 6) for q in (2.5, 97.5)],
                                  'cells': int(len(nc)), 'pitches_in_cells_with_two_or_more': int(keep_c.sum())}
        if params.get('channels'):
            # VALUE-14: through which outcomes the own part works: the primary design (within hitter, neighbor-held) on
            # each outcome class's indicator instead of the run value; the classes' coefficients times their linear
            # weights add up to the run value's
            Xk = np.column_stack([Xd[:, :-2], loo(Dw * outside), loo(Dw * ~outside), Dw * outside, Dw * ~outside])
            y_save = y
            ch = {}
            for k_, nm_ in enumerate(OUT7):
                y = (T['out7'][ix] == k_).astype(float)
                bk_ = np.linalg.lstsq(Xk, y, rcond=None)[0]; ci_ = boot(Xk, reps=100)
                ch[nm_] = {'outside_per_point': round(float(bk_[-2]), 7), 'outside_interval': ci_[0], 'inside_per_point': round(float(bk_[-1]), 7), 'inside_interval': ci_[1],
                           'rate': round(float(y.mean()), 4)}
            y = y_save
            ch['weights_check_outside'] = round(float(sum(LW7[k_] * ch[nm_]['outside_per_point'] for k_, nm_ in enumerate(OUT7))), 7)
            chk['channels'] = ch
        res['coefficient_checks'] = chk
    bands_rp = bool(params.get('reprice_bands'))             # VALUE-17: the run value per point by where the pitch crosses (six distance bands), for pricing aims
    BAND_EDGES_IN = (-2.0, 0.0, 1.0, 2.0, 4.0)               # inches from the zone edge, positive outside: deep inside, near inside, 0-1, 1-2, 2-4, 4+ outside
    band_of = lambda e_ft: np.searchsorted(np.asarray(BAND_EDGES_IN), e_ft * 12.0, side='right')
    if bands_rp:
        bnd = band_of(e); nb6 = len(BAND_EDGES_IN) + 1
        Xb6 = np.column_stack([Xd[:, :-2]] + [D * (bnd == k_) for k_ in range(nb6)]); pb6 = Xb6.shape[1]
        bb6 = np.linalg.lstsq(Xb6, y, rcond=None)[0][-nb6:]
        XtX6 = np.zeros((len(ug), pb6, pb6))
        for a_ in range(pb6):
            for c_ in range(a_, pb6):
                v_ = np.bincount(gi, weights=Xb6[:, a_] * Xb6[:, c_], minlength=len(ug)); XtX6[:, a_, c_] = v_; XtX6[:, c_, a_] = v_
        Xty6 = np.column_stack([np.bincount(gi, weights=Xb6[:, a_] * y, minlength=len(ug)) for a_ in range(pb6)])
        rb6 = np.random.default_rng(19); dr6 = []
        for _ in range(int(params.get('reps', 200))):
            w_ = np.bincount(rb6.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
            dr6.append(np.linalg.solve(np.tensordot(w_, XtX6, 1) + 1e-9 * np.eye(pb6), w_ @ Xty6)[-nb6:])
        dr6 = np.asarray(dr6)
        band_names = ['inside_2in_plus', 'inside_0_2in', 'outside_0_1in', 'outside_1_2in', 'outside_2_4in', 'outside_4in_plus']
        res['coefficients_by_band'] = {band_names[k_]: {'runs_per_point': round(float(bb6[k_]), 6), 'interval': [round(float(np.percentile(dr6[:, k_], q)), 6) for q in (2.5, 97.5)],
                                                        'pitches': int((bnd == k_).sum())} for k_ in range(nb6)}
    structural = bool(params.get('reprice_structural'))     # VALUE-18: each scattered aim priced by the engine's own components, not a regression coefficient
    PM = None
    if structural:
        # calibration of the own part on the test season's swings, out of sample for the maps: how much of a fitted point shows up in
        # the hitter's actual swings (near 1 when the maps are real, near 0 when they are noise); by side of the zone, game bootstrap
        base_sw = np.zeros(len(ix))
        for h, rr in gb.items():
            base_sw[rr] = sig(off[ix[rr]] + Bm[ix[rr]] @ mbar[h]) if specific else p_l[rr]
        r_sw = swing[ix] - base_sw; x_sw = D / 100.0
        lam = {}; lam_draws = {}
        rl = np.random.default_rng(31)
        for zn_, m_ in (('outside', outside), ('inside', ~outside)):
            sxy = np.bincount(gi, weights=(x_sw * r_sw) * m_, minlength=len(ug)); sxx = np.bincount(gi, weights=(x_sw * x_sw) * m_, minlength=len(ug))
            lam[zn_] = float(sxy.sum() / max(sxx.sum(), 1e-12))
            dl = []
            for _ in range(int(params.get('reps', 200))):
                w_ = np.bincount(rl.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
                dl.append(float((w_ @ sxy) / max(w_ @ sxx, 1e-12)))
            lam_draws[zn_] = np.asarray(dl)
        res['own_part_calibration'] = {zn_: {'slope': round(lam[zn_], 4), 'interval': [round(float(np.percentile(lam_draws[zn_], q)), 4) for q in (2.5, 97.5)]} for zn_ in lam}
        res['own_part_calibration']['note'] = 'test-season swing residual (actual minus the league with the shared shape) regressed on the own part in probability units; 1 means a fitted point is a real point'
        PM = PAModels(T, tr, np.random.default_rng(23), {'league_n': int(params.get('league_n', 500000)), 'min_pitches': 300}, stage)
        xoff_all, zoff_all = xp - T['px'].astype(np.float64), zp - T['pz'].astype(np.float64)   # decision-moment projection minus crossing, per pitch
        # VALUE-18J: the maps' own noise in the interval. Maps refitted on training games resampled with replacement (the league model and the
        # engine held fixed), each draw with its own shared shape and calibration; the structural figure is recomputed per draw in the aiming loop
        map_draws = []
        n_draws = int(params.get('map_draws', 0))
        if n_draws > 0:
            rd = np.random.default_rng(37)
            if not specific:
                ub_, bi_ = np.unique(T['batter'][tr], return_inverse=True)
                side_of = dict(zip(ub_.tolist(), np.round(np.bincount(bi_, weights=T['stand_r'][tr]) / np.bincount(bi_)).astype(int).tolist()))
            tr_idx = np.flatnonzero(tr); g_tr = T['game'][tr_idx]
            ug_tr, g_inv = np.unique(g_tr, return_inverse=True)
            order_g = np.argsort(g_inv, kind='stable'); starts = np.searchsorted(g_inv[order_g], np.arange(len(ug_tr) + 1))
            rows_by_game = [tr_idx[order_g[starts[i_]:starts[i_ + 1]]] for i_ in range(len(ug_tr))]
            shrink_ = float(params.get('shrinkage', FAMILY_SHRINKAGE['location_by_family'] if fam else 10.0))
            for d_ in range(n_draws):
                pick = rd.integers(0, len(ug_tr), len(ug_tr))
                idx_d = np.concatenate([rows_by_game[i_] for i_ in pick])
                o_ = np.argsort(T['batter'][idx_d], kind='stable'); idx_d = idx_d[o_]; kb = T['batter'][idx_d]
                cut = np.flatnonzero(np.diff(kb)) + 1
                groups_d = {int(kb[a_]): idx_d[a_:b_] for a_, b_ in zip(np.r_[0, cut], np.r_[cut, len(idx_d)])}
                maps_d = {h_: _ridge_offset(Bm[r_], swing[r_], off[r_], shrink_) for h_, r_ in groups_d.items() if len(r_) >= 300 and h_ in maps}
                Ssum_d = {0: 0.0, 1: 0.0}; Nn_d = {0: 0, 1: 0}
                for h_, m_ in maps_d.items():
                    Ssum_d[side_of[h_]] = Ssum_d[side_of[h_]] + m_; Nn_d[side_of[h_]] += 1
                mbar_d = {h_: (Ssum_d[side_of[h_]] - m_) / max(Nn_d[side_of[h_]] - 1, 1) for h_, m_ in maps_d.items()}
                D_d = np.zeros(len(ix)); base_d = np.zeros(len(ix)); have_d = np.zeros(len(ix), bool)
                for h_, rr_ in gb.items():
                    if h_ in maps_d:
                        base_d[rr_] = sig(off[ix[rr_]] + Bm[ix[rr_]] @ mbar_d[h_]); D_d[rr_] = (sig(off[ix[rr_]] + Bm[ix[rr_]] @ maps_d[h_]) - base_d[rr_]) * 100; have_d[rr_] = True
                lam_d = {}
                for zn_, m_ in (('outside', outside), ('inside', ~outside)):
                    mm_ = m_ & have_d; x_ = D_d[mm_] / 100.0; r_ = swing[ix][mm_] - base_d[mm_]
                    lam_d[zn_] = float(np.dot(x_, r_) / max(np.dot(x_, x_), 1e-12))
                map_draws.append((maps_d, mbar_d, lam_d))
            stage(f'map draws {n_draws}')
        if True:
            # the structural per-point value at the test pitches themselves (its mean beside the regression's coefficient; the synthetic worlds compare it with the generator's)
            blk_t = PM.blocks_at(xp[ix], zp[ix], T['px'][ix].astype(np.float64), T['pz'][ix].astype(np.float64), T['stand_r'][ix], T['throw_r'][ix], T['group'][ix],
                                 T['v0'][ix].astype(np.float64), T['balls'][ix], T['strikes'][ix])
            tau_t = np.zeros(len(ix))
            for h, rr in gb.items():
                sub = {k_: (v_[rr] if isinstance(v_, np.ndarray) and len(v_) == len(ix) else v_) for k_, v_ in blk_t.items()}
                pids = T['pitcher'][ix[rr]]
                ps_ = np.array([PM.p_scalar.get(int(q), (0.0, PM.lg_bip))[0] for q in pids]); pb_ = np.array([PM.p_scalar.get(int(q), (0.0, PM.lg_bip))[1] for q in pids])
                tau_t[rr] = 0.01 * PM.swing_minus_take(sub, int(h), (ps_, pb_), T['balls'][ix[rr]], T['strikes'][ix[rr]], cv)
            if params.get('diag_hook') is not None:
                params['diag_hook'](ix=ix, D=D, outside=outside, y=y, Xd=Xd, b=b, tau_struct=tau_t)
            # the structural value regressed the regression's way: the test pitches' structural own-part value (calibrated) on the fitted part with the
            # same controls gives the per-point coefficient the structure implies, beside the realized-outcome regression's
            v_struct = D * tau_t * np.where(outside, lam['outside'], lam['inside'])
            bs_ = np.linalg.lstsq(Xd, v_struct, rcond=None)[0]
            res['structural_at_test_pitches'] = {'mean_tau_outside': round(float(tau_t[outside].mean()), 6), 'mean_tau_inside': round(float(tau_t[~outside].mean()), 6),
                                                 'sd_tau_outside': round(float(tau_t[outside].std()), 6),
                                                 'implied_runs_per_point_outside': round(float(bs_[-2]), 6), 'implied_runs_per_point_inside': round(float(bs_[-1]), 6),
                                                 'regression_runs_per_point_outside': round(b_out, 6), 'regression_runs_per_point_inside': round(b_in, 6)}
            if pitch_estimand:
                # VALUE-19: is the structure calibrated against what happened? The realized telescoping value regressed on the structure's expected value
                # of the same pitch (slope 1 means the engine's pieces price a pitch as the outcomes do), overall and for the swing and take parts
                ev_t = np.zeros(len(ix)); vs_t = np.zeros(len(ix)); vt_t = np.zeros(len(ix)); ps_t = np.zeros(len(ix))
                for h, rr in gb.items():
                    sub = {k_: (v_[rr] if isinstance(v_, np.ndarray) and len(v_) == len(ix) else v_) for k_, v_ in blk_t.items()}
                    pids = T['pitcher'][ix[rr]]
                    ps_ = np.array([PM.p_scalar.get(int(q), (0.0, PM.lg_bip))[0] for q in pids]); pb_ = np.array([PM.p_scalar.get(int(q), (0.0, PM.lg_bip))[1] for q in pids])
                    ev_t[rr], vs_t[rr], vt_t[rr], ps_t[rr] = PM.expected_value(sub, int(h), (ps_, pb_), T['balls'][ix[rr]], T['strikes'][ix[rr]], cv)
                sl = lambda x_, y_: float(np.dot(x_ - x_.mean(), y_ - y_.mean()) / max(np.dot(x_ - x_.mean(), x_ - x_.mean()), 1e-12))
                cal = {'slope_realized_on_expected': round(sl(ev_t, y), 4), 'mean_expected': round(float(ev_t.mean()), 5), 'mean_realized': round(float(y.mean()), 5),
                       'slope_outside': round(sl(ev_t[outside], y[outside]), 4), 'slope_inside': round(sl(ev_t[~outside], y[~outside]), 4)}
                sw_t = swing[ix] == 1
                # the swing value against realized values on actual swings, the take value on actual takes (each conditional on the decision)
                cal['slope_swing_value_on_swings'] = round(sl(vs_t[sw_t], y[sw_t]), 4); cal['mean_swing_value_minus_realized_on_swings'] = round(float((vs_t[sw_t] - y[sw_t]).mean()), 5)
                cal['slope_take_value_on_takes'] = round(sl(vt_t[~sw_t], y[~sw_t]), 4); cal['mean_take_value_minus_realized_on_takes'] = round(float((vt_t[~sw_t] - y[~sw_t]).mean()), 5)
                cal['outside_swings'] = {'mean_swing_value_minus_realized': round(float((vs_t[sw_t & outside] - y[sw_t & outside]).mean()), 5), 'n': int((sw_t & outside).sum())}
                cal['outside_takes'] = {'mean_take_value_minus_realized': round(float((vt_t[~sw_t & outside] - y[~sw_t & outside]).mean()), 5), 'n': int((~sw_t & outside).sum())}
                cal['swing_minus_take_structural_outside_mean'] = round(float((vs_t - vt_t)[outside].mean()), 5)
                res['structural_calibration'] = cal
    stage('coefficients')
    sigmas = [float(v) for v in params.get('sigmas', (0.0, 0.3, 0.6, 0.9))]
    K = 16
    jit = np.random.default_rng(3).standard_normal((K, 2))
    jit = (jit - jit.mean(0)) / jit.std(0)
    cg = np.where(T['strikes'][ix] == 2, 2, np.where(T['balls'][ix] > T['strikes'][ix], 1, 0))
    pit = T['pitcher'][ix]; stand = T['stand_r'][ix]; strikes = T['strikes'][ix]
    hit_side = {h: int(np.round(stand[rr].mean())) for h, rr in gb.items()}
    acc = {(zn, sg): [0.0, 0.0] for zn in ('outside', 'inside') for sg in sigmas}
    acc_b = {}
    gp = _groups(pit, np.ones(len(ix), bool))
    within_type = bool(params.get('within_type'))            # VALUE-04: aim only among the same pitch group's spots
    pgroup = np.clip(T['group'][ix], 0, 6)
    type_levels = range(7) if within_type else (None,)
    pool_train = bool(params.get('pool_from_train'))         # VALUE-17: the aim menu is the pitcher's training-season spots, never the evaluation period's
    reprice = bool(params.get('reprice_boundary'))           # VALUE-17: a scattered pitch is priced on the side of the zone it actually crosses
    aim_hook = params.get('aim_hook')                        # VALUE-17 (synthetic worlds): called with every aim choice so the generator can price it with its truth
    if pool_train:
        gp_tr = _groups(T['pitcher'], tr)
        xt_all, zt_all = T['px'].astype(np.float64), T['pz'].astype(np.float64)
        u_all = np.where(T['stand_r'] == 1, xt_all, -xt_all)
        outside_all = np.maximum(np.maximum(np.abs(u_all) - ZONE_HALF, zt_all - ZONE_TOP), ZONE_BOT - zt_all) > 0
        cg_all = np.where(T['strikes'] == 2, 2, np.where(T['balls'] > T['strikes'], 1, 0)); pg_all = np.clip(T['group'], 0, 6)
    acc_runs = {}; acc_expo = {}; acc_expo_s = {}; acc_hit = {}; acc_draw = {}
    for pid, rr in gp.items():
        hs_here = np.unique(T['batter'][ix[rr]])
        for sd in (0, 1):
            for c3 in (0, 1, 2):
              for tg in type_levels:
                for zn, zmask in (('outside', outside), ('inside', ~outside)):
                    if pool_train:
                        rtr = gp_tr.get(int(pid))
                        if rtr is None:
                            continue
                        pool_rows = rtr[(T['stand_r'][rtr] == sd) & (cg_all[rtr] == c3) & (outside_all[rtr] == (zn == 'outside')) & ((pg_all[rtr] == tg) if tg is not None else True)]
                        if len(pool_rows) < 30:
                            continue
                        if len(pool_rows) > 300:
                            pool_rows = rng.choice(pool_rows, 300, replace=False)
                        pool = None; prow = pool_rows
                    else:
                        pool = rr[(stand[rr] == sd) & (cg[rr] == c3) & zmask[rr] & ((pgroup[rr] == tg) if tg is not None else True)]
                        if len(pool) < 30:
                            continue
                        if len(pool) > 300:
                            pool = rng.choice(pool, 300, replace=False)
                        prow = ix[pool]
                    p_pool = sig(off[prow])
                    third = np.searchsorted(np.percentile(p_pool, [33.3, 66.7]), p_pool)
                    lb0 = league_loc(LB[prow], Bh[prow], T['group'][prow])
                    mats = {}
                    for sg in sigmas:
                        xj = (xp[prow][:, None] + sg * jit[None, :, 0]).ravel(); zj = (zp[prow][:, None] + sg * jit[None, :, 1]).ravel()
                        sj = np.repeat(np.full(len(prow), sd), K); kj = np.repeat(T['strikes'][prow], K); gj = np.repeat(T['group'][prow], K)
                        Bj = hitter_basis(xj, zj, sj, kj)
                        offj = np.repeat(off[prow] - lb0, K) + league_loc(location_block(xj, zj, sj, kj), Bj, gj)
                        if fam:
                            Bj = family_basis(Bj, gj)
                        xtj = ztj = coef_side = coef_band = band_j = None
                        if reprice or bands_rp or structural or aim_hook is not None:
                            # the scattered pitch's true crossing moves with the same scatter; where it crosses sets its coefficient
                            xtj = (T['px'][prow].astype(np.float64)[:, None] + sg * jit[None, :, 0]).ravel(); ztj = (T['pz'][prow].astype(np.float64)[:, None] + sg * jit[None, :, 1]).ravel()
                            uj = np.where(sd == 1, xtj, -xtj)
                            e_j = np.maximum(np.maximum(np.abs(uj) - ZONE_HALF, ztj - ZONE_TOP), ZONE_BOT - ztj)
                            if reprice:
                                coef_side = np.where(e_j > 0, b_out, b_in)
                            if bands_rp:
                                band_j = band_of(e_j); coef_band = bb6[band_j]
                        blk_s = None
                        if structural:
                            blk_s = PM.blocks_at(xtj + np.repeat(xoff_all[prow], K), ztj + np.repeat(zoff_all[prow], K), xtj, ztj, np.full(len(xtj), sd), np.repeat(T['throw_r'][prow], K),
                                                 gj, np.repeat(T['v0'][prow].astype(np.float64), K), np.repeat(T['balls'][prow], K), kj)
                        mats[sg] = (offj, Bj, coef_side, coef_band, band_j, xtj, ztj, blk_s, e_j if (reprice or bands_rp or structural or aim_hook is not None) else None)
                    for h in hs_here:
                        if int(h) not in maps or hit_side.get(int(h)) != sd:
                            continue
                        n_here = int(np.sum((T['batter'][ix[rr]] == h) & (stand[rr] == sd) & (cg[rr] == c3) & zmask[rr] & ((pgroup[rr] == tg) if tg is not None else True)))
                        if n_here == 0:
                            continue
                        mh = maps[int(h)]
                        for sg in sigmas:
                            offj, Bj, coef_side, coef_band, band_j, xtj, ztj, blk_s, e_j_s = mats[sg]
                            base_j = sig(offj + Bj @ mbar[int(h)]) if specific else sig(offj)
                            dev_jk = (sig(offj + Bj @ mh) - base_j) * 100
                            dj = dev_jk.reshape(len(prow), K).mean(1)
                            # every pricing picks its own best third (lower is better for the pitcher): points by the own part itself, the
                            # repriced variants by the run value of the scattered pitch
                            vals = {'points': -dj if zn == 'outside' else dj}
                            if coef_side is not None:
                                vals['side'] = (dev_jk * coef_side).reshape(len(prow), K).mean(1)
                            if coef_band is not None:
                                vals['bands'] = (dev_jk * coef_band).reshape(len(prow), K).mean(1)
                                expo = np.stack([(dev_jk * (band_j == k_)).reshape(len(prow), K).mean(1) for k_ in range(nb6)], 1)   # points by band, per aim
                            if blk_s is not None:
                                tau_j = 0.01 * PM.swing_minus_take(blk_s, int(h), PM.p_scalar.get(int(pid), (0.0, PM.lg_bip)), np.repeat(T['balls'][prow], K), kj, cv)
                                lam_j = np.where(e_j_s > 0, lam['outside'], lam['inside'])
                                vals['structural'] = (dev_jk * lam_j * tau_j).reshape(len(prow), K).mean(1)
                                for d_, (maps_d, mbar_d, lam_d) in enumerate(map_draws):
                                    if int(h) not in maps_d:
                                        continue
                                    dev_d = (sig(offj + Bj @ maps_d[int(h)]) - sig(offj + Bj @ mbar_d[int(h)])) * 100
                                    v_d = (dev_d * np.where(e_j_s > 0, lam_d['outside'], lam_d['inside']) * tau_j).reshape(len(prow), K).mean(1)
                                    parts_d = []
                                    for t3 in range(3):
                                        sel3 = third == t3
                                        if int(sel3.sum()) >= 3:
                                            k3 = max(1, int(sel3.sum()) // 3); i3 = np.flatnonzero(sel3); pick = i3[np.argsort(v_d[i3])[:k3]]
                                            parts_d.append(float(v_d[pick].mean() - v_d[i3].mean()))
                                    if parts_d:
                                        a_ = acc_draw.setdefault((d_, zn, sg), [0.0, 0.0]); a_[0] += n_here * float(np.mean(parts_d)); a_[1] += n_here
                            parts = {nm_: [] for nm_ in vals}; chosen = {nm_: np.zeros(len(prow), bool) for nm_ in vals}; graded = np.zeros(len(prow), bool)
                            expo_parts = []; expo_parts_s = []
                            for t3 in range(3):
                                sel3 = third == t3
                                if int(sel3.sum()) >= 3:
                                    k3 = max(1, int(sel3.sum()) // 3); i3 = np.flatnonzero(sel3); graded[i3] = True
                                    for nm_, v_ in vals.items():
                                        o3 = np.argsort(v_[i3]); pick = i3[o3[:k3]]; chosen[nm_][pick] = True
                                        if nm_ == 'points':
                                            parts[nm_].append(float(dj[pick].mean() - dj[i3].mean()))
                                        else:
                                            parts[nm_].append(float(v_[pick].mean() - v_[i3].mean()))
                                    if coef_band is not None:
                                        pick_b = np.flatnonzero(chosen['bands'] & sel3)
                                        expo_parts.append(expo[pick_b].mean(0) - expo[i3].mean(0))
                                        if 'structural' in chosen:
                                            pick_s = np.flatnonzero(chosen['structural'] & sel3)
                                            expo_parts_s.append(expo[pick_s].mean(0) - expo[i3].mean(0))
                            if parts['points'] and aim_hook is not None:
                                aim_hook(sg=sg, zone=zn, hitter=int(h), pitcher=int(pid), rows=prow, K=K, x_true=xtj, z_true=ztj,
                                         balls=np.repeat(T['balls'][prow], K), strikes=np.repeat(T['strikes'][prow], K), group=np.repeat(T['group'][prow], K),
                                         chosen=chosen, graded=graded, third=third, weight=n_here, dev=dj, coef=b_out if zn == 'outside' else b_in)
                            if parts['points']:
                                acc[(zn, sg)][0] += n_here * float(np.mean(parts['points'])); acc[(zn, sg)][1] += n_here
                                for bk in (('count', c3), ('type', tg)):
                                    a_ = acc_b.setdefault((zn, sg) + bk, [0.0, 0.0]); a_[0] += n_here * float(np.mean(parts['points'])); a_[1] += n_here
                            for nm_ in ('side', 'bands', 'structural'):
                                if parts.get(nm_):
                                    a_ = acc_runs.setdefault((nm_, zn, sg), [0.0, 0.0]); a_[0] += n_here * float(np.mean(parts[nm_])); a_[1] += n_here
                                    if nm_ == 'structural':
                                        hc = acc_hit.setdefault((zn, sg), {}); c_h = hc.setdefault(int(h), [0.0, 0.0]); c_h[0] += n_here * float(np.mean(parts[nm_])); c_h[1] += n_here
                            if expo_parts:
                                a_ = acc_expo.setdefault((zn, sg), np.zeros(nb6)); a_ += n_here * np.mean(expo_parts, 0)
                            if expo_parts_s:
                                a_ = acc_expo_s.setdefault((zn, sg), np.zeros(nb6)); a_ += n_here * np.mean(expo_parts_s, 0)
    out = {}
    for sg in sigmas:
        go = acc[('outside', sg)][0] / max(acc[('outside', sg)][1], 1); gi_ = acc[('inside', sg)][0] / max(acc[('inside', sg)][1], 1)
        per_pa = b_out * go * res['coefficients']['outside_pitches_per_pa'] + b_in * gi_ * res['coefficients']['inside_pitches_per_pa']
        both_draws = (draws[:, 0] * go * res['coefficients']['outside_pitches_per_pa'] + draws[:, 1] * gi_ * res['coefficients']['inside_pitches_per_pa']) * 6200
        out_draws = draws[:, 0] * go * res['coefficients']['outside_pitches_per_pa'] * 6200
        out[str(sg)] = {'outside_gain_points': round(go, 3), 'inside_gain_points': round(gi_, 3),
                        'runs_per_6200_outside_only_interval': [round(float(np.percentile(out_draws, 2.5)), 1), round(float(np.percentile(out_draws, 97.5)), 1)],
                        'runs_per_6200_both_interval': [round(float(np.percentile(both_draws, 2.5)), 1), round(float(np.percentile(both_draws, 97.5)), 1)],
                        'runs_per_6200_outside_only': round(b_out * go * res['coefficients']['outside_pitches_per_pa'] * 6200, 1),
                        'runs_per_6200_inside_only': round(b_in * gi_ * res['coefficients']['inside_pitches_per_pa'] * 6200, 1),
                        'runs_per_6200_both': round(per_pa * 6200, 1)}
    res['by_command_sd_ft'] = out
    if reprice:
        rp_ = {}
        for sg in sigmas:
            ao = acc_runs.get(('side', 'outside', sg), [0.0, 0.0]); ai = acc_runs.get(('side', 'inside', sg), [0.0, 0.0])
            gr_o = ao[0] / max(ao[1], 1); gr_i = ai[0] / max(ai[1], 1)
            rp_[str(sg)] = {'runs_per_outside_pitch': round(gr_o, 6), 'runs_per_inside_pitch': round(gr_i, 6),
                            'runs_per_6200_outside_only': round(gr_o * res['coefficients']['outside_pitches_per_pa'] * 6200, 1),
                            'runs_per_6200_outside_only_interval': [round(float(np.percentile(draws[:, 0] / b_out * gr_o * res['coefficients']['outside_pitches_per_pa'] * 6200, q)), 1) for q in (2.5, 97.5)] if b_out != 0 else None}
        res['repriced_by_side'] = rp_
        res['repriced_note'] = 'each scattered pitch priced with the coefficient of the side of the zone it crosses (VALUE-17); the interval scales the outside coefficient draws only'
    if bands_rp:
        rb_ = {}
        for sg in sigmas:
            ao = acc_runs.get(('bands', 'outside', sg), [0.0, 0.0]); ai = acc_runs.get(('bands', 'inside', sg), [0.0, 0.0])
            gr_o = ao[0] / max(ao[1], 1); gr_i = ai[0] / max(ai[1], 1)
            ex_o = acc_expo.get(('outside', sg), np.zeros(nb6)) / max(ao[1], 1)
            dr_runs = dr6 @ ex_o * res['coefficients']['outside_pitches_per_pa'] * 6200       # the selection held fixed, the band coefficients resampled by game
            rb_[str(sg)] = {'runs_per_outside_pitch': round(gr_o, 6), 'runs_per_inside_pitch': round(gr_i, 6),
                            'runs_per_6200_outside_only': round(gr_o * res['coefficients']['outside_pitches_per_pa'] * 6200, 1),
                            'runs_per_6200_outside_only_interval': [round(float(np.percentile(dr_runs, q)), 1) for q in (2.5, 97.5)],
                            'chosen_points_by_band_outside': {band_names[k_]: round(float(ex_o[k_]), 3) for k_ in range(nb6)}}
        res['repriced_by_band'] = rb_
        res['repriced_by_band_note'] = 'each scattered pitch priced with the run value per point of the distance band it crosses in (VALUE-17); the interval resamples the band coefficients by game with the chosen spots fixed'
    if structural:
        rs_ = {}; rh = np.random.default_rng(29)
        for sg in sigmas:
            ao = acc_runs.get(('structural', 'outside', sg), [0.0, 0.0]); ai = acc_runs.get(('structural', 'inside', sg), [0.0, 0.0])
            gr_o = ao[0] / max(ao[1], 1); gr_i = ai[0] / max(ai[1], 1)
            hc = acc_hit.get(('outside', sg), {}); hv = np.array([v_ for v_ in hc.values()]) if hc else np.zeros((0, 2))
            if len(hv):
                dr_h = []
                for _ in range(400):
                    pick = rh.integers(0, len(hv), len(hv)); dr_h.append(hv[pick, 0].sum() / max(hv[pick, 1].sum(), 1.0))
                dr_h = np.asarray(dr_h) * res['coefficients']['outside_pitches_per_pa'] * 6200
                hint = [round(float(np.percentile(dr_h, q)), 1) for q in (2.5, 97.5)]
                # with the calibration's own uncertainty (its game draws scale the figure; the two sources taken as independent)
                ld = lam_draws['outside'][rh.integers(0, len(lam_draws['outside']), len(dr_h))] / (lam['outside'] if lam['outside'] != 0 else 1.0)
                hint2 = [round(float(np.percentile(dr_h * ld, q)), 1) for q in (2.5, 97.5)]
            else:
                hint = hint2 = None
            rs_[str(sg)] = {'runs_per_outside_pitch': round(gr_o, 6), 'runs_per_inside_pitch': round(gr_i, 6),
                            'runs_per_6200_outside_only': round(gr_o * res['coefficients']['outside_pitches_per_pa'] * 6200, 1),
                            'runs_per_6200_inside_only': round(gr_i * res['coefficients']['inside_pitches_per_pa'] * 6200, 1),
                            'runs_per_6200_outside_only_interval_by_hitter': hint, 'runs_per_6200_outside_only_interval': hint2, 'hitters': int(len(hv))}
            if map_draws:
                figs = [acc_draw[(d_, 'outside', sg)][0] / max(acc_draw[(d_, 'outside', sg)][1], 1) * res['coefficients']['outside_pitches_per_pa'] * 6200
                        for d_ in range(len(map_draws)) if (d_, 'outside', sg) in acc_draw]
                figs_in = [acc_draw[(d_, 'inside', sg)][0] / max(acc_draw[(d_, 'inside', sg)][1], 1) * res['coefficients']['inside_pitches_per_pa'] * 6200
                           for d_ in range(len(map_draws)) if (d_, 'inside', sg) in acc_draw]
                pt_ = gr_o * res['coefficients']['outside_pitches_per_pa'] * 6200
                sd_maps = float(np.std(figs, ddof=1)) if len(figs) > 1 else 0.0
                sd_hit = float(np.std(dr_h * ld)) if len(hv) else 0.0
                sd_all = float(np.sqrt(sd_maps ** 2 + sd_hit ** 2))
                rs_[str(sg)]['map_draws'] = {'draws': int(len(figs)), 'outside_figures': [round(float(v_), 1) for v_ in figs], 'inside_figures': [round(float(v_), 1) for v_ in figs_in],
                                             'sd_from_maps': round(sd_maps, 1), 'sd_from_hitters_and_calibration': round(sd_hit, 1),
                                             'runs_per_6200_outside_only_interval_with_maps': [round(pt_ - 1.96 * sd_all, 1), round(pt_ + 1.96 * sd_all, 1)],
                                             'calibration_outside_by_draw': [round(md[2]['outside'], 3) for md in map_draws]}
            if bands_rp and ('outside', sg) in acc_expo_s:
                # VALUE-19: the same structural choices priced by the realized-outcome regression by band (outcome-anchored value of the structural choice)
                ao_s = acc_runs.get(('structural', 'outside', sg), [0.0, 0.0])
                ex_s = acc_expo_s[('outside', sg)] / max(ao_s[1], 1)
                dr_s = dr6 @ ex_s * res['coefficients']['outside_pitches_per_pa'] * 6200
                rs_[str(sg)]['structural_choice_priced_by_band_regression'] = {'runs_per_6200_outside_only': round(float(bb6 @ ex_s * res['coefficients']['outside_pitches_per_pa'] * 6200), 1),
                                                                               'interval': [round(float(np.percentile(dr_s, q)), 1) for q in (2.5, 97.5)],
                                                                               'chosen_points_by_band_outside': {band_names[k_]: round(float(ex_s[k_]), 3) for k_ in range(nb6)}}
        res['repriced_structural'] = rs_
        res['repriced_structural_note'] = ('each scattered aim priced by the engine: the hitter\'s own swing part, scaled by its out-of-sample calibration on the test season\'s swings, times '
                                           '(value if swing minus value if take) from the whiff, called-strike, foul and contact models with the hitter\'s and pitcher\'s terms and the training count '
                                           'values (VALUE-18); the interval resamples hitters and the calibration slope by game')
    if 'coefficient_checks' in res:
        for nm, v in res['coefficient_checks'].items():
            if isinstance(v, dict) and 'outside' in v:
                for sg in sigmas:
                    go = acc[('outside', sg)][0] / max(acc[('outside', sg)][1], 1)
                    v[f'runs_per_6200_outside_at_{sg}'] = round(v['outside'] * go * res['coefficients']['outside_pitches_per_pa'] * 6200, 1)
                    if 'outside_interval' in v:
                        v[f'runs_per_6200_outside_interval_at_{sg}'] = [round(c * go * res['coefficients']['outside_pitches_per_pa'] * 6200, 1) for c in v['outside_interval']]
    if params.get('breakdown'):
        # runs per 6,200 plate appearances from each count group (and, aiming within type, each pitch group), at each sigma;
        # count groups use their own run value per point (the regression with the deviation split by count group)
        names_c = {0: 'ahead_or_even', 1: 'behind', 2: 'two_strikes'}
        Xc = np.column_stack([Xd[:, :-2]] + [D * outside * (cg == k) for k in range(3)] + [D * ~outside * (cg == k) for k in range(3)])
        bc = np.linalg.lstsq(Xc, y, rcond=None)[0]
        coef_c = {('outside', k): float(bc[-6 + k]) for k in range(3)}; coef_c.update({('inside', k): float(bc[-3 + k]) for k in range(3)})
        res['coefficients_by_count'] = {f'{zn}_{names_c[k]}': round(v, 6) for (zn, k), v in coef_c.items()}
        bd = {}
        for (zn, sg, kind, lev), (sm, nn_) in acc_b.items():
            if lev is None or nn_ == 0:
                continue
            sel = (outside if zn == 'outside' else ~outside) & ((cg == lev) if kind == 'count' else (np.clip(T['group'][ix], 0, 6) == lev))
            per_pa = float(sel.sum() / n_pa)
            coef = coef_c[(zn, lev)] if kind == 'count' else (b_out if zn == 'outside' else b_in)
            label = names_c[lev] if kind == 'count' else f'group_{lev}'
            bd.setdefault(str(sg), {}).setdefault(kind, {})[f'{zn}_{label}'] = {'gain_points': round(sm / nn_, 3), 'pitches_per_pa': round(per_pa, 3),
                                                                                'runs_per_6200': round(coef * (sm / nn_) * per_pa * 6200, 1)}
        res['breakdown'] = bd
    stage('aiming with scatter')
    return res


# ---------------------------------------------------------------- ABS-01: did 2026 raise the price of a chase?
def abs_study(T: dict, params: dict, stage) -> dict:
    """VALUE-11 found a hitter's own chase spots worth 1.65 times as much in 2026 (through July) as in 2025. Candidate
    mechanism: with the 2026 ball-strike challenge system a taken pitch outside the zone is called a ball more
    reliably, so a chase gives up more against a take. Measured on 2025 and 2026 through July (development data):
    the called-strike chance of taken pitches by distance outside (and inside) the zone edge, by season; and the
    plate appearance's run value for a swing against a take on pitches outside the zone, holding the count, the
    distance band, the pitch group, both players' earlier run values and platoon, by season, with the difference's
    game-bootstrap interval."""
    res = {}
    T = take(T, np.isin(T['season'], (2025, 2026)))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); del F
    swing = ((T['call'] == 1) | (T['call'] == 2)); take_ = T['call'] == 0; cs = T['cs'] == 1
    xt, zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    u_t = np.where(T['stand_r'] == 1, xt, -xt)
    e = np.maximum(np.maximum(np.abs(u_t) - ZONE_HALF, zt - ZONE_TOP), ZONE_BOT - zt)
    edges = np.array([-4, -2, -1, 0, 1, 2, 4, 8, 99]) / 12.0
    band = np.searchsorted(edges, e)                 # 0: deeper than 4 in inside ... 8: beyond 8 in outside
    names = ['inside_4in_plus', 'inside_2_4in', 'inside_1_2in', 'inside_0_1in', 'outside_0_1in', 'outside_1_2in', 'outside_2_4in', 'outside_4_8in', 'outside_8in_plus']
    cso = {}
    for ssn in (2025, 2026):
        m = (T['season'] == ssn) & take_
        cso[ssn] = {names[b]: {'called_strike': round(float(cs[m & (band == b)].mean()), 4), 'takes': int((m & (band == b)).sum())} for b in range(len(names)) if (m & (band == b)).sum() > 100}
    res['called_strike_on_takes'] = cso
    # the side edges only (heights 2.0 to 3.0 ft, where the plate's width is the edge in both seasons and the 2026 zone's batter-specific top and bottom do not enter)
    mid = (zt >= 2.0) & (zt <= 3.0)
    es = np.abs(u_t) - ZONE_HALF; band_s = np.searchsorted(edges, es)
    res['called_strike_on_takes_side_edges'] = {ssn: {names[b]: {'called_strike': round(float(cs[(T['season'] == ssn) & take_ & mid & (band_s == b)].mean()), 4),
                                                                   'takes': int(((T['season'] == ssn) & take_ & mid & (band_s == b)).sum())}
                                                        for b in range(len(names)) if ((T['season'] == ssn) & take_ & mid & (band_s == b)).sum() > 100} for ssn in (2025, 2026)}
    stage('called strikes')
    out = (e > 0) & (T['out7'] >= 0)
    y = LW7[np.clip(T['out7'], 0, 6).astype(int)]
    first = (T['pitch_no'] == 0) & (T['out7'] >= 0)
    rv_all = y
    ix = np.flatnonzero(out)
    def prior_rv(key):
        nn, ss = _prior_by_day(np.r_[key[first], key[ix]].astype(np.int64), np.r_[T['day'][first], T['day'][ix]].astype(np.int64),
                               np.r_[rv_all[first], np.zeros(len(ix))], np.r_[np.ones(int(first.sum()), bool), np.zeros(len(ix), bool)])
        nn, ss = nn[int(first.sum()):], ss[int(first.sum()):]
        lg_ = float(rv_all[first].mean()); return (ss + 200 * lg_) / (nn + 200)
    rv_b, rv_p = prior_rv(T['batter']), prior_rv(T['pitcher'])
    cc = np.clip(T['balls'][ix], 0, 3) * 3 + np.clip(T['strikes'][ix], 0, 2)
    s26 = (T['season'][ix] == 2026).astype(float); sw = swing[ix].astype(float)
    cols = [np.ones(len(ix)), s26]
    for k in range(1, 12):
        cols += [(cc == k).astype(float), (cc == k) * s26]
    for b in range(5, 9):
        cols += [(band[ix] == b).astype(float), (band[ix] == b) * s26]
    for g in range(1, 7):
        cols.append((np.clip(T['group'][ix], 0, 6) == g).astype(float))
    cols += [rv_b, rv_p, (T['stand_r'][ix] == T['throw_r'][ix]).astype(float), sw, sw * s26]
    X = np.column_stack(cols); yy = y[ix]
    bb = np.linalg.lstsq(X, yy, rcond=None)[0]
    ug, gi = np.unique(T['game'][ix], return_inverse=True); pk = X.shape[1]
    XtX = np.zeros((len(ug), pk, pk))
    for a_ in range(pk):
        for c_ in range(a_, pk):
            v_ = np.bincount(gi, weights=X[:, a_] * X[:, c_], minlength=len(ug)); XtX[:, a_, c_] = v_; XtX[:, c_, a_] = v_
    Xty = np.column_stack([np.bincount(gi, weights=X[:, a_] * yy, minlength=len(ug)) for a_ in range(pk)])
    rng = np.random.default_rng(5); dr = []
    for _ in range(int(params.get('reps', 200))):
        w_ = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
        dr.append(np.linalg.solve(np.tensordot(w_, XtX, 1) + 1e-9 * np.eye(pk), w_ @ Xty)[-2:])
    dr = np.asarray(dr)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]
    res['swing_vs_take_outside'] = {'pitches': int(len(ix)), 'runs_2025': round(float(bb[-2]), 5), 'runs_2025_interval': q(dr[:, 0]),
                                    'runs_2026': round(float(bb[-2] + bb[-1]), 5), 'runs_2026_interval': q(dr[:, 0] + dr[:, 1]),
                                    'change_2026': round(float(bb[-1]), 5), 'change_2026_interval': q(dr[:, 1])}
    stage('swing against take')
    return res


# ---------------------------------------------------------------- ABS-02: who the 2026 challenge system would hurt, measured in 2025
def abs2_study(T: dict, params: dict, stage) -> dict:
    """ABS-01 found the 2026 challenge system took away most borderline strikes on taken pitches and gave back a few
    just inside. A rule change is a shock whose exposure can be measured before it: each pitcher's and hitter's 2025
    taken pitches by distance from the zone edge, times the league's change in called-strike chance in that band
    (2026 through July against 2025), per plate appearance, is the strikes the new calling would have taken from him
    (pitchers) or given him back (hitters) on his own 2025 pitches. Outcome: the change in walk rate, strikeout rate
    and run value per plate appearance from 2025 to 2026 (through July), for players with at least 150 plate
    appearances in each, holding their 2024 and 2025 rates (regression to the mean) and the other outcome rates;
    weighted by plate appearances; intervals from resampling players."""
    res = {}
    T = take(T, np.isin(T['season'], (2024, 2025, 2026)))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0); del F
    take_ = (T['call'] == 0) & keep; cs = T['cs'] == 1
    xt, zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    u_t = np.where(T['stand_r'] == 1, xt, -xt)
    e = np.maximum(np.maximum(np.abs(u_t) - ZONE_HALF, zt - ZONE_TOP), ZONE_BOT - zt)
    edges = np.array([-4, -2, -1, 0, 1, 2, 4, 8, 99]) / 12.0
    band = np.searchsorted(edges, e)
    rate = {}
    for ssn in (2025, 2026):
        m = take_ & (T['season'] == ssn)
        rate[ssn] = np.array([cs[m & (band == b)].mean() if (m & (band == b)).sum() > 100 else 0.0 for b in range(len(edges))])
    dcs = rate[2026] - rate[2025]                       # change in called-strike chance by band (negative outside)
    res['called_strike_change_by_band'] = [round(float(v), 4) for v in dcs]
    pa = (T['pitch_no'] == 0) & (T['out7'] >= 0)
    y7 = T['out7'].astype(int)
    out = {}
    for role, key in (('pitchers', 'pitcher'), ('hitters', 'batter')):
        ids = T[key]
        def per(ssn, w):
            m = T['season'] == ssn
            u_, i_ = np.unique(ids[m], return_inverse=True)
            return dict(zip(u_.tolist(), np.bincount(i_, weights=w[m]).tolist()))
        n_pa = {s_: per(s_, pa.astype(float)) for s_ in (2024, 2025, 2026)}
        bb = {s_: per(s_, (pa & (y7 == 2)).astype(float)) for s_ in (2024, 2025, 2026)}
        kk = {s_: per(s_, (pa & (y7 == 1)).astype(float)) for s_ in (2024, 2025, 2026)}
        rv = {s_: per(s_, np.where(pa, LW7[np.clip(y7, 0, 6)], 0.0)) for s_ in (2024, 2025, 2026)}
        lost = per(2025, np.where(take_ & (band >= 4), -dcs[np.clip(band, 0, len(dcs) - 1)], 0.0))      # outside: strikes the new calling takes away
        gain = per(2025, np.where(take_ & (band <= 3), dcs[np.clip(band, 0, len(dcs) - 1)], 0.0))       # inside: strikes it adds
        rows = []
        for pid, n25 in n_pa[2025].items():
            n26 = n_pa[2026].get(pid, 0.0)
            if n25 < 150 or n26 < 150:
                continue
            n24 = n_pa[2024].get(pid, 0.0)
            r = lambda d, s_, n: d[s_].get(pid, 0.0) / n if n > 0 else np.nan
            lg24 = lambda d: sum(d[2024].values()) / max(sum(n_pa[2024].values()), 1)
            rows.append([pid, n25, n26, (lost.get(pid, 0.0) - gain.get(pid, 0.0)) / n25,
                         r(bb, 2025, n25), r(bb, 2026, n26), r(bb, 2024, n24) if n24 >= 50 else lg24(bb),
                         r(kk, 2025, n25), r(kk, 2026, n26), r(kk, 2024, n24) if n24 >= 50 else lg24(kk),
                         r(rv, 2025, n25), r(rv, 2026, n26), r(rv, 2024, n24) if n24 >= 50 else lg24(rv)])
        R = np.asarray(rows, float)
        if len(R) < 30:
            out[role] = {'players': int(len(R))}; continue
        expo = R[:, 3]; w = 1 / (1 / R[:, 1] + 1 / R[:, 2])
        rng = np.random.default_rng(7)
        res_role = {'players': int(len(R)), 'exposure_mean_per_pa': round(float(np.average(expo, weights=w)), 4), 'exposure_sd_per_pa': round(float(np.sqrt(np.cov(expo, aweights=w))), 4)}
        for nm, c25, c26, c24 in (('walk_rate', 4, 5, 6), ('strikeout_rate', 7, 8, 9), ('runs_per_pa', 10, 11, 12)):
            yv = R[:, c26] - R[:, c25]
            X = np.column_stack([np.ones(len(R)), expo, R[:, c25], R[:, c24], R[:, 4], R[:, 7]])
            def wls(idx):
                Xw = X[idx] * np.sqrt(w[idx])[:, None]; yw = yv[idx] * np.sqrt(w[idx])
                return np.linalg.lstsq(Xw, yw, rcond=None)[0][1]
            b0 = wls(np.arange(len(R)))
            bs = np.array([wls(rng.integers(0, len(R), len(R))) for _ in range(int(params.get('reps', 500)))])
            q = np.std(expo)
            res_role[nm] = {'per_strike_per_pa': round(float(b0), 4), 'interval': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                            'per_sd_of_exposure': round(float(b0 * q), 5)}
        # top and bottom fifth by exposure: average changes
        qq = np.percentile(expo, [20, 80])
        for nm_, sel in (('most_exposed_fifth', expo >= qq[1]), ('least_exposed_fifth', expo <= qq[0])):
            res_role[nm_] = {'exposure': round(float(np.average(expo[sel], weights=w[sel])), 4), 'walk_rate_change': round(float(np.average(R[sel, 5] - R[sel, 4], weights=w[sel])), 4),
                             'strikeout_rate_change': round(float(np.average(R[sel, 8] - R[sel, 7], weights=w[sel])), 4)}
        out[role] = res_role
        stage('role ' + role)
    res.update(out)
    return res


# ---------------------------------------------------------------- FRAMING-01: what the 2026 challenge system did to catcher framing
def load_pa_states(repo, token, key_hex):
    """The plate-appearance history from the sealed data package (game state, catcher), for joins to the pitch table;
    stays on the runner."""
    import importlib.util, tempfile
    spec = importlib.util.spec_from_file_location('brl_entrypoint', ROOT / 'brl_engine' / 'entrypoint.py')
    ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)
    manifest = json.loads((ROOT / 'brl_engine' / 'data_package.json').read_text())
    root = Path(tempfile.mkdtemp(prefix='brl-pkg-'))
    ep.restore_package(repo, token, key_hex, manifest, root)
    path = next(root.rglob('plate_appearances.csv.gz'))
    import pandas as pd
    h = pd.read_csv(path, usecols=['game_pk', 'at_bat_number', 'fielder_2', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'bat_score_diff', 'inning', 'date_key'], low_memory=False)
    return h


def framing_study(T: dict, H, params: dict, stage) -> dict:
    """ABS-01 found the 2026 challenge system took away borderline strikes. Framing is a catcher's share of borderline
    calls; if challenges overturn the framed ones, catchers' framing should shrink toward zero in 2026 beyond the usual
    year-to-year regression. Taken pitches within 2 inches of the zone edge (either side), called-strike chance by
    half-inch band and season (league); each pitch's surplus (called strike minus that chance) split into catcher and
    pitcher parts by backfitting within a season; catchers with at least 800 such takes in a season. Spread of catcher
    parts by season (extra strikes per 1,000 borderline takes) and persistence from one season to the next (2024 to 2025
    against 2025 to 2026), with intervals from resampling catchers."""
    res = {}
    import pandas as pd
    keyT = T['game'].astype(np.int64) * 1000 + T['ab'].astype(np.int64)
    Hk = (H['game_pk'].astype(np.int64) * 1000 + H['at_bat_number'].astype(np.int64) - 1).to_numpy()
    cat = pd.Series(H['fielder_2'].to_numpy(), index=Hk)
    cat = cat[~cat.index.duplicated()]
    catcher = cat.reindex(keyT).to_numpy()
    take_ = (T['call'] == 0)
    ut = np.where(T['stand_r'] == 1, T['px'].astype(np.float64), -T['px'].astype(np.float64)); zt = T['pz'].astype(np.float64)
    e = np.maximum(np.maximum(np.abs(ut) - ZONE_HALF, zt - ZONE_TOP), ZONE_BOT - zt)
    border = take_ & np.isfinite(e) & (np.abs(e) <= 2.0 / 12.0) & np.isfinite(catcher)
    band = np.clip(np.floor((e + 2.0 / 12.0) / (0.5 / 12.0)), 0, 7).astype(int)
    cs = (T['cs'] == 1).astype(float)
    res['matched_share'] = round(float(np.isfinite(catcher[take_]).mean()), 4)
    parts = {}
    for ssn in (2024, 2025, 2026):
        m = border & (T['season'] == ssn)
        if m.sum() < 1000:
            continue
        pb = np.array([cs[m & (band == b)].mean() if (m & (band == b)).any() else 0.0 for b in range(8)])
        surplus = cs[m] - pb[band[m]]
        cid = catcher[m].astype(np.int64); pid = T['pitcher'][m].astype(np.int64)
        uc, ic = np.unique(cid, return_inverse=True); up, ip = np.unique(pid, return_inverse=True)
        nc, np_ = np.bincount(ic), np.bincount(ip)
        ec = np.zeros(len(uc)); epp = np.zeros(len(up))
        for _ in range(20):
            ec = np.bincount(ic, weights=surplus - epp[ip]) / nc
            epp = np.bincount(ip, weights=surplus - ec[ic]) / np_
        keep = nc >= 800
        parts[ssn] = dict(zip(uc[keep].tolist(), (ec[keep] * 1000).tolist()))
        res.setdefault('by_season', {})[str(ssn)] = {'borderline_takes': int(m.sum()), 'called_strike_rate': round(float(cs[m].mean()), 4),
                                                     'catchers': int(keep.sum()), 'sd_extra_strikes_per_1000': round(float(np.std(ec[keep] * 1000)), 2),
                                                     'band_rates': [round(float(v), 3) for v in pb]}
    stage('framing parts')
    rng = np.random.default_rng(7)
    for a, b in ((2024, 2025), (2025, 2026)):
        if a not in parts or b not in parts:
            continue
        common = sorted(set(parts[a]) & set(parts[b]))
        x = np.array([parts[a][c] for c in common]); y = np.array([parts[b][c] for c in common])
        slope = float(np.polyfit(x, y, 1)[0]); corr = float(np.corrcoef(x, y)[0, 1])
        bs = []
        for _ in range(int(params.get('reps', 1000))):
            j = rng.integers(0, len(x), len(x)); bs.append(np.polyfit(x[j], y[j], 1)[0])
        res[f'persistence_{a}_to_{b}'] = {'catchers': len(common), 'slope': [round(slope, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                         'correlation': round(corr, 3)}
    if 'persistence_2024_to_2025' in res and 'persistence_2025_to_2026' in res:
        res['persistence_ratio'] = round(res['persistence_2025_to_2026']['slope'][0] / max(res['persistence_2024_to_2025']['slope'][0], 1e-9), 3)
    if '2025' in res.get('by_season', {}) and '2026' in res.get('by_season', {}):
        res['spread_ratio_2026_over_2025'] = round(res['by_season']['2026']['sd_extra_strikes_per_1000'] / res['by_season']['2025']['sd_extra_strikes_per_1000'], 3)
    return res


# ---------------------------------------------------------------- PRESSURE-01: do hitters decide differently under pressure?
def pressure_study(T: dict, H, params: dict, stage) -> dict:
    """Hitter maps pool every game state. If hitters press (or tighten) with runners in scoring position or late in close
    games, their swings at the same apparent spots differ from the map there, and an aim plan should know it.
    MATCHUP-05F's representation fitted on 2023-2024; every 2025 pitch to a hitter with a map, joined to the plate
    appearance's state (outs, runners, inning, score); his swing with the full map's prediction as offset, on runners in
    scoring position, late and close (seventh inning on, within one run), two outs, and each of the first two times the
    own part at this pitch (does the own part pull harder under pressure?), separately outside and inside the zone;
    game-bootstrap intervals."""
    res = {}
    import pandas as pd
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26); del F
    Bh = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    X = np.hstack([location_block(xp, zp, T['stand_r'], T['strikes']), (Bh[:, :-1] * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32),
                   (Bh[:, :-1] * (T['group'] == 5)[:, None]).astype(np.float32), control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    tr = np.isin(T['season'], (2023, 2024))
    rng = np.random.default_rng(11)
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
    off = fit_logistic(X[idx], swing[idx]).decision_function(X); del X
    Bm = family_basis(Bh, T['group']); del Bh
    maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], tr), 10.0, 300)
    ub_, bi_ = np.unique(T['batter'][tr], return_inverse=True)
    side_of = dict(zip(ub_.tolist(), np.round(np.bincount(bi_, weights=T['stand_r'][tr]) / np.bincount(bi_)).astype(int).tolist()))
    Ss = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
    for h, m in maps.items():
        Ss[side_of[h]] = Ss[side_of[h]] + m; Nn[side_of[h]] += 1
    mbar = {h: (Ss[side_of[h]] - m) / max(Nn[side_of[h]] - 1, 1) for h, m in maps.items()}
    stage(f'maps {len(maps)}')
    te = (T['season'] == 2025) & np.isin(T['batter'], np.asarray(list(maps), dtype=np.int64))
    ix = np.flatnonzero(te)
    lo_map = np.zeros(len(ix)); own = np.zeros(len(ix))
    for h, rr in _groups(T['batter'][ix], np.ones(len(ix), bool)).items():
        a = ix[rr]; lo_map[rr] = off[a] + Bm[a] @ maps[h]; own[rr] = Bm[a] @ (maps[h] - mbar[h])
    key = T['game'][ix].astype(np.int64) * 1000 + T['ab'][ix].astype(np.int64)
    Hk = (H['game_pk'].astype(np.int64) * 1000 + H['at_bat_number'].astype(np.int64) - 1).to_numpy()
    st = pd.DataFrame({'outs': H['outs_when_up'].to_numpy(), 'r2': H['on_2b'].notna().to_numpy() if H['on_2b'].dtype == object else np.isfinite(H['on_2b'].to_numpy(dtype=float)),
                       'r3': H['on_3b'].notna().to_numpy() if H['on_3b'].dtype == object else np.isfinite(H['on_3b'].to_numpy(dtype=float)),
                       'inning': H['inning'].to_numpy(), 'diff': H['bat_score_diff'].to_numpy()}, index=Hk)
    st = st[~st.index.duplicated()].reindex(key)
    have = st['outs'].notna().to_numpy()
    risp = (st['r2'].fillna(False).to_numpy(bool) | st['r3'].fillna(False).to_numpy(bool)).astype(float)
    late_close = ((st['inning'].fillna(0).to_numpy() >= 7) & (np.abs(st['diff'].fillna(99).to_numpy()) <= 1)).astype(float)
    two_out = (st['outs'].fillna(0).to_numpy() == 2).astype(float)
    sig = lambda v: 1 / (1 + np.exp(-v))
    xt, zt = T['px'][ix], T['pz'][ix]
    u_t = np.where(T['stand_r'][ix] == 1, xt, -xt)
    outside = (np.abs(u_t) > ZONE_HALF) | (zt > ZONE_TOP) | (zt < ZONE_BOT)
    def fit(Zs, ys, bs):
        b = np.zeros(Zs.shape[1])
        for _ in range(30):
            p_ = sig(bs + Zs @ b); W = p_ * (1 - p_)
            step = np.linalg.solve((Zs * W[:, None]).T @ Zs + 1e-6 * np.eye(Zs.shape[1]), Zs.T @ (ys - p_)); b += step
            if np.max(np.abs(step)) < 1e-9:
                break
        return b
    names = ['intercept', 'scoring_position', 'late_and_close', 'two_outs', 'own_part', 'own_part_x_scoring_position', 'own_part_x_late_and_close']
    for zn, zm in (('outside', outside), ('inside', ~outside)):
        m = zm & have
        Z = np.column_stack([np.ones(m.sum()), risp[m], late_close[m], two_out[m], own[m], own[m] * risp[m], own[m] * late_close[m]])
        yy = swing[ix][m]; bs_ = lo_map[m]
        bfull = fit(Z, yy, bs_)
        games = T['game'][ix][m]; ug, gi = np.unique(games, return_inverse=True); draws = []
        for _ in range(int(params.get('reps', 60))):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi]; sel = np.repeat(np.arange(len(yy)), w)
            draws.append(fit(Z[sel], yy[sel], bs_[sel]))
        draws = np.asarray(draws)
        res[zn] = {'pitches': int(m.sum()), 'share_scoring_position': round(float(risp[m].mean()), 3), 'share_late_close': round(float(late_close[m].mean()), 3),
                   'coefficients_logodds': {n_: [round(float(bfull[j]), 4), round(float(np.percentile(draws[:, j], 2.5)), 4), round(float(np.percentile(draws[:, j], 97.5)), 4)] for j, n_ in enumerate(names)},
                   'points_at_mean': {n_: round(float(bfull[j] * np.mean(sig(bs_) * (1 - sig(bs_))) * 100), 2) for j, n_ in enumerate(names[1:4], start=1)}}
    res['matched_share'] = round(float(have.mean()), 4)
    stage('pressure')
    return res


# ---------------------------------------------------------------- VALUE-12: a hitter's own whiff holes, priced
# ---------------------------------------------------------------- COMMAND-02: execution scatter from repeated intent
def _repeat_em(D: np.ndarray, iters: int = 400, floor: float = 0.06, init_s: float = 0.3, init_pi: float = 0.3):
    """Location change between consecutive pitches of one type (feet, 2 columns) as a two-part mixture: a same-intent
    part N(0, diag(2 sx^2, 2 sz^2)), where only execution differs, and a changed-intent part N(mu, S). EM from a narrow
    start. Returns sx, sz, the same-intent share and each pair's posterior same-intent probability."""
    n = len(D)
    a = b = 2.0 * init_s ** 2; pi = init_pi; fl = 2.0 * floor ** 2
    mu = D.mean(0); S = np.cov(D.T) + np.eye(2) * 1e-3
    r = np.full(n, pi)
    for _ in range(iters):
        fn = np.exp(-0.5 * (D[:, 0] ** 2 / a + D[:, 1] ** 2 / b)) / (2.0 * np.pi * np.sqrt(a * b))
        Si = np.linalg.inv(S); dd = D - mu
        q = dd[:, 0] ** 2 * Si[0, 0] + 2.0 * dd[:, 0] * dd[:, 1] * Si[0, 1] + dd[:, 1] ** 2 * Si[1, 1]
        fb = np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(max(np.linalg.det(S), 1e-12)))
        num = pi * fn; r = num / (num + (1.0 - pi) * fb + 1e-300)
        sr = float(r.sum()); w = 1.0 - r; sb = float(w.sum())
        pi_new = sr / n
        a = max(float((r * D[:, 0] ** 2).sum()) / max(sr, 1e-9), fl); b = max(float((r * D[:, 1] ** 2).sum()) / max(sr, 1e-9), fl)
        mu = (w[:, None] * D).sum(0) / max(sb, 1e-9); dd = D - mu
        S = np.array([[(w * dd[:, 0] ** 2).sum(), (w * dd[:, 0] * dd[:, 1]).sum()], [(w * dd[:, 0] * dd[:, 1]).sum(), (w * dd[:, 1] ** 2).sum()]]) / max(sb, 1e-9) + np.eye(2) * 1e-3
        done = abs(pi_new - pi) < 1e-7
        pi = pi_new
        if done:
            break
    if a > S[0, 0] and b > S[1, 1]:                   # the parts swapped: no narrow same-intent part found
        return np.nan, np.nan, np.nan, r
    return float(np.sqrt(a / 2.0)), float(np.sqrt(b / 2.0)), float(pi), r


def _gmm_xctrl(L: np.ndarray, kmax: int = 4, iters: int = 80, seed: int = 0) -> float:
    """The closest published measure (xCTRL, arXiv 2508.19184): a Gaussian mixture of one pitcher's locations for one
    pitch type and batter side, components chosen by BIC (1 to kmax), and the mean over pitches of the distance to each
    component's center weighted by the pitch's posterior. Feet."""
    n = len(L); rng = np.random.default_rng(seed); best = (np.inf, None)
    for K in range(1, kmax + 1):
        c = [L[rng.integers(n)]]
        for _ in range(1, K):                            # k-means++ start
            d2 = np.min(((L[:, None, :] - np.asarray(c)[None, :, :]) ** 2).sum(-1), 1)
            c.append(L[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else L[rng.integers(n)])
        mus = np.asarray(c, dtype=np.float64); covs = np.repeat((np.cov(L.T) + np.eye(2) * 1e-3)[None], K, 0); ws = np.full(K, 1.0 / K)
        for _ in range(iters):
            dens = np.empty((n, K))
            for k in range(K):
                Si = np.linalg.inv(covs[k]); dd = L - mus[k]
                q = dd[:, 0] ** 2 * Si[0, 0] + 2.0 * dd[:, 0] * dd[:, 1] * Si[0, 1] + dd[:, 1] ** 2 * Si[1, 1]
                dens[:, k] = ws[k] * np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(max(np.linalg.det(covs[k]), 1e-12)))
            tot = dens.sum(1) + 1e-300; R = dens / tot[:, None]; nk = R.sum(0) + 1e-9
            ws = nk / n; mus = (R.T @ L) / nk[:, None]
            for k in range(K):
                dd = L - mus[k]
                covs[k] = (R[:, k, None, None] * np.einsum('ij,ik->ijk', dd, dd)).sum(0) / nk[k] + np.eye(2) * 1e-3
        ll = float(np.log(tot).sum()); bic = -2.0 * ll + (6 * K - 1) * np.log(n)
        if bic < best[0]:
            dist = np.sqrt(((L[:, None, :] - mus[None, :, :]) ** 2).sum(-1))
            best = (bic, float((R * dist).sum(1).mean()))
    return best[1]


def _group_rows(key: np.ndarray, sel: np.ndarray):
    """Yield (key, row indices) for each key among the selected rows (one sort, no per-key scans)."""
    rows = np.flatnonzero(sel)
    if not len(rows):
        return
    ks, inv = np.unique(key[rows], return_inverse=True)
    order = np.argsort(inv, kind='stable'); cnt = np.bincount(inv); starts = np.r_[0, np.cumsum(cnt)[:-1]]
    for j, k in enumerate(ks):
        yield int(k), rows[order[starts[j]:starts[j] + cnt[j]]]


def command2_study(T: dict, params: dict, stage) -> dict:
    """COMMAND-02. Command is the scatter of a pitch around where it was aimed; the aim is not in public data. When a
    pitcher throws the same pitch type twice in a row, sometimes he aims at the same spot again; then the change in
    location between the two pitches is pure execution (variance 2 sigma^2 per axis, the pitcher's lasting miss
    direction cancels). The change over all consecutive same-type pairs is a mixture of that narrow same-intent part
    and a broad changed-intent part; its narrow width is an execution-scatter estimate (an upper bound: re-aiming a
    little differently counts as scatter). Per pitcher, season and pitch group (at least min_pairs pairs), against the
    raw location spread, an xCTRL-style mixture distance and the spread of 3-0 and 3-1 fastballs; reliability (game
    halves, seasons), whether the narrow part is larger after two-strike fouls (count unchanged, the same aim most
    likely), and whether each measure predicts next season's walks and hit batters beyond walk and zone rates, and the
    other half's walks within a season."""
    res = {}
    min_pairs = int(params.get('min_pairs', 80)); rng = np.random.default_rng(7)
    ok = (T['group'] >= 0) & (T['group'] <= 5) & np.isfinite(T['px']) & np.isfinite(T['pz']) & (T['call'] != 3) & (T['bunt_pa'] == 0)
    pak = T['game'].astype(np.int64) * 1000 + T['ab'].astype(np.int64)
    order = np.lexsort((T['pitch_no'], pak)); o1, o2 = order[:-1], order[1:]
    pair = (pak[o1] == pak[o2]) & (T['pitch_no'][o2] == T['pitch_no'][o1] + 1) & ok[o1] & ok[o2] & (T['group'][o1] == T['group'][o2]) & (T['pitcher'][o1] == T['pitcher'][o2])
    i1, i2 = o1[pair], o2[pair]
    D = np.column_stack([T['px'][i2].astype(np.float64) - T['px'][i1], T['pz'][i2].astype(np.float64) - T['pz'][i1]])
    same_count = (T['balls'][i1] == T['balls'][i2]) & (T['strikes'][i1] == T['strikes'][i2])
    p_pit, p_ssn, p_grp = T['pitcher'][i1].astype(np.int64), T['season'][i1].astype(np.int64), T['group'][i1].astype(np.int64)
    p_half = (T['game'][i1].astype(np.int64) % 2)
    res['pairs'] = {'all': int(len(i1)), 'count_unchanged': int(same_count.sum()), 'share_of_pitches_in_a_pair': round(float(len(i1) / max(ok.sum(), 1)), 3)}
    stage('pairs')

    def fit_cells(sel):
        key = (p_pit * 10000 + p_ssn) * 10 + p_grp
        out = {}; post = np.full(len(D), np.nan)
        for k, rows in _group_rows(key, sel):
            if len(rows) < min_pairs:
                continue
            sx, sz, pi, r = _repeat_em(D[rows])
            if np.isfinite(sx):
                out[k] = (sx, sz, pi, int(len(rows))); post[rows] = r
        return out, post
    cells, post = fit_cells(np.ones(len(D), bool))
    stage(f'repeat mixtures ({len(cells)} cells)')
    halves = {h: fit_cells(p_half == h)[0] for h in (0, 1)}
    stage('half-season mixtures')
    grp_names = {0: 'four_seam', 1: 'sinker', 2: 'cutter', 3: 'slider', 4: 'curveball', 5: 'changeup_splitter'}
    sig = lambda v: float(np.sqrt((v[0] ** 2 + v[1] ** 2) / 2.0))
    res['scatter_ft'] = {}
    for g, name in grp_names.items():
        v = [(c[0], c[1], c[2]) for k, c in cells.items() if k % 10 == g]
        if len(v) < 10:
            continue
        a = np.asarray(v)
        res['scatter_ft'][name] = {'cells': len(v), 'median_sigma': round(float(np.median(np.sqrt((a[:, 0] ** 2 + a[:, 1] ** 2) / 2))), 3),
                                   'quartiles_sigma': [round(float(np.percentile(np.sqrt((a[:, 0] ** 2 + a[:, 1] ** 2) / 2), q)), 3) for q in (25, 75)],
                                   'median_sigma_x': round(float(np.median(a[:, 0])), 3), 'median_sigma_z': round(float(np.median(a[:, 1])), 3),
                                   'median_same_intent_share': round(float(np.median(a[:, 2])), 3)}
    # same-intent reading: the posterior same-intent probability after a two-strike foul (count unchanged) against other pairs
    fitted = np.isfinite(post)
    keyc = ((p_pit * 10000 + p_ssn) * 10 + p_grp)
    foul1 = (T['call'][i1] == 1) & (T['last_in_pa'][i1] == 0)       # the first pitch fouled off: the same selection on both sides
    a_ = fitted & same_count & foul1; b_ = fitted & ~same_count & foul1

    def cell_means(sel):
        u, iv = np.unique(keyc[sel], return_inverse=True)
        return u, np.bincount(iv, weights=post[sel]) / np.bincount(iv)
    ua, ma_all = cell_means(a_); ub, mb_all = cell_means(b_)
    both, ia, ib = np.intersect1d(ua, ub, return_indices=True)
    if len(both) > 20:
        ma, mb = ma_all[ia], mb_all[ib]
        dlt = ma - mb; bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(1000)]
        res['same_intent_after_two_strike_foul'] = {'comparison': 'pairs after a two-strike foul (count unchanged) against pairs after a foul with fewer strikes', 'cells': int(len(dlt)), 'mean_posterior_count_unchanged': round(float(ma.mean()), 4),
                                                     'mean_posterior_after_earlier_fouls': round(float(mb.mean()), 4),
                                                     'difference': [round(float(dlt.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    stage('same-intent check')

    # other measures per pitcher, season, group (and side for the location ones)
    base_ok = ok & (T['group'] <= 5)
    blob = {}; xct = {}
    ck = ((T['pitcher'].astype(np.int64) * 10000 + T['season'].astype(np.int64)) * 10 + T['group'].astype(np.int64)) * 2 + T['stand_r'].astype(np.int64)
    xmax = int(params.get('xctrl_cells', 6000)); n_x = 0
    for k, rows in _group_rows(ck, base_ok):
        if len(rows) < 60:
            continue
        L = np.column_stack([T['px'][rows], T['pz'][rows]]).astype(np.float64)
        cell = k // 2
        blob.setdefault(cell, []).append((float(np.sqrt((L[:, 0].var() + L[:, 1].var()) / 2.0)), len(rows)))
        if len(rows) >= 250 and n_x < xmax:
            xct.setdefault(cell, []).append((_gmm_xctrl(L), len(rows))); n_x += 1
    stage(f'location spread and xCTRL-style cells ({n_x})')
    blob = {k: (sum(s * n for s, n in v) / sum(n for _, n in v), sum(n for _, n in v)) for k, v in blob.items()}
    xct = {k: (sum(s * n for s, n in v) / sum(n for _, n in v), sum(n for _, n in v)) for k, v in xct.items()}
    # must-strike spread: four-seamers and sinkers at 3-0 and 3-1, per pitcher, season and side
    ms_ok = ok & np.isin(T['group'], (0, 1)) & (T['balls'] == 3) & (T['strikes'] <= 1)
    mk = (T['pitcher'].astype(np.int64) * 10000 + T['season'].astype(np.int64)) * 2 + T['stand_r'].astype(np.int64)
    must = {}
    for k, rows in _group_rows(mk, ms_ok):
        if len(rows) < 12:
            continue
        L = np.column_stack([T['px'][rows], T['pz'][rows]]).astype(np.float64)
        must.setdefault(k // 2, []).append((float(np.sqrt((L[:, 0].var() + L[:, 1].var()) / 2.0)), int(len(rows))))
    must = {k: (sum(s * n for s, n in v) / sum(n for _, n in v), sum(n for _, n in v)) for k, v in must.items()}
    stage('must-strike spread')

    def index(cellmap, value_of):
        """Pitcher-season index: each group's value over that group's league median, weighted by count."""
        med = {}
        for g in range(6):
            v = [value_of(c) for k, c in cellmap.items() if k % 10 == g]
            if len(v) >= 10:
                med[g] = float(np.median(v))
        acc = {}
        for k, c in cellmap.items():
            g = k % 10
            if g not in med:
                continue
            ps = k // 10; n_ = c[-1]
            s0, n0 = acc.get(ps, (0.0, 0))
            acc[ps] = (s0 + n_ * value_of(c) / med[g], n0 + n_)
        return {ps: s / n for ps, (s, n) in acc.items()}
    idx = {'repeat_scatter': index(cells, sig), 'location_spread': index(blob, lambda c: c[0]), 'xctrl_style': index(xct, lambda c: c[0])}
    idx_half = {h: index(halves[h], sig) for h in (0, 1)}
    med_ms = float(np.median([v[0] for v in must.values()])) if must else np.nan
    idx['must_strike_spread'] = {ps: v[0] / med_ms for ps, v in must.items()}

    # outcomes per pitcher-season and per half
    last = (T['last_in_pa'] == 1) & (T['out7'] >= 0)
    psk = T['pitcher'].astype(np.int64) * 10000 + T['season'].astype(np.int64)
    halfk = psk * 2 + (T['game'].astype(np.int64) % 2)
    inz = (T['zone'] >= 1) & (T['zone'] <= 9); zk = T['zone'] >= 1

    def tally(key, sel):
        u, iv = np.unique(key[sel], return_inverse=True)
        return dict(zip(u.tolist(), np.bincount(iv).tolist()))
    outz = (T['zone'] >= 11); chase_ = outz & ok & ((T['call'] == 1) | (T['call'] == 2))
    PA = tally(psk, last); BB = tally(psk, last & (T['out7'] == 2)); Z = tally(psk, zk & ok); ZI = tally(psk, zk & ok & inz)
    OZ = tally(psk, outz & ok); CH = tally(psk, chase_)
    PAh = tally(halfk, last); BBh = tally(halfk, last & (T['out7'] == 2)); Zh = tally(halfk, zk & ok); ZIh = tally(halfk, zk & ok & inz)
    OZh = tally(halfk, outz & ok); CHh = tally(halfk, chase_)
    lw = np.where(last, np.asarray([-0.26, -0.28, 0.32, 0.47, 0.78, 1.40, 0.45])[np.clip(T['out7'], 0, 6)], 0.0)
    u_, iv_ = np.unique(psk[last], return_inverse=True); RV = dict(zip(u_.tolist(), np.bincount(iv_, weights=lw[last]).tolist()))
    lg_bb = sum(BB.values()) / max(sum(PA.values()), 1)
    res['league_walk_rate'] = round(lg_bb, 4)

    # reliability
    rel = {}
    common = sorted(set(idx_half[0]) & set(idx_half[1]))
    if len(common) > 30:
        x = np.array([idx_half[0][k] for k in common]); y = np.array([idx_half[1][k] for k in common]); r_ = float(np.corrcoef(x, y)[0, 1])
        rel['repeat_scatter_game_halves'] = {'pitcher_seasons': len(common), 'correlation': round(r_, 3), 'spearman_brown': round(2 * r_ / (1 + r_), 3)}
    for name, d in idx.items():
        pr = [(d[ps], d[ps + 1]) for ps in d if ps + 1 in d]
        if len(pr) > 30:
            a = np.asarray(pr); rel[f'{name}_season_to_season'] = {'pairs': len(pr), 'correlation': round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3)}
    names = list(idx)
    for i_, a_n in enumerate(names):
        for b_n in names[i_ + 1:]:
            cm = sorted(set(idx[a_n]) & set(idx[b_n]))
            if len(cm) > 30:
                rel[f'corr_{a_n}_vs_{b_n}'] = round(float(np.corrcoef([idx[a_n][k] for k in cm], [idx[b_n][k] for k in cm])[0, 1]), 3)
    res['reliability'] = rel
    stage('reliability')

    # prediction 1: next season's walks and hit batters per plate appearance
    kbb = 150.0; min_pa = int(params.get('min_pa', 150))

    target = 'walks'

    def design(ps, nxt, measure):
        pa0, pa1 = PA.get(ps, 0), PA.get(nxt, 0)
        if pa0 < min_pa or pa1 < min_pa or Z.get(ps, 0) < 200:
            return None
        bb0 = (BB.get(ps, 0) + kbb * lg_bb) / (pa0 + kbb)
        row = [1.0, bb0, ZI.get(ps, 0) / Z[ps], CH.get(ps, 0) / max(OZ.get(ps, 0), 1)] + ([measure[ps]] if measure is not None else [])
        return row, (BB.get(nxt, 0) / pa1 if target == 'walks' else RV.get(nxt, 0.0) / pa1), pa1

    def predictive(measure, common_set=None):
        out = {}
        data = {}
        for ps in (measure if measure is not None else PA):
            ssn = ps % 10000
            if common_set is not None and ps not in common_set:
                continue
            r0 = design(ps, ps + 1, None); r1 = design(ps, ps + 1, measure)
            if r0 is None or r1 is None:
                continue
            data.setdefault(ssn, []).append((r0[0], r1[0], r1[1], r1[2], ps // 10000))
        tr = data.get(2023, []); te = data.get(2024, []) + data.get(2025, [])
        if len(tr) < 60 or len(te) < 60:
            return {'error': 'too few pitchers', 'train': len(tr), 'test': len(te)}
        def arr(rows):
            return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), np.array([r[3] for r in rows], float), np.array([r[4] for r in rows]))
        X0, X1, y, w, pid = arr(tr); Z0, Z1, yt, wt, pidt = arr(te)
        def wls(X, y, w):
            return np.linalg.solve((X * w[:, None]).T @ X + np.eye(X.shape[1]) * 1e-9, (X * w[:, None]).T @ y)
        b0 = wls(X0, y, w); b1 = wls(X1, y, w)
        e0 = wt * (yt - Z0 @ b0) ** 2; e1 = wt * (yt - Z1 @ b1) ** 2
        red = (e0 - e1) / wt.sum() * 1e4                  # squared error reduction, (walk rate points)^2 x 1e4 / PA
        up = np.unique(pidt); gi = np.searchsorted(up, pidt); bsr = []; bsc = []
        for _ in range(1000):
            wb = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[gi].astype(float)
            bsr.append(float((wb * (e0 - e1)).sum() / max((wb * wt).sum(), 1e-9) * 1e4))
        upt = np.unique(pid); git = np.searchsorted(upt, pid)
        for _ in range(300):
            wb = np.bincount(rng.integers(0, len(upt), len(upt)), minlength=len(upt))[git].astype(float)
            bsc.append(float(wls(X1, y, w * wb)[-1]))
        out = {'train_pitchers': int(len(y)), 'test_pitcher_seasons': int(len(yt)),
               'coefficient_per_unit_index': [round(float(b1[-1]), 4), round(float(np.percentile(bsc, 2.5)), 4), round(float(np.percentile(bsc, 97.5)), 4)],
               'squared_error_reduction_x1e4': [round(float(red.sum()), 4), round(float(np.percentile(bsr, 2.5)), 4), round(float(np.percentile(bsr, 97.5)), 4)],
               'test_r2_base': round(float(1 - e0.sum() / (wt * (yt - np.average(yt, weights=wt)) ** 2).sum()), 4),
               'test_r2_with': round(float(1 - e1.sum() / (wt * (yt - np.average(yt, weights=wt)) ** 2).sum()), 4)}
        return out
    res['next_season_walks'] = {name: predictive(d) for name, d in idx.items()}
    target = 'runs'
    res['next_season_run_value_secondary'] = {name: predictive(d) for name, d in idx.items() if name in ('repeat_scatter', 'location_spread')}
    target = 'walks'
    common_set = set(idx['repeat_scatter']) & set(idx['location_spread']) & set(idx['xctrl_style'])
    res['next_season_walks_common_pitchers'] = {name: predictive(d, common_set) for name, d in idx.items() if name != 'must_strike_spread'}
    # the repeat index beyond the location spread: the spread enters the base
    both_ = {ps: v for ps, v in idx['repeat_scatter'].items() if ps in idx['location_spread']}
    if both_:
        bx = {}
        for ps in both_:
            bx[ps] = (idx['location_spread'][ps], idx['repeat_scatter'][ps])
        rows = []
        for ps, (ls_, rs_) in bx.items():
            r0 = design(ps, ps + 1, None)
            if r0 is None:
                continue
            rows.append((r0[0] + [ls_], r0[0] + [ls_, rs_], r0[1], r0[2], ps // 10000, ps % 10000))
        tr = [r for r in rows if r[5] == 2023]; te = [r for r in rows if r[5] in (2024, 2025)]
        if len(tr) >= 60 and len(te) >= 60:
            def arr2(rr):
                return np.array([r[0] for r in rr]), np.array([r[1] for r in rr]), np.array([r[2] for r in rr]), np.array([r[3] for r in rr], float), np.array([r[4] for r in rr])
            X0, X1, y, w, pid = arr2(tr); Z0, Z1, yt, wt, pidt = arr2(te)
            b0 = np.linalg.solve((X0 * w[:, None]).T @ X0, (X0 * w[:, None]).T @ y); b1 = np.linalg.solve((X1 * w[:, None]).T @ X1, (X1 * w[:, None]).T @ y)
            e0 = wt * (yt - Z0 @ b0) ** 2; e1 = wt * (yt - Z1 @ b1) ** 2
            up = np.unique(pidt); gi = np.searchsorted(up, pidt); bsr = []
            for _ in range(1000):
                wb = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[gi].astype(float)
                bsr.append(float((wb * (e0 - e1)).sum() / max((wb * wt).sum(), 1e-9) * 1e4))
            res['next_season_walks_repeat_beyond_spread'] = {'test_pitcher_seasons': int(len(yt)), 'coefficient': round(float(b1[-1]), 4),
                                                             'squared_error_reduction_x1e4': [round(float((e0 - e1).sum() / wt.sum() * 1e4), 4), round(float(np.percentile(bsr, 2.5)), 4), round(float(np.percentile(bsr, 97.5)), 4)]}
    stage('next-season walks')

    # prediction 2: within a season, one half's repeat scatter and the other half's walks
    rows = []
    for h in (0, 1):
        for ps, v in idx_half[h].items():
            a_k, b_k = ps * 2 + h, ps * 2 + (1 - h)
            if PAh.get(a_k, 0) < 80 or PAh.get(b_k, 0) < 80 or Zh.get(a_k, 0) < 150:
                continue
            bb0 = (BBh.get(a_k, 0) + kbb * lg_bb) / (PAh[a_k] + kbb)
            rows.append(([1.0, bb0, ZIh.get(a_k, 0) / Zh[a_k], CHh.get(a_k, 0) / max(OZh.get(a_k, 0), 1)], v, BBh.get(b_k, 0) / PAh[b_k], PAh[b_k], ps // 10000, ps % 10000))
    tr = [r for r in rows if r[5] in (2023, 2024)]; te = [r for r in rows if r[5] in (2025, 2026)]
    if len(tr) >= 60 and len(te) >= 60:
        X0 = np.array([r[0] for r in tr]); X1 = np.column_stack([X0, [r[1] for r in tr]]); y = np.array([r[2] for r in tr]); w = np.array([r[3] for r in tr], float)
        Z0 = np.array([r[0] for r in te]); Z1 = np.column_stack([Z0, [r[1] for r in te]]); yt = np.array([r[2] for r in te]); wt = np.array([r[3] for r in te], float)
        pidt = np.array([r[4] for r in te])
        b0 = np.linalg.solve((X0 * w[:, None]).T @ X0, (X0 * w[:, None]).T @ y); b1 = np.linalg.solve((X1 * w[:, None]).T @ X1, (X1 * w[:, None]).T @ y)
        e0 = wt * (yt - Z0 @ b0) ** 2; e1 = wt * (yt - Z1 @ b1) ** 2
        up = np.unique(pidt); gi = np.searchsorted(up, pidt); bsr = []
        for _ in range(1000):
            wb = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[gi].astype(float)
            bsr.append(float((wb * (e0 - e1)).sum() / max((wb * wt).sum(), 1e-9) * 1e4))
        res['other_half_walks'] = {'train_rows': int(len(y)), 'test_rows': int(len(yt)), 'coefficient': round(float(b1[-1]), 4),
                                   'squared_error_reduction_x1e4': [round(float((e0 - e1).sum() / wt.sum() * 1e4), 4), round(float(np.percentile(bsr, 2.5)), 4), round(float(np.percentile(bsr, 97.5)), 4)]}
    stage('other-half walks')
    if params.get('export'):
        ex = {}
        for k, c in cells.items():
            ps = k // 10
            ex.setdefault(str(ps), {})[str(k % 10)] = [round(c[0], 3), round(c[1], 3), round(c[2], 3), c[3]]
        res['export'] = {'by_pitcher_season': ex, 'index': {str(ps): round(v, 3) for ps, v in idx['repeat_scatter'].items()},
                         'note': 'key pitcher*10000+season; group: [sigma_x ft, sigma_z ft, same-intent share, pairs]'}
    return res


# ---------------------------------------------------------------- WARMUP-01: whose decline is the times-through-the-order penalty?
def warmup_study(T: dict, H, params: dict, stage) -> dict:
    """Hitters do better the more often they have batted in a game. Three explanations are tangled in a starter's
    times through the order: the hitter warming up to live pitching from anyone (general exposure), the hitter learning
    this pitcher (pair familiarity) and the pitcher tiring (his pitches thrown). Relievers separate them: a reliever's
    first meeting with a hitter has no pair familiarity and little load, while the hitter may have batted three times
    already; relievers who enter early (short starts, openers, long relief) face hitters who have batted little.
    Substitutes come in late and cold. Plate appearances 2023 to July 2026 (feed), with outs, runners and score from the
    plate-appearance history. Talent enters as batter-season and pitcher-season fixed effects, not as earlier rates,
    which absorb each player's usual contexts (relievers mostly face hitters who have batted; starters tire).
    (S1) Joint run-value model: steps for the hitter's earlier plate appearances in the game (k = 1, 2, 3, 4 or more),
    his earlier meetings with this pitcher (F = 1, 2 or more), the pitcher's pitches thrown per 100, a substitute's first
    plate appearance and the hitter's pitches seen beyond 3.9 per earlier plate appearance, with platoon, home, outs,
    runners and score; game-bootstrap intervals; with inning and in linear form as secondary readings. (S2) Relievers'
    first meetings only. (S4) Out-of-sample log loss (fit 2023-2024, scored 2025 and 2026 through July; earlier rates as
    talent, as the simulator has them) of the decomposition added to the simulator's structure (starter times through by
    batters faced, reliever, inning). A substitutes-batting-twice contrast was dropped before registering: conditioning on
    a second plate appearance selects innings where the first went well."""
    res = {}
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    real = T['group'] != -1                                            # pickoffs and other non-pitches do not count
    gh = T['game'].astype(np.int64) * 2 + T['half'].astype(np.int64)
    order = np.lexsort((T['pitch_no'], T['ab'], gh)); T = take(T, order); real = real[order]
    pak = T['game'].astype(np.int64) * 1000 + T['ab'].astype(np.int64)
    first = np.r_[True, pak[1:] != pak[:-1]]
    pa_id = np.cumsum(first) - 1
    npitch = np.bincount(pa_id, weights=real.astype(float))
    # the pitcher's pitches thrown in this game before each pitch
    gp = T['game'].astype(np.int64) * 10_000_000 + T['pitcher'].astype(np.int64)
    o2 = np.lexsort((T['pitch_no'], T['ab'], gp)); gps = gp[o2]
    st2 = np.r_[True, gps[1:] != gps[:-1]]; grp2 = np.cumsum(st2) - 1
    cr = np.cumsum(real[o2].astype(float)); cr0 = cr - real[o2]
    load = np.empty(len(gp)); load[o2] = cr0 - (cr0[np.flatnonzero(st2)])[grp2]
    # starters: the first pitcher of each half-inning side in each game
    ghs = T['game'].astype(np.int64) * 2 + T['half'].astype(np.int64)
    starter_of = {}
    for g_, p_ in zip(ghs[first], T['pitcher'][first]):
        starter_of.setdefault(int(g_), int(p_))
    P = {k: v[first] for k, v in T.items()}
    P['load'] = load[first]; P['np'] = npitch
    P['starter'] = np.asarray([starter_of[int(g_)] == int(p_) for g_, p_ in zip(ghs[first], P['pitcher'])])
    df = pd.DataFrame({'game': P['game'], 'half': P['half'], 'ab': P['ab'], 'batter': P['batter'], 'pitcher': P['pitcher'], 'np': P['np']})
    df['k'] = df.groupby(['game', 'batter']).cumcount()
    df['E'] = df.groupby(['game', 'batter'])['np'].cumsum() - df['np']
    df['F'] = df.groupby(['game', 'batter', 'pitcher']).cumcount()
    df['team_pa'] = df.groupby(['game', 'half']).cumcount()
    df['sub'] = df.groupby(['game', 'batter'])['team_pa'].transform('min') >= 9
    df['bf'] = df.groupby(['game', 'pitcher']).cumcount()               # the pitcher's batters faced before this one
    k = df['k'].to_numpy(); E = df['E'].to_numpy(float); F = df['F'].to_numpy(); sub = df['sub'].to_numpy(); bf = df['bf'].to_numpy()
    stage('exposure')
    # base-out state and score from the plate-appearance history
    Hk = (H['game_pk'].astype(np.int64) * 1000 + H['at_bat_number'].astype(np.int64) - 1).to_numpy()
    Hs = H.set_index(pd.Index(Hk))
    Hs = Hs[~Hs.index.duplicated()]
    key = P['game'].astype(np.int64) * 1000 + P['ab'].astype(np.int64)
    st = Hs.reindex(key)
    outs = st['outs_when_up'].to_numpy(float); r1 = st['on_1b'].notna().to_numpy(float); r2 = st['on_2b'].notna().to_numpy(float); r3 = st['on_3b'].notna().to_numpy(float)
    sd_ = st['bat_score_diff'].to_numpy(float)
    have_state = np.isfinite(outs) & np.isfinite(sd_)
    res['state_matched_share'] = round(float(have_state.mean()), 4)
    ok = (P['out7'] >= 0) & have_state & (P['bunt_pa'] == 0)
    y7 = P['out7'].astype(int)
    # both players' earlier rates by outcome class (shrunk, log ratios to the league)
    league = np.bincount(y7[ok], minlength=7) / ok.sum()

    def rates(key_, kk):
        cols_ = []
        for c in range(7):
            nn, ss = _prior_by_day(key_, P['day'].astype(np.int64), (y7 == c).astype(float), ok)
            cols_.append(np.log((ss + kk * league[c]) / (nn + kk) / league[c]))
        return np.column_stack(cols_)
    rb = rates(P['batter'].astype(np.int64), 150.0); rp = rates(P['pitcher'].astype(np.int64), 300.0)
    stage('rates')
    inn = np.clip(P['inning'], 1, 10)
    C = np.column_stack([rb, rp, (P['stand_r'] == P['throw_r']).astype(float), (P['half'] == 1).astype(float)]
                        + [(inn == i).astype(float) for i in range(2, 11)]
                        + [(P['season'] == s).astype(float) for s in (2024, 2025, 2026)]
                        + [np.nan_to_num(outs) == o for o in (1, 2)] + [r1, r2, r3, r1 * r2, r2 * r3, r1 * r3]
                        + [np.clip(np.nan_to_num(sd_), -4, 4) == v for v in (-4, -3, -2, -1, 1, 2, 3, 4)]).astype(float)
    rel = (~P['starter']).astype(float)
    K = np.minimum(k, 4).astype(float)[:, None]                       # earlier plate appearances in the game (to 4)
    Fm = np.minimum(F, 2).astype(float)[:, None]                      # earlier meetings with this pitcher (to 2)
    L = (np.minimum(P['load'].astype(float), 120.0) / 100.0)[:, None]   # the pitcher's pitches thrown, per 100
    extra = ((E - 3.9 * k) / 10.0)[:, None]
    subf = (sub & (k == 0)).astype(float)[:, None]
    # game-state controls for the fixed-effects designs (talent is in the player-season effects, not in prior rates,
    # which absorb each player's usual contexts: relievers mostly face hitters who have batted, starters tire)
    G = np.column_stack([(P['stand_r'] == P['throw_r']).astype(float), (P['half'] == 1).astype(float)]
                        + [np.nan_to_num(outs) == o for o in (1, 2)] + [r1, r2, r3, r1 * r2, r2 * r3, r1 * r3]
                        + [np.clip(np.nan_to_num(sd_), -4, 4) == v for v in (-4, -3, -2, -1, 1, 2, 3, 4)]).astype(float)
    inn_d = np.column_stack([(inn == i) for i in range(2, 11)]).astype(float)
    rv = LW7[np.clip(y7, 0, 6)]
    games = P['game']
    rng = np.random.default_rng(5)
    bsk = P['batter'].astype(np.int64) * 10000 + P['season']; psk = P['pitcher'].astype(np.int64) * 10000 + P['season']

    def fe_fit(rows, X, names, reps):
        """Run value on X with batter-season and pitcher-season effects estimated from the player's other games
        (leave-game-out), so a player's effect never contains the errors of the plate appearance being explained:
        with effects from the same games, a reliever's bad outing both lengthens it (more hitters, higher k) and lowers
        his effect, which biases the exposure steps (seen in the synthetic null). Two rounds: plain two-way demeaning for
        a first beta, then the effects from the residual, taken out game by game, and least squares on the remainder.
        Intervals from resampling games with the effects held."""
        ub, ib = np.unique(bsk[rows], return_inverse=True); up, ip = np.unique(psk[rows], return_inverse=True)
        cb = np.bincount(ib).astype(float); cp = np.bincount(ip).astype(float)
        Xr = X[rows].astype(np.float64); yr = rv[rows]
        Z = np.column_stack([Xr, yr])
        for _ in range(int(params.get('fe_iters', 40))):
            for ids, cnt in ((ib, cb), (ip, cp)):
                for j in range(Z.shape[1]):
                    Z[:, j] -= (np.bincount(ids, weights=Z[:, j], minlength=len(cnt)) / cnt)[ids]
        beta = np.linalg.lstsq(Z[:, :-1], Z[:, -1], rcond=None)[0]
        gr = games[rows]
        gb_key = ib.astype(np.int64) * 10_000_000 + (gr % 10_000_000); gp_key = ip.astype(np.int64) * 10_000_000 + (gr % 10_000_000)
        ugb, igb = np.unique(gb_key, return_inverse=True); ugp, igp = np.unique(gp_key, return_inverse=True)
        cgb = np.bincount(igb).astype(float); cgp = np.bincount(igp).astype(float)
        keep = (cb[ib] - cgb[igb] >= 1) & (cp[ip] - cgp[igp] >= 1)
        for _ in range(2):
            r = yr - Xr @ beta; a_ = np.zeros(len(cb)); b_ = np.zeros(len(cp))
            for _ in range(30):
                a_ = np.bincount(ib, weights=r - b_[ip], minlength=len(cb)) / cb
                b_ = np.bincount(ip, weights=r - a_[ib], minlength=len(cp)) / cp
            ra = r - b_[ip]; rb_ = r - a_[ib]
            sa = np.bincount(ib, weights=ra, minlength=len(cb)); sag = np.bincount(igb, weights=ra, minlength=len(cgb))
            sp = np.bincount(ip, weights=rb_, minlength=len(cp)); spg = np.bincount(igp, weights=rb_, minlength=len(cgp))
            with np.errstate(invalid='ignore', divide='ignore'):
                a_lgo = (sa[ib] - sag[igb]) / (cb[ib] - cgb[igb]); b_lgo = (sp[ip] - spg[igp]) / (cp[ip] - cgp[igp])
            yadj = (yr - a_lgo - b_lgo)[keep]; A = np.column_stack([np.ones(keep.sum()), Xr[keep]])
            beta = np.linalg.lstsq(A, yadj, rcond=None)[0][1:]
        A = np.column_stack([np.ones(keep.sum()), Xr[keep]])
        gk = gr[keep]; ug = np.unique(gk); gi = np.searchsorted(ug, gk); draws = []
        for _ in range(reps):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
            Aw = A * w[:, None]
            draws.append(np.linalg.solve(Aw.T @ A + np.eye(A.shape[1]) * 1e-9, Aw.T @ yadj)[1:])
        draws = np.asarray(draws)
        out = {nm: [round(float(beta[i]), 5), round(float(np.percentile(draws[:, i], 2.5)), 5), round(float(np.percentile(draws[:, i], 97.5)), 5)] for i, nm in enumerate(names)}
        out['rows_used'] = int(keep.sum())
        return out, None
    reps = int(params.get('reps', 150))
    res['rows'] = {'plate_appearances': int(ok.sum()), 'reliever_share': round(float(rel[ok].mean()), 3), 'substitute_first_pa': int(subf[ok, 0].sum()),
                   'mean_pitches_per_pa': round(float(P['np'][ok].mean()), 3),
                   'k_shares': [round(float(((k == v) if v < 4 else (k >= 4))[ok].mean()), 3) for v in range(5)]}
    Kd = np.column_stack([(k == 1), (k == 2), (k == 3), (k >= 4), (F == 1), (F >= 2)]).astype(float)
    names1 = ['k1', 'k2', 'k3', 'k4plus', 'F1', 'F2plus', 'per_100_pitches_thrown', 'sub_first', 'extra_pitches_per10']
    res['S1_joint_run_value'], _ = fe_fit(ok, np.hstack([Kd, L, subf, extra, G]), names1, reps)
    res['S1_with_inning_secondary'], _ = fe_fit(ok, np.hstack([Kd, L, subf, extra, G, inn_d]), names1, max(40, reps // 3))
    res['S1_linear_secondary'], _ = fe_fit(ok, np.hstack([K, Fm, L, subf, extra, G]), ['per_earlier_pa', 'per_earlier_meeting', 'per_100_pitches_thrown', 'sub_first', 'extra_pitches_per10'], max(40, reps // 3))
    # identification: how many plate appearances against relievers come early in the hitter's game
    res['reliever_meetings_by_k'] = [int((ok & (rel == 1) & (F == 0) & ~sub & ((k == v) if v < 4 else (k >= 4))).sum()) for v in range(5)]
    stage('S1 joint')
    # S2: relievers' first meetings
    r2_ = ok & (rel == 1) & (F == 0)
    K4 = np.column_stack([(k == 1), (k == 2), (k == 3), (k >= 4)]).astype(float)
    res['S2_relievers_first_meeting'], _ = fe_fit(r2_, np.hstack([K4, L, subf, extra, G]), ['k1', 'k2', 'k3', 'k4plus', 'per_100_pitches_thrown', 'sub_first', 'extra_pitches_per10'], reps)
    res['S2_linear_secondary'], _ = fe_fit(r2_, np.hstack([K, L, subf, extra, G]), ['per_earlier_pa', 'per_100_pitches_thrown', 'sub_first', 'extra_pitches_per10'], max(40, reps // 3))
    res['S2_relievers_first_meeting']['plate_appearances'] = int(r2_.sum())
    res['S2_raw_run_value_by_k'] = [round(float(rv[r2_ & ((k == v) if v < 4 else (k >= 4))].mean()), 4) for v in range(5)]
    stage('S2 relievers')
    # how the starter's times-through rise splits: the simulator-style terms alone, then the decomposition
    tto_p = np.minimum(bf // 9, 3)
    T0 = np.column_stack([(tto_p == 1) & (rel == 0), (tto_p == 2) & (rel == 0), (tto_p >= 3) & (rel == 0), rel]).astype(float)
    res['S1_simulator_terms_only'], _ = fe_fit(ok, np.hstack([T0[:, :3], G, inn_d]), ['starter_tto2', 'starter_tto3', 'starter_tto4plus'], max(40, reps // 3))
    stage('simulator terms')
    # S4: out-of-sample log loss, the decomposition against the simulator's structure
    X0m = np.hstack([T0, C]); X1m = np.hstack([T0, K, Fm, L, subf, extra, C])
    tr = ok & np.isin(P['season'], (2023, 2024))
    m0 = LogisticRegression(max_iter=600, C=10.0).fit(X0m[tr], y7[tr]); m1 = LogisticRegression(max_iter=600, C=10.0).fit(X1m[tr], y7[tr])
    res['S4_logloss_gain_nats_per_1000'] = {}
    for tname, tm in (('2025', ok & (P['season'] == 2025)), ('2026_through_july', ok & (P['season'] == 2026))):
        if tm.sum() < 1000:
            continue
        l0 = -np.log(np.clip(m0.predict_proba(X0m[tm])[np.arange(tm.sum()), y7[tm]], 1e-9, 1)); l1 = -np.log(np.clip(m1.predict_proba(X1m[tm])[np.arange(tm.sum()), y7[tm]], 1e-9, 1))
        ci = clustered_ci(l0 - l1, games[tm])
        res['S4_logloss_gain_nats_per_1000'][tname] = {'plate_appearances': int(tm.sum()), 'gain': [round(v * 1000, 3) for v in ci]}
        # where it acts: relievers' plate appearances
        tr_ = tm & (rel == 1)
        ci_r = clustered_ci(l0[rel[tm] == 1] - l1[rel[tm] == 1], games[tr_])
        res['S4_logloss_gain_nats_per_1000'][tname]['relievers_only'] = [round(v * 1000, 3) for v in ci_r]
    tp = ok & np.isin(P['season'], (2025, 2026))
    if tp.sum() >= 1000:
        l0 = -np.log(np.clip(m0.predict_proba(X0m[tp])[np.arange(tp.sum()), y7[tp]], 1e-9, 1)); l1 = -np.log(np.clip(m1.predict_proba(X1m[tp])[np.arange(tp.sum()), y7[tp]], 1e-9, 1))
        res['S4_logloss_gain_nats_per_1000']['pooled'] = {'plate_appearances': int(tp.sum()), 'gain': [round(v * 1000, 3) for v in clustered_ci(l0 - l1, games[tp])]}
    stage('S4 out of sample')
    return res


# ---------------------------------------------------------------- CHANGE-01: timing carried across a pitching change
def _fe_lgo(rows, X, y, bkey, pkey, games, names, reps, rng, iters=40):
    """Least squares of y on X with batter-season and pitcher-season effects estimated from the player's other games
    (leave-game-out, WARMUP-01's design), intervals from resampling games with the effects held."""
    ub, ib = np.unique(bkey[rows], return_inverse=True); up, ip = np.unique(pkey[rows], return_inverse=True)
    cb = np.bincount(ib).astype(float); cp = np.bincount(ip).astype(float)
    Xr = X[rows].astype(np.float64); yr = y[rows].astype(np.float64)
    Z = np.column_stack([Xr, yr])
    for _ in range(iters):
        for ids, cnt in ((ib, cb), (ip, cp)):
            for j in range(Z.shape[1]):
                Z[:, j] -= (np.bincount(ids, weights=Z[:, j], minlength=len(cnt)) / cnt)[ids]
    beta = np.linalg.lstsq(Z[:, :-1], Z[:, -1], rcond=None)[0]
    gr = games[rows]
    gb_key = ib.astype(np.int64) * 10_000_000 + (gr % 10_000_000); gp_key = ip.astype(np.int64) * 10_000_000 + (gr % 10_000_000)
    ugb, igb = np.unique(gb_key, return_inverse=True); ugp, igp = np.unique(gp_key, return_inverse=True)
    cgb = np.bincount(igb).astype(float); cgp = np.bincount(igp).astype(float)
    keep = (cb[ib] - cgb[igb] >= 1) & (cp[ip] - cgp[igp] >= 1)
    for _ in range(2):
        r = yr - Xr @ beta; a_ = np.zeros(len(cb)); b_ = np.zeros(len(cp))
        for _ in range(30):
            a_ = np.bincount(ib, weights=r - b_[ip], minlength=len(cb)) / cb
            b_ = np.bincount(ip, weights=r - a_[ib], minlength=len(cp)) / cp
        ra = r - b_[ip]; rb_ = r - a_[ib]
        sa = np.bincount(ib, weights=ra, minlength=len(cb)); sag = np.bincount(igb, weights=ra, minlength=len(cgb))
        sp = np.bincount(ip, weights=rb_, minlength=len(cp)); spg = np.bincount(igp, weights=rb_, minlength=len(cgp))
        with np.errstate(invalid='ignore', divide='ignore'):
            a_lgo = (sa[ib] - sag[igb]) / (cb[ib] - cgb[igb]); b_lgo = (sp[ip] - spg[igp]) / (cp[ip] - cgp[igp])
        yadj = (yr - a_lgo - b_lgo)[keep]; A = np.column_stack([np.ones(keep.sum()), Xr[keep]])
        beta = np.linalg.lstsq(A, yadj, rcond=None)[0][1:]
    A = np.column_stack([np.ones(keep.sum()), Xr[keep]])
    gk = gr[keep]; ug = np.unique(gk); gi = np.searchsorted(ug, gk); draws = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
        Aw = A * w[:, None]
        draws.append(np.linalg.solve(Aw.T @ A + np.eye(A.shape[1]) * 1e-9, Aw.T @ yadj)[1:])
    draws = np.asarray(draws)
    out = {nm: [round(float(beta[i]), 6), round(float(np.percentile(draws[:, i], 2.5)), 6), round(float(np.percentile(draws[:, i], 97.5)), 6)] for i, nm in enumerate(names)}
    out['rows_used'] = int(keep.sum())
    return out


def change_study(T: dict, H, params: dict, stage) -> dict:
    """CHANGE-01. Does a hitter's timing, calibrated on the pitcher he has been facing, carry into his first plate
    appearances against a reliever? TIMING-01 to 03F: hitters do not fully re-time between one pitcher's pitches and the
    last pitch's speed moves the next contact. If the calibration also carries across a pitching change, a reliever who
    throws harder than the pitcher he replaces meets hitters who are late, one who throws softer meets hitters who are
    early, for his first batters, fading as the hitters watch him.
    Speed gap = the reliever's usual fastball speed (four-seam and sinker on his earlier days of the season, at least 30)
    minus the mean of the last 10 fastballs of the pitcher he replaced (at least 5). Plate appearances in relievers' first
    meetings with each hitter (feed, development seasons). Run value per plate appearance with leave-game-out
    batter-season and pitcher-season effects (WARMUP-01's design): the reliever's own level and the hitter's are absorbed,
    so the gap acts through who pitched before him. Pitch level: whiff per swing on his fastballs to those hitters.
    Secondary: the absolute gap (it also picks up the curvature of the run scale), the previous pitcher's usual speed
    instead of his last fastballs, and the out-of-sample log loss of the gap terms on a plate-appearance model with both
    players' earlier rates. Two designs were dropped after the synthetic checks: the gap to the pitcher who follows
    (selected by how the reliever did: -0.0017 with nothing planted) and a within-entry gap from each hitter's own last
    fastballs (+0.0014 to +0.0037 with nothing planted)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    res = {}
    N_LAST = int(params.get('last_fastballs', 10)); MIN_LAST = int(params.get('min_last', 5)); MIN_NORM = int(params.get('min_norm', 30))
    real = T['group'] != -1
    gh = T['game'].astype(np.int64) * 2 + T['half'].astype(np.int64)
    order = np.lexsort((T['pitch_no'], T['ab'], gh)); T = take(T, order); real = real[order]; gh = gh[order]
    n = len(gh)
    v0 = T['v0'].astype(np.float64)
    fb = real & np.isin(T['group'], (0, 1)) & np.isfinite(v0)
    # pitcher blocks: contiguous runs of one pitcher inside a fielding side's pitch sequence
    newblk = np.r_[True, (gh[1:] != gh[:-1]) | (T['pitcher'][1:] != T['pitcher'][:-1])]
    blk = np.cumsum(newblk) - 1; nblk = int(blk[-1]) + 1
    bstart = np.flatnonzero(newblk)
    side_new = np.r_[True, gh[bstart][1:] != gh[bstart][:-1]]
    bpos = np.arange(nblk) - np.maximum.accumulate(np.where(side_new, np.arange(nblk), 0))   # 0 = the side's first pitcher
    # first and last fastballs of each block
    fi = np.flatnonzero(fb); bb = blk[fi]; vv = v0[fi]
    cnt_b = np.bincount(bb, minlength=nblk)
    fpos_new = np.r_[True, bb[1:] != bb[:-1]]
    pos = np.arange(len(bb)) - np.maximum.accumulate(np.where(fpos_new, np.arange(len(bb)), 0)); rpos = cnt_b[bb] - 1 - pos
    lastm = rpos < N_LAST
    last_n = np.bincount(bb[lastm], minlength=nblk); last_mean = np.bincount(bb[lastm], weights=vv[lastm], minlength=nblk) / np.maximum(last_n, 1)
    # each pitcher's usual fastball speed: his earlier days of the season
    pks = T['pitcher'].astype(np.int64) * 10000 + T['season'].astype(np.int64)
    on, ov = _prior_by_day(pks, T['day'].astype(np.int64), v0, fb)
    norm_n = on[bstart]; norm = np.where(norm_n > 0, ov[bstart] / np.maximum(norm_n, 1), np.nan)
    have_norm = norm_n >= MIN_NORM
    prev_ok = (bpos > 0) & np.r_[False, last_n[:-1] >= MIN_LAST]
    prev_seen = np.r_[np.nan, last_mean[:-1]]
    gap_b = np.where(prev_ok & have_norm, norm - prev_seen, np.nan)
    prev_norm = np.r_[np.nan, norm[:-1]]; prev_norm_ok = np.r_[False, have_norm[:-1]] & (bpos > 0)
    stage('blocks')
    # plate appearances
    pak = T['game'].astype(np.int64) * 1000 + T['ab'].astype(np.int64)
    first = np.r_[True, pak[1:] != pak[:-1]]
    pa_id = np.cumsum(first) - 1
    P = {k: v[first] for k, v in T.items()}
    P['blk'] = blk[first]
    npitch = np.bincount(pa_id, weights=real.astype(float))
    df = pd.DataFrame({'game': P['game'], 'half': P['half'], 'batter': P['batter'], 'pitcher': P['pitcher'], 'np': npitch})
    df['k'] = df.groupby(['game', 'batter']).cumcount()
    df['F'] = df.groupby(['game', 'batter', 'pitcher']).cumcount()
    df['bf'] = df.groupby(['game', 'pitcher']).cumcount()
    df['team_pa'] = df.groupby(['game', 'half']).cumcount()
    df['sub'] = df.groupby(['game', 'batter'])['team_pa'].transform('min') >= 9
    df['load'] = df.groupby(['game', 'pitcher'])['np'].cumsum() - df['np']
    k = df['k'].to_numpy(); F = df['F'].to_numpy(); bf = df['bf'].to_numpy(); sub = df['sub'].to_numpy()
    load = df['load'].to_numpy(float)
    b_of = P['blk']
    rel = bpos[b_of] > 0
    gap = gap_b[b_of]; nrm = norm[b_of]
    # base-out state and score from the plate-appearance history
    Hk = (H['game_pk'].astype(np.int64) * 1000 + H['at_bat_number'].astype(np.int64) - 1).to_numpy()
    Hs = H.set_index(pd.Index(Hk)); Hs = Hs[~Hs.index.duplicated()]
    st = Hs.reindex(P['game'].astype(np.int64) * 1000 + P['ab'].astype(np.int64))
    outs = st['outs_when_up'].to_numpy(float); r1 = st['on_1b'].notna().to_numpy(float); r2 = st['on_2b'].notna().to_numpy(float); r3 = st['on_3b'].notna().to_numpy(float)
    sd_ = st['bat_score_diff'].to_numpy(float)
    have_state = np.isfinite(outs) & np.isfinite(sd_)
    ok = (P['out7'] >= 0) & have_state & (P['bunt_pa'] == 0)
    y7 = P['out7'].astype(int); rv = LW7[np.clip(y7, 0, 6)]
    inn = np.clip(P['inning'], 1, 10)
    G = np.column_stack([(P['stand_r'] == P['throw_r']).astype(float), (P['half'] == 1).astype(float)]
                        + [np.nan_to_num(outs) == o for o in (1, 2)] + [r1, r2, r3, r1 * r2, r2 * r3, r1 * r3]
                        + [np.clip(np.nan_to_num(sd_), -4, 4) == v for v in (-4, -3, -2, -1, 1, 2, 3, 4)]
                        + [(inn == i) for i in range(2, 11)]).astype(float)
    early = (bf <= 2).astype(float); later = ((bf >= 3) & (bf <= 5)).astype(float); late = (bf >= 6).astype(float)
    Bd = np.column_stack([(bf == 1), (bf == 2), (bf >= 3) & (bf <= 5), (bf >= 6), (k == 1), (k == 2), (k >= 3), sub & (k == 0),
                          np.minimum(load, 60.0) / 100.0]).astype(float)
    bsk = P['batter'].astype(np.int64) * 10000 + P['season']; psk = P['pitcher'].astype(np.int64) * 10000 + P['season']
    games = P['game']; rng = np.random.default_rng(int(params.get('seed', 11))); reps = int(params.get('reps', 150))
    cap = float(params.get('gap_cap', 10.0))
    g_ = np.clip(np.nan_to_num(gap), -cap, cap)
    base = ok & rel & (F == 0) & np.isfinite(gap)
    res['rows'] = {'plate_appearances': int(ok.sum()), 'reliever_first_meetings_with_gap': int(base.sum()),
                   'reliever_entries_with_gap': int(len(np.unique(b_of[base]))),
                   'first_three_batters': int((base & (bf <= 2)).sum()),
                   'gap_percentiles_mph': [round(float(v), 2) for v in np.percentile(gap[base], [5, 25, 50, 75, 95])],
                   'gap_sd_mph': round(float(np.std(gap[base])), 3),
                   'share_abs_gap_4plus': round(float((np.abs(gap[base]) >= 4).mean()), 4),
                   'corr_gap_inning': round(float(np.corrcoef(gap[base], inn[base])[0, 1]), 4),
                   'corr_gap_score_diff': round(float(np.corrcoef(gap[base], np.nan_to_num(sd_[base]))[0, 1]), 4),
                   'corr_gap_reliever_norm': round(float(np.corrcoef(gap[base], nrm[base])[0, 1]), 4),
                   'mean_run_value': round(float(rv[base].mean()), 5)}
    stage('plate appearances')
    # (1) run value: the gap for his first three batters and for later ones
    names = ['gap_first3', 'gap_batters4to6', 'gap_batters7plus']
    X = np.column_stack([g_ * early, g_ * later, g_ * late, Bd, G])
    res['R1_run_value_per_mph'] = _fe_lgo(base, X, rv, bsk, psk, games, names, reps, rng)
    # finer: each of the first three batters
    Xf = np.column_stack([g_ * (bf == 0), g_ * (bf == 1), g_ * (bf == 2), g_ * (bf >= 3), Bd, G])
    res['R1_by_batter'] = _fe_lgo(base, Xf, rv, bsk, psk, games, ['gap_batter1', 'gap_batter2', 'gap_batter3', 'gap_batter4plus'], max(40, reps // 3), rng)
    # absolute gap (a change of pace either way)
    Xa = np.column_stack([g_ * early, np.abs(g_) * early, g_ * (1 - early), np.abs(g_) * (1 - early), Bd, G])
    res['R1_signed_and_absolute'] = _fe_lgo(base, Xa, rv, bsk, psk, games, ['gap_first3', 'abs_gap_first3', 'gap_later', 'abs_gap_later'], max(40, reps // 3), rng)
    # strikeouts and walks (linear probability)
    for nm, yy in (('strikeout', (y7 == 1).astype(float)), ('walk', (y7 == 2).astype(float))):
        res['R1_' + nm + '_per_mph'] = _fe_lgo(base, X, yy, bsk, psk, games, names, max(40, reps // 3), rng)
    # the previous pitcher's usual speed instead of what was seen (what hitters were calibrated on, net of his drift today)
    gn = np.where(prev_norm_ok[b_of] & have_norm[b_of], nrm - prev_norm[b_of], np.nan)
    bn = base & np.isfinite(gn)
    res['R1_usual_speeds_gap'] = _fe_lgo(bn, np.column_stack([np.clip(np.nan_to_num(gn), -cap, cap) * early, np.clip(np.nan_to_num(gn), -cap, cap) * (1 - early), Bd, G]),
                                         rv, bsk, psk, games, ['gap_first3', 'gap_later'], max(40, reps // 3), rng)
    stage('run value')
    # (3) pitch level: whiff per swing on his fastballs in the hitter's first plate appearance against him
    on_pitch = base[pa_id]
    sw = on_pitch & fb & np.isin(T['call'], (1, 2))
    yv = (T['call'] == 2).astype(float)
    gp_ = g_[pa_id]; early_p = early[pa_id]
    cnt_d = np.column_stack([(T['balls'] == b_) & (T['strikes'] == s_) for b_ in range(4) for s_ in range(3)][1:]).astype(float)
    zin = ((T['zone'] >= 1) & (T['zone'] <= 9)).astype(float)
    effort = np.clip(np.nan_to_num(v0 - nrm[pa_id]), -6, 6)
    first2 = (T['pitch_no'] <= 1).astype(float)
    Xp = np.column_stack([gp_ * early_p * first2, gp_ * early_p * (1 - first2), gp_ * (1 - early_p), effort, zin, cnt_d,
                          (T['stand_r'] == T['throw_r']).astype(float)])
    bskp = T['batter'].astype(np.int64) * 10000 + T['season']; pskp = T['pitcher'].astype(np.int64) * 10000 + T['season']
    res['R3_whiff_per_swing_per_mph'] = _fe_lgo(sw, Xp, yv, bskp, pskp, T['game'], ['gap_first3_pitches1to2', 'gap_first3_later_pitches', 'gap_later_batters', 'own_speed_vs_usual'],
                                                max(40, reps // 3), rng)
    res['R3_whiff_per_swing_per_mph']['swings'] = int(sw.sum()); res['R3_whiff_per_swing_per_mph']['mean_whiff'] = round(float(yv[sw].mean()), 4)
    stage('pitch level')
    # (4) out of sample: the gap terms on a plate-appearance model with both players' earlier rates
    league = np.bincount(y7[ok], minlength=7) / ok.sum()

    def rates(key_, kk):
        cols_ = []
        for c in range(7):
            nn, ss = _prior_by_day(key_, P['day'].astype(np.int64), (y7 == c).astype(float), ok)
            cols_.append(np.log((ss + kk * league[c]) / (nn + kk) / league[c]))
        return np.column_stack(cols_)
    rb = rates(P['batter'].astype(np.int64), 150.0); rp = rates(P['pitcher'].astype(np.int64), 300.0)
    X0 = np.hstack([rb, rp, Bd, G]); X1 = np.hstack([X0, np.column_stack([g_ * early, g_ * (1 - early)])])
    tr = base & np.isin(P['season'], (2023, 2024))
    res['R4_logloss_gain_nats_per_1000'] = {}
    if tr.sum() > 5000:
        m0 = LogisticRegression(max_iter=600, C=10.0).fit(X0[tr], y7[tr]); m1 = LogisticRegression(max_iter=600, C=10.0).fit(X1[tr], y7[tr])
        for tname, tm in (('2025', base & (P['season'] == 2025)), ('2026_development', base & (P['season'] == 2026))):
            if tm.sum() < 1000:
                continue
            l0 = -np.log(np.clip(m0.predict_proba(X0[tm])[np.arange(tm.sum()), y7[tm]], 1e-9, 1))
            l1 = -np.log(np.clip(m1.predict_proba(X1[tm])[np.arange(tm.sum()), y7[tm]], 1e-9, 1))
            res['R4_logloss_gain_nats_per_1000'][tname] = {'plate_appearances': int(tm.sum()), 'gain': [round(v * 1000, 3) for v in clustered_ci(l0 - l1, games[tm])]}
    stage('out of sample')
    return res


# ---------------------------------------------------------------- SWINGMAP-01: do hitters chase where their bat goes?
def _bat_intrinsic(S: dict, seasons, min_swings: int = 150) -> tuple[dict, dict]:
    """Per hitter-season, each bat-tracking measure net of the pitch it met: every competitive swing's value (bat speed at
    least 50 mph) regressed, within a season, on the pitch's height (and its square), its side toward or away from the
    hitter (and its square), contact depth (for the angles), speed, two strikes and pitch group; the hitter's mean
    residual is his intrinsic value. Attack direction is oriented toward the pull side by the data (the sign, by batter
    side, that makes it agree with where his balls in play go). Returns {(batter, season): {measure: value, 'n': swings}}
    and the orientation check."""
    desc = np.asarray(S['description__vocab'])[S['description']] if 'description__vocab' in S else np.asarray(S['description'])
    stand = np.asarray(S['stand__vocab'])[S['stand']] if 'stand__vocab' in S else np.asarray(S['stand'])
    ptype = np.asarray(S['pitch_type__vocab'])[S['pitch_type']] if 'pitch_type__vocab' in S else np.asarray(S['pitch_type'])
    swing = np.isin(desc, ('swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'))
    day = S['day'].astype(np.int64); year = np.asarray([date.fromordinal(int(d_)).year for d_ in day])
    stand_r = stand == 'R'
    px = S['plate_x'].astype(np.float64); pz = S['plate_z'].astype(np.float64)
    u = np.where(stand_r, px, -px)
    bs = S['bat_speed'].astype(np.float64)
    ok = swing & np.isfinite(bs) & (bs >= 50) & np.isfinite(px) & np.isfinite(pz) & np.isin(year, seasons)
    grpv = np.asarray([TYPE_GROUPS.get(str(t_ or '').upper(), 6) for t_ in ptype])
    G = np.column_stack([(grpv == g_) for g_ in (1, 2, 3, 4, 5, 6)]).astype(float)
    depth = S['intercept_ball_minus_batter_pos_y_inches'].astype(np.float64)
    speed = S['release_speed'].astype(np.float64); two = (S['strikes'] == 2).astype(float)
    # orientation of attack direction: its correlation with pull-side spray on balls in play, by batter side
    check = {}
    sign = {}
    hx, hy = S['hc_x'].astype(np.float64), S['hc_y'].astype(np.float64)
    spray = np.degrees(np.arctan2(hx - 125.42, 198.27 - hy))           # positive toward right field
    pull = np.where(stand_r, -spray, spray)
    ad = S['attack_direction'].astype(np.float64)
    for side, m_side in (('R', stand_r), ('L', ~stand_r)):
        m = ok & m_side & np.isfinite(ad) & np.isfinite(pull) & (desc == 'hit_into_play')
        c = float(np.corrcoef(ad[m], pull[m])[0, 1]) if m.sum() > 100 else 0.0
        sign[side] = 1.0 if c >= 0 else -1.0
        check[side] = {'corr_raw_with_pull_spray': round(c, 3), 'balls_in_play': int(m.sum())}
    adir = ad * np.where(stand_r, sign['R'], sign['L'])
    measures = {'attack_angle': S['attack_angle'].astype(np.float64), 'swing_path_tilt': S['swing_path_tilt'].astype(np.float64),
                'attack_direction_pull': adir, 'contact_depth_in': depth, 'bat_speed': bs, 'swing_length': S['swing_length'].astype(np.float64)}
    out = {}
    bat = S['batter'].astype(np.int64)
    for ssn in seasons:
        ms = ok & (year == ssn)
        base = np.column_stack([np.ones(len(u)), pz, pz ** 2, u, u ** 2, np.nan_to_num(speed, nan=float(np.nanmean(speed))), two, G])
        for name, v in measures.items():
            use_depth = name in ('attack_angle', 'swing_path_tilt', 'attack_direction_pull')
            X = np.column_stack([base, np.nan_to_num(depth)]) if use_depth else base
            m = ms & np.isfinite(v) & (np.isfinite(depth) if use_depth else True)
            if m.sum() < 1000:
                continue
            beta = np.linalg.lstsq(X[m], v[m], rcond=None)[0]
            r = v[m] - X[m] @ beta
            ub, ib = np.unique(bat[m], return_inverse=True); cnt = np.bincount(ib); mean_r = np.bincount(ib, weights=r) / cnt
            for b_, n_, mr in zip(ub, cnt, mean_r):
                if n_ >= min_swings:
                    d_ = out.setdefault((int(b_), int(ssn)), {}); d_[name] = float(mr); d_['n'] = max(d_.get('n', 0), int(n_))
    return out, check


def _batter_positions(S: dict, seasons, min_swings: int = 150, params_grid_lo: float = -1.5) -> tuple[dict, dict]:
    """STANCE-01: where each hitter stands, from bat tracking. Savant's intercept fields give the ball's position minus
    the batter's (his center of mass) when the bat meets the ball, sideways (ix) and toward the pitcher (iy). The ball's
    position along its fitted flight is known at any distance from the plate, so for one hitter-season his position
    (x_b, y_b) is the point that makes ball_x(y_b + iy) - ix agree across all his swings: a grid over y_b with x_b the
    mean, identified by how the flight's sideways slope varies across his pitches. Distance off the plate is measured
    from the plate's inside edge (0.708 ft from its center) to x_b, depth from the front of the plate back to y_b (both
    in inches, at the moment of contact, so after the stride). The sign of the sideways intercept for each batter side is
    set by the data (right-handed hitters stand on the third-base side, negative x from the catcher's view). Returns
    {(batter, season): {'off_plate_in', 'depth_in', 'n_stance'}} and checks: the conventions, positions by side, the
    share of fits at the grid's edge and the split-half reliability (odd and even games)."""
    sv = savant_module()
    desc = np.asarray(S['description__vocab'])[S['description']] if 'description__vocab' in S else np.asarray(S['description'])
    stand = np.asarray(S['stand__vocab'])[S['stand']] if 'stand__vocab' in S else np.asarray(S['stand'])
    swing = np.isin(desc, ('swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'))
    ix = S['intercept_ball_minus_batter_pos_x_inches'].astype(np.float64) / 12.0
    iy = S['intercept_ball_minus_batter_pos_y_inches'].astype(np.float64) / 12.0
    bs = S['bat_speed'].astype(np.float64)
    yr = np.asarray([date.fromordinal(int(d_)).year for d_ in S['day']])
    flight = ('vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'plate_x', 'plate_z')
    ok = swing & np.isfinite(ix) & np.isfinite(iy) & np.isfinite(bs) & (bs >= 50)
    for k in flight:
        ok &= np.isfinite(S[k].astype(np.float64))
    stand_r = stand == 'R'
    bat = S['batter'].astype(np.int64); gpk = S['game_pk'].astype(np.int64)
    grid = np.round(np.arange(float(params_grid_lo), 4.0001, 0.05), 3)
    out, check = {}, {'grid_ft': [float(grid[0]), float(grid[-1])], 'by_season': {}}
    for ssn in seasons:
        m = ok & (yr == ssn)
        if m.sum() < 2000:
            continue
        c = {k: S[k][m].astype(np.float64) for k in flight}
        sv.anchor(c, int(ssn))
        ixm, iym, sr, bm, half = ix[m], iy[m], stand_r[m], bat[m], (gpk[m] % 2)
        # the sideways convention: which sign puts right-handed hitters at x < 0 and left-handed at x > 0
        x_c0 = sv.at(c, 0.7 + iym)[0]
        sign = {}
        conv = {}
        for side, ms in (('R', sr), ('L', ~sr)):
            med_minus = float(np.median((x_c0 - ixm)[ms])); med_plus = float(np.median((x_c0 + ixm)[ms]))
            want = -1.0 if side == 'R' else 1.0
            sign[side] = 1.0 if med_minus * want > med_plus * want else -1.0          # x_b = x_c - sign * ix
            conv[side] = {'median_x_ball_minus_ix_ft': round(med_minus, 3), 'median_x_ball_plus_ix_ft': round(med_plus, 3), 'sign_used': sign[side]}
        sgn = np.where(sr, sign['R'], sign['L'])
        # groups: hitter and side (a switch hitter's two sides are fitted apart; his majority side is kept)
        key = bm * 2 + sr.astype(np.int64)
        res_by = {}
        for tag, gkey in (('all', key), ('half', key * 2 + half)):
            ug, ig = np.unique(gkey, return_inverse=True); ng = np.bincount(ig).astype(float)
            sse = np.empty((len(grid), len(ug))); mean_ = np.empty((len(grid), len(ug)))
            for j, yb in enumerate(grid):
                r = sv.at(c, yb + iym)[0] - sgn * ixm
                s1 = np.bincount(ig, weights=r, minlength=len(ug)); s2 = np.bincount(ig, weights=r * r, minlength=len(ug))
                mean_[j] = s1 / ng; sse[j] = s2 - s1 * s1 / ng
            jb = np.argmin(sse, axis=0)
            res_by[tag] = (ug, ng, grid[jb], mean_[jb, np.arange(len(ug))], (jb == 0) | (jb == len(grid) - 1))
        ug, ng, yb, xb, edge = res_by['all']
        off = (np.abs(xb) - 0.708) * 12.0; dep = (sv.FRONT - yb) * 12.0
        b_of = ug // 2; side_of = ug % 2
        keep = (ng >= min_swings) & ~edge
        best = {}
        for b_, n_, o_, d_, s_ in zip(b_of[keep], ng[keep], off[keep], dep[keep], side_of[keep]):
            if int(b_) not in best or n_ > best[int(b_)][0]:
                best[int(b_)] = (n_, o_, d_, s_)
        for b_, (n_, o_, d_, s_) in best.items():
            out[(b_, int(ssn))] = {'off_plate_in': float(o_), 'depth_in': float(d_), 'n_stance': int(n_)}
        # split-half reliability: the same fit on odd and even games
        ugh, ngh, ybh, xbh, edgeh = res_by['half']
        offh = (np.abs(xbh) - 0.708) * 12.0; deph = (sv.FRONT - ybh) * 12.0
        hk = {int(k_): (o_, d_) for k_, n_, o_, d_, e_ in zip(ugh, ngh, offh, deph, edgeh) if n_ >= min_swings / 2 and not e_}
        pairs_h = [(hk[k_ * 2], hk[k_ * 2 + 1]) for k_ in {k_ // 2 for k_ in hk} if k_ * 2 in hk and k_ * 2 + 1 in hk]
        rel = {}
        for i_, nm in ((0, 'off_plate'), (1, 'depth')):
            a_ = np.array([p_[0][i_] for p_ in pairs_h]); b2 = np.array([p_[1][i_] for p_ in pairs_h])
            rel[nm] = round(float(np.corrcoef(a_, b2)[0, 1]), 3) if len(a_) > 20 else None
        vals = np.array([[v['off_plate_in'], v['depth_in']] for k_, v in out.items() if k_[1] == int(ssn)])
        check['by_season'][str(ssn)] = {'swings': int(m.sum()), 'convention': conv, 'hitters': int(len(best)),
                                        'edge_share': round(float(edge[ng >= min_swings].mean()), 4) if (ng >= min_swings).any() else None,
                                        'off_plate_in_percentiles': [round(float(v), 2) for v in np.percentile(vals[:, 0], [5, 25, 50, 75, 95])] if len(vals) else None,
                                        'depth_in_percentiles': [round(float(v), 2) for v in np.percentile(vals[:, 1], [5, 25, 50, 75, 95])] if len(vals) else None,
                                        'split_half_r': rel, 'split_half_hitters': len(pairs_h)}
    return out, check


def swingmap_study(T: dict, S: dict, params: dict, stage) -> dict:
    """SWINGMAP-01. Each hitter's own chase spots at the decision moment (his swing map minus the same-side mean map of
    the other hitters, MATCHUP-01's representation, VALUE-08's own part) against his swing path from bat tracking, net
    of the pitches each swing met. Map features per hitter-season (2025 and 2026 through July, maps fitted per season
    with shrinkage 10, at least 500 pitches), read for a first-pitch four-seamer: high minus low (mean own part above
    the zone, 3.75 and 4.0 ft, minus below, 1.0 and 1.25 ft, across the plate), away minus inside (outside the plate
    at mid heights) and the overall outside level, in points of swing chance. Correlations with intrinsic attack angle,
    swing path tilt, pull-side attack direction, contact depth, bat speed and swing length by season (bootstrap over
    hitters), and the same for each hitter's change from 2025 to 2026."""
    res = {}
    seasons = tuple(int(s) for s in params.get('map_seasons', (2025, 2026)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025, 2026)))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    Lb = location_block(xp, zp, T['stand_r'], T['strikes']); prop = swing_propensity(T); pprop = pitcher_propensity(T)
    X = np.hstack([Lb, control_block(T, prop), pprop[:, None].astype(np.float32)]); i_prop = Lb.shape[1] + 32
    rng = np.random.default_rng(13)
    idx = rng.choice(len(swing), min(len(swing), 700000), replace=False)
    league = fit_logistic(X[idx], swing[idx]); off = league.decision_function(X)
    Bm = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    stage('league model')
    # the reference read: first-pitch four-seamer to a hitter of each side, league-average swing level
    def ref_points(us, zs):
        UU, ZZ = np.meshgrid(np.asarray(us, float), np.asarray(zs, float)); return UU.ravel(), ZZ.ravel()
    bands = {'high': ref_points((-0.6, 0.0, 0.6), (3.75, 4.0)), 'low': ref_points((-0.6, 0.0, 0.6), (1.0, 1.25)),
             'away': ref_points((1.05, 1.25), (2.0, 2.5, 3.0)), 'inside': ref_points((-1.05, -1.25), (2.0, 2.5, 3.0))}
    lg_prop = float(np.mean(prop)); lg_pp = float(np.mean(pprop))
    ref = {}
    for sd in (0, 1):
        for name, (uu, zz) in bands.items():
            n_ = len(uu); px_ = uu if sd == 1 else -uu
            g_t = {'balls': np.zeros(n_, np.int64), 'strikes': np.zeros(n_, np.int64), 'group': np.zeros(n_, np.int64), 'v0': np.full(n_, 94.0),
                   'stand_r': np.full(n_, sd, np.int64), 'throw_r': np.ones(n_, np.int64)}
            Xg = np.hstack([location_block(px_, zz, g_t['stand_r'], g_t['strikes']), control_block(g_t, np.full(n_, lg_prop)), np.full((n_, 1), lg_pp, np.float32)])
            ref[(sd, name)] = (league.decision_function(Xg).astype(np.float64), hitter_basis(px_, zz, g_t['stand_r'], g_t['strikes']))
    feats = {}
    for ssn in seasons:
        rows_s = T['season'] == ssn
        maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], rows_s), 10.0, int(params.get('min_pitches', 500)))
        side = {h: int(np.round(T['stand_r'][r].mean())) for h, r in _groups(T['batter'], rows_s).items() if h in maps}
        tot = {0: 0.0, 1: 0.0}; nn = {0: 0, 1: 0}
        for h, m in maps.items():
            tot[side[h]] = tot[side[h]] + m; nn[side[h]] += 1
        for h, m in maps.items():
            sd = side[h]; mbar = (tot[sd] - m) / max(nn[sd] - 1, 1)
            own = {}
            for name in bands:
                lo, Bg = ref[(sd, name)]
                own[name] = float(np.mean(1 / (1 + np.exp(-(lo + Bg @ m))) - 1 / (1 + np.exp(-(lo + Bg @ mbar))))) * 100
            feats[(int(h), int(ssn))] = {'high_minus_low': own['high'] - own['low'], 'away_minus_inside': own['away'] - own['inside'],
                                         'outside_level': float(np.mean([own[k] for k in bands])), 'side': sd}
        stage(f'maps {ssn}: {len(maps)}')
    bat, check = _bat_intrinsic(S, seasons, int(params.get('min_swings', 150)))
    res['attack_direction_orientation'] = check
    if params.get('stance'):
        # STANCE-01: each hitter-season's distance off the plate and depth at contact, from the intercept fields
        pos, pos_check = _batter_positions(S, seasons, int(params.get('min_swings', 150)))
        res['stance_check'] = pos_check
        for k_, v_ in pos.items():
            if k_ in bat:
                bat[k_].update(v_)
        stage('stance')
    if params.get('zones'):
        # ZONEMAP-01: each hitter-season's strike zone from Savant (median top and bottom over the pitches he saw)
        yr = np.asarray([date.fromordinal(int(d_)).year for d_ in S['day']])
        top_, bot_ = S['sz_top'].astype(np.float64), S['sz_bot'].astype(np.float64); bat_ = S['batter'].astype(np.int64)
        for ssn in seasons:
            mz = (yr == ssn) & np.isfinite(top_) & np.isfinite(bot_)
            for b_, rr in _groups(bat_, mz).items():
                if len(rr) >= 300 and (int(b_), int(ssn)) in bat:
                    bat[(int(b_), int(ssn))]['zone_mid'] = float(np.median((top_[rr] + bot_[rr]) / 2.0))
                    bat[(int(b_), int(ssn))]['zone_height'] = float(np.median(top_[rr] - bot_[rr]))
                    bat[(int(b_), int(ssn))]['zone_top'] = float(np.median(top_[rr]))
                    bat[(int(b_), int(ssn))]['zone_bottom'] = float(np.median(bot_[rr]))
        stage('zones')
    stage(f'bat measures {len(bat)}')
    keys = sorted(set(feats) & set(bat))
    res['hitter_seasons'] = {str(s): int(sum(1 for k in keys if k[1] == s)) for s in seasons}
    pairs_ = [('attack_angle', 'high_minus_low'), ('swing_path_tilt', 'high_minus_low'), ('attack_direction_pull', 'away_minus_inside'),
              ('contact_depth_in', 'away_minus_inside'), ('bat_speed', 'outside_level'), ('swing_length', 'outside_level'),
              ('attack_angle', 'outside_level'), ('contact_depth_in', 'high_minus_low')]
    if params.get('zones'):
        pairs_ = [('zone_mid', 'high_minus_low'), ('zone_top', 'high_minus_low'), ('zone_bottom', 'high_minus_low'), ('zone_height', 'high_minus_low'),
                  ('zone_height', 'outside_level'), ('zone_mid', 'away_minus_inside')] + pairs_
    if params.get('stance'):
        pairs_ = [('off_plate_in', 'away_minus_inside'), ('depth_in', 'outside_level'), ('off_plate_in', 'outside_level'), ('depth_in', 'away_minus_inside'),
                  ('off_plate_in', 'high_minus_low'), ('depth_in', 'high_minus_low')] + pairs_

    def corr_ci(x, y, reps=2000):
        x, y = np.asarray(x, float), np.asarray(y, float); ok_ = np.isfinite(x) & np.isfinite(y); x, y = x[ok_], y[ok_]
        if len(x) < 30:
            return None
        r0 = float(np.corrcoef(x, y)[0, 1]); bs_ = []
        for _ in range(reps):
            j = rng.integers(0, len(x), len(x)); bs_.append(np.corrcoef(x[j], y[j])[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        return {'n': int(len(x)), 'r': [round(r0, 3), round(float(np.percentile(bs_, 2.5)), 3), round(float(np.percentile(bs_, 97.5)), 3)], 'slope_points_per_unit': round(slope, 4)}
    res['by_season'] = {}
    for s in seasons:
        ks = [k for k in keys if k[1] == s]
        res['by_season'][str(s)] = {f'{a}__{b}': corr_ci([bat[k].get(a, np.nan) for k in ks], [feats[k][b] for k in ks]) for a, b in pairs_}
        # within each batter side, the headline pair
        for sd, nm in ((1, 'R'), (0, 'L')):
            ks2 = [k for k in ks if feats[k]['side'] == sd]
            res['by_season'][str(s)][f'attack_angle__high_minus_low__{nm}'] = corr_ci([bat[k].get('attack_angle', np.nan) for k in ks2], [feats[k]['high_minus_low'] for k in ks2])
            if params.get('stance'):
                res['by_season'][str(s)][f'off_plate_in__away_minus_inside__{nm}'] = corr_ci([bat[k].get('off_plate_in', np.nan) for k in ks2], [feats[k]['away_minus_inside'] for k in ks2])
    # change from the first to the second season, within hitter
    if len(seasons) == 2:
        s0, s1 = seasons
        hs = sorted({k[0] for k in keys if (k[0], s0) in bat and (k[0], s1) in bat and (k[0], s0) in feats and (k[0], s1) in feats})
        res['change'] = {'hitters': len(hs)}
        for a, b in pairs_[:4]:
            dx = [bat[(h, s1)].get(a, np.nan) - bat[(h, s0)].get(a, np.nan) for h in hs]; dy = [feats[(h, s1)][b] - feats[(h, s0)][b] for h in hs]
            res['change'][f'{a}__{b}'] = corr_ci(dx, dy)
        # how much each map feature and bat measure repeat between the seasons (their reliability bounds any correlation)
        res['repeat'] = {}
        for b in ('high_minus_low', 'away_minus_inside', 'outside_level'):
            res['repeat'][b] = corr_ci([feats[(h, s0)][b] for h in hs], [feats[(h, s1)][b] for h in hs], 500)
        for a in ('attack_angle', 'swing_path_tilt', 'attack_direction_pull', 'contact_depth_in') + (('off_plate_in', 'depth_in') if params.get('stance') else ()):
            res['repeat'][a] = corr_ci([bat[(h, s0)].get(a, np.nan) for h in hs], [bat[(h, s1)].get(a, np.nan) for h in hs], 500)
    stage('correlations')
    return res


def value_whiff_study(T: dict, params: dict, stage) -> dict:
    """VALUE-08's design for the whiff map on the true crossing (MATCHUP-03's representation: league whiff model on
    swings, hitter maps on location, family and height bands, shrinkage 30, at least 250 swings). Each pitch's whiff
    deviation (his extra chance of missing if he swings, in points) is split into the shared part (the same-side mean
    map of the other hitters) and his own part; the chase deviation's two parts (MATCHUP-05F's swing representation)
    are held in the regression, so the whiff coefficient is net of chasing. Checks: placebo (another same-side hitter's
    whiff map net of the same mean map), within hitter, pitch group and zone side, the plate appearance's other pitches
    held fixed, the same spot. Aiming: among the pitcher's own spots of the same pitch group to that side, count group
    and side of the zone edge, with 0.6 ft of command scatter around where the ball arrives, the best third for this
    hitter by his own whiff part within thirds of the league's whiff chance, against all of them; value from the
    primary coefficient (within hitter, neighbor-held) times the gain and pitches per plate appearance."""
    res = {}
    final = bool(params.get('final_eval'))
    if final and not params.get('frozen_commit'):
        raise ValueError('the final whiff value scoring runs only as the registered evaluation of a frozen commit')
    if final:
        T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | ((T['season'] == 2026) & (T['day'] >= date(2026, 8, 1).toordinal())))
    else:
        post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64); whiff = (T['call'] == 2).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26); del F
    xt, zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    tr = np.isin(T['season'], (2023, 2024, 2025)) if final else np.isin(T['season'], (2023, 2024))
    test_season = 2026 if final else 2025
    rng = np.random.default_rng(11)
    sig = lambda v: 1 / (1 + np.exp(-v))
    def sample(rows, n=600000):
        idx = np.flatnonzero(rows); return rng.choice(idx, min(len(idx), n), replace=False) if len(idx) > n else idx
    ub_, bi_ = np.unique(T['batter'][tr], return_inverse=True)
    side_of = dict(zip(ub_.tolist(), np.round(np.bincount(bi_, weights=T['stand_r'][tr]) / np.bincount(bi_)).astype(int).tolist()))
    def own_maps(Bm, y, off, rows, lam, min_n):
        maps = _hitter_maps(Bm, y, off, _groups(T['batter'], rows), lam, min_n)
        Ss = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
        for h, m in maps.items():
            Ss[side_of[h]] = Ss[side_of[h]] + m; Nn[side_of[h]] += 1
        return maps, {h: (Ss[side_of[h]] - m) / max(Nn[side_of[h]] - 1, 1) for h, m in maps.items()}
    # chases: MATCHUP-05F's representation (held in the regression)
    prop_s = swing_propensity(T)
    Ls = location_block(xp, zp, T['stand_r'], T['strikes']); Bh = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    Xs = np.hstack([Ls, (Bh[:, :-1] * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32), (Bh[:, :-1] * (T['group'] == 5)[:, None]).astype(np.float32),
                    control_block(T, prop_s), pitcher_propensity(T)[:, None].astype(np.float32)]); del Ls
    i = sample(tr); off_s = fit_logistic(Xs[i], swing[i]).decision_function(Xs); del Xs
    Bs = family_basis(Bh, T['group']); del Bh
    maps_s, mbar_s = own_maps(Bs, swing, off_s, tr, 10.0, 300)
    # whiffs: MATCHUP-03's representation on the true crossing
    tw = dict(T); tw['call'] = np.where(whiff == 1, 1, np.where(swing == 1, 0, 3)); prop_w = swing_propensity(tw, 200.0); del tw
    grp7 = np.zeros((len(swing), 7), np.float32); grp7[np.arange(len(swing)), np.clip(T['group'], 0, 6)] = 1
    Lw = location_block(xt, zt, T['stand_r'], T['strikes']); nLw = Lw.shape[1]
    Xw = np.hstack([Lw, grp7, hats(T['v0'].astype(np.float64), V_KNOTS), (T['strikes'] == 2)[:, None].astype(np.float32), prop_w[:, None].astype(np.float32),
                    (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)])
    i = sample(tr & (swing == 1)); m_w = fit_logistic(Xw[i], whiff[i]); off_w = m_w.decision_function(Xw); wLw = m_w.coef_[0][:nLw].astype(np.float64); del Xw, grp7
    famc = np.column_stack([np.isin(T['group'], (0, 1, 2)), np.isin(T['group'], (3, 4)), np.isin(T['group'], (5,))]).astype(np.float64)
    hz4 = (1.0, 2.0, 3.0, 4.0)
    Bw = np.hstack([hitter_basis(xt, zt, T['stand_r'], T['strikes']), famc, hats(zt, hz4).astype(np.float64)])
    maps_w, mbar_w = own_maps(Bw, whiff, off_w, tr & (swing == 1), 30.0, 250)
    stage(f'maps {len(maps_s)} swing, {len(maps_w)} whiff')
    have = np.asarray(sorted(set(maps_s) & set(maps_w)), dtype=np.int64)
    te = (T['season'] == test_season) & (T['out7'] >= 0) & np.isin(T['batter'], have)
    ix = np.flatnonzero(te)
    p_l = sig(off_s[ix]); p_lw = sig(off_w[ix])
    DC = np.zeros(len(ix)); DCm = np.zeros(len(ix)); DW = np.zeros(len(ix)); DWm = np.zeros(len(ix))
    gb = _groups(T['batter'][ix], np.ones(len(ix), bool))
    for h, rr in gb.items():
        a = ix[rr]
        ms_ = sig(off_s[a] + Bs[a] @ mbar_s[h]); mw_ = sig(off_w[a] + Bw[a] @ mbar_w[h])
        DCm[rr] = (ms_ - p_l[rr]) * 100; DC[rr] = (sig(off_s[a] + Bs[a] @ maps_s[h]) - ms_) * 100
        DWm[rr] = (mw_ - p_lw[rr]) * 100; DW[rr] = (sig(off_w[a] + Bw[a] @ maps_w[h]) - mw_) * 100
    u_t = np.where(T['stand_r'][ix] == 1, xt[ix], -xt[ix]); zt_ = zt[ix]
    e = np.maximum(np.maximum(np.abs(u_t) - ZONE_HALF, zt_ - ZONE_TOP), ZONE_BOT - zt_)
    outside = e > 0
    y = LW7[T['out7'][ix].astype(int)]
    first = (T['pitch_no'] == 0) & (T['out7'] >= 0)
    rv_all = LW7[np.clip(T['out7'], 0, 6).astype(int)]
    def prior_rv(key):
        nn, ss = _prior_by_day(np.r_[key[first], key[ix]].astype(np.int64), np.r_[T['day'][first], T['day'][ix]].astype(np.int64),
                               np.r_[rv_all[first], np.zeros(len(ix))], np.r_[np.ones(int(first.sum()), bool), np.zeros(len(ix), bool)])
        nn, ss = nn[int(first.sum()):], ss[int(first.sum()):]
        lg_ = float(rv_all[first].mean()); return (ss + 200 * lg_) / (nn + 200)
    rv_b, rv_p = prior_rv(T['batter']), prior_rv(T['pitcher'])
    cnt = np.zeros((len(ix), 11)); cc = np.clip(T['balls'][ix], 0, 3) * 3 + np.clip(T['strikes'][ix], 0, 2)
    for k in range(1, 12):
        cnt[:, k - 1] = cc == k
    gdum = np.zeros((len(ix), 6)); g_ = np.clip(T['group'][ix], 0, 6)
    for k in range(1, 7):
        gdum[:, k - 1] = g_ == k
    base = np.column_stack([np.ones(len(ix)), cnt, gdum, p_l, np.log(p_l / (1 - p_l)), p_lw, np.log(p_lw / (1 - p_lw)), outside, hats(e, (-0.8, -0.4, -0.15, 0.0, 0.15, 0.4, 0.8, 1.5)),
                            rv_b, rv_p, (T['stand_r'][ix] == T['throw_r'][ix]).astype(float), DCm * outside, DCm * ~outside, DC * outside, DC * ~outside, DWm * outside, DWm * ~outside])
    ug, gi = np.unique(T['game'][ix], return_inverse=True)
    rc = np.random.default_rng(5)
    def fit(Xm, reps=None):
        bb = np.linalg.lstsq(Xm, y, rcond=None)[0]
        pk = Xm.shape[1]; XtX_ = np.zeros((len(ug), pk, pk))
        for a_ in range(pk):
            for c_ in range(a_, pk):
                v_ = np.bincount(gi, weights=Xm[:, a_] * Xm[:, c_], minlength=len(ug)); XtX_[:, a_, c_] = v_; XtX_[:, c_, a_] = v_
        Xty_ = np.column_stack([np.bincount(gi, weights=Xm[:, a_] * y, minlength=len(ug)) for a_ in range(pk)])
        dr = []
        for _ in range(int(reps or params.get('reps', 200))):
            w_ = np.bincount(rc.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
            dr.append(np.linalg.solve(np.tensordot(w_, XtX_, 1) + 1e-9 * np.eye(pk), w_ @ Xty_)[-2:])
        dr = np.asarray(dr)
        return bb, [[round(float(np.percentile(dr[:, j], q)), 6) for q in (2.5, 97.5)] for j in (0, 1)]
    n_pa = max(int((T['pitch_no'][ix] == 0).sum()), 1)
    per_pa = {'outside': float(outside.sum() / n_pa), 'inside': float((~outside).sum() / n_pa)}
    chk = {}
    bb, ci = fit(np.column_stack([base, DW * outside, DW * ~outside]))
    chk['own_part'] = {'outside': round(float(bb[-2]), 6), 'inside': round(float(bb[-1]), 6), 'outside_interval': ci[0], 'inside_interval': ci[1],
                       'chase_own_part_outside': round(float(bb[-6]), 6), 'shared_whiff_outside': round(float(bb[-4]), 6), 'sd_points_outside': round(float(DW[outside].std()), 3),
                       'sd_points_inside': round(float(DW[~outside].std()), 3), 'sd_shared_outside': round(float(DWm[outside].std()), 3)}
    by_side = {}
    for h, rr in gb.items():
        by_side.setdefault(side_of.get(h, int(np.round(T['stand_r'][ix[rr]].mean()))), []).append(h)
    perm = {}
    for hs in by_side.values():
        sh = list(hs); rc.shuffle(sh); perm.update({h: sh[(k + 1) % len(sh)] for k, h in enumerate(hs)})
    Dp = np.zeros(len(ix))
    for h, rr in gb.items():
        a = ix[rr]; Dp[rr] = (sig(off_w[a] + Bw[a] @ maps_w[perm[h]]) - sig(off_w[a] + Bw[a] @ mbar_w[h])) * 100
    bb, ci = fit(np.column_stack([base, Dp * outside, Dp * ~outside]))
    chk['placebo_other_hitter'] = {'outside': round(float(bb[-2]), 6), 'inside': round(float(bb[-1]), 6), 'outside_interval': ci[0], 'inside_interval': ci[1]}
    _, ki = np.unique((T['batter'][ix].astype(np.int64) * 10 + np.clip(T['group'][ix], 0, 6)) * 2 + outside, return_inverse=True)
    Dw_ = DW - (np.bincount(ki, weights=DW) / np.bincount(ki))[ki]
    pa_i = np.unique(T['game'][ix].astype(np.int64) * 1000 + T['ab'][ix].astype(np.int64), return_inverse=True)[1]
    loo = lambda v: np.bincount(pa_i, weights=v)[pa_i] - v
    for nm, dd in (('neighbor_held', DW), ('within_hitter_neighbor_held', Dw_)):
        bb, ci = fit(np.column_stack([base, loo(dd * outside), loo(dd * ~outside), dd * outside, dd * ~outside]))
        chk[nm] = {'outside': round(float(bb[-2]), 6), 'inside': round(float(bb[-1]), 6), 'outside_interval': ci[0], 'inside_interval': ci[1]}
    cgp = np.where(T['strikes'][ix] == 2, 2, np.where(T['balls'][ix] > T['strikes'][ix], 1, 0))
    cx = np.clip(np.floor((u_t + 2.5) / 0.2), 0, 25).astype(np.int64); cz = np.clip(np.floor(zt_ / 0.2), 0, 30).astype(np.int64)
    cell = ((((T['stand_r'][ix].astype(np.int64) * 3 + cgp) * 7 + np.clip(T['group'][ix], 0, 6)) * 26 + cx) * 31 + cz)
    _, ci_ = np.unique(cell, return_inverse=True); nc = np.bincount(ci_).astype(float)
    dm = lambda v: v - (np.bincount(ci_, weights=v) / nc)[ci_]
    Xsp = np.column_stack([dm(c) for c in np.column_stack([base[:, 1:], DW * outside, DW * ~outside]).T])
    bb = np.linalg.lstsq(Xsp, dm(y), rcond=None)[0]
    chk['same_spot'] = {'outside': round(float(bb[-2]), 6), 'inside': round(float(bb[-1]), 6)}
    res['coefficient_checks'] = chk
    res['test'] = {'season': test_season, 'pitches': int(len(ix)), 'outside_pitches_per_pa': round(per_pa['outside'], 3), 'inside_pitches_per_pa': round(per_pa['inside'], 3)}
    stage('coefficients')
    # aiming by the own whiff part, within pitch group, scatter around where the ball arrives
    sgm = float(params.get('sigma', 0.6)); K = 16
    jit = np.random.default_rng(3).standard_normal((K, 2)); jit = (jit - jit.mean(0)) / jit.std(0)
    pit = T['pitcher'][ix]; stand = T['stand_r'][ix]; strikes = T['strikes'][ix]; pg = np.clip(T['group'][ix], 0, 6)
    acc = {'outside': [0.0, 0.0], 'inside': [0.0, 0.0]}
    for pid, rr in _groups(pit, np.ones(len(ix), bool)).items():
        hs_here = np.unique(T['batter'][ix[rr]])
        for sd in (0, 1):
            for c3 in (0, 1, 2):
                for tg in range(7):
                    for zn, zmask in (('outside', outside), ('inside', ~outside)):
                        sel_ = (stand[rr] == sd) & (cgp[rr] == c3) & zmask[rr] & (pg[rr] == tg)
                        pool = rr[sel_]
                        if len(pool) < 30:
                            continue
                        if len(pool) > 300:
                            pool = rng.choice(pool, 300, replace=False)
                        a = ix[pool]
                        third = np.searchsorted(np.percentile(p_lw[pool], [33.3, 66.7]), p_lw[pool])
                        xj = (xt[a][:, None] + sgm * jit[None, :, 0]).ravel(); zj = (zt[a][:, None] + sgm * jit[None, :, 1]).ravel()
                        sj = np.repeat(np.full(len(a), sd), K); kj = np.repeat(strikes[pool], K)
                        offj = np.repeat(off_w[a] - Lw[a].astype(np.float64) @ wLw, K) + location_block(xj, zj, sj, kj).astype(np.float64) @ wLw
                        Bj = np.hstack([hitter_basis(xj, zj, sj, kj), np.repeat(famc[a], K, axis=0), hats(zj, hz4).astype(np.float64)])
                        bat_rr = T['batter'][ix[rr]][sel_]
                        for h in hs_here:
                            h = int(h)
                            if h not in maps_w or side_of.get(h) is None:
                                continue
                            n_here = int(np.sum(bat_rr == h))
                            if n_here == 0:
                                continue
                            dj = ((sig(offj + Bj @ maps_w[h]) - sig(offj + Bj @ mbar_w[h])) * 100).reshape(len(a), K).mean(1)
                            parts = []
                            for t3 in range(3):
                                d3 = dj[third == t3]
                                if len(d3) >= 3:
                                    k3 = max(1, len(d3) // 3); parts.append(float(np.sort(d3)[::-1][:k3].mean() - d3.mean()))
                            if parts:
                                acc[zn][0] += n_here * float(np.mean(parts)); acc[zn][1] += n_here
    aim = {}
    for zn in ('outside', 'inside'):
        g_ = acc[zn][0] / max(acc[zn][1], 1)
        aim[zn] = {'gain_points': round(g_, 3)}
        for nm in ('own_part', 'within_hitter_neighbor_held'):
            c = chk[nm]
            aim[zn]['runs_per_6200_' + nm] = round(c[zn] * g_ * per_pa[zn] * 6200, 1)
            aim[zn]['runs_per_6200_' + nm + '_interval'] = [round(v * g_ * per_pa[zn] * 6200, 1) for v in c[zn + '_interval']]
    res['aiming'] = aim
    stage('aiming')
    return res


# ---------------------------------------------------------------- VALUE-03: the edge net of hitters tightening up
def value3_study(T: dict, params: dict, stage) -> dict:
    """VALUE-02's outside edge at command scatter sigma (default 0.6 ft), net of ADAPT-01's general tightening: a pitcher
    who aims at a hitter's spots shows him more high spots than chance, and each extra one lowers the log-odds of his
    swinging at any later outside pitch in the game by tighten (default 0.065, ADAPT-01). Each 2025 outside pitch to a
    hitter with a map keeps its pitcher, side and count group; for that cell and hitter, aiming at the best third of
    the pitcher's own spots (under scatter) gives a gain in the hitter's deviation and a share of pitches landing in
    his high spots (top quarter of deviations); walking each hitter-game in order, the expected extra high spots seen
    before each pitch, beyond his own share, sets the tightening on it. Net runs = run value per point (VALUE-01's
    regression, re-estimated) times (gain minus tightening), per plate appearance, scaled to 6,200. Also with the
    tightening capped at three extra high spots (ADAPT-01's data rarely go beyond). Development data (2025)."""
    res = {}
    sgm = float(params.get('sigma', 0.6)); tighten = float(params.get('tighten', 0.065))
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    LB = location_block(xp, zp, T['stand_r'], T['strikes']); nL = LB.shape[1]
    X = np.hstack([LB, control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    tr = np.isin(T['season'], (2023, 2024))
    rng = np.random.default_rng(11)
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
    league = fit_logistic(X[idx], swing[idx]); off = league.decision_function(X); wL = league.coef_[0][:nL].astype(np.float64)
    Bm = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], tr), 10.0, 300)
    stage(f'maps {len(maps)}')
    te = (T['season'] == 2025) & (T['out7'] >= 0) & np.isin(T['batter'], np.asarray(list(maps), dtype=np.int64))
    ix = np.flatnonzero(te)
    sig = lambda v: 1 / (1 + np.exp(-v))
    p_l = sig(off[ix]); D = np.zeros(len(ix)); lo_map = np.zeros(len(ix))
    gb = _groups(T['batter'][ix], np.ones(len(ix), bool))
    for h, rr in gb.items():
        dev = Bm[ix[rr]] @ maps[h]; lo_map[rr] = off[ix[rr]] + dev
        D[rr] = (sig(off[ix[rr]] + dev) - p_l[rr]) * 100
    xt, zt = T['px'][ix].astype(np.float64), T['pz'][ix].astype(np.float64)
    u_t = np.where(T['stand_r'][ix] == 1, xt, -xt)
    e = np.maximum(np.maximum(np.abs(u_t) - ZONE_HALF, zt - ZONE_TOP), ZONE_BOT - zt)
    outside = e > 0
    thr = float(np.percentile(D[outside], 75))
    # run value per point outside (VALUE-01's regression)
    y = LW7[T['out7'][ix].astype(int)]
    first = (T['pitch_no'] == 0) & (T['out7'] >= 0)
    rv_all = LW7[np.clip(T['out7'], 0, 6).astype(int)]
    def prior_rv(key):
        nn, ss = _prior_by_day(np.r_[key[first], key[ix]].astype(np.int64), np.r_[T['day'][first], T['day'][ix]].astype(np.int64),
                               np.r_[rv_all[first], np.zeros(len(ix))], np.r_[np.ones(int(first.sum()), bool), np.zeros(len(ix), bool)])
        nn, ss = nn[int(first.sum()):], ss[int(first.sum()):]
        lg_ = float(rv_all[first].mean()); return (ss + 200 * lg_) / (nn + 200)
    rv_b, rv_p = prior_rv(T['batter']), prior_rv(T['pitcher'])
    cnt = np.zeros((len(ix), 11)); cc = np.clip(T['balls'][ix], 0, 3) * 3 + np.clip(T['strikes'][ix], 0, 2)
    for k in range(1, 12):
        cnt[:, k - 1] = cc == k
    grp = np.zeros((len(ix), 6)); g_ = np.clip(T['group'][ix], 0, 6)
    for k in range(1, 7):
        grp[:, k - 1] = g_ == k
    Xd = np.column_stack([np.ones(len(ix)), cnt, grp, p_l, np.log(p_l / (1 - p_l)), outside, hats(e, (-0.8, -0.4, -0.15, 0.0, 0.15, 0.4, 0.8, 1.5)), rv_b, rv_p,
                          (T['stand_r'][ix] == T['throw_r'][ix]).astype(float), D * outside, D * ~outside])
    b_out = float(np.linalg.lstsq(Xd, y, rcond=None)[0][-2])
    stage('coefficient')
    # per hitter and cell: gain and high-spot share when aiming at the best third under scatter
    K = 16; jit = np.random.default_rng(3).standard_normal((K, 2)); jit = (jit - jit.mean(0)) / jit.std(0)
    cg = np.where(T['strikes'][ix] == 2, 2, np.where(T['balls'][ix] > T['strikes'][ix], 1, 0))
    pit = T['pitcher'][ix]; stand = T['stand_r'][ix]; strikes = T['strikes'][ix]; bat = T['batter'][ix]
    hit_side = {h: int(np.round(stand[rr].mean())) for h, rr in gb.items()}
    cell_gain, cell_high = {}, {}
    for pid, rr in _groups(pit, np.ones(len(ix), bool)).items():
        hs_here = np.unique(bat[rr])
        for sd in (0, 1):
            for c3 in (0, 1, 2):
                pool = rr[(stand[rr] == sd) & (cg[rr] == c3) & outside[rr]]
                if len(pool) < 30:
                    continue
                if len(pool) > 300:
                    pool = rng.choice(pool, 300, replace=False)
                third = np.searchsorted(np.percentile(p_l[pool], [33.3, 66.7]), p_l[pool])
                lb0 = LB[ix[pool]].astype(np.float64) @ wL
                xj = (xp[ix[pool]][:, None] + sgm * jit[None, :, 0]).ravel(); zj = (zp[ix[pool]][:, None] + sgm * jit[None, :, 1]).ravel()
                sj = np.repeat(np.full(len(pool), sd), K); kj = np.repeat(strikes[pool], K)
                offj = np.repeat(off[ix[pool]] - lb0, K) + location_block(xj, zj, sj, kj).astype(np.float64) @ wL
                Bj = hitter_basis(xj, zj, sj, kj)
                for h in hs_here:
                    if int(h) not in maps or hit_side.get(int(h)) != sd:
                        continue
                    dmat = ((sig(offj + Bj @ maps[int(h)]) - sig(offj)) * 100).reshape(len(pool), K)
                    dj = dmat.mean(1); hj = (dmat >= thr).mean(1)
                    gains, highs = [], []
                    for t3 in range(3):
                        sel = np.flatnonzero(third == t3)
                        if len(sel) >= 3:
                            k3 = max(1, len(sel) // 3); best = sel[np.argsort(dj[sel])[::-1][:k3]]
                            gains.append(float(dj[best].mean() - dj[sel].mean())); highs.append(float(hj[best].mean()))
                    if gains:
                        cell_gain[(int(h), int(pid), c3)] = float(np.mean(gains)); cell_high[(int(h), int(pid), c3)] = float(np.mean(highs))
    stage(f'cells {len(cell_gain)}')
    # each hitter's own share of high spots among his outside pitches (what chance gives)
    uh, ih = np.unique(bat[outside], return_inverse=True)
    qh = dict(zip(uh.tolist(), (np.bincount(ih, weights=(D[outside] >= thr).astype(float)) / np.bincount(ih)).tolist()))
    # walk each hitter-game in order
    o = np.lexsort((T['pitch_no'][ix], T['ab'][ix], bat, T['game'][ix]))
    gross = tight = tight_cap = 0.0; n_used = 0; extra_hist = []
    cur = None; extra = 0.0
    for j in o:
        key = (int(T['game'][ix][j]), int(bat[j]))
        if key != cur:
            cur = key; extra = 0.0
        if not outside[j]:
            continue
        c = (int(bat[j]), int(pit[j]), int(cg[j]))
        if c not in cell_gain:
            continue
        gnn = cell_gain[c]; p_aim = min(max(sig(lo_map[j]) + gnn / 100.0, 1e-4), 1 - 1e-4)
        lo_aim = np.log(p_aim / (1 - p_aim))
        t_full = (p_aim - sig(lo_aim - tighten * extra)) * 100
        t_cap = (p_aim - sig(lo_aim - tighten * min(extra, 3.0))) * 100
        gross += gnn; tight += t_full; tight_cap += t_cap; n_used += 1; extra_hist.append(extra)
        extra += cell_high[c] - qh.get(int(bat[j]), 0.25)
    n_pa = max(int((T['pitch_no'][ix] == 0).sum()), 1)
    scale = 6200.0 / n_pa
    res['inputs'] = {'sigma_ft': sgm, 'tighten_per_extra_high_spot': tighten, 'high_threshold_points': round(thr, 2), 'runs_per_point_outside': round(b_out, 6),
                     'outside_pitches_valued': n_used, 'plate_appearances': n_pa}
    eh = np.asarray(extra_hist) if extra_hist else np.zeros(1)
    res['extra_high_spots_before_a_pitch'] = {'mean': round(float(eh.mean()), 3), 'p90': round(float(np.percentile(eh, 90)), 3), 'max': round(float(eh.max()), 3)}
    res['runs_per_6200'] = {'gross': round(b_out * gross * scale, 1), 'tightening': round(-b_out * tight * scale, 1), 'net': round(b_out * (gross - tight) * scale, 1),
                            'net_tightening_capped_at_3': round(b_out * (gross - tight_cap) * scale, 1)}
    res['points'] = {'gain_per_pitch': round(gross / max(n_used, 1), 3), 'tightening_per_pitch': round(tight / max(n_used, 1), 3)}
    stage('net')
    return res


# ---------------------------------------------------------------- ADAPT-01: do hitters learn their own chase spots within a game?
def adapt_study(T: dict, params: dict, stage) -> dict:
    """If pitchers started aiming at a hitter's own chase spots (VALUE-01F), would he adapt? Pitchers do not aim today
    (EXPLOIT-01), so how many of his chase spots a hitter has already been shown in a game is chance. Every 2025 pitch
    outside the zone to a hitter with a map (2023-2024): his swing on it, with the map's prediction as offset, on the
    number of earlier pitches in the same game that fell in his high spots (outside the zone, his own deviation in the
    top quarter), those he chased, and their products with this pitch's deviation (does his excess chasing at his own
    spots shrink after he has been shown them?), holding fixed the outside pitches seen so far, times through the
    order and the pitch number of the plate appearance. Game-bootstrap intervals."""
    res = {}
    post_mode = bool(params.get('postseason'))          # EXPLOIT-03: maps from the 2023-2025 regular seasons, targeting in those postseasons
    if 'post' not in T:
        T = dict(T); T['post'] = np.zeros(len(T['season']), np.int64)
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    X = np.hstack([location_block(xp, zp, T['stand_r'], T['strikes']), control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    tr = np.isin(T['season'], (2023, 2024))
    rng = np.random.default_rng(11)
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
    off = fit_logistic(X[idx], swing[idx]).decision_function(X)
    Bm = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], tr), 10.0, 300)
    stage(f'maps {len(maps)}')
    te = (T['season'] == 2025) & np.isin(T['batter'], np.asarray(list(maps), dtype=np.int64))
    ix = np.flatnonzero(te)
    sig = lambda v: 1 / (1 + np.exp(-v))
    lo_map = np.zeros(len(ix)); d = np.zeros(len(ix))
    for h, rr in _groups(T['batter'][ix], np.ones(len(ix), bool)).items():
        dev = Bm[ix[rr]] @ maps[h]; lo_map[rr] = off[ix[rr]] + dev
        d[rr] = (sig(off[ix[rr]] + dev) - sig(off[ix[rr]])) * 100
    xt, zt = T['px'][ix], T['pz'][ix]
    u_t = np.where(T['stand_r'][ix] == 1, xt, -xt)
    outside = (np.abs(u_t) > ZONE_HALF) | (zt > ZONE_TOP) | (zt < ZONE_BOT)
    thr = float(np.percentile(d[outside], 75))
    high = outside & (d >= thr)
    y = swing[ix]
    # earlier pitches in the same game to the same hitter (order: game, hitter, plate appearance, pitch)
    o = np.lexsort((T['pitch_no'][ix], T['ab'][ix], T['batter'][ix], T['game'][ix]))
    g_, b_ = T['game'][ix][o], T['batter'][ix][o]
    start = np.ones(len(o), bool); start[1:] = (g_[1:] != g_[:-1]) | (b_[1:] != b_[:-1])
    def prev_cum(v):
        c = np.cumsum(v[o].astype(float)); seg = np.maximum.accumulate(np.where(start, np.arange(len(o)), 0))
        before = c - v[o].astype(float) - np.where(seg > 0, c[seg - 1], 0.0)
        out_ = np.empty(len(o)); out_[o] = before; return out_
    n_out = prev_cum(outside); n_high = prev_cum(high); n_high_ch = prev_cum(high & (y == 1)); n_low = prev_cum(outside & ~high)
    # times through the order: plate appearances this hitter has had in the game before this one
    pa_key = T['game'][ix].astype(np.int64) * 10000 + T['ab'][ix].astype(np.int64)
    first_pitch = T['pitch_no'][ix] == 0
    n_pa_prev = prev_cum(first_pitch) - 0.0
    m = outside
    dd = d[m] / 10.0
    # how many of the outside pitches already seen fell in his high spots, beyond what his own share of high spots
    # would give by chance (hitters with more distinct maps have more high spots, which would otherwise stand in for
    # how distinct, and how well calibrated, the map is)
    bat_ix = T['batter'][ix]
    uh, ih = np.unique(bat_ix[outside], return_inverse=True)
    qh = dict(zip(uh.tolist(), (np.bincount(ih, weights=high[outside].astype(float)) / np.bincount(ih)).tolist()))
    q = np.asarray([qh.get(int(h_), 0.25) for h_ in bat_ix])
    excess = n_high - n_out * q
    Z = np.column_stack([np.ones(m.sum()), n_out[m], excess[m], excess[m] * dd, q[m] * dd, n_out[m] * dd, n_high_ch[m], n_high_ch[m] * dd,
                         np.minimum(n_pa_prev[m], 3), np.minimum(T['pitch_no'][ix][m], 6)])
    names = ['intercept', 'outside_seen_before', 'high_spots_seen_beyond_chance', 'high_spots_beyond_chance_x_deviation', 'own_high_share_x_deviation',
             'outside_seen_x_deviation', 'high_spots_chased', 'high_spots_chased_x_deviation', 'earlier_plate_appearances', 'pitch_number']
    yy = y[m]; base = lo_map[m]
    from sklearn.linear_model import LogisticRegression
    def fit(Zs, ys, bs):
        # logistic with offset: fold the offset into the working response by Newton steps
        bcoef = np.zeros(Zs.shape[1])
        for _ in range(25):
            p_ = sig(bs + Zs @ bcoef); W = p_ * (1 - p_)
            H = (Zs * W[:, None]).T @ Zs + 1e-6 * np.eye(Zs.shape[1]); g = Zs.T @ (ys - p_)
            step = np.linalg.solve(H, g); bcoef += step
            if np.max(np.abs(step)) < 1e-8:
                break
        return bcoef
    bfull = fit(Z, yy, base)
    games = T['game'][ix][m]; ug, gi = np.unique(games, return_inverse=True); draws = []
    for _ in range(int(params.get('reps', 60))):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi]; sel = np.repeat(np.arange(len(yy)), w)
        draws.append(fit(Z[sel], yy[sel], base[sel]))
    draws = np.asarray(draws)
    res['coefficients'] = {n_: [round(float(bfull[k]), 4), round(float(np.percentile(draws[:, k], 2.5)), 4), round(float(np.percentile(draws[:, k], 97.5)), 4)] for k, n_ in enumerate(names)}
    # the excess chase at a high spot, after 0 to 3 earlier high spots, model against map
    hi_m = high[m]
    tab = []
    for k in range(4):
        sel = hi_m & (np.minimum(n_high[m], 3) == k)            # descriptive only: mixes chance with how distinct the hitter's map is
        if sel.sum() > 200:
            tab.append({'earlier_high_spots': k, 'pitches': int(sel.sum()), 'observed_swing': round(float(yy[sel].mean()), 4),
                        'map_predicted': round(float(sig(base[sel]).mean()), 4), 'league_predicted': round(float(sig(off[ix][m][sel]).mean()), 4)})
    res['high_spot_pitches_by_earlier_exposure'] = tab
    res['rows'] = {'outside_pitches': int(m.sum()), 'high_threshold_points': round(thr, 2), 'high_share': round(float(hi_m.mean()), 4)}
    stage('adaptation')
    return res


# ---------------------------------------------------------------- SERIES-01: does the own-spot edge fade as a hitter keeps facing a pitcher?
def series_study(T: dict, params: dict, stage) -> dict:
    """In a series the same hitters face the same pitchers again. ADAPT-01 found no learning of own spots within a game;
    this asks across games. MATCHUP-05F's representation fitted on 2023-2024 (league with its family part, family maps,
    shrinkage 10) and each hitter's own part (his map minus the same-side mean map). Every 2025 pitch outside the zone
    to a hitter with a map: his swing, with the full map's prediction as offset, on the plate appearances he had
    against this pitcher in earlier games of 2025 (capped at 10), its product with the own part at this pitch (log
    odds; negative means the own part's pull fades with familiarity), the plate appearances against him earlier in this
    game, and the pitch number. Game-bootstrap intervals; and the observed swing at his high own spots against the map,
    by earlier meetings."""
    res = {}
    post_mode = bool(params.get('postseason'))          # EXPLOIT-03: maps from the 2023-2025 regular seasons, targeting in those postseasons
    if 'post' not in T:
        T = dict(T); T['post'] = np.zeros(len(T['season']), np.int64)
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64)
    xp, zp = projected(T, F, None, 'straight', 0.26); del F
    Bh = hitter_basis(xp, zp, T['stand_r'], T['strikes'])
    X = np.hstack([location_block(xp, zp, T['stand_r'], T['strikes']), (Bh[:, :-1] * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32),
                   (Bh[:, :-1] * (T['group'] == 5)[:, None]).astype(np.float32), control_block(T, swing_propensity(T)), pitcher_propensity(T)[:, None].astype(np.float32)])
    tr = np.isin(T['season'], (2023, 2024))
    rng = np.random.default_rng(11)
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
    off = fit_logistic(X[idx], swing[idx]).decision_function(X); del X
    Bm = family_basis(Bh, T['group']); del Bh
    maps = _hitter_maps(Bm, swing, off, _groups(T['batter'], tr), 10.0, 300)
    ub_, bi_ = np.unique(T['batter'][tr], return_inverse=True)
    side_of = dict(zip(ub_.tolist(), np.round(np.bincount(bi_, weights=T['stand_r'][tr]) / np.bincount(bi_)).astype(int).tolist()))
    Ss = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
    for h, m in maps.items():
        Ss[side_of[h]] = Ss[side_of[h]] + m; Nn[side_of[h]] += 1
    mbar = {h: (Ss[side_of[h]] - m) / max(Nn[side_of[h]] - 1, 1) for h, m in maps.items()}
    stage(f'maps {len(maps)}')
    te = (T['season'] == 2025) & np.isin(T['batter'], np.asarray(list(maps), dtype=np.int64))
    ix = np.flatnonzero(te)
    sig = lambda v: 1 / (1 + np.exp(-v))
    lo_map = np.zeros(len(ix)); own = np.zeros(len(ix))
    for h, rr in _groups(T['batter'][ix], np.ones(len(ix), bool)).items():
        a = ix[rr]; full = Bm[a] @ maps[h]; lo_map[rr] = off[a] + full; own[rr] = Bm[a] @ (maps[h] - mbar[h])
    xt, zt = T['px'][ix], T['pz'][ix]
    u_t = np.where(T['stand_r'][ix] == 1, xt, -xt)
    outside = (np.abs(u_t) > ZONE_HALF) | (zt > ZONE_TOP) | (zt < ZONE_BOT)
    # plate appearances of this hitter against this pitcher in earlier games of the season, and earlier in this game
    first = T['pitch_no'][ix] == 0
    pair = T['batter'][ix].astype(np.int64) * 10_000_000 + T['pitcher'][ix].astype(np.int64)
    o = np.lexsort((T['pitch_no'][ix], T['ab'][ix], T['game'][ix], T['day'][ix], pair))
    pr, gm, fp = pair[o], T['game'][ix][o], first[o]
    newpair = np.ones(len(o), bool); newpair[1:] = pr[1:] != pr[:-1]
    newgame = newpair.copy(); newgame[1:] |= gm[1:] != gm[:-1]
    c = np.cumsum(fp.astype(float))
    pair_start = np.maximum.accumulate(np.where(newpair, np.arange(len(o)), 0)); game_start = np.maximum.accumulate(np.where(newgame, np.arange(len(o)), 0))
    before_pair = np.where(pair_start > 0, c[pair_start - 1], 0.0); before_game = np.where(game_start > 0, c[game_start - 1], 0.0)
    earlier_games = np.empty(len(o)); earlier_games[o] = before_game - before_pair
    same_game = np.empty(len(o)); same_game[o] = (c - fp.astype(float)) - before_game
    m = outside
    k = np.minimum(earlier_games[m], 10.0)
    Z = np.column_stack([np.ones(m.sum()), k, own[m] * k, own[m], np.minimum(same_game[m], 3), np.minimum(T['pitch_no'][ix][m], 6)])
    names = ['intercept', 'earlier_meetings', 'own_part_x_earlier_meetings', 'own_part_extra', 'earlier_this_game', 'pitch_number']
    yy = swing[ix][m]; base = lo_map[m]
    def fit(Zs, ys, bs):
        bcoef = np.zeros(Zs.shape[1])
        for _ in range(30):
            p_ = sig(bs + Zs @ bcoef); W = p_ * (1 - p_)
            step = np.linalg.solve((Zs * W[:, None]).T @ Zs + 1e-6 * np.eye(Zs.shape[1]), Zs.T @ (ys - p_)); bcoef += step
            if np.max(np.abs(step)) < 1e-9:
                break
        return bcoef
    bfull = fit(Z, yy, base)
    games = T['game'][ix][m]; ug, gi = np.unique(games, return_inverse=True); draws = []
    for _ in range(int(params.get('reps', 60))):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi]; sel = np.repeat(np.arange(len(yy)), w)
        draws.append(fit(Z[sel], yy[sel], base[sel]))
    draws = np.asarray(draws)
    res['coefficients'] = {n_: [round(float(bfull[j]), 4), round(float(np.percentile(draws[:, j], 2.5)), 4), round(float(np.percentile(draws[:, j], 97.5)), 4)] for j, n_ in enumerate(names)}
    res['own_part_sd_logodds'] = round(float(own[m].std()), 3)
    thr = float(np.percentile(own[m], 80)); hi = own[m] >= thr
    tab = []
    for lo_, hi_, lab in ((0, 0, '0'), (1, 3, '1-3'), (4, 9, '4-9'), (10, 99, '10+')):
        sel = hi & (earlier_games[m] >= lo_) & (earlier_games[m] <= hi_)
        if sel.sum() > 200:
            tab.append({'earlier_meetings': lab, 'pitches': int(sel.sum()), 'observed_swing': round(float(yy[sel].mean()), 4), 'map_predicted': round(float(sig(base[sel]).mean()), 4)})
    res['high_own_spots_by_meetings'] = tab
    res['rows'] = {'outside_pitches': int(m.sum()), 'with_earlier_meetings': int((earlier_games[m] > 0).sum()), 'mean_earlier_meetings': round(float(earlier_games[m].mean()), 2)}
    stage('series')
    return res


# ---------------------------------------------------------------- DAMAGE-01: where a hitter does damage, and which clock it follows
def damage_study(T: dict, params: dict, stage) -> dict:
    """The third part of the matchup engine: damage on contact. Each ball in play's expected run value from its launch
    speed and angle (a league table of run value by speed and angle bins fitted on 2023-2024, so fielding drops out).
    League model: the pitch's location bands by count, group, speed and platoon, plus the hitter's earlier damage
    (shrunk), linear. Hitter damage maps: ridge regression of the residual on the 22-column hitter basis (at least 150
    training balls in play), shrinkage chosen on the second half of 2024. Each on the true crossing and at the decision
    moment. Trained 2023-2024, scored on 2025, squared-error reduction per 1,000 balls in play, paired by game. Also the
    run value of a ball in play by the hitter's map value at that spot (calibration across quintiles)."""
    res = {}
    final = bool(params.get('final_eval'))
    if final and not params.get('frozen_commit'):
        raise ValueError('the final damage scoring runs only as the registered evaluation of a frozen commit')
    if final:      # DAMAGE-01F: fitted on 2023-2025 (shrinkage on the second half of 2025), scored on 2026 from August 1
        T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | ((T['season'] == 2026) & (T['day'] >= date(2026, 8, 1).toordinal())))
    else:
        T = take(T, np.isin(T['season'], (2023, 2024, 2025)))
    F = rebuild(T)
    bip = F['ok'] & (T['group'] >= 0) & (T['call'] == 1) & (T['last_in_pa'] == 1) & np.isfinite(T['ls']) & np.isfinite(T['la']) & (T['ls'] > 20) & (T['bunt_pa'] == 0) & np.isin(T['out7'], (0, 3, 4, 5, 6))
    T = take(T, bip); F = {k: v[bip] for k, v in F.items()}
    rv = LW7[T['out7'].astype(int)]
    last_train = 2025 if final else 2024
    tr = (T['season'] <= last_train); te = T['season'] == last_train + 1
    # expected run value by launch speed and angle (league table from training seasons)
    sb = np.clip(((T['ls'] - 40) // 4).astype(int), 0, 18); ab = np.clip(((T['la'] + 40) // 6).astype(int), 0, 19)
    cell = sb * 20 + ab
    sums = np.bincount(cell[tr], weights=rv[tr], minlength=19 * 20); cnts = np.bincount(cell[tr], minlength=19 * 20)
    xrv_tab = (sums + 20 * rv[tr].mean()) / (cnts + 20)
    y = xrv_tab[cell]
    stage('expected run value table')
    # hitter's earlier damage (shrunk), from earlier days
    nn, ss = _prior_by_day(T['batter'].astype(np.int64), T['day'].astype(np.int64), y, np.ones(len(y), bool))
    prior_h = (ss + 100 * y[tr].mean()) / (nn + 100)
    xp, zp = projected(T, F, None, 'straight', 0.26)
    xt, zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    grp = np.zeros((len(y), 7), np.float32); grp[np.arange(len(y)), np.clip(T['group'], 0, 6)] = 1
    C = np.hstack([grp, hats(T['v0'].astype(np.float64), V_KNOTS), (T['stand_r'] == T['throw_r'])[:, None], prior_h[:, None]]).astype(np.float64)
    mid = date(last_train, 7, 1).toordinal()
    fit_a = tr & ~((T['season'] == last_train) & (T['day'] >= mid)); val = tr & (T['season'] == last_train) & (T['day'] >= mid)
    games = T['game'][te]; yt = y[te]
    out, preds = {}, {}
    for name, (x, z) in (('true', (xt, zt)), ('percept', (xp, zp))):
        X = np.hstack([location_block(x, z, T['stand_r'], T['strikes']).astype(np.float64), C])
        def league(rows):
            A = X[rows]; b = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y[rows]); return X @ b
        base_a, base = league(fit_a), league(tr)
        Bm = hitter_basis(x, z, T['stand_r'], T['strikes'])
        def maps_for(rows, basev, lam):
            mp = {}
            for h, r in _groups(T['batter'], rows).items():
                if len(r) >= 150:
                    A = Bm[r]; mp[h] = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[r] - basev[r]))
            return mp
        val_mse = {}
        for lam in (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0):
            mp = maps_for(fit_a, base_a, lam); pr = base_a.copy()
            for h, r in _groups(T['batter'], val).items():
                if h in mp:
                    pr[r] += Bm[r] @ mp[h]
            val_mse[lam] = float(np.mean((y[val] - pr[val]) ** 2))
        best = min(val_mse, key=val_mse.get)
        mp = maps_for(tr, base, best); ph = base.copy(); cov = np.zeros(len(y), bool)
        for h, r in _groups(T['batter'], te).items():
            if h in mp:
                ph[r] += Bm[r] @ mp[h]; cov[r] = True
        preds[name] = (base[te], ph[te])
        out[name] = {'shrinkage': best, 'validation_mse': {str(k): round(v, 6) for k, v in val_mse.items()}, 'covered': round(float(cov[te].mean()), 4),
                     'test_mse_league': round(float(np.mean((yt - base[te]) ** 2)), 6), 'test_mse_hitter_maps': round(float(np.mean((yt - ph[te]) ** 2)), 6)}
        stage('damage ' + name)
    se = lambda p_: (yt - p_) ** 2
    cc = lambda dlt: [round(v * 1000, 4) for v in clustered_ci(dlt, games)]
    res['representations'] = out
    res['balls_in_play_test'] = int(te.sum()); res['test_season'] = int(last_train + 1); res['expected_run_value_sd'] = round(float(yt.std()), 4)
    res['squared_error_reduction_per_1000'] = {'hitter_maps_over_league_true': cc(se(preds['true'][0]) - se(preds['true'][1])),
                                               'hitter_maps_over_league_percept': cc(se(preds['percept'][0]) - se(preds['percept'][1])),
                                               'true_over_percept_hitter_maps': cc(se(preds['percept'][1]) - se(preds['true'][1]))}
    # calibration of the hitter-map deviation (true crossing) across quintiles
    dev = preds['true'][1] - preds['true'][0]
    q = np.percentile(dev, [20, 40, 60, 80]); qi = np.searchsorted(q, dev)
    res['by_map_deviation_quintile'] = [{'map_deviation': round(float(dev[qi == k].mean()), 4), 'observed_minus_league': round(float((yt - preds['true'][0])[qi == k].mean()), 4),
                                         'balls_in_play': int((qi == k).sum())} for k in range(5)]
    return res


# ---------------------------------------------------------------- SCOUT-01: per-player decision-moment profiles and postseason matchups
def _postseason_rosters(season: int) -> dict:
    """Postseason teams of the season (public schedule), their active rosters split into hitters and pitchers, and the
    pairs of teams that meet."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('brl_backfill_for_scout', ROOT / 'tools' / 'brl_bookkeeping_backfill.py')
    bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
    sched = bf.get_json(f'{bf.API}/schedule?sportId=1&season={season}&gameType=F,D,L,W')
    meets = set(); teams = {}
    for d in sched.get('dates', []):
        for g in d.get('games', []):
            a = (g.get('teams') or {}).get('away', {}).get('team', {}); h = (g.get('teams') or {}).get('home', {}).get('team', {})
            if a.get('id') and h.get('id'):
                teams[int(a['id'])] = a.get('name', ''); teams[int(h['id'])] = h.get('name', '')
                meets.add(tuple(sorted((int(a['id']), int(h['id'])))))
    rosters = {}
    for tid in teams:
        try:
            r = bf.get_json(f'{bf.API}/teams/{tid}/roster?rosterType=active&season={season}')
        except Exception:
            continue
        hit, pit = [], []
        for e in r.get('roster', []):
            pid = int((e.get('person') or {}).get('id') or 0); pos = (e.get('position') or {}).get('type', '')
            if not pid:
                continue
            (pit if pos == 'Pitcher' else hit).append(pid)
            if pos == 'Two-Way Player':
                pit.append(pid)
        rosters[tid] = {'name': teams[tid], 'hitters': hit, 'pitchers': pit}
    return {'teams': rosters, 'meets': sorted(list(m) for m in meets)}


def scout_export(T: dict, params: dict, stage) -> dict:
    """Per-player profiles from the credited decision-moment representation (MATCHUP-01F) and the miss maps
    (MATCHUP-03), fitted on 2025 and 2026 through July (the program's untouched months stay out), and the postseason's
    hitter-pitcher matchups: each hitter's swing map at the decision moment and whiff map on the true crossing as
    log-odds deviations from the league on a 7 by 7 grid (feet; side measured away from the hitter), his chase,
    zone-swing and miss tendencies read on one standard set of pitches, his top chase cells; each postseason pitcher's
    arsenal; and for every postseason hitter against every pitcher of a team he meets: the extra chase his map predicts
    on that pitcher's pitches (times the calibrated slope 0.95), and the count-by-count chain's strikeout and walk
    changes (times the ENGINE-01 calibration, 0.40 and 0.53), split into hitter, pitcher and pair parts over these
    pairs. Model outputs and per-player summaries only."""
    res = {}
    T = take(T, np.isin(T['season'], (2025, 2026)))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    Tk = take(T, keep); Fk = {k: v[keep] for k, v in F.items()}
    swing = ((Tk['call'] == 1) | (Tk['call'] == 2)).astype(np.float64); whiff = (Tk['call'] == 2).astype(np.float64)
    take_ = Tk['call'] == 0; cs = (Tk['cs'] == 1).astype(np.float64); contact = Tk['call'] == 1
    foul = (contact & (Tk['last_in_pa'] == 0)).astype(np.float64)
    xp, zp = projected(Tk, Fk, None, 'straight', 0.26)
    xt, zt = Tk['px'].astype(np.float64), Tk['pz'].astype(np.float64)
    u_t = np.where(Tk['stand_r'] == 1, xt, -xt)
    outside = (np.abs(u_t) > ZONE_HALF) | (zt > ZONE_TOP) | (zt < ZONE_BOT)
    rng = np.random.default_rng(11)
    def sample(rows, n=600000):
        idx = np.flatnonzero(rows); return rng.choice(idx, min(len(idx), n), replace=False) if len(idx) > n else idx
    prop_s = swing_propensity(Tk); prop_p = pitcher_propensity(Tk)
    Ls = location_block(xp, zp, Tk['stand_r'], Tk['strikes'])
    famv = bool(params.get('family'))             # SCOUT-02: MATCHUP-05F's representation (league family part, hitter family maps)
    Bh0 = hitter_basis(xp, zp, Tk['stand_r'], Tk['strikes']); nh = Bh0.shape[1] - 1
    lf_parts = [(Bh0[:, :-1] * np.isin(Tk['group'], (3, 4))[:, None]).astype(np.float32), (Bh0[:, :-1] * (Tk['group'] == 5)[:, None]).astype(np.float32)] if famv else []
    Xs = np.hstack([Ls] + lf_parts + [control_block(Tk, prop_s), prop_p[:, None].astype(np.float32)]); i_ps = Ls.shape[1] + (2 * nh if famv else 0) + 32
    del lf_parts
    allr = np.ones(len(swing), bool)
    i = sample(allr); m_s = fit_logistic(Xs[i], swing[i]); off_s = m_s.decision_function(Xs)
    Bs = family_basis(Bh0, Tk['group']) if famv else Bh0
    maps_s = _hitter_maps(Bs, swing, off_s, _groups(Tk['batter'], allr), 10.0, int(params.get('min_pitches', 500)))
    mbar = {}
    if famv:
        # each hitter's same-side mean map over the other hitters with maps (VALUE-08's shared part)
        gb0 = _groups(Tk['batter'], allr)
        side0 = {h: int(np.round(Tk['stand_r'][gb0[h]].mean())) for h in maps_s}
        Ssum = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
        for h, m in maps_s.items():
            Ssum[side0[h]] = Ssum[side0[h]] + m; Nn[side0[h]] += 1
        mbar = {h: (Ssum[side0[h]] - m) / max(Nn[side0[h]] - 1, 1) for h, m in maps_s.items()}
        mbar_side = {sd: Ssum[sd] / max(Nn[sd], 1) for sd in (0, 1)}
    tw = dict(Tk); tw['call'] = np.where(whiff == 1, 1, np.where(swing == 1, 0, 3)); prop_w = swing_propensity(tw, 200.0)
    grp = np.zeros((len(swing), 7), np.float32); grp[np.arange(len(swing)), np.clip(Tk['group'], 0, 6)] = 1
    Lw = location_block(xt, zt, Tk['stand_r'], Tk['strikes'])
    Xw = np.hstack([Lw, grp, hats(Tk['v0'].astype(np.float64), V_KNOTS), (Tk['strikes'] == 2)[:, None].astype(np.float32), prop_w[:, None].astype(np.float32),
                    (Tk['stand_r'] == Tk['throw_r'])[:, None].astype(np.float32)]); i_pw = Lw.shape[1] + 7 + len(V_KNOTS) + 1
    sw_rows = swing == 1
    i = sample(sw_rows); m_w = fit_logistic(Xw[i], whiff[i]); off_w = m_w.decision_function(Xw)
    fam = np.column_stack([np.isin(Tk['group'], (0, 1, 2)), np.isin(Tk['group'], (3, 4)), np.isin(Tk['group'], (5,))]).astype(np.float64)
    Bw = np.hstack([hitter_basis(xt, zt, Tk['stand_r'], Tk['strikes']), fam, hats(zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
    maps_w = _hitter_maps(Bw, whiff, off_w, _groups(Tk['batter'], sw_rows), 30.0, 250)
    Xc = np.hstack([Lw, (Tk['stand_r'] == Tk['throw_r'])[:, None].astype(np.float32)])
    i = sample(take_); p_cs = fit_logistic(Xc[i], cs[i]).predict_proba(Xc)[:, 1]
    i = sample(contact); p_fo = fit_logistic(Xw[i], foul[i]).predict_proba(Xw)[:, 1]
    stage(f'maps {len(maps_s)} swing, {len(maps_w)} whiff')
    # grid in feet: side away from the hitter, height; the hitter basis at two strikes off
    gu = np.array([-1.25, -0.83, -0.42, 0.0, 0.42, 0.83, 1.25]); gz = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    UU, ZZ = np.meshgrid(gu, gz); uu, zz = UU.ravel(), ZZ.ravel()
    Bg = hitter_basis(uu, zz, np.ones(len(uu), np.int64), np.zeros(len(uu), np.int64))
    fam_ff = np.tile(np.array([1.0, 0.0, 0.0]), (len(uu), 1))
    Bgw = np.hstack([Bg, fam_ff, hats(zz, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
    # league probabilities on the grid for a first-pitch four-seamer at 94 mph to a right-handed hitter, average propensities
    g_t = {'balls': np.zeros(len(uu), np.int64), 'strikes': np.zeros(len(uu), np.int64), 'group': np.zeros(len(uu), np.int64), 'v0': np.full(len(uu), 94.0),
           'stand_r': np.ones(len(uu), np.int64), 'throw_r': np.ones(len(uu), np.int64)}
    lg_ps = float(np.mean(prop_s)); lg_pp = float(np.mean(prop_p)); lg_pw = float(np.mean(prop_w[sw_rows]))
    Xg = np.hstack([location_block(uu, zz, g_t['stand_r'], g_t['strikes']), control_block(g_t, np.full(len(uu), lg_ps)), np.full((len(uu), 1), lg_pp, np.float32)])
    Xgw = np.hstack([location_block(uu, zz, g_t['stand_r'], g_t['strikes']), np.tile(np.eye(7, dtype=np.float32)[0], (len(uu), 1)), hats(np.full(len(uu), 94.0), V_KNOTS),
                     np.zeros((len(uu), 1), np.float32), np.full((len(uu), 1), lg_pw, np.float32), np.ones((len(uu), 1), np.float32)])
    if famv:
        # by family: first pitch to a right-handed hitter from a right-hander, typical speeds; the average hitter adds the mean map
        famgrid = {}
        for fname, gcode, spd in (('fastball', 0, 94.0), ('breaking', 3, 85.0), ('offspeed', 5, 86.0)):
            gt_f = dict(g_t); gt_f['group'] = np.full(len(uu), gcode, np.int64); gt_f['v0'] = np.full(len(uu), spd)
            Bgf = family_basis(Bg, gt_f['group'])
            Xgf = np.hstack([location_block(uu, zz, gt_f['stand_r'], gt_f['strikes']), (Bg[:, :-1] * (gcode in (3, 4))).astype(np.float32), (Bg[:, :-1] * (gcode == 5)).astype(np.float32),
                             control_block(gt_f, np.full(len(uu), lg_ps)), np.full((len(uu), 1), lg_pp, np.float32)])
            lo_l = m_s.decision_function(Xgf).astype(np.float64)
            famgrid[fname] = {'league_swing': [round(float(v), 4) for v in 1 / (1 + np.exp(-lo_l))],
                              'average_hitter_swing': [round(float(v), 4) for v in 1 / (1 + np.exp(-(lo_l + Bgf @ mbar_side[1])))]}
        Xg = np.hstack([location_block(uu, zz, g_t['stand_r'], g_t['strikes']), np.zeros((len(uu), 2 * nh), np.float32), control_block(g_t, np.full(len(uu), lg_ps)), np.full((len(uu), 1), lg_pp, np.float32)])
    res['grid'] = {'side_ft': gu.tolist(), 'height_ft': gz.tolist(), 'league_swing': [round(float(v), 4) for v in 1 / (1 + np.exp(-m_s.decision_function(Xg).astype(np.float64)))],
                   'league_whiff': [round(float(v), 4) for v in 1 / (1 + np.exp(-m_w.decision_function(Xgw).astype(np.float64)))],
                   'context': 'first pitch, four-seamer at 94 mph, right-handed hitter and pitcher, league-average swing and miss levels'}
    if famv:
        res['grid']['by_family'] = famgrid
    # standard reference pitches by side for the hitter summaries
    ref = rng.choice(len(swing), min(len(swing), 40000), replace=False)
    ref_side = {sd: ref[Tk['stand_r'][ref] == sd] for sd in (0, 1)}
    base_s = {}; base_w = {}
    for sd, R in ref_side.items():
        a = Xs[R].astype(np.float64).copy(); a[:, i_ps] = 0.0; base_s[sd] = m_s.decision_function(a)
        b = Xw[R].astype(np.float64).copy(); b[:, i_pw] = 0.0; base_w[sd] = m_w.decision_function(b)
    c_s, c_w = float(m_s.coef_[0][i_ps]), float(m_w.coef_[0][i_pw])
    lgt = lambda r: np.log(r / (1 - r))
    lg_sw = float(swing.mean()); lg_wh = float(whiff[sw_rows].mean())
    rost = {}
    try:
        rost = _postseason_rosters(int(params.get('season', 2026)))
    except Exception as e:
        res['roster_error'] = repr(e)[:300]
    res['postseason'] = rost
    want_h = set(h for t in (rost.get('teams') or {}).values() for h in t['hitters']) if rost else set()
    want_p = set(p_ for t in (rost.get('teams') or {}).values() for p_ in t['pitchers']) if rost else set()
    gb = _groups(Tk['batter'], allr)
    hitters = {}
    for h, r in gb.items():
        if h not in maps_s or h not in maps_w:
            continue
        if want_h and h not in want_h and len(r) < int(params.get('min_pitches_all', 2500)):
            continue
        sd = int(np.round(Tk['stand_r'][r].mean())); R = ref_side[sd]
        ps = (swing[r].sum() + 300 * lg_sw) / (len(r) + 300); pw = (whiff[r].sum() + 200 * lg_wh) / (swing[r].sum() + 200)
        p_sw = 1 / (1 + np.exp(-(base_s[sd] + c_s * lgt(ps) + Bs[R] @ maps_s[h])))
        p_wh = 1 / (1 + np.exp(-(base_w[sd] + c_w * lgt(pw) + Bw[R] @ maps_w[h])))
        o = outside[R]
        out_cells = [k for k in range(len(uu)) if (abs(uu[k]) > ZONE_HALF or zz[k] > ZONE_TOP or zz[k] < ZONE_BOT)]
        dw = Bgw @ maps_w[h]
        if famv:
            own = {}
            for fname, gcode in (('fastball', 0), ('breaking', 3), ('offspeed', 5)):
                dgf = family_basis(Bg, np.full(len(uu), gcode, np.int64)) @ (maps_s[h] - mbar[h])
                own[fname] = {'own_dev_grid': [round(float(v), 3) for v in dgf],
                              'top_chase_cells': [[float(uu[k]), float(zz[k]), round(float(dgf[k]), 3)] for k in sorted(out_cells, key=lambda k: -dgf[k])[:3]]}
            dg = family_basis(Bg, np.zeros(len(uu), np.int64)) @ maps_s[h]
        else:
            dg = Bg @ maps_s[h]
        top = sorted(out_cells, key=lambda k: -dg[k])[:3]
        hitters[int(h)] = {'side': 'R' if sd == 1 else 'L', 'pitches': int(len(r)), 'swings': int(swing[r].sum()),
                           'map_chase': round(float(p_sw[o].mean()), 4), 'map_zone_swing': round(float(p_sw[~o].mean()), 4),
                           'map_whiff': round(float((p_sw * p_wh).sum() / p_sw.sum()), 4),
                           'raw_chase': round(float(swing[r][outside[r]].mean()), 4) if outside[r].any() else None,
                           'raw_zone_swing': round(float(swing[r][~outside[r]].mean()), 4) if (~outside[r]).any() else None,
                           'raw_whiff': round(float(whiff[r].sum() / max(swing[r].sum(), 1)), 4),
                           'swing_dev_grid': [round(float(v), 3) for v in dg], 'whiff_dev_grid': [round(float(v), 3) for v in dw],
                           'top_chase_cells': [[float(uu[k]), float(zz[k]), round(float(dg[k]), 3)] for k in top]}
        if famv:
            hitters[int(h)]['own_by_family'] = own
    res['hitters'] = hitters
    stage(f'hitters {len(hitters)}')
    # pitchers: arsenal by family and side, from 2025 and 2026 through July
    gp = _groups(Tk['pitcher'], allr)
    famname = np.where(np.isin(Tk['group'], (0, 1, 2)), 0, np.where(np.isin(Tk['group'], (3, 4)), 1, np.where(Tk['group'] == 5, 2, 3)))
    pitchers = {}
    for p_, r in gp.items():
        if (want_p and p_ not in want_p) or len(r) < 300:
            continue
        mix = {}
        for f_, nm in ((0, 'fastball'), (1, 'breaking'), (2, 'offspeed')):
            mm = famname[r] == f_
            if mm.any():
                mix[nm] = {'share': round(float(mm.mean()), 3), 'speed': round(float(Tk['v0'][r][mm].mean()), 1)}
        dm_in_true_out = float(np.mean((~((np.abs(np.where(Tk['stand_r'][r] == 1, xp[r], -xp[r])) > ZONE_HALF) | (zp[r] > ZONE_TOP) | (zp[r] < ZONE_BOT))) & outside[r]))
        pitchers[int(p_)] = {'throws': 'R' if Tk['throw_r'][r][0] == 1 else 'L', 'pitches': int(len(r)), 'mix': mix,
                             'looks_in_ends_out': round(dm_in_true_out, 4), 'chase_rate_against': round(float(swing[r][outside[r]].mean()), 4) if outside[r].any() else None}
    res['pitchers'] = pitchers
    stage(f'pitchers {len(pitchers)}')
    # pairs: postseason hitters against the pitchers of every team they meet
    if rost and rost.get('teams'):
        team_of_h = {h: tid for tid, t in rost['teams'].items() for h in t['hitters']}
        opp = {}
        for a_, b_ in rost.get('meets', []):
            opp.setdefault(a_, set()).add(b_); opp.setdefault(b_, set()).add(a_)
        cidx = (Tk['balls'].astype(np.int64) * 3 + Tk['strikes'].astype(np.int64))
        sig = lambda v: 1 / (1 + np.exp(-v))
        def chain_from(ps_, pw_, pcs_, pfo_, ci_):
            use_c = np.bincount(ci_, minlength=12) >= 15; ki = ci_ % 3
            def agg(w):
                return np.where(use_c, np.bincount(ci_, weights=w, minlength=12), np.bincount(ki, weights=w, minlength=3)[np.arange(12) % 3])
            n_ = agg(np.ones(len(ps_))); s_ = agg(ps_); sw_ = agg(ps_ * pw_); t_ = agg(1 - ps_); tc_ = agg((1 - ps_) * pcs_)
            c_ = agg(ps_ * (1 - pw_)); cf_ = agg(ps_ * (1 - pw_) * pfo_)
            d = lambda v: {(b, k): float(v[b * 3 + k]) for b in range(4) for k in range(3)}
            return _count_chain(d(s_ / np.maximum(n_, 1e-9)), d(sw_ / np.maximum(s_, 1e-9)), d(tc_ / np.maximum(t_, 1e-9)), d(cf_ / np.maximum(c_, 1e-9)))
        rows = []
        cache = {}
        for h, tid in team_of_h.items():
            if h not in maps_s or h not in maps_w:
                continue
            sd = int(np.round(Tk['stand_r'][gb[h]].mean())) if h in gb else 1
            for ot in opp.get(int(tid), ()):
                for p_ in rost['teams'].get(ot, {}).get('pitchers', []):
                    if p_ not in gp or len(gp[p_]) < 300:
                        continue
                    key = (p_, sd)
                    if key not in cache:
                        r = gp[p_]; side = Tk['stand_r'][r] == sd
                        r = r[side] if side.sum() >= 100 else r
                        A = {'r': r, 'ci': cidx[r], 'pcs': p_cs[r], 'pfo': p_fo[r], 'os': off_s[r], 'ow': off_w[r], 'Bs': Bs[r], 'Bw': Bw[r], 'out': outside[r]}
                        A['league'] = chain_from(sig(A['os']), sig(A['ow']), A['pcs'], A['pfo'], A['ci'])
                        cache[key] = A
                    A = cache[key]
                    ph = sig(A['os'] + A['Bs'] @ maps_s[h]); pl = sig(A['os'])
                    hit = chain_from(ph, sig(A['ow'] + A['Bw'] @ maps_w[h]), A['pcs'], A['pfo'], A['ci'])
                    chase = float((ph - pl)[A['out']].mean()) if A['out'].any() else 0.0
                    zsw = float((ph - pl)[~A['out']].mean()) if (~A['out']).any() else 0.0
                    rows.append((int(h), int(p_), int(tid), int(ot), chase, zsw, hit[0] - A['league'][0], hit[1] - A['league'][1], A['league'][0], A['league'][1], int(len(A['r']))))
        if rows:
            Rr = np.asarray([r_[4:10] for r_ in rows], float)
            hid = np.asarray([r_[0] for r_ in rows]); pid = np.asarray([r_[1] for r_ in rows])
            def two_way(v, a, b, iters=12):
                ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
                na, nb = np.bincount(ia), np.bincount(ib); mu = float(v.mean()); ea = np.zeros(len(ua)); eb = np.zeros(len(ub))
                for _ in range(iters):
                    ea = np.bincount(ia, weights=v - mu - eb[ib]) / na; eb = np.bincount(ib, weights=v - mu - ea[ia]) / nb
                return ea[ia], eb[ib], v - mu - ea[ia] - eb[ib]
            hk, pk, rk = two_way(Rr[:, 2] * 100, hid, pid); hb_, pb_, rb = two_way(Rr[:, 3] * 100, hid, pid)
            res['pairs'] = [{'hitter': r_[0], 'pitcher': r_[1], 'team': r_[2], 'opponent': r_[3], 'arsenal_pitches': r_[10],
                             'chase_points': round(r_[4] * 100 * (1.0 if famv else 0.95), 2), 'zone_swing_points': round(r_[5] * 100, 2),
                             'k_points': round((r_[6] * 100) * 0.40, 2), 'bb_points': round((r_[7] * 100) * 0.53, 2),
                             'k_pair_points': round(float(rk[j]) * 0.40, 2), 'bb_pair_points': round(float(rb[j]) * 0.53, 2),
                             'league_chain_k': round(r_[8], 4), 'league_chain_bb': round(r_[9], 4)} for j, r_ in enumerate(rows)]
            res['pair_scale_note'] = (('chase from family maps against the league with its family part, unscaled (MATCHUP-05F; no pair calibration measured for family maps yet)' if famv else 'chase times 0.95 (MATCHUP-01F slope)')
                                      + '; strikeout and walk changes times 0.40 and 0.53 (ENGINE-01 coefficients over calibrated); pair parts after removing hitter and pitcher parts over these pairs')
        stage(f'pairs {len(rows)}')
        if params.get('aim') and rows and famv:
            # SCOUT-02 aiming (VALUE-08F): outside the zone only, by the hitter's own part (his map minus the same-side mean map),
            # among the pitcher's own spots of the same pitch group to that side and count group, under 0.6 ft of command
            # scatter, best third within thirds of the league's swing chance; runs per 100 plate appearances from VALUE-08F's
            # primary coefficient for family maps; aim cells by family where the aimed pitches arrive
            nL = Ls.shape[1]; cf_ = m_s.coef_[0].astype(np.float64); wL = cf_[:nL]; wB = cf_[nL:nL + nh]; wO = cf_[nL + nh:nL + 2 * nh]
            def league_loc(LBm, Bh_m, g):
                return LBm.astype(np.float64) @ wL + np.isin(g, (3, 4)) * (Bh_m[:, :-1] @ wB) + (g == 5) * (Bh_m[:, :-1] @ wO)
            b_out, n_out = -0.000657, 1.876
            K = 16; jit = np.random.default_rng(3).standard_normal((K, 2)); jit = (jit - jit.mean(0)) / jit.std(0); sgm = float(params.get('sigma', 0.6))
            cgrp = np.where(Tk['strikes'] == 2, 2, np.where(Tk['balls'] > Tk['strikes'], 1, 0))
            gu_ = np.asarray(res['grid']['side_ft']); gz_ = np.asarray(res['grid']['height_ft'])
            famof = lambda g: 'fastball' if g in (0, 1, 2, 6) else ('breaking' if g in (3, 4) else 'offspeed')
            pools = {}
            def pool_for(p_, sd, c3, tg):
                key = (p_, sd, c3, tg)
                if key in pools:
                    return pools[key]
                r = gp[p_]; r = r[(Tk['stand_r'][r] == sd) & (cgrp[r] == c3) & outside[r] & (np.clip(Tk['group'][r], 0, 6) == tg)]
                if len(r) < 15:
                    pools[key] = None; return None
                n_all = len(r)
                if len(r) > 250:
                    r = rng.choice(r, 250, replace=False)
                pl_ = sig(off_s[r]); third = np.searchsorted(np.percentile(pl_, [33.3, 66.7]), pl_)
                lb0 = league_loc(Ls[r], Bh0[r], Tk['group'][r])
                xj = (xp[r][:, None] + sgm * jit[None, :, 0]).ravel(); zj = (zp[r][:, None] + sgm * jit[None, :, 1]).ravel()
                sj = np.repeat(np.full(len(r), sd), K); kj = np.repeat(Tk['strikes'][r], K); gj = np.repeat(Tk['group'][r], K)
                Bj0 = hitter_basis(xj, zj, sj, kj)
                offj = np.repeat(off_s[r] - lb0, K) + league_loc(location_block(xj, zj, sj, kj), Bj0, gj)
                uu_ = np.where(sd == 1, xt[r], -xt[r])
                cell = np.argmin(np.abs(uu_[:, None] - gu_[None, :]), 1) + len(gu_) * np.argmin(np.abs(zt[r][:, None] - gz_[None, :]), 1)
                pools[key] = (offj, family_basis(Bj0, gj), third, cell, len(r), n_all)
                return pools[key]
            aim = []
            for pr_ in res['pairs']:
                h, p_ = int(pr_['hitter']), int(pr_['pitcher'])
                if h not in maps_s or p_ not in gp:
                    continue
                sd = int(np.round(Tk['stand_r'][gb[h]].mean())) if h in gb else 1
                rp = gp[p_][(Tk['stand_r'][gp[p_]] == sd) & outside[gp[p_]]]
                if len(rp) < 40:
                    continue
                tot = 0.0; wsum = 0.0; cells = {f_: np.zeros(len(gu_) * len(gz_)) for f_ in ('fastball', 'breaking', 'offspeed')}
                for c3 in (0, 1, 2):
                    for tg in range(7):
                        P_ = pool_for(p_, sd, c3, tg)
                        if P_ is None:
                            continue
                        offj, Bj, third, cell, npool, n_all = P_
                        dj = ((sig(offj + Bj @ maps_s[h]) - sig(offj + Bj @ mbar[h])) * 100).reshape(npool, K).mean(1)
                        gains = []
                        for t3 in range(3):
                            sel = np.flatnonzero(third == t3)
                            if len(sel) >= 3:
                                k3 = max(1, len(sel) // 3)
                                best = sel[np.argsort(dj[sel])[::-1][:k3]]
                                gains.append(float(dj[best].mean() - dj[sel].mean()))
                                np.add.at(cells[famof(tg)], cell[best], n_all / len(rp))
                        if gains:
                            tot += n_all * float(np.mean(gains)); wsum += n_all
                if wsum > 0:
                    g_ = tot / wsum
                    aim.append({'hitter': h, 'pitcher': p_, 'runs_per_100_pa': round(float(b_out * g_ * n_out) * 100, 2), 'gain_points': round(g_, 2),
                                'aim_cells': {f_: [[float(gu_[k % len(gu_)]), float(gz_[k // len(gu_)])] for k in np.argsort(c_)[::-1][:3] if c_[k] > 0] for f_, c_ in cells.items()}})
            res['aim'] = aim
            res['aim_note'] = ('runs per 100 plate appearances saved by aiming outside pitches (negative = fewer runs for the hitter) at 0.6 ft of command scatter, '
                               'by the hitter\'s own part of his family map, within each pitch group, coefficient from VALUE-08F (family maps, -0.000657 per point); '
                               'cells are where the aimed pitches arrive (true crossing), by family; in-zone aiming is not stated (its placebo failed)')
            stage(f'aim {len(aim)}')
        elif params.get('aim') and rows:
            # where each pitcher should aim against each hitter he may face (VALUE-02F): among his own spots to that side
            # and count group, the best third for this hitter under 0.6 ft of command scatter (outside: highest extra
            # chase; in the zone: lowest extra swing), within thirds of the league's swing chance; runs per 100 plate
            # appearances from the credited coefficients
            nL = Ls.shape[1]; wL = m_s.coef_[0][:nL].astype(np.float64)
            b_out, b_in, n_out, n_in = -0.000945, 0.000384, 1.876, 2.04
            K = 16; jit = np.random.default_rng(3).standard_normal((K, 2)); jit = (jit - jit.mean(0)) / jit.std(0); sgm = float(params.get('sigma', 0.6))
            cgrp = np.where(Tk['strikes'] == 2, 2, np.where(Tk['balls'] > Tk['strikes'], 1, 0))
            gu_ = np.asarray(res['grid']['side_ft']); gz_ = np.asarray(res['grid']['height_ft'])
            pools = {}
            def pool_for(p_, sd, c3, zone_out):
                key = (p_, sd, c3, zone_out)
                if key in pools:
                    return pools[key]
                r = gp[p_]; r = r[(Tk['stand_r'][r] == sd) & (cgrp[r] == c3) & (outside[r] == zone_out)]
                if len(r) < 20:
                    pools[key] = None; return None
                if len(r) > 250:
                    r = rng.choice(r, 250, replace=False)
                pl_ = sig(off_s[r]); third = np.searchsorted(np.percentile(pl_, [33.3, 66.7]), pl_)
                lb0 = Ls[r].astype(np.float64) @ wL
                xj = (xp[r][:, None] + sgm * jit[None, :, 0]).ravel(); zj = (zp[r][:, None] + sgm * jit[None, :, 1]).ravel()
                sj = np.repeat(np.full(len(r), sd), K); kj = np.repeat(Tk['strikes'][r], K)
                offj = np.repeat(off_s[r] - lb0, K) + location_block(xj, zj, sj, kj).astype(np.float64) @ wL
                uu_ = np.where(sd == 1, xt[r], -xt[r])                      # aim cells where the pitch arrives
                cell = np.argmin(np.abs(uu_[:, None] - gu_[None, :]), 1) + len(gu_) * np.argmin(np.abs(zt[r][:, None] - gz_[None, :]), 1)
                pools[key] = (offj, hitter_basis(xj, zj, sj, kj), third, cell, len(r))
                return pools[key]
            aim = []
            for pr_ in res['pairs']:
                h, p_ = int(pr_['hitter']), int(pr_['pitcher'])
                if h not in maps_s or p_ not in gp:
                    continue
                sd = int(np.round(Tk['stand_r'][gb[h]].mean())) if h in gb else 1
                rp = gp[p_][Tk['stand_r'][gp[p_]] == sd]
                if len(rp) < 60:
                    continue
                wc = np.bincount(cgrp[rp], minlength=3) / len(rp)
                runs = 0.0; cells_out = np.zeros(len(gu_) * len(gz_)); cells_in = np.zeros(len(gu_) * len(gz_)); used = 0.0
                for c3 in (0, 1, 2):
                    for zone_out in (True, False):
                        P_ = pool_for(p_, sd, c3, zone_out)
                        if P_ is None:
                            continue
                        offj, Bj, third, cell, npool = P_
                        dj = ((sig(offj + Bj @ maps_s[h]) - sig(offj)) * 100).reshape(npool, K).mean(1)
                        gains = []
                        for t3 in range(3):
                            sel = np.flatnonzero(third == t3)
                            if len(sel) >= 3:
                                k3 = max(1, len(sel) // 3)
                                order = np.argsort(dj[sel]); best = sel[order[::-1][:k3]] if zone_out else sel[order[:k3]]
                                gains.append(float(dj[best].mean() - dj[sel].mean()))
                                np.add.at(cells_out if zone_out else cells_in, cell[best], wc[c3])
                        if gains:
                            g_ = float(np.mean(gains))
                            runs += wc[c3] * (b_out * g_ * n_out if zone_out else b_in * g_ * n_in)
                            used += wc[c3] / 2
                if used > 0:
                    top_out = [int(k) for k in np.argsort(cells_out)[::-1][:3] if cells_out[k] > 0]
                    top_in = [int(k) for k in np.argsort(cells_in)[::-1][:3] if cells_in[k] > 0]
                    aim.append({'hitter': h, 'pitcher': p_, 'runs_per_100_pa': round(float(runs) * 100, 2),
                                'aim_outside_cells': [[float(gu_[k % len(gu_)]), float(gz_[k // len(gu_)])] for k in top_out],
                                'aim_strike_cells': [[float(gu_[k % len(gu_)]), float(gz_[k // len(gu_)])] for k in top_in]})
            res['aim'] = aim
            res['aim_note'] = 'runs per 100 plate appearances saved by aiming (negative = fewer runs for the hitter), 0.6 ft command scatter, coefficients from VALUE-02F; cells are where the aimed pitches arrive (true crossing), weighted by count group'
            stage(f'aim {len(aim)}')
    return res


# ---------------------------------------------------------------- matchup tables for the simulator
def matchup_pairs(T: dict, train_seasons, arsenal_season: int, target_pairs, params: dict, stage) -> dict:
    """{(batter, pitcher): (chase points, zone-swing points)} for the target pairs: hitter maps and the league map at
    the decision moment fitted on train_seasons, each pitcher's arsenal from arsenal_season (to the hitter's side when
    he threw at least 100 pitches to it). Nothing from the target season is used."""
    use = np.isin(T['season'], tuple(train_seasons) + (arsenal_season,))
    Tk = take(T, use)
    F = rebuild(Tk)
    keep = F['ok'] & (Tk['group'] >= 0) & (Tk['call'] <= 2) & (Tk['balls'] >= 0) & (Tk['strikes'] >= 0) & ~((Tk['bunt_pa'] == 1) & (Tk['last_in_pa'] == 1))
    Tk = take(Tk, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((Tk['call'] == 1) | (Tk['call'] == 2)).astype(np.float64)
    xp, zp = projected(Tk, F, None, 'straight', float(params.get('tau', 0.26)))
    C = np.hstack([control_block(Tk, swing_propensity(Tk)), pitcher_propensity(Tk)[:, None].astype(np.float32)])
    X = np.hstack([location_block(xp, zp, Tk['stand_r'], Tk['strikes']), C])
    tr = np.isin(Tk['season'], tuple(train_seasons))
    rng = np.random.default_rng(int(params.get('seed', 11)))
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), int(params.get('league_n', 600000))), replace=False)
    off = fit_logistic(X[idx], swing[idx]).decision_function(X)
    Bm = hitter_basis(xp, zp, Tk['stand_r'], Tk['strikes'])
    maps = _hitter_maps(Bm, swing, off, _groups(Tk['batter'], tr), float(params.get('lam', 10.0)), int(params.get('min_pitches', 300)))
    stage(f'maps {len(maps)}')
    u_true = np.where(Tk['stand_r'] == 1, Tk['px'], -Tk['px'])
    outside = (np.abs(u_true) > ZONE_HALF) | (Tk['pz'] > ZONE_TOP) | (Tk['pz'] < ZONE_BOT)
    ars = {}
    for pid, r in _groups(Tk['pitcher'], Tk['season'] == arsenal_season).items():
        if len(r) >= int(params.get('min_arsenal', 300)):
            ars[pid] = r if len(r) <= 1500 else rng.choice(r, 1500, replace=False)
    stage(f'arsenals {len(ars)}')
    out = {}
    for (b, p_, stand) in target_pairs:
        if b not in maps or p_ not in ars:
            continue
        r = ars[p_]
        side = Tk['stand_r'][r] == stand
        if side.sum() >= 100:
            r = r[side]
        dev = Bm[r] @ maps[b]
        pl = 1 / (1 + np.exp(-off[r])); ph = 1 / (1 + np.exp(-(off[r] + dev)))
        o = outside[r]
        out[(b, p_)] = (round(float((ph - pl)[o].mean()) * 100, 3) if o.any() else 0.0, round(float((ph - pl)[~o].mean()) * 100, 3) if (~o).any() else 0.0)
    stage(f'pairs {len(out)}')
    return out


def matchup_table(T: dict, params: dict, stage, store) -> dict:
    """Pair table for one target season (for replays): pairs that met in that season, from earlier seasons only."""
    target = int(params.get('target', 2025))
    prior = [int(s) for s in params.get('train', [s for s in (2023, 2024, 2025) if s < target])]
    pa = (T['season'] == target) & (T['pitch_no'] == 0)
    trip = np.unique(np.column_stack([T['batter'][pa], T['pitcher'][pa], T['stand_r'][pa]]), axis=0)
    pairs = matchup_pairs(T, prior, target - 1, [(int(a), int(b), int(c)) for a, b, c in trip], params, stage)
    doc = {'schema': 'brl.matchup-table.v1', 'target_season': target, 'train_seasons': prior, 'arsenal_season': target - 1,
           'units': 'points of swing rate (hitter map minus league), outside and inside the zone', 'pairs': [[b, p_, c, z] for (b, p_), (c, z) in pairs.items()]}
    path = store(target, doc)
    vals = np.asarray([v for v in pairs.values()]) if pairs else np.zeros((0, 2))
    return {'target_season': target, 'pairs_met': int(len(trip)), 'pairs_with_values': len(pairs), 'path': path,
            'chase_sd_points': round(float(vals[:, 0].std()), 3) if len(vals) else None, 'zone_swing_sd_points': round(float(vals[:, 1].std()), 3) if len(vals) else None}


# ---------------------------------------------------------------- DRIFT-01: swing maps that move before strikeouts and walks do
def drift_study(T: dict, params: dict, stage) -> dict:
    """Each hitter's swing map at the decision moment, refitted on each 30-day window, read on one fixed set of pitches
    outside the zone (so what he was thrown does not move it): a pitch-mix-free chase propensity. Does its change in a
    window, against his earlier windows, predict his strikeouts and walks in the next window beyond his recent and
    earlier strikeout and walk rates, and better than the raw chase rate's change? Train 2023-2024, score 2025."""
    res = {}
    post_seasons = tuple(int(x) for x in params.get('post_seasons', (2023, 2024, 2025)))
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) | (post_mode & (T['season'] >= 2023) & (T['season'] <= max(post_seasons))))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0)
    Tk = take(T, keep); F = {k: v[keep] for k, v in F.items()}
    swing = ((Tk['call'] == 1) | (Tk['call'] == 2)).astype(np.float64)
    xp, zp = projected(Tk, F, None, 'straight', float(params.get('tau', 0.26)))
    X = np.hstack([location_block(xp, zp, Tk['stand_r'], Tk['strikes']), control_block(Tk, np.zeros(len(swing)))])
    rng = np.random.default_rng(11)
    lg = np.flatnonzero(Tk['season'] == 2023); lg = rng.choice(lg, min(len(lg), 500000), replace=False)
    off = fit_logistic(X[lg], swing[lg]).decision_function(X)
    Bm = hitter_basis(xp, zp, Tk['stand_r'], Tk['strikes'])
    u = np.where(Tk['stand_r'] == 1, Tk['px'], -Tk['px'])
    outside = (np.abs(u) > ZONE_HALF) | (Tk['pz'] > ZONE_TOP) | (Tk['pz'] < ZONE_BOT)
    ref = rng.choice(np.flatnonzero(outside & (Tk['season'] == 2023)), 4000, replace=False)
    Bref, oref = Bm[ref], off[ref]
    p_ref0 = (1 / (1 + np.exp(-oref))).mean()
    # 30-day windows from each season's start
    start = {s: int(Tk['day'][Tk['season'] == s].min()) for s in (2023, 2024, 2025)}
    win = np.asarray([(int(d) - start[int(s)]) // 30 for d, s in zip(Tk['day'], Tk['season'])])
    lam = float(params.get('lam', 30.0)); min_n = int(params.get('min_pitches', 150))
    rows = {}
    key = Tk['batter'].astype(np.int64) * 100000 + (Tk['season'].astype(np.int64) - 2000) * 100 + win
    for k, r in _groups(key, np.ones(len(key), bool)).items():
        if len(r) < min_n:
            continue
        b = _ridge_offset(Bm[r], swing[r], off[r], lam)
        chase_map = float((1 / (1 + np.exp(-(oref + Bref @ b)))).mean() - p_ref0)
        o = outside[r]
        rows[k] = (chase_map, float(swing[r][o].mean()) if o.any() else np.nan, int(len(r)))
    stage(f'window maps {len(rows)}')
    # plate appearances with their window and outcomes
    P = take(T, (T['pitch_no'] == 0) & (T['out7'] >= 0))
    wp = np.asarray([(int(d) - start[int(s)]) // 30 for d, s in zip(P['day'], P['season'])])
    kp = P['batter'].astype(np.int64) * 100000 + (P['season'].astype(np.int64) - 2000) * 100 + wp
    y7 = P['out7'].astype(int); K = (y7 == 1).astype(float); BB = (y7 == 2).astype(float)
    # window rates of K and BB per hitter-window
    uk, inv = np.unique(kp, return_inverse=True)
    nk = np.bincount(inv); kk = np.bincount(inv, weights=K); bb = np.bincount(inv, weights=BB)
    win_rate = {int(k_): (kk[i] / nk[i], bb[i] / nk[i], nk[i]) for i, k_ in enumerate(uk)}
    feats = []
    for i in range(len(y7)):
        k_now = int(kp[i]); w = int(wp[i])
        if w < 2:
            continue
        prev = k_now - 1
        earlier = [k_now - j for j in range(2, w + 1)]
        if prev not in rows or prev not in win_rate or not all(e in rows for e in earlier[:1]):
            continue
        cm_prev, cr_prev, _ = rows[prev]
        cm_old = np.mean([rows[e][0] for e in earlier if e in rows]); cr_old = np.nanmean([rows[e][1] for e in earlier if e in rows])
        kr, br, n_ = win_rate[prev]
        ko = [win_rate[e] for e in earlier if e in win_rate]
        if not ko:
            continue
        ko_k = sum(a[0] * a[2] for a in ko) / sum(a[2] for a in ko); ko_b = sum(a[1] * a[2] for a in ko) / sum(a[2] for a in ko)
        feats.append((i, kr, br, ko_k, ko_b, cm_prev - cm_old, cr_prev - cr_old))
    Fm = np.asarray(feats, float)
    idx = Fm[:, 0].astype(int)
    lgt = lambda v: np.log(np.clip(v, 0.02, 0.98) / (1 - np.clip(v, 0.02, 0.98)))
    season = P['season'][idx]; games = P['game'][idx]
    out = {'plate_appearances': int(len(idx)), 'map_change_sd_points': round(float(np.std(Fm[:, 5]) * 100), 3), 'raw_chase_change_sd_points': round(float(np.nanstd(Fm[:, 6]) * 100), 3),
           'corr_map_change_raw_change': round(float(np.corrcoef(Fm[:, 5], np.nan_to_num(Fm[:, 6]))[0, 1]), 3)}
    from sklearn.linear_model import LogisticRegression
    tr, te = np.isin(season, (2023, 2024)), season == 2025
    for name, y, rcols in (('strikeout', K[idx], (1, 3)), ('walk', BB[idx], (2, 4))):
        base = np.column_stack([lgt(Fm[:, rcols[0]]), lgt(Fm[:, rcols[1]])])
        designs = {'rates': base, 'rates_raw_chase_change': np.column_stack([base, np.nan_to_num(Fm[:, 6]) * 100]),
                   'rates_map_change': np.column_stack([base, Fm[:, 5] * 100]), 'rates_both': np.column_stack([base, Fm[:, 5] * 100, np.nan_to_num(Fm[:, 6]) * 100])}
        ll = {}; coefs = {}
        for nm, Xd in designs.items():
            m = LogisticRegression(C=1e4, max_iter=500).fit(Xd[tr], y[tr])
            ll[nm] = logloss_vec(m.predict_proba(Xd[te])[:, 1], y[te]); coefs[nm] = [round(float(c), 4) for c in m.coef_[0]]
        g = games[te]
        out[name] = {'gain_raw_chase_change': [round(v * 1000, 3) for v in clustered_ci(ll['rates'] - ll['rates_raw_chase_change'], g)],
                     'gain_map_change': [round(v * 1000, 3) for v in clustered_ci(ll['rates'] - ll['rates_map_change'], g)],
                     'map_over_raw': [round(v * 1000, 3) for v in clustered_ci(ll['rates_raw_chase_change'] - ll['rates_both'], g)],
                     'coefs': coefs}
    res.update(out)
    stage('drift')
    return res


# ---------------------------------------------------------------- FATIGUE-01: the pitcher's state inside the game
LW7 = np.array([-0.26, -0.28, 0.32, 0.47, 0.78, 1.40, 0.45])


def _prior_by_day(key, day, value, use):
    """Count and sum of value over rows of the same key on earlier days where use."""
    n = len(key); order = np.lexsort((day, key)); k, d = key[order], day[order]
    u = use[order].astype(np.float64); v = np.where(use[order], np.nan_to_num(value[order]), 0.0)
    cu, cv = np.cumsum(u), np.cumsum(v)
    first = np.ones(n, bool); first[1:] = (k[1:] != k[:-1]) | (d[1:] != d[:-1])
    start = np.ones(n, bool); start[1:] = k[1:] != k[:-1]
    i_f = np.maximum.accumulate(np.where(first, np.arange(n), 0)); i_k = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    b = lambda c: np.where(i_f > 0, c[i_f - 1], 0.0) - np.where(i_k > 0, c[i_k - 1], 0.0)
    on, ov = np.empty(n), np.empty(n); on[order], ov[order] = b(cu), b(cv)
    return on, ov


def fatigue(T: dict, params: dict, stage) -> dict:
    """Does the starter's measured physical drift inside the game (fastball speed, release height, spin, extension over
    his last eight fastballs against his first ten) predict the next plate appearances beyond pitch count, times
    through the order and both players' rates? And does a start that begins below his usual speed run worse all game?
    Starters only; train 2023-2024, score 2025 and 2026 through July."""
    res = {}
    T = take(T, (T['day'] < date(2026, 8, 1).toordinal()) & (T['out7'] >= -1))
    order = np.lexsort((T['pitch_no'], T['ab'], T['pitcher'], T['game']))
    T = take(T, order)
    gp = T['game'].astype(np.int64) * 10_000_000 + T['pitcher'].astype(np.int64)
    n = len(gp)
    start = np.ones(n, bool); start[1:] = gp[1:] != gp[:-1]
    gidx = np.cumsum(start) - 1
    first_row = np.flatnonzero(start)
    pos = np.arange(n) - first_row[gidx]                         # pitches thrown in this game before this one
    # starters: the first pitcher of each team in each game
    gh = T['game'].astype(np.int64) * 2 + T['half']
    o2 = np.lexsort((T['pitch_no'], T['ab'], gh))
    starter_of = {}
    ghs = gh[o2]; ps = T['pitcher'][o2]
    keep_first = np.r_[True, ghs[1:] != ghs[:-1]]
    for g_, p_ in zip(ghs[keep_first], ps[keep_first]):
        starter_of[int(g_)] = int(p_)
    is_starter = np.asarray([starter_of.get(int(g_), -1) == int(p_) for g_, p_ in zip(gh, T['pitcher'])])
    fb = np.isin(T['group'], (0, 1)) & np.isfinite(T['v0'])
    feats = {'v0': T['v0'].astype(np.float64), 'z0': T['z0'].astype(np.float64), 'spin': T['spin'].astype(np.float64), 'ext': T['ext'].astype(np.float64)}
    # cumulative fastball sums within each game-pitcher group, before each pitch
    def before_cum(x, mask):
        c = np.cumsum(np.where(mask, x, 0.0)); c0 = c - np.where(mask, x, 0.0)
        return c0 - (c[first_row] - np.where(mask, x, 0.0)[first_row])[gidx]
    nfb = before_cum(np.ones(n), fb)
    pa_start = (T['pitch_no'] == 0) & (T['out7'] >= 0) & is_starter
    idx = np.flatnonzero(pa_start)
    # fastball sequences per group, to read the first ten and the last eight before each plate appearance
    fb_rows = np.flatnonzero(fb)
    fb_g = gidx[fb_rows]
    fb_first = np.searchsorted(fb_g, np.arange(gidx[-1] + 1), side='left')
    out = {name: np.full(len(idx), np.nan) for name in ('drift_v', 'drift_z', 'drift_spin', 'drift_ext', 'start_v')}
    cums = {name: np.r_[0.0, np.cumsum(np.nan_to_num(feats[name][fb_rows]))] for name in feats}
    fcnt = np.r_[0.0, np.cumsum(np.isfinite(feats['spin'][fb_rows]).astype(float))]
    for j, i in enumerate(idx):
        k = int(nfb[i]); g0 = fb_first[gidx[i]]
        if k >= 10:
            out['start_v'][j] = (cums['v0'][g0 + 10] - cums['v0'][g0]) / 10.0
        if k >= 18:
            for name, key in (('drift_v', 'v0'), ('drift_z', 'z0'), ('drift_spin', 'spin'), ('drift_ext', 'ext')):
                base = (cums[key][g0 + 10] - cums[key][g0]) / 10.0
                last = (cums[key][g0 + k] - cums[key][g0 + k - 8]) / 8.0
                out[name][j] = last - base
    stage('drift')
    P = {k: v[idx] for k, v in T.items()}
    P['pitch_count'] = pos[idx].astype(np.float64)
    # times this batter has faced this starter earlier in the game
    bk = P['game'].astype(np.int64) * 10_000_000 + P['batter'].astype(np.int64)
    o3 = np.lexsort((P['ab'], bk)); tto = np.zeros(len(idx))
    bks = bk[o3]; st3 = np.r_[True, bks[1:] != bks[:-1]]; grp3 = np.cumsum(st3) - 1
    tto[o3] = np.arange(len(o3)) - np.flatnonzero(st3)[grp3]
    # the starter's usual fastball speed: his fastballs on earlier days
    nv, sv_ = _prior_by_day(T['pitcher'].astype(np.int64), T['day'].astype(np.int64), feats['v0'], fb)
    norm_v = np.where(nv[idx] >= 50, sv_[idx] / np.maximum(nv[idx], 1), np.nan)
    today = out['start_v'] - norm_v
    # both players' earlier rates by outcome class (shrunk shares, as log ratios to the league)
    y7 = P['out7'].astype(int)
    league = np.bincount(y7, minlength=7) / len(y7)
    def rates(key, k=150.0):
        cols_ = []
        for c in range(7):
            nn, ss = _prior_by_day(key, P['day'].astype(np.int64), (y7 == c).astype(float), np.ones(len(y7), bool))
            cols_.append(np.log((ss + k * league[c]) / (nn + k) / league[c]))
        return np.column_stack(cols_)
    rb = rates(P['batter'].astype(np.int64)); rp = rates(P['pitcher'].astype(np.int64), 300.0)
    stage('rates')
    pc = hats(P['pitch_count'], (0, 15, 30, 45, 60, 75, 90, 105, 120))
    tto_d = np.column_stack([(tto == 1), (tto >= 2)]).astype(float)
    plat = (P['stand_r'] == P['throw_r']).astype(float)[:, None]
    have = np.isfinite(out['drift_v'])
    D = np.column_stack([np.where(have, out[k], 0.0) for k in ('drift_v', 'drift_z', 'drift_spin', 'drift_ext')] + [have.astype(float)])
    D[:, 2] /= 100.0
    tv = np.isfinite(today)
    TD = np.column_stack([np.where(tv, today, 0.0), tv.astype(float)])
    X0 = np.hstack([pc, tto_d, rb, rp, plat])
    designs = {'M0_count_tto_rates': X0, 'M1_plus_drift': np.hstack([X0, D]), 'M2_plus_drift_and_start': np.hstack([X0, D, TD])}
    cmd = bool(params.get('command'))              # COMMAND-01: the starter's location in his first 20 pitches against his norm
    if cmd:
        ut_ = np.where(T['stand_r'] == 1, T['px'].astype(np.float64), -T['px'].astype(np.float64)); zt_ = T['pz'].astype(np.float64)
        e_ = np.maximum(np.maximum(np.abs(ut_) - ZONE_HALF, zt_ - ZONE_TOP), ZONE_BOT - zt_)
        okloc = np.isfinite(e_); inz = okloc & (e_ <= 0); waste = okloc & (e_ > 0.5)
        cz = np.r_[0.0, np.cumsum(inz.astype(float))]; cw = np.r_[0.0, np.cumsum(waste.astype(float))]; cn = np.r_[0.0, np.cumsum(okloc.astype(float))]
        fr = first_row[gidx[idx]]; late = pos[idx] >= 20
        f20 = np.where(late, fr + 20, fr)
        n20 = cn[f20] - cn[fr]; z20 = (cz[f20] - cz[fr]) / np.maximum(n20, 1); w20 = (cw[f20] - cw[fr]) / np.maximum(n20, 1)
        nz_, sz_ = _prior_by_day(T['pitcher'].astype(np.int64), T['day'].astype(np.int64), inz.astype(float), okloc)
        nw_, sw2_ = _prior_by_day(T['pitcher'].astype(np.int64), T['day'].astype(np.int64), waste.astype(float), okloc)
        normz = np.where(nz_[idx] >= 300, sz_[idx] / np.maximum(nz_[idx], 1), np.nan); normw = np.where(nw_[idx] >= 300, sw2_[idx] / np.maximum(nw_[idx], 1), np.nan)
        have_c = late & (n20 >= 15) & np.isfinite(normz) & np.isfinite(normw)
        zdev = np.where(have_c, z20 - normz, np.nan); wdev = np.where(have_c, w20 - normw, np.nan)
        CM = np.column_stack([np.where(have_c, zdev, 0.0) * 10, np.where(have_c, wdev, 0.0) * 10, have_c.astype(float)])
        designs['M3_plus_command'] = np.hstack([X0, D, TD, CM])
        res['command_rows'] = {'plate_appearances_with_command': int(have_c.sum()), 'zone_dev_sd_points': round(float(np.nanstd(zdev) * 100), 2),
                               'waste_dev_sd_points': round(float(np.nanstd(wdev) * 100), 2)}
    tr = np.isin(P['season'], (2023, 2024))
    tests = {'2025': P['season'] == 2025, '2026_through_july': P['season'] == 2026}
    from sklearn.linear_model import LogisticRegression
    res['rows'] = {'plate_appearances': int(len(idx)), 'with_drift': int(have.sum()), 'with_start_vs_norm': int(tv.sum()),
                   'drift_v_sd': round(float(np.nanstd(out['drift_v'])), 3), 'drift_v_mean': round(float(np.nanmean(out['drift_v'])), 3),
                   'start_vs_norm_sd': round(float(np.nanstd(today)), 3)}
    fits = {}
    for name, X in designs.items():
        m = LogisticRegression(max_iter=500, C=10.0).fit(X[tr], y7[tr]); fits[name] = m
    stage('multinomial fits')
    res['test_logloss_gain_nats_per_1000_pa'] = {}
    for tname, tm in tests.items():
        if not tm.any():
            continue
        ll = {k: -np.log(np.clip(m.predict_proba(designs[k][tm])[np.arange(tm.sum()), y7[tm]], 1e-9, 1)) for k, m in fits.items()}
        g = P['game'][tm]
        res['test_logloss_gain_nats_per_1000_pa'][tname] = {
            'drift': [round(v * 1000, 3) for v in clustered_ci(ll['M0_count_tto_rates'] - ll['M1_plus_drift'], g)],
            'start_vs_norm': [round(v * 1000, 3) for v in clustered_ci(ll['M1_plus_drift'] - ll['M2_plus_drift_and_start'], g)],
            'plate_appearances': int(tm.sum())}
        if cmd:
            res['test_logloss_gain_nats_per_1000_pa'][tname]['command'] = [round(v * 1000, 3) for v in clustered_ci(ll['M2_plus_drift_and_start'] - ll['M3_plus_command'], g)]
    # run value per plate appearance on the same features (least squares), coefficients with game-clustered intervals
    rv = LW7[y7]
    names = [f'pc{i}' for i in range(pc.shape[1])] + ['tto2', 'tto3plus'] + [f'b{c}' for c in range(7)] + [f'p{c}' for c in range(7)] + ['platoon',
             'drift_v', 'drift_z', 'drift_spin100', 'drift_ext', 'has_drift', 'start_vs_norm_v', 'has_start'] + (['zone_rate_dev_per_10pts', 'waste_rate_dev_per_10pts', 'has_command'] if cmd else [])
    X = designs['M3_plus_command'] if cmd else designs['M2_plus_drift_and_start']
    def ols(rows):
        A = np.column_stack([np.ones(rows.sum()), X[rows]]); return np.linalg.lstsq(A, rv[rows], rcond=None)[0][1:]
    allr = tr | tests['2025'] | tests['2026_through_july']
    beta = ols(allr)
    games_all = np.unique(P['game'][allr]); rng = np.random.default_rng(5); draws = []
    gi = np.searchsorted(games_all, P['game'][allr]); rows_all = np.flatnonzero(allr)
    for _ in range(int(params.get('reps', 150))):
        pick = rng.integers(0, len(games_all), len(games_all)); w = np.bincount(pick, minlength=len(games_all))[gi]
        sel = np.repeat(rows_all, w)
        A = np.column_stack([np.ones(len(sel)), X[sel]]); draws.append(np.linalg.lstsq(A, rv[sel], rcond=None)[0][1:])
    draws = np.asarray(draws)
    keep_n = ('tto2', 'tto3plus', 'drift_v', 'drift_z', 'drift_spin100', 'drift_ext', 'start_vs_norm_v') + (('zone_rate_dev_per_10pts', 'waste_rate_dev_per_10pts') if cmd else ())
    res['run_value_per_pa'] = {nm: [round(float(beta[names.index(nm)]), 5), round(float(np.percentile(draws[:, names.index(nm)], 2.5)), 5),
                                    round(float(np.percentile(draws[:, names.index(nm)], 97.5)), 5)] for nm in keep_n}
    # how much of the times-through-the-order penalty the measured drift accounts for
    X0r = designs['M0_count_tto_rates'][allr]; A0 = np.column_stack([np.ones(allr.sum()), X0r]); b0 = np.linalg.lstsq(A0, rv[allr], rcond=None)[0][1:]
    i2, i3 = pc.shape[1], pc.shape[1] + 1
    res['tto_without_drift'] = {'tto2': round(float(b0[i2]), 5), 'tto3plus': round(float(b0[i3]), 5)}
    res['tto_with_drift'] = {'tto2': round(float(beta[i2]), 5), 'tto3plus': round(float(beta[i3]), 5)}
    stage('run value')
    return res


# ---------------------------------------------------------------- SEQ-01: how unexpected a pitch is to the hitter
def surprise_study(T: dict, params: dict, stage) -> dict:
    """The hitter can know the pitcher's habits: how often he throws each pitch type in this count bucket, after this
    previous pitch, to this side. Surprise = -log of that probability for the pitch actually thrown (the pitcher's
    earlier days, shrunk toward the league in the same context). Does surprise predict swings, whiffs on swings and
    called strikes on takes beyond the pitch's own physics, location, count and the hitter's habits? Train 2023-2024,
    score 2025 (feed locations, one reference)."""
    res = {}
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['strikes'] >= 0))
    order = np.lexsort((T['pitch_no'], T['ab'], T['game']))
    T = take(T, order)
    n = len(T['game'])
    # previous pitch type within the plate appearance (6 = none)
    same_pa = np.r_[False, (T['game'][1:] == T['game'][:-1]) & (T['ab'][1:] == T['ab'][:-1])]
    prev = np.where(same_pa, np.r_[6, np.clip(T['group'][:-1], 0, 5)], 6)
    b, k = T['balls'], T['strikes']
    bucket = np.select([(b == 0) & (k == 0), (k > b) & ~((b == 3) & (k == 2)), (b == k) & (b > 0), (b == 3) & (k == 2)], [0, 1, 2, 4], 3)
    hand = (T['stand_r'] == T['throw_r']).astype(np.int64)
    ctx = (bucket * 7 + prev) * 2 + hand                                   # 70 contexts
    g6 = np.clip(T['group'], 0, 5)
    key = T['pitcher'].astype(np.int64) * 100 + ctx
    day = T['day'].astype(np.int64)
    train_rows = np.isin(T['season'], (2023, 2024))
    lc = np.zeros((70, 6))
    for c in range(70):
        m = train_rows & (ctx == c)
        lc[c] = (np.bincount(g6[m], minlength=6) + 1.0) / (m.sum() + 6.0)
    kk = float(params.get('k_usage', 60.0))
    probs = np.zeros((n, 6))
    tot_n = None
    for g in range(6):
        nn, ss = _prior_by_day(key, day, (g6 == g).astype(float), np.ones(n, bool))
        probs[:, g] = ss; tot_n = nn
    probs = (probs + kk * lc[ctx]) / (tot_n[:, None] + kk)
    p_act = probs[np.arange(n), g6]
    surprise = -np.log(np.clip(p_act, 1e-4, 1))
    entropy = -(probs * np.log(np.clip(probs, 1e-9, 1))).sum(1)
    stage('usage priors')
    res['surprise'] = {'mean_bits': round(float(surprise.mean() / np.log(2)), 3), 'sd_bits': round(float(surprise.std() / np.log(2)), 3),
                       'entropy_mean_bits': round(float(entropy.mean() / np.log(2)), 3)}
    swing = (T['call'] == 1) | (T['call'] == 2); whiff = T['call'] == 2; take_ = T['call'] == 0
    u = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    prop = swing_propensity(T)
    X0 = np.column_stack([T['v0'], T['pfx_x'] * np.where(T['throw_r'] == 1, 1, -1), T['pfx_z'], u, T['pz'], T['spin'], T['x0'] * np.where(T['throw_r'] == 1, 1, -1),
                          T['z0'], T['ext'], b, k, g6, hand, prop, prev])
    X1 = np.column_stack([X0, surprise, entropy, probs])
    from sklearn.ensemble import HistGradientBoostingClassifier
    hp = dict(max_iter=int(params.get('gbm_iter', 300)), learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=300, l2_regularization=1.0, random_state=11)
    tr = train_rows; te = T['season'] == 2025
    rng = np.random.default_rng(11)
    out = {}
    for name, rows, y in (('swing', np.ones(n, bool), swing), ('whiff_on_swing', swing, whiff), ('called_strike_on_take', take_, T['cs'] == 1)):
        a = np.flatnonzero(tr & rows); a = rng.choice(a, min(len(a), int(params.get('train_n', 1200000))), replace=False)
        t_ = te & rows
        yy = y.astype(float)
        m0 = HistGradientBoostingClassifier(**hp).fit(X0[a], yy[a]); p0 = m0.predict_proba(X0[t_])[:, 1]
        m1 = HistGradientBoostingClassifier(**hp).fit(X1[a], yy[a]); p1 = m1.predict_proba(X1[t_])[:, 1]
        l0, l1 = logloss_vec(p0, yy[t_]), logloss_vec(p1, yy[t_])
        q = np.nanpercentile(surprise[t_], [20, 40, 60, 80]); qi = np.searchsorted(q, surprise[t_])
        out[name] = {'test_rows': int(t_.sum()), 'gain_nats_per_1000': [round(v * 1000, 3) for v in clustered_ci(l0 - l1, T['game'][t_])],
                     'by_surprise_quintile': [{'surprise_bits': round(float(surprise[t_][qi == j].mean() / np.log(2)), 3), 'observed': round(float(yy[t_][qi == j].mean()), 4),
                                               'predicted_without': round(float(p0[qi == j].mean()), 4)} for j in range(5)]}
        stage('surprise ' + name)
    res['models'] = out
    return res


# ---------------------------------------------------------------- SEQ-02: does the carried timing after a called strike reach the outcome?
def seq2_study(T: dict, params: dict, stage) -> dict:
    """SEQ-02. TIMING-03F (credited): after a called strike the hitter meets the next pitch as if expecting the other
    family (contact farther out front the slower that strike was); after a miss his swing's timing persists; a foul
    carries nothing. If that reaches outcomes, then on the pitch after a called strike, repeating the previous family
    should draw more misses than changing it, beyond what the pitch itself predicts, and the same contrast after a
    taken ball (no carried timing) should be smaller. Swings with a previous pitch in the plate appearance (feed,
    seasons by params). Whiff on a swing: logistic model with the pitch's location on the true crossing (whiff
    representation), group, speed, two strikes, both players' earlier whiff levels, platoon, the count, the previous
    pitch's family and this pitch's family, and the sequence terms: previous outcome kind (ball, called strike, foul,
    miss) times repeat-family and times the speed change (this pitch minus the previous, per 10 mph). Fitted on the
    training seasons; coefficients with game-bootstrap intervals; out-of-sample log loss gain of the sequence terms on
    the test seasons (clustered by game); the observed whiff rate after called strikes, repeat against change, within
    bins of the model's prediction without the sequence terms. Contact quality the same way (launch speed on balls
    in play, OLS). Flexible check: boosting with and without the sequence terms."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    res = {}
    final = bool(params.get('final_eval'))           # SEQ-03F: trained on everything before August 1, 2026, scored once on the untouched months
    if final and not params.get('frozen_commit'):
        raise ValueError('seq2_study scores the untouched months only as the registered evaluation of a frozen commit')
    train = tuple(int(v) for v in params.get('train', (2023, 2024))); tests = tuple(int(v) for v in params.get('tests', (2025, 2026)))
    if final:
        train = (2023, 2024, 2025, 2026); tests = (2026,)
    T = take(T, np.isin(T['season'], train + tests) & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2))
    if final:
        T = take(T, T['post'] == 0)
    order = np.lexsort((T['pitch_no'], T['ab'], T['game'])); T = take(T, order)
    n = len(T['game'])
    test_from = date(2026, 8, 1).toordinal()
    same_pa = np.r_[False, (T['game'][1:] == T['game'][:-1]) & (T['ab'][1:] == T['ab'][:-1])]
    fam = np.where(np.isin(T['group'], (0, 1, 2)), 0, np.where(np.isin(T['group'], (3, 4)), 1, 2))
    prev_fam = np.r_[-1, fam[:-1]]; prev_v0 = np.r_[np.nan, T['v0'][:-1].astype(np.float64)]
    prev_call = np.r_[-1, T['call'][:-1]]; prev_cs = np.r_[0, T['cs'][:-1]]
    # previous outcome kind: 0 ball, 1 called strike, 2 foul (contact that did not end the previous pitch's plate appearance), 3 miss
    prev_kind = np.where(prev_call == 0, np.where(prev_cs == 1, 1, 0), np.where(prev_call == 2, 3, 2))
    swing = (T['call'] == 1) | (T['call'] == 2); whiff = (T['call'] == 2).astype(np.float64)
    v0 = T['v0'].astype(np.float64)
    ok = same_pa & swing & np.isfinite(v0) & np.isfinite(prev_v0) & (prev_fam >= 0)
    dv = np.where(ok, (v0 - prev_v0) / 10.0, 0.0); repeat = (fam == prev_fam).astype(np.float64)
    # both players' earlier whiff levels (prior dates, shrunk)
    lg_w = float(whiff[swing].mean())
    def prop(key, k_):
        nn, ss = _prior_by_day(key.astype(np.int64), T['day'].astype(np.int64), whiff, swing)
        r_ = (ss + k_ * lg_w) / (nn + k_); return np.log(r_ / (1 - r_))
    pw_b = prop(T['batter'], 150.0); pw_p = prop(T['pitcher'], 300.0)
    xt, zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
    Lw = location_block(xt, zt, T['stand_r'], T['strikes'])
    grp = np.zeros((n, 7), np.float32); grp[np.arange(n), np.clip(T['group'], 0, 6)] = 1
    cnt = np.zeros((n, 12), np.float32); cnt[np.arange(n), np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)] = 1
    pf = np.zeros((n, 3), np.float32); pf[np.arange(n), np.clip(prev_fam, 0, 2)] = 1
    pk = np.zeros((n, 4), np.float32); pk[np.arange(n), prev_kind] = 1
    base = np.hstack([Lw, grp, hats(v0, V_KNOTS), (T['strikes'] == 2)[:, None].astype(np.float32), pw_b[:, None].astype(np.float32), pw_p[:, None].astype(np.float32),
                      (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32), cnt, pf, pk])
    # the sequence terms: for each previous kind, repeat and the speed change
    seq_names = []; seq_cols = []
    for kname, kv in (('ball', 0), ('called_strike', 1), ('foul', 2), ('miss', 3)):
        m = (prev_kind == kv).astype(np.float64)
        seq_cols += [m * repeat, m * dv]; seq_names += [f'{kname}_x_repeat', f'{kname}_x_speed_change_per10']
    S = np.column_stack(seq_cols).astype(np.float32)
    X0 = base; X1 = np.hstack([base, S])
    tr = ok & ((T['day'] < test_from) if final else np.isin(T['season'], train))
    rng = np.random.default_rng(int(params.get('seed', 11)))
    idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), int(params.get('train_n', 900000))), replace=False)
    stage(f'features: {int(ok.sum())} swings with a previous pitch, train {len(idx)}')
    m1 = LogisticRegression(C=1.0, max_iter=400, tol=1e-6).fit(X1[idx], whiff[idx]); m0 = LogisticRegression(C=1.0, max_iter=400, tol=1e-6).fit(X0[idx], whiff[idx])
    cs = m1.coef_[0][-len(seq_names):].astype(np.float64)
    # game bootstrap of the sequence coefficients (refit on resampled games, a modest number of draws)
    games_tr = T['game'][idx]; ug = np.unique(games_tr); gi = np.searchsorted(ug, games_tr); draws = []
    for _ in range(int(params.get('reps', 30))):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi]
        mb = LogisticRegression(C=1.0, max_iter=300, tol=1e-5, warm_start=False).fit(X1[idx], whiff[idx], sample_weight=w)
        draws.append(mb.coef_[0][-len(seq_names):])
    draws = np.asarray(draws)
    res['sequence_coefficients_logodds'] = {nm: [round(float(cs[i]), 4), round(float(np.percentile(draws[:, i], 2.5)), 4), round(float(np.percentile(draws[:, i], 97.5)), 4)] for i, nm in enumerate(seq_names)}
    i_cs, i_b, i_f = seq_names.index('called_strike_x_repeat'), seq_names.index('ball_x_repeat'), seq_names.index('foul_x_repeat')
    d_ = draws[:, i_cs] - draws[:, i_b]
    res['called_strike_minus_ball_repeat'] = [round(float(cs[i_cs] - cs[i_b]), 4), round(float(np.percentile(d_, 2.5)), 4), round(float(np.percentile(d_, 97.5)), 4)]
    d2 = draws[:, i_cs] - draws[:, i_f]
    res['called_strike_minus_foul_repeat'] = [round(float(cs[i_cs] - cs[i_f]), 4), round(float(np.percentile(d2, 2.5)), 4), round(float(np.percentile(d2, 97.5)), 4)]
    stage('coefficients')
    # out of sample
    res['test'] = {}
    for s_ in tests:
        te = ok & ((T['day'] >= test_from) if final else (T['season'] == s_))
        if te.sum() < 5000:
            continue
        p0 = m0.predict_proba(X0[te])[:, 1]; p1 = m1.predict_proba(X1[te])[:, 1]; y = whiff[te]
        l0, l1 = logloss_vec(p0, y), logloss_vec(p1, y)
        out = {'swings': int(te.sum()), 'gain_nats_per_1000_swings': [round(v * 1000, 3) for v in clustered_ci(l0 - l1, T['game'][te])]}
        # observed whiff after called strikes, repeat against change, within bins of the no-sequence prediction
        pkt = prev_kind[te]; rp = repeat[te]
        for kname, kv in (('after_called_strike', 1), ('after_ball', 0), ('after_foul', 2), ('after_miss', 3)):
            mk_ = pkt == kv
            if mk_.sum() < 2000:
                continue
            q = np.percentile(p0[mk_], [25, 50, 75]); qb = np.searchsorted(q, p0[mk_])
            rows = []
            for j in range(4):
                for r_, rname in ((1.0, 'repeat'), (0.0, 'change')):
                    sel = mk_.copy(); sel[mk_] = (qb == j) & (rp[mk_] == r_)
                    if sel.sum() >= 200:
                        rows.append({'bin': j, 'kind': rname, 'n': int(sel.sum()), 'observed': round(float(y[sel].mean()), 4), 'predicted_no_sequence': round(float(p0[sel].mean()), 4)})
            # the contrast: observed minus predicted, repeat minus change, pooled over bins (weighted by swings)
            def omp(r_):
                sel = mk_ & (rp == r_); return float((y[sel] - p0[sel]).mean()) if sel.sum() else float('nan')
            out[kname] = {'repeat_minus_change_observed_minus_predicted': round(omp(1.0) - omp(0.0), 4), 'swings_repeat': int((mk_ & (rp == 1)).sum()), 'swings_change': int((mk_ & (rp == 0)).sum()), 'bins': rows}
        res['test'][str(s_)] = out
        stage(f'test {s_}')
    # contact quality: launch speed on balls in play (the last pitch), same design, OLS
    ls = T['ls'].astype(np.float64); inplay = ok & np.isfinite(ls) & (T['last_in_pa'] == 1) & (T['call'] == 1)
    trc = inplay & ((T['day'] < test_from) if final else np.isin(T['season'], train))
    if trc.sum() > 20000:
        A1 = np.column_stack([np.ones(int(trc.sum())), X1[trc]]); b1 = np.linalg.lstsq(A1, ls[trc], rcond=None)[0]
        cq = b1[-len(seq_names):]
        res['launch_speed_coefficients_mph'] = {nm: round(float(cq[i]), 3) for i, nm in enumerate(seq_names)}
        res['launch_speed_balls_in_play_train'] = int(trc.sum())
    # flexible check: boosting with the raw sequence facts (previous kind, previous family, repeat, speed change)
    if params.get('gbm', True):
        raw = np.column_stack([v0, T['pfx_x'], T['pfx_z'], xt, zt, T['balls'], T['strikes'], T['group'], T['stand_r'], T['throw_r'], pw_b, pw_p, prev_fam])
        raw1 = np.column_stack([raw, prev_kind, repeat, dv])
        hp = dict(max_iter=int(params.get('gbm_iter', 300)), learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=300, l2_regularization=1.0, random_state=11)
        g0 = HistGradientBoostingClassifier(**hp).fit(raw[idx], whiff[idx]); g1 = HistGradientBoostingClassifier(**hp).fit(raw1[idx], whiff[idx])
        res['boosting'] = {}
        for s_ in tests:
            te = ok & ((T['day'] >= test_from) if final else (T['season'] == s_))
            if te.sum() < 5000:
                continue
            l0 = logloss_vec(g0.predict_proba(raw[te])[:, 1], whiff[te]); l1 = logloss_vec(g1.predict_proba(raw1[te])[:, 1], whiff[te])
            res['boosting'][str(s_)] = {'sequence_terms_gain_nats_per_1000_swings': [round(v * 1000, 3) for v in clustered_ci(l0 - l1, T['game'][te])]}
        stage('boosting')
    return res


# ---------------------------------------------------------------- PLAN-01: the plate appearance as a sequential decision problem
E_NONE = 0
E_KINDS = ('ball', 'called', 'foul', 'miss')


def e_index(fam: int, kind: int) -> int:
    """Expectation state after a pitch: 0 none; 1 + family * 4 + kind (families 0-2, kinds ball, called, foul, miss)."""
    return 1 + int(fam) * 4 + int(kind)


class PAModels:
    """The credited pieces fitted on the training rows, evaluated for any pitch at any count: swing at the decision
    moment (league with its family part, hitter family maps), miss on the true crossing (league, hitter whiff maps),
    called strike on a take, foul on contact, the run value of a ball in play. Hitter-level rates (swing, miss, contact
    value) are scalars any plan may use; the maps are what makes a plan hitter-specific."""

    def __init__(self, T: dict, tr: np.ndarray, rng, params: dict, stage):
        self.T = T; self.tr = tr; self.rng = rng
        F = rebuild(T)
        self.xp, self.zp = projected(T, F, None, 'straight', 0.26)
        self.xt, self.zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
        n = len(T['call'])
        swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64); whiff = (T['call'] == 2).astype(np.float64)
        self.swing, self.whiff = swing, whiff
        contact = (T['call'] == 1); foul = (contact & (T['last_in_pa'] == 0)).astype(np.float64); take_ = T['call'] == 0
        self.inplay = contact & (T['last_in_pa'] == 1) & np.isin(T['out7'], (0, 3, 4, 5, 6))
        self.bip_value = np.where(self.inplay, LW7[np.clip(T['out7'], 0, 6)], 0.0)
        # hitter and pitcher levels from earlier dates (the rows' own propensities) and as scalars per player (training rows)
        self.prop_s = swing_propensity(T); self.prop_p = pitcher_propensity(T)
        tw = dict(T); tw['call'] = np.where(whiff == 1, 1, np.where(swing == 1, 0, 3)); self.prop_w = swing_propensity(tw, 200.0)
        lg_bip = float(self.bip_value[self.inplay & tr].mean())
        def prior_bip(key, k_):
            nn, ss = _prior_by_day(key.astype(np.int64), T['day'].astype(np.int64), self.bip_value, self.inplay)
            return (ss + k_ * lg_bip) / (nn + k_)
        self.bip_b = prior_bip(T['batter'], 80.0); self.bip_p = prior_bip(T['pitcher'], 150.0)
        self.lg_bip = lg_bip
        # swing model: location at the decision moment with the league's family part, controls, pitcher propensity
        Ls = location_block(self.xp, self.zp, T['stand_r'], T['strikes']); Bh0 = hitter_basis(self.xp, self.zp, T['stand_r'], T['strikes']); nh = Bh0.shape[1] - 1
        brk = np.isin(T['group'], (3, 4))[:, None]; ofs = (T['group'] == 5)[:, None]
        Xs = np.hstack([Ls, (Bh0[:, :-1] * brk).astype(np.float32), (Bh0[:, :-1] * ofs).astype(np.float32), control_block(T, self.prop_s), self.prop_p[:, None].astype(np.float32)])
        idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), int(params.get('league_n', 600000))), replace=False)
        self.m_s = fit_logistic(Xs[idx], swing[idx]); self.off_s = self.m_s.decision_function(Xs)
        self.nL, self.nh = Ls.shape[1], nh
        self.Bs = family_basis(Bh0, T['group'])
        gb_tr = _groups(T['batter'], tr)
        self.maps_s = _hitter_maps(self.Bs, swing, self.off_s, gb_tr, float(params.get('lam_s', 10.0)), int(params.get('min_pitches', 300)))
        self.side = {h: int(np.round(T['stand_r'][r].mean())) for h, r in gb_tr.items()}
        del Xs
        stage(f'swing model, {len(self.maps_s)} maps')
        # whiff model on the true crossing
        grp = np.zeros((n, 7), np.float32); grp[np.arange(n), np.clip(T['group'], 0, 6)] = 1
        Lw = location_block(self.xt, self.zt, T['stand_r'], T['strikes'])
        Xw = np.hstack([Lw, grp, hats(T['v0'].astype(np.float64), V_KNOTS), (T['strikes'] == 2)[:, None].astype(np.float32), self.prop_w[:, None].astype(np.float32),
                        (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)])
        sw_tr = tr & (swing == 1); idx = np.flatnonzero(sw_tr); idx = rng.choice(idx, min(len(idx), int(params.get('league_n', 600000))), replace=False)
        self.m_w = fit_logistic(Xw[idx], whiff[idx]); self.off_w = self.m_w.decision_function(Xw)
        fam3 = np.column_stack([np.isin(T['group'], (0, 1, 2)), np.isin(T['group'], (3, 4)), np.isin(T['group'], (5,))]).astype(np.float64)
        self.Bw = np.hstack([hitter_basis(self.xt, self.zt, T['stand_r'], T['strikes']), fam3, hats(self.zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
        self.maps_w = _hitter_maps(self.Bw, whiff, self.off_w, _groups(T['batter'], sw_tr), float(params.get('lam_w', 30.0)), int(params.get('min_swings', 200)))
        # called strike on a take; foul on contact
        Xc = np.hstack([Lw, (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)])
        tk = tr & take_; idx = np.flatnonzero(tk); idx = rng.choice(idx, min(len(idx), 400000), replace=False)
        self.m_c = fit_logistic(Xc[idx], (T['cs'] == 1).astype(np.float64)[idx]); self.p_c = self.m_c.predict_proba(Xc)[:, 1]
        ct = tr & contact; idx = np.flatnonzero(ct); idx = rng.choice(idx, min(len(idx), 400000), replace=False)
        self.m_f = fit_logistic(Xw[idx], foul[idx]); self.p_f = self.m_f.predict_proba(Xw)[:, 1]
        # the run value of a ball in play: linear on location, group, speed, count, platoon and both players' contact values
        cnt = np.zeros((n, 12), np.float32); cnt[np.arange(n), np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)] = 1
        Xb = np.hstack([np.ones((n, 1), np.float32), Lw, grp, hats(T['v0'].astype(np.float64), V_KNOTS), cnt, (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32),
                        (self.bip_b - lg_bip)[:, None].astype(np.float32), (self.bip_p - lg_bip)[:, None].astype(np.float32)])
        bp = tr & self.inplay
        A = Xb[bp].astype(np.float64); self.beta_b = np.linalg.lstsq(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ self.bip_value[bp], rcond=None)[0]
        self.v_bip = Xb @ self.beta_b
        del Xw, Xc, Xb, Lw, grp, cnt
        # hitter scalars (training rows): swing, whiff, contact value, each as the row propensities would read at the end of training
        self.h_scalar = {}
        for h, r in gb_tr.items():
            if h in self.maps_s:
                self.h_scalar[h] = (float(self.prop_s[r[-1]]), float(self.prop_w[r[-1]]), float(self.bip_b[r[-1]]))
        gp_tr = _groups(T['pitcher'], tr)
        self.p_scalar = {p: (float(self.prop_p[r[-1]]), float(self.bip_p[r[-1]])) for p, r in gp_tr.items()}
        self.gp = gp_tr
        # the coefficients the per-hitter scalars multiply
        cs_ = self.m_s.coef_[0].astype(np.float64); i_prop_s = self.nL + 2 * nh + 32   # control_block: 12 count + 7 group + 7 group x two strikes + 6 speed = 32, then prop
        self.w_prop_s = float(cs_[i_prop_s]); self.w_prop_p = float(cs_[-1])
        cw_ = self.m_w.coef_[0].astype(np.float64); self.w_prop_w = float(cw_[-2])
        self.w_bip_b = float(self.beta_b[-2]); self.w_bip_p = float(self.beta_b[-1])
        self.delta = float(params.get('delta_repeat_after_called', 0.0))     # SEQ-02's log-odds on a repeat after a called strike
        self.cache = {}
        stage('models fitted')

    def _blocks(self, r: np.ndarray, b: int, k: int):
        """Model logits for rows r evaluated as if at count (b, k), with the hitter-level scalars taken out (added per hitter)."""
        T = self.T; n = len(r)
        return self.blocks_at(self.xp[r], self.zp[r], self.xt[r], self.zt[r], T['stand_r'][r], T['throw_r'][r], T['group'][r], T['v0'][r].astype(np.float64),
                              np.full(n, b, np.int64), np.full(n, k, np.int64))

    def blocks_at(self, xp, zp, xt, zt, stand, throw, grp_, v0, balls, strikes):
        """The same for any pitches given by their decision-moment projection, true crossing, side, hand, group, speed and count arrays
        (VALUE-18 prices scattered aims this way)."""
        n = len(xt); strikes = np.asarray(strikes, np.int64); balls = np.asarray(balls, np.int64)
        tt = {'balls': balls, 'strikes': strikes, 'group': grp_, 'v0': v0, 'stand_r': stand, 'throw_r': throw}
        Ls = location_block(xp, zp, stand, strikes); Bh0 = hitter_basis(xp, zp, stand, strikes)
        brk = np.isin(grp_, (3, 4))[:, None]; ofs = (grp_ == 5)[:, None]
        Xs = np.hstack([Ls, (Bh0[:, :-1] * brk).astype(np.float32), (Bh0[:, :-1] * ofs).astype(np.float32), control_block(tt, np.zeros(n)), np.zeros((n, 1), np.float32)])
        lo_s = self.m_s.decision_function(Xs).astype(np.float64)
        Bs = family_basis(Bh0, grp_)
        grp = np.zeros((n, 7), np.float32); grp[np.arange(n), np.clip(grp_, 0, 6)] = 1
        Lw = location_block(xt, zt, stand, strikes)
        plat = (stand == throw)[:, None].astype(np.float32)
        Xw = np.hstack([Lw, grp, hats(v0, V_KNOTS), (strikes == 2)[:, None].astype(np.float32), np.zeros((n, 1), np.float32), plat])
        lo_w = self.m_w.decision_function(Xw).astype(np.float64)
        fam3 = np.column_stack([np.isin(grp_, (0, 1, 2)), np.isin(grp_, (3, 4)), np.isin(grp_, (5,))]).astype(np.float64)
        Bw = np.hstack([hitter_basis(xt, zt, stand, strikes), fam3, hats(zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
        p_c = self.m_c.predict_proba(np.hstack([Lw, plat]))[:, 1]
        p_f = self.m_f.predict_proba(Xw)[:, 1]
        cnt = np.zeros((n, 12), np.float32); cnt[np.arange(n), np.clip(balls, 0, 3) * 3 + np.clip(strikes, 0, 2)] = 1
        Xb = np.hstack([np.ones((n, 1), np.float32), Lw, grp, hats(v0, V_KNOTS), cnt, plat, np.zeros((n, 2), np.float32)])
        v_b = Xb.astype(np.float64) @ self.beta_b
        fam = np.where(np.isin(grp_, (0, 1, 2)), 0, np.where(np.isin(grp_, (3, 4)), 1, 2))
        return {'lo_s': lo_s, 'Bs': Bs, 'lo_w': lo_w, 'Bw': Bw, 'c': p_c, 'f': p_f, 'v': v_b, 'fam': fam}

    def swing_minus_take(self, blk: dict, h: int | None, p_scalar, balls, strikes, cv: np.ndarray):
        """VALUE-18: the value of the plate appearance if the hitter swings minus if he takes, at these pitches, for hitter h against a
        pitcher with these scalars, under the count values cv (12, by balls*3+strikes): the structural run value of one unit of swing chance."""
        _, p_w, p_c, p_f, v, _ = self.probs(blk, h, p_scalar)
        balls = np.asarray(balls, np.int64); strikes = np.asarray(strikes, np.int64)
        v_strike = np.where(strikes == 2, LW7[1], cv[np.clip(balls * 3 + strikes + 1, 0, 11)])
        v_foul = np.where(strikes == 2, cv[np.clip(balls * 3 + 2, 0, 11)], cv[np.clip(balls * 3 + strikes + 1, 0, 11)])
        v_ball = np.where(balls == 3, LW7[2], cv[np.clip((balls + 1) * 3 + strikes, 0, 11)])
        v_swing = p_w * v_strike + (1 - p_w) * (p_f * v_foul + (1 - p_f) * v)
        v_take = p_c * v_strike + (1 - p_c) * v_ball
        return v_swing - v_take

    def expected_value(self, blk: dict, h: int | None, p_scalar, balls, strikes, cv: np.ndarray):
        """The structural expectation of a pitch's realized (telescoping) value: swing chance times the swing value plus take chance times the take
        value, each relative to the current count's value; (expected value, swing value, take value, swing chance)."""
        p_s, p_w, p_c, p_f, v, _ = self.probs(blk, h, p_scalar)
        balls = np.asarray(balls, np.int64); strikes = np.asarray(strikes, np.int64)
        v_strike = np.where(strikes == 2, LW7[1], cv[np.clip(balls * 3 + strikes + 1, 0, 11)])
        v_foul = np.where(strikes == 2, cv[np.clip(balls * 3 + 2, 0, 11)], cv[np.clip(balls * 3 + strikes + 1, 0, 11)])
        v_ball = np.where(balls == 3, LW7[2], cv[np.clip((balls + 1) * 3 + strikes, 0, 11)])
        v_now = cv[np.clip(balls * 3 + strikes, 0, 11)]
        v_swing = p_w * v_strike + (1 - p_w) * (p_f * v_foul + (1 - p_f) * v) - v_now
        v_take = p_c * v_strike + (1 - p_c) * v_ball - v_now
        return p_s * v_swing + (1 - p_s) * v_take, v_swing, v_take, p_s

    def pool(self, p: int, sd: int, max_per_group: int = 120):
        """The pitcher's own pitches to a side (training rows), a sample per count group with the group's weight."""
        key = (p, sd)
        if key in self.cache:
            return self.cache[key]
        T = self.T; r = self.gp.get(p)
        if r is None:
            self.cache[key] = None; return None
        r = r[T['stand_r'][r] == sd]
        if len(r) < 150:
            self.cache[key] = None; return None
        c3 = np.where(T['strikes'][r] == 2, 2, np.where(T['balls'][r] > T['strikes'][r], 1, 0))
        rows = []; cg = []
        for c in (0, 1, 2):
            rr = r[c3 == c]
            if len(rr) > max_per_group:
                rr = self.rng.choice(rr, max_per_group, replace=False)
            rows.append(rr); cg.append(np.full(len(rr), c))
        rows = np.concatenate(rows); cg = np.concatenate(cg)
        blocks = {(b, k): self._blocks(rows, b, k) for b in range(4) for k in range(3)}
        out = {'rows': rows, 'cg': cg, 'blocks': blocks, 'p_scalar': self.p_scalar.get(p, (0.0, self.lg_bip))}
        self.cache[key] = out
        return out

    def probs(self, blk: dict, h: int | None, p_scalar, hs=None):
        """(swing, whiff, called, foul, value, fam) for a block, for hitter h (maps) or the league plan (no maps); hs
        overrides the hitter scalars (used by the actual-pitch evaluation, whose rows carry their own)."""
        ps_, pw_, pb_ = hs if hs is not None else (self.h_scalar[h] if h is not None and h in self.h_scalar else (0.0, 0.0, self.lg_bip))
        lo_s = blk['lo_s'] + self.w_prop_s * ps_ + self.w_prop_p * p_scalar[0]
        lo_w = blk['lo_w'] + self.w_prop_w * pw_
        v = blk['v'] + self.w_bip_b * (pb_ - self.lg_bip) + self.w_bip_p * (p_scalar[1] - self.lg_bip)
        if h is not None and h in self.maps_s:
            lo_s = lo_s + blk['Bs'] @ self.maps_s[h]
            if h in self.maps_w:
                lo_w = lo_w + blk['Bw'] @ self.maps_w[h]
        return 1 / (1 + np.exp(-lo_s)), 1 / (1 + np.exp(-lo_w)), blk['c'], blk['f'], v, blk['fam']

    def solve(self, P: dict, h: int | None, hs=None):
        """Value iteration over counts and expectation states for one pitcher pool against one hitter. Returns the
        optimal value, the actual-policy value (the pitcher's own mix in each count group) and the optimal action."""
        nE = 1 + 3 * 4; pr = {}
        for (b, k), blk in P['blocks'].items():
            pr[(b, k)] = self.probs(blk, h, P['p_scalar'], hs)
        cg = P['cg']; V = np.zeros((4, 3, nE)); Va = np.zeros((4, 3, nE)); act = {}
        def q_of(b, k, e, s, w, c, f, v, fam):
            # whiff adjustment: a repeat of the family after a called strike (SEQ-02)
            if self.delta and e != E_NONE:
                ef, ek = (e - 1) // 4, (e - 1) % 4
                if ek == 1:
                    w = np.where(fam == ef, 1 / (1 + np.exp(-(np.log(w / (1 - w)) + self.delta))), w)
            idx_e = lambda kind: 1 + fam * 4 + kind
            vK = LW7[1]; vBB = LW7[2]
            q = s * (1 - w) * (1 - f) * v
            q += s * w * (vK if k == 2 else V[b, k + 1][idx_e(3)])
            q += s * (1 - w) * f * (V[b, 2][idx_e(2)] if k == 2 else V[b, k + 1][idx_e(2)])
            q += (1 - s) * c * (vK if k == 2 else V[b, k + 1][idx_e(1)])
            q += (1 - s) * (1 - c) * (vBB if b == 3 else V[b + 1, k][idx_e(0)])
            return q
        for total in range(5, -1, -1):
            for b in range(4):
                k = total - b
                if k < 0 or k > 2:
                    continue
                s, w, c, f, v, fam = pr[(b, k)]
                c3 = 2 if k == 2 else (1 if b > k else 0); wts = (cg == c3).astype(np.float64); wts = wts / max(wts.sum(), 1e-9)
                for _ in range(40 if k == 2 else 1):
                    for e in range(nE):
                        q = q_of(b, k, e, s, w, c, f, v, fam)
                        j = int(np.argmin(q)); V[b, k, e] = q[j]; act[(b, k, e)] = j
                        Va[b, k, e] = float(np.dot(wts, q)) if wts.sum() > 0 else q.mean()
        return V, Va, act


def plan_study(T: dict, params: dict, stage) -> dict:
    """PLAN-01. The plate appearance as a sequential decision problem: states are the count and what the hitter just
    saw (family and outcome of the previous pitch), actions are the pitcher's own pitches (type and spot), the
    transition model is the credited pieces, and the value is the plate appearance's run value. Solved backward from
    the end of the plate appearance for each hitter-pitcher pair, with the hitter's maps (hitter-specific plan) and
    without them (league plan, which still knows his swing, miss and contact levels). Claims: the runs per plate
    appearance the hitter-specific plan saves over the league plan, and over the pitcher's own mix. Test (the natural
    experiment, as VALUE-08): on the test seasons' actual pitches, the hitter-specific advantage of the pitch actually
    thrown (its Q under the hitter's plan minus the pool average, minus the same under the league plan) against the
    pitch's realized run value (count-state values from the training seasons); pitchers do not aim at the maps, so the
    slope measures whether the plan's hitter-specific differences are real; placebo with another hitter's maps."""
    import pandas as pd
    res = {}
    train = tuple(int(v) for v in params.get('train', (2023, 2024))); tests = tuple(int(v) for v in params.get('tests', (2025, 2026)))
    T = take(T, np.isin(T['season'], train + tests))
    F = rebuild(T)
    keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
    T = take(T, keep)
    order = np.lexsort((T['pitch_no'], T['ab'], T['game'])); T = take(T, order)
    tr = np.isin(T['season'], train)
    rng = np.random.default_rng(int(params.get('seed', 11)))
    M = PAModels(T, tr, rng, params, stage)
    n = len(T['call'])
    # realized pitch values: count-state values from the training seasons (mean final value of plate appearances passing through the count)
    pak = T['game'].astype(np.int64) * 1000 + T['ab'].astype(np.int64)
    first = np.r_[True, pak[1:] != pak[:-1]]; pa_id = np.cumsum(first) - 1
    last_idx = np.flatnonzero(T['last_in_pa'] == 1)
    pa_final = np.full(pa_id.max() + 1, np.nan)
    pa_final[pa_id[last_idx]] = np.where(T['out7'][last_idx] >= 0, LW7[np.clip(T['out7'][last_idx], 0, 6)], np.nan)
    fin = pa_final[pa_id]
    ci = np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)
    cv = np.zeros(12)
    for c in range(12):
        m = tr & (ci == c) & np.isfinite(fin); cv[c] = float(fin[m].mean()) if m.any() else 0.0
    res['count_values'] = {f'{c // 3}-{c % 3}': round(float(cv[c]), 4) for c in range(12)}
    # next state value of each pitch: the next pitch's count value in the same plate appearance, else the terminal value
    nxt = np.r_[cv[ci[1:]], 0.0]; same_next = np.r_[pak[1:] == pak[:-1], False]
    term = np.where(T['out7'] >= 0, LW7[np.clip(T['out7'], 0, 6)], np.nan)
    realized = np.where(same_next, nxt, term) - cv[ci]
    # the expectation state of each pitch from the previous pitch in the plate appearance
    fam = np.where(np.isin(T['group'], (0, 1, 2)), 0, np.where(np.isin(T['group'], (3, 4)), 1, 2))
    kind = np.where(T['call'] == 0, np.where(T['cs'] == 1, 1, 0), np.where(T['call'] == 2, 3, 2))
    same_prev = np.r_[False, pak[1:] == pak[:-1]]
    e_state = np.where(same_prev, 1 + np.r_[0, fam[:-1]] * 4 + np.r_[0, kind[:-1]], 0)
    stage('states')
    # pairs on the test seasons: hitters with maps against pitchers with pools, the most frequent pairs first
    te = np.isin(T['season'], tests)
    pair = T['batter'].astype(np.int64) * 1_000_000 + T['pitcher'].astype(np.int64)
    up, cnt_p = np.unique(pair[te], return_counts=True)
    order_p = np.argsort(-cnt_p); up = up[order_p]; cnt_p = cnt_p[order_p]
    max_pairs = int(params.get('max_pairs', 3000)); min_pair = int(params.get('min_pair_pitches', 12))
    chosen = [(int(k // 1_000_000), int(k % 1_000_000)) for k, c in zip(up, cnt_p) if c >= min_pair][:max_pairs]
    stage(f'{len(chosen)} pairs')
    values = []; adv_rows = []
    sides = {}
    t_pairs = 0
    for h, p in chosen:
        if h not in M.maps_s or h not in M.h_scalar:
            continue
        sd = M.side[h]; P = M.pool(p, sd)
        if P is None:
            continue
        V_h, Va_h, act_h = M.solve(P, h); V_l, Va_l, act_l = M.solve(P, None, M.h_scalar[h])
        # the league plan's actions evaluated under the hitter's models: follow act_l, value with h's probabilities
        pr_h = {(b, k): M.probs(P['blocks'][(b, k)], h, P['p_scalar']) for (b, k) in P['blocks']}
        V_lh = np.zeros_like(V_h); nE = V_h.shape[2]
        for total in range(5, -1, -1):
            for b in range(4):
                k = total - b
                if k < 0 or k > 2:
                    continue
                s, w, c, f, v, fm = pr_h[(b, k)]
                for _ in range(40 if k == 2 else 1):
                    for e in range(nE):
                        j = act_l[(b, k, e)]
                        ef = fm[j]; wj = w[j]
                        if M.delta and e != E_NONE and (e - 1) % 4 == 1 and (e - 1) // 4 == ef:
                            wj = 1 / (1 + np.exp(-(np.log(wj / (1 - wj)) + M.delta)))
                        q = s[j] * (1 - wj) * (1 - f[j]) * v[j]
                        q += s[j] * wj * (LW7[1] if k == 2 else V_lh[b, k + 1, 1 + ef * 4 + 3])
                        q += s[j] * (1 - wj) * f[j] * (V_lh[b, 2, 1 + ef * 4 + 2] if k == 2 else V_lh[b, k + 1, 1 + ef * 4 + 2])
                        q += (1 - s[j]) * c[j] * (LW7[1] if k == 2 else V_lh[b, k + 1, 1 + ef * 4 + 1])
                        q += (1 - s[j]) * (1 - c[j]) * (LW7[2] if b == 3 else V_lh[b + 1, k, 1 + ef * 4 + 0])
                        V_lh[b, k, e] = q
        values.append((h, p, float(Va_h[0, 0, 0]), float(V_lh[0, 0, 0]), float(V_h[0, 0, 0])))
        # the natural experiment: the actual test pitches of this pair
        rows = np.flatnonzero(te & (T['batter'] == h) & (T['pitcher'] == p) & np.isfinite(realized))
        if len(rows) == 0:
            continue
        blk_a = {}
        for (b, k) in set(zip(T['balls'][rows].tolist(), T['strikes'][rows].tolist())):
            rr = rows[(T['balls'][rows] == b) & (T['strikes'][rows] == k)]
            blk = M._blocks(rr, int(b), int(k))
            hs = (M.h_scalar[h][0], M.h_scalar[h][1], M.h_scalar[h][2])
            for which, hh in (('h', h), ('l', None)):
                s, w, c, f, v, fm = M.probs(blk, hh, P['p_scalar'], hs if which == 'l' else None)
                Vuse = V_h if which == 'h' else V_l
                for i_, ri in enumerate(rr):
                    e = int(e_state[ri]); ef = int(fm[i_]); wj = float(w[i_])
                    if M.delta and e != E_NONE and (e - 1) % 4 == 1 and (e - 1) // 4 == ef:
                        wj = 1 / (1 + np.exp(-(np.log(wj / (1 - wj)) + M.delta)))
                    q = s[i_] * (1 - wj) * (1 - f[i_]) * v[i_]
                    q += s[i_] * wj * (LW7[1] if k == 2 else Vuse[b, k + 1, 1 + ef * 4 + 3])
                    q += s[i_] * (1 - wj) * f[i_] * (Vuse[b, 2, 1 + ef * 4 + 2] if k == 2 else Vuse[b, k + 1, 1 + ef * 4 + 2])
                    q += (1 - s[i_]) * c[i_] * (LW7[1] if k == 2 else Vuse[b, k + 1, 1 + ef * 4 + 1])
                    q += (1 - s[i_]) * (1 - c[i_]) * (LW7[2] if b == 3 else Vuse[b + 1, k, 1 + ef * 4 + 0])
                    base_q = (Va_h if which == 'h' else Va_l)[b, k, e]
                    blk_a.setdefault(int(ri), {})[which] = float(q - base_q)
        for ri, d in blk_a.items():
            if 'h' in d and 'l' in d:
                adv_rows.append((ri, d['h'], d['l'], h, p))
        t_pairs += 1
        if t_pairs % 200 == 0:
            stage(f'pairs solved {t_pairs}')
    stage(f'pairs solved {t_pairs}, pitches {len(adv_rows)}')
    vals = np.asarray([v[2:] for v in values])
    if len(vals):
        pa_w = np.asarray([int(((T['batter'] == h) & (T['pitcher'] == p) & te & (T['last_in_pa'] == 1)).sum()) for h, p, *_ in values], float)
        wavg = lambda x: float(np.average(x, weights=np.maximum(pa_w, 1)))
        res['value_runs_per_pa'] = {'pairs': int(len(vals)), 'actual_mix': round(wavg(vals[:, 0]), 4), 'league_plan': round(wavg(vals[:, 1]), 4), 'hitter_plan': round(wavg(vals[:, 2]), 4),
                                    'league_plan_over_actual': round(wavg(vals[:, 0] - vals[:, 1]), 4), 'hitter_plan_over_league_plan': round(wavg(vals[:, 1] - vals[:, 2]), 4),
                                    'per_team_season_6100_pa': {'league_plan_over_actual': round(wavg(vals[:, 0] - vals[:, 1]) * 6100, 1), 'hitter_plan_over_league_plan': round(wavg(vals[:, 1] - vals[:, 2]) * 6100, 1)}}
    if adv_rows:
        A = np.asarray(adv_rows); ri = A[:, 0].astype(int); adv_h = A[:, 1]; adv_l = A[:, 2]; d = adv_h - adv_l; y = realized[ri]
        games = T['game'][ri]
        def reg(x1, x2, yy, cl):
            X = np.column_stack([np.ones(len(x1)), x1, x2]); beta = np.linalg.lstsq(X, yy, rcond=None)[0]
            ug, gi = np.unique(cl, return_inverse=True); draws = []
            for _ in range(int(params.get('reps', 300))):
                w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
                Xw = X * w[:, None]; draws.append(np.linalg.solve(Xw.T @ X + 1e-9 * np.eye(3), Xw.T @ yy))
            draws = np.asarray(draws)
            return {'hitter_specific_slope': [round(float(beta[1]), 3), round(float(np.percentile(draws[:, 1], 2.5)), 3), round(float(np.percentile(draws[:, 1], 97.5)), 3)],
                    'league_advantage_slope': [round(float(beta[2]), 3), round(float(np.percentile(draws[:, 2], 2.5)), 3), round(float(np.percentile(draws[:, 2], 97.5)), 3)]}
        res['natural_experiment'] = {}
        for s_ in tests:
            m = T['season'][ri] == s_
            if m.sum() < 5000:
                continue
            out = reg(d[m], adv_l[m], y[m], games[m]); out['pitches'] = int(m.sum())
            out['sd_hitter_specific_advantage_runs'] = round(float(d[m].std()), 4); out['sd_league_advantage_runs'] = round(float(adv_l[m].std()), 4)
            res['natural_experiment'][str(s_)] = out
        # placebos and reliability (PLAN-01b): the hitter-specific advantage recomputed under other map sets, for a
        # sample of the pitches. 'shuffle': another same-side hitter's maps; 'mean': the same-side mean maps (the shared
        # shape only); 'noise': random maps with each hitter's own-part variance; 'flip': the own part negated;
        # 'half_a' and 'half_b': maps fitted on each training season alone (their agreement bounds the slope a noisy
        # claim can reach: the validated slope is attenuated by the maps' reliability)
        hs_ = sorted({int(v[0]) for v in values}); by_side = {0: [], 1: []}
        for h in hs_:
            by_side[M.side[h]].append(h)
        maps_s0, maps_w0 = M.maps_s, M.maps_w
        Ss = {0: 0.0, 1: 0.0}; Ns = {0: 0, 1: 0}; Sw = {0: 0.0, 1: 0.0}; Nw = {0: 0, 1: 0}
        for h, m in maps_s0.items():
            sd_ = M.side.get(h, 1); Ss[sd_] = Ss[sd_] + m; Ns[sd_] += 1
        for h, m in maps_w0.items():
            sd_ = M.side.get(h, 1); Sw[sd_] = Sw[sd_] + m; Nw[sd_] += 1
        mean_s = {sd_: Ss[sd_] / max(Ns[sd_], 1) for sd_ in (0, 1)}; mean_w = {sd_: Sw[sd_] / max(Nw[sd_], 1) for sd_ in (0, 1)}
        own_sd_s = {sd_: np.std(np.array([maps_s0[h] - mean_s[sd_] for h in maps_s0 if M.side.get(h, 1) == sd_]), axis=0) for sd_ in (0, 1)}
        own_sd_w = {sd_: np.std(np.array([maps_w0[h] - mean_w[sd_] for h in maps_w0 if M.side.get(h, 1) == sd_]), axis=0) for sd_ in (0, 1)}
        kinds = list(params.get('placebo_kinds', ('shuffle', 'mean', 'noise', 'flip', 'half_a', 'half_b')))
        half_maps = {}
        if any(k_.startswith('half') for k_ in kinds) and len(train) >= 2:
            for tag, ssn in (('half_a', train[0]), ('half_b', train[-1])):
                rows_h = tr & (T['season'] == ssn)
                ms_ = _hitter_maps(M.Bs, M.swing, M.off_s, _groups(T['batter'], rows_h), float(params.get('lam_s', 10.0)), max(150, int(params.get('min_pitches', 300)) // 2))
                mw_ = _hitter_maps(M.Bw, M.whiff, M.off_w, _groups(T['batter'], rows_h & (M.swing == 1)), float(params.get('lam_w', 30.0)), max(100, int(params.get('min_swings', 200)) // 2))
                half_maps[tag] = (ms_, mw_)
        sample = rng.choice(len(ri), min(len(ri), int(params.get('placebo_n', 150000))), replace=False)
        res['placebo'] = {}
        d_half = {}
        for kind in kinds:
            if kind == 'shuffle':
                perm = {}
                for sd_, lst in by_side.items():
                    sh = list(lst); rng.shuffle(sh); perm.update(dict(zip(lst, sh)))
                M.maps_s = {h: maps_s0[perm[h]] for h in hs_ if perm[h] in maps_s0}; M.maps_w = {h: maps_w0[perm[h]] for h in hs_ if perm[h] in maps_w0}
            elif kind == 'mean':
                M.maps_s = {h: mean_s[M.side[h]] for h in hs_}; M.maps_w = {h: mean_w[M.side[h]] for h in hs_ if h in maps_w0}
            elif kind == 'noise':
                M.maps_s = {h: mean_s[M.side[h]] + rng.standard_normal(len(mean_s[M.side[h]])) * own_sd_s[M.side[h]] for h in hs_}
                M.maps_w = {h: mean_w[M.side[h]] + rng.standard_normal(len(mean_w[M.side[h]])) * own_sd_w[M.side[h]] for h in hs_ if h in maps_w0}
            elif kind == 'flip':
                M.maps_s = {h: 2 * mean_s[M.side[h]] - maps_s0[h] for h in hs_ if h in maps_s0}; M.maps_w = {h: 2 * mean_w[M.side[h]] - maps_w0[h] for h in hs_ if h in maps_w0}
            elif kind in half_maps:
                M.maps_s, M.maps_w = half_maps[kind]
            else:
                continue
            d_pl = np.full(len(ri), np.nan); cacheV = {}
            for jj in sample:
                h, p = int(A[jj, 3]), int(A[jj, 4]); key = (h, p)
                if h not in M.maps_s:
                    continue
                if key not in cacheV:
                    P = M.pool(p, M.side[h]); V_h2, Va_h2, _ = M.solve(P, h); cacheV[key] = (P, V_h2, Va_h2)
                P, V_h2, Va_h2 = cacheV[key]
                r_ = np.array([ri[jj]]); b, k = int(T['balls'][ri[jj]]), int(T['strikes'][ri[jj]])
                blk = M._blocks(r_, b, k); s, w, c, f, v, fm = M.probs(blk, h, P['p_scalar'])
                e = int(e_state[ri[jj]]); ef = int(fm[0]); wj = float(w[0])
                q = s[0] * (1 - wj) * (1 - f[0]) * v[0]
                q += s[0] * wj * (LW7[1] if k == 2 else V_h2[b, k + 1, 1 + ef * 4 + 3])
                q += s[0] * (1 - wj) * f[0] * (V_h2[b, 2, 1 + ef * 4 + 2] if k == 2 else V_h2[b, k + 1, 1 + ef * 4 + 2])
                q += (1 - s[0]) * c[0] * (LW7[1] if k == 2 else V_h2[b, k + 1, 1 + ef * 4 + 1])
                q += (1 - s[0]) * (1 - c[0]) * (LW7[2] if b == 3 else V_h2[b + 1, k, 1 + ef * 4 + 0])
                d_pl[jj] = (q - Va_h2[b, k, e]) - adv_l[jj]
            M.maps_s, M.maps_w = maps_s0, maps_w0
            mp = np.isfinite(d_pl)
            if mp.sum() > 5000:
                out = reg(d_pl[mp], adv_l[mp], y[mp], games[mp]); out['pitches'] = int(mp.sum())
                out['corr_with_real_advantage'] = round(float(np.corrcoef(d_pl[mp], d[mp])[0, 1]), 4)
                res['placebo'][kind] = out
            if kind.startswith('half'):
                d_half[kind] = d_pl
            stage(f'placebo {kind}')
        if 'half_a' in d_half and 'half_b' in d_half:
            mh = np.isfinite(d_half['half_a']) & np.isfinite(d_half['half_b'])
            if mh.sum() > 5000:
                r_ab = float(np.corrcoef(d_half['half_a'][mh], d_half['half_b'][mh])[0, 1])
                res['reliability'] = {'corr_half_a_half_b': round(r_ab, 4), 'pitches': int(mh.sum()),
                                      'note': 'the claim from two-season maps has reliability about 2r/(1+r) of this; the validated slope is attenuated by it'}
        stage('natural experiment')
    return res


# ---------------------------------------------------------------- EXPOSURE-01: does a hitter learn a pitch type within the game
def exposure_study(T: dict, params: dict, stage) -> dict:
    """Times through the order, pitch by pitch: for each pitch, how many pitches of the same type this hitter has already
    seen from this pitcher in the game (same-type exposure), how many of any type (total exposure), and how many of
    the same type from other pitchers earlier in the game (other-pitcher exposure, the control: learning a pitcher's
    pitch should not transfer fully). Whiffs on swings, chases and launch speed on balls in play, beyond the pitch's
    physics, location, count, pitch count and times through the order. Train 2023-2024, score 2025."""
    res = {}
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)) & (T['group'] >= 0) & (T['group'] <= 5) & (T['call'] <= 2) & (T['balls'] >= 0))
    order = np.lexsort((T['pitch_no'], T['ab'], T['game']))
    T = take(T, order)
    n = len(T['game'])
    g = np.unique(T['game'], return_inverse=True)[1].astype(np.int64); b = np.unique(T['batter'], return_inverse=True)[1].astype(np.int64)
    p = np.unique(T['pitcher'], return_inverse=True)[1].astype(np.int64); grp = T['group'].astype(np.int64)
    NB, NP = int(b.max()) + 1, int(p.max()) + 1
    def running(key):
        o = np.lexsort((np.arange(n), key)); k = key[o]
        st = np.r_[True, k[1:] != k[:-1]]; idx = np.arange(n) - np.flatnonzero(st)[np.cumsum(st) - 1]
        out = np.empty(n); out[o] = idx; return out
    pa_key = (g * NB + b) * NP + p                                        # this batter against this pitcher in this game
    same = running(pa_key * 8 + grp)
    total = running(pa_key)
    any_same = running((g * NB + b) * 8 + grp)                            # same type from anyone in the game
    other_same = any_same - same
    # times through the order against this pitcher: plate appearances started before this one
    first = T['pitch_no'] == 0
    pa_count = running(np.where(first, pa_key, -1 - np.arange(n)))
    tto = np.zeros(n); fi = np.flatnonzero(first); tto[fi] = pa_count[fi]
    # carry the plate appearance's count to its later pitches
    pa_id = np.cumsum(first) - 1
    tto = tto[np.flatnonzero(first)][np.maximum(pa_id, 0)]
    pc = running(g * NP + p)                                              # pitcher's pitch count in the game
    swing = (T['call'] == 1) | (T['call'] == 2); whiff = T['call'] == 2
    u = np.where(T['stand_r'] == 1, T['px'], -T['px'])
    outside = (np.abs(u) > ZONE_HALF) | (T['pz'] > ZONE_TOP) | (T['pz'] < ZONE_BOT)
    bip = np.isfinite(T['ls'])
    prop = swing_propensity(T)
    X0 = np.column_stack([T['v0'], T['pfx_x'] * np.where(T['throw_r'] == 1, 1, -1), T['pfx_z'], u, T['pz'], T['spin'], T['z0'], T['ext'],
                          T['balls'], T['strikes'], grp, (T['stand_r'] == T['throw_r']), prop, pc, np.minimum(tto, 3), total])
    X1 = np.column_stack([X0, same, other_same])
    res['exposure'] = {'same_type_mean': round(float(same.mean()), 3), 'same_type_p90': float(np.percentile(same, 90)), 'other_pitcher_same_type_mean': round(float(other_same.mean()), 3)}
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    hp = dict(max_iter=int(params.get('gbm_iter', 300)), learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=300, l2_regularization=1.0, random_state=11)
    tr = np.isin(T['season'], (2023, 2024)); te = T['season'] == 2025
    rng = np.random.default_rng(11)
    out = {}
    for name, rows, y in (('whiff_on_swing', swing, whiff), ('chase', outside, swing)):
        a = np.flatnonzero(tr & rows); a = rng.choice(a, min(len(a), int(params.get('train_n', 1200000))), replace=False)
        t_ = te & rows; yy = y.astype(float)
        m0 = HistGradientBoostingClassifier(**hp).fit(X0[a], yy[a]); m1 = HistGradientBoostingClassifier(**hp).fit(X1[a], yy[a])
        p0, p1 = m0.predict_proba(X0[t_])[:, 1], m1.predict_proba(X1[t_])[:, 1]
        # the exposure effect read as the change in prediction when same-type exposure is raised by 3, other features held
        Xs = X1[t_].copy(); Xs[:, -2] += 3; ps = m1.predict_proba(Xs)[:, 1]
        Xo = X1[t_].copy(); Xo[:, -1] += 3; po = m1.predict_proba(Xo)[:, 1]
        out[name] = {'test_rows': int(t_.sum()), 'gain_nats_per_1000': [round(v * 1000, 3) for v in clustered_ci(logloss_vec(p0, yy[t_]) - logloss_vec(p1, yy[t_]), T['game'][t_])],
                     'effect_of_3_more_same_type_from_this_pitcher_points': round(float((ps - p1).mean() * 100), 3),
                     'effect_of_3_more_same_type_from_other_pitchers_points': round(float((po - p1).mean() * 100), 3),
                     'by_same_type_exposure': [{'seen': k, 'observed': round(float(yy[t_][np.minimum(same[t_], 9) == k].mean()), 4) if np.any(np.minimum(same[t_], 9) == k) else None,
                                                'predicted_without': round(float(p0[np.minimum(same[t_], 9) == k].mean()), 4) if np.any(np.minimum(same[t_], 9) == k) else None,
                                                'rows': int(np.sum(np.minimum(same[t_], 9) == k))} for k in (0, 1, 2, 3, 5, 7, 9)]}
        stage('exposure ' + name)
    a = np.flatnonzero(tr & bip); t_ = te & bip
    r0 = HistGradientBoostingRegressor(**hp).fit(X0[a], T['ls'][a]); r1 = HistGradientBoostingRegressor(**hp).fit(X1[a], T['ls'][a])
    e0 = (r0.predict(X0[t_]) - T['ls'][t_]) ** 2; e1 = (r1.predict(X1[t_]) - T['ls'][t_]) ** 2
    Xs = X1[t_].copy(); Xs[:, -2] += 3
    out['launch_speed_on_contact'] = {'test_rows': int(t_.sum()), 'mse_reduction': [round(v, 4) for v in clustered_ci(e0 - e1, T['game'][t_])],
                                      'effect_of_3_more_same_type_mph': round(float((r1.predict(Xs) - r1.predict(X1[t_])).mean()), 3)}
    res['models'] = out
    stage('exposure contact')
    return res


# ---------------------------------------------------------------- ZONE-01: pitch location across the 2026 definition change
YMID = 8.5 / 12.0          # the middle of the plate, Statcast's reference for plate_x and plate_z from 2026


def zone_audit(T: dict, params: dict, stage) -> dict:
    """Statcast documents that plate_x and plate_z moved from the front of the plate (through 2025) to its middle (2026)
    and that the zone's top and bottom became the ABS zone. Is the move in our feed's pitch locations, and how big is it?
    Same pitcher and pitch type in consecutive seasons: the change in mean height at the plate (2025 to 2026, with 2023
    to 2024 and 2024 to 2025 as controls) against the drop the ball makes between the front and the middle of the plate,
    computed from each pitch's own flight (pitches that arrive steeper drop more in those 8.5 inches, so a definition
    change scales with the pitch's descent; a change in where pitchers aim does not). Also the Statcast zone field's
    in-zone share and called strikes by zone and height, by season. Aggregates only."""
    res = {}
    F = rebuild(T)
    ok = F['ok'] & (T['group'] >= 0) & np.isfinite(T['v1'])
    vy = T['v1'].astype(np.float64) * FT_PER_MPH
    with np.errstate(invalid='ignore', divide='ignore'):
        dt = (YPLATE - YMID) / vy
    dz_pred = (F['vz0'] + F['az'] * F['tf']) * dt          # height at the middle minus height at the front (ft)
    dx_pred = (F['vx0'] + F['ax'] * F['tf']) * dt
    ok &= np.isfinite(dz_pred) & np.isfinite(dx_pred)
    seasons = [int(s) for s in np.unique(T['season'])]
    res['predicted_drop_in'] = {GROUP_NAMES[g]: round(float(np.nanmean(dz_pred[ok & (T['group'] == g) & (T['season'] <= 2025)]) * 12.0), 3)
                                for g in range(6)}
    key = T['pitcher'].astype(np.int64) * 100 + T['sub'].astype(np.int64)
    stats = {}
    for s in seasons:
        m = ok & (T['season'] == s)
        uk, inv = np.unique(key[m], return_inverse=True)
        n = np.bincount(inv, minlength=len(uk))
        stats[s] = {'keys': uk, 'n': n, 'pz': np.bincount(inv, weights=T['pz'][m].astype(np.float64), minlength=len(uk)) / np.maximum(n, 1),
                    'dz': np.bincount(inv, weights=dz_pred[m], minlength=len(uk)) / np.maximum(n, 1),
                    'grp': np.bincount(inv, weights=T['group'][m].astype(np.float64), minlength=len(uk)) / np.maximum(n, 1)}
    min_n = int(params.get('min_pitches', 50))
    pairs = {}
    for a, b in zip(seasons[:-1], seasons[1:]):
        # the predicted move comes from a third season's flights: a pitch's computed descent depends on its own measured
        # height, so taking it from either season of the pair would tie it to that season's noise (regression to the mean)
        c = min((x for x in seasons if x not in (a, b)), key=lambda x: (abs(x - a), x))
        A, B, Cs = stats[a], stats[b], stats[c]
        common, ia, ib = np.intersect1d(A['keys'], B['keys'], return_indices=True)
        common, ja, jc = np.intersect1d(common, Cs['keys'], return_indices=True)
        ia, ib = ia[ja], ib[ja]
        keep = (A['n'][ia] >= min_n) & (B['n'][ib] >= min_n) & (Cs['n'][jc] >= min_n)
        ia, ib, jc = ia[keep], ib[keep], jc[keep]
        w = np.minimum(A['n'][ia], B['n'][ib]).astype(np.float64)
        change = B['pz'][ib] - A['pz'][ia]
        pred = Cs['dz'][jc]
        grp = np.rint(A['grp'][ia]).astype(int)
        out = {'pairs': int(len(ia)), 'pitches': int(w.sum()), 'predicted_from_season': int(c),
               'mean_change_in': round(float(np.average(change, weights=w) * 12.0), 3),
               'by_group': {}}
        X = np.c_[np.ones(len(ia)), pred]
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], change * np.sqrt(w), rcond=None)[0]
        rng = np.random.default_rng(int(params.get('seed', 11))); draws = []
        for _ in range(int(params.get('reps', 300))):
            j = rng.integers(0, len(ia), len(ia))
            draws.append(np.linalg.lstsq(X[j] * np.sqrt(w[j])[:, None], change[j] * np.sqrt(w[j]), rcond=None)[0][1])
        out['slope_on_predicted_drop'] = [round(float(beta[1]), 3), round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)]
        for g in range(6):
            mg = grp == g
            if mg.sum() >= 20:
                out['by_group'][GROUP_NAMES[g]] = {'pairs': int(mg.sum()), 'change_in': round(float(np.average(change[mg], weights=w[mg]) * 12.0), 3),
                                                   'predicted_in': round(float(np.average(pred[mg], weights=w[mg]) * 12.0), 3)}
        pairs[f'{a}_to_{b}'] = out
    res['same_pitcher_and_type'] = pairs
    stage('paired location change')
    zone = T['zone']; called = (T['call'] == 0) & np.isin(T['zone'], np.arange(1, 15))
    cs = T['cs'] == 1
    by_season = {}
    for s in seasons:
        m = (T['season'] == s) & (zone >= 1)
        inz = m & (zone <= 9)
        ms = called & (T['season'] == s)
        z = T['pz'][ms].astype(np.float64)
        bands = {}
        for lo, hi in ((1.2, 1.4), (1.4, 1.6), (1.6, 1.8), (3.2, 3.4), (3.4, 3.6), (3.6, 3.8)):
            b = (z >= lo) & (z < hi) & (np.abs(T['px'][ms]) <= 0.5)
            bands[f'{lo:.1f}-{hi:.1f}ft'] = {'taken': int(b.sum()), 'called_strike': round(float(cs[ms][b].mean()), 4) if b.any() else None}
        by_season[s] = {'pitches_with_zone': int(m.sum()), 'zone_missing_share': round(float(np.mean(zone[T['season'] == s] < 1)), 4),
                        'in_zone_share': round(float(inz.sum() / max(m.sum(), 1)), 4),
                        'called_strike_in_zone_1_9': round(float(cs[ms & (zone <= 9)].mean()), 4) if (ms & (zone <= 9)).any() else None,
                        'called_strike_outside_11_14': round(float(cs[ms & (zone >= 11)].mean()), 4) if (ms & (zone >= 11)).any() else None,
                        'mean_pz_ft': round(float(np.nanmean(T['pz'][T['season'] == s])), 4),
                        'called_strike_by_height_middle_third': bands}
    res['zone_field_and_calls'] = by_season
    stage('zone field')
    return res



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
    params = json.loads((ROOT / 'tools' / 'discovery_params.json').read_text())
    experiment = str(params.get('experiment') or 'horizon')
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    receipt = {'schema': 'brl.discovery-receipt.v1', 'experiment': experiment, 'run_id': run_id, 'params': params,
               'started_at': datetime.now(timezone.utc).isoformat(), 'stages': []}
    receipt['environment'] = environment_record()
    t0 = time.time()

    def stage(name):
        receipt['stages'].append({'stage': name, 'at_seconds': round(time.time() - t0, 1)}); print(name, round(time.time() - t0), 's', flush=True)
    try:
        if experiment == 'savant_pull':
            from cloud.security import seal
            sv = savant_module(); results = {}
            for year in params.get('seasons', (2024, 2025, 2026)):
                year = int(year); start, end = sv.SEASONS[year]
                stage(f'savant {year}')
                cols, notes = sv.pull(year, start, end, 'R', int(params.get('workers', 3)), log=lambda m: print(m, flush=True))
                notes['errors'] = notes['errors'][:20]
                if cols is None:
                    results[year] = {'notes': notes}; continue
                results[year] = {'notes': notes, 'coverage': sv.coverage(cols), 'plate_reference': sv.reference_check(cols), 'stored': {}}
                for part, sub in sv.split_parts(cols, year).items():
                    sealed = seal(sv.to_bytes(sub), key, sv.purpose(year, part))
                    put_bytes(repo, token, sv.path(year, part), sealed, branch, f'BRL: Savant pitches {year} part {part}')
                    results[year]['stored'][part] = {'path': sv.path(year, part), 'bytes': len(sealed), 'rows': int(len(sub['day']))}
                del cols
            receipt['results'] = results
            raise StopIteration
        if experiment == 'feed_vs_savant':
            import importlib.util
            spec = importlib.util.spec_from_file_location('brl_matchup', ROOT / 'tools' / 'brl_matchup.py')
            mx = importlib.util.module_from_spec(spec); spec.loader.exec_module(mx)
            from cloud.security import unseal
            sv = savant_module(); out = {}
            for year in params.get('seasons', (2025, 2026)):
                year = int(year)
                stage(f'load {year}')
                S = mx.merge(load_savant(repo, token, branch, key, year))
                if year == 2026:
                    keep_ = S['day'] < mx.UNTOUCHED_FROM
                    S = {k: (v[keep_] if not k.endswith('__vocab') else v) for k, v in S.items()}
                raw = read_blob(repo, token, study_path(year), branch)
                doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(year))))
                Tf = pitch_table(doc, year); del doc, raw
                if year == 2026:
                    keep_ = Tf['day'] < mx.UNTOUCHED_FROM
                    Tf = take(Tf, keep_)
                kf = (Tf['game'].astype(np.int64) * 1000 + Tf['ab'].astype(np.int64)) * 100 + Tf['pitch_no'].astype(np.int64)
                ks = (S['game_pk'].astype(np.int64) * 1000 + (S['at_bat_number'].astype(np.int64) - 1)) * 100 + (S['pitch_number'].astype(np.int64) - 1)
                common, i_f, i_s = np.intersect1d(kf, ks, return_indices=True)
                c = {k: S[k][i_s].astype(np.float64) for k in ('vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'plate_x', 'plate_z')}
                sv.anchor(c, year)
                xf, zf, _, _ = sv.at(c, sv.FRONT); xm, zm, _, _ = sv.at(c, sv.MIDDLE)
                fz, fx = Tf['pz'][i_f].astype(np.float64), Tf['px'][i_f].astype(np.float64)
                ok = np.isfinite(fz) & np.isfinite(zf) & np.isfinite(c['plate_z'])
                med = lambda v: round(float(np.median(v[ok])) * 12.0, 3)
                out[year] = {'matched': int(len(common)), 'feed_pitches': int(len(kf)), 'savant_pitches': int(len(ks)),
                             'feed_minus_savant_plate_z_in': med(fz - c['plate_z']), 'feed_minus_front_z_in': med(fz - zf), 'feed_minus_middle_z_in': med(fz - zm),
                             'abs_feed_minus_savant_plate_z_p90_in': round(float(np.percentile(np.abs(fz - c['plate_z'])[ok], 90)) * 12.0, 3),
                             'feed_minus_savant_plate_x_in': round(float(np.median((fx - c['plate_x'])[ok])) * 12.0, 3),
                             'savant_plate_reference': sv.reference_check(S)}
                stage(f'compared {year}')
            receipt['results'] = out
            raise StopIteration
        if experiment == 'contact':
            import importlib.util
            spec = importlib.util.spec_from_file_location('brl_matchup', ROOT / 'tools' / 'brl_matchup.py')
            mx = importlib.util.module_from_spec(spec); spec.loader.exec_module(mx)
            sv = savant_module(); got = []
            receipt['savant'] = {}
            for year in params.get('seasons', (2024, 2025, 2026)):
                stage(f'load savant {year}')
                parts = load_savant(repo, token, branch, key, int(year))
                if parts:
                    one = mx.merge(parts)
                    if int(year) == 2026:                     # the program's untouched set stays out of every summary
                        keep_ = one['day'] < mx.UNTOUCHED_FROM
                        one = {k: (v[keep_] if not k.endswith('__vocab') else v) for k, v in one.items()}
                    receipt['savant'][int(year)] = {'coverage': sv.coverage(one), 'plate_reference': sv.reference_check(one)}
                    del one
                got.extend(parts)
            cols = mx.merge(got); del got
            receipt['rows'] = int(len(cols['day']))
            study = params.get('study') or ('decompose' if params.get('decompose') else 'contact')
            run_study = {'contact': mx.contact_study, 'decompose': mx.contact_decompose, 'timing': mx.timing_study}.get(study)
            if run_study is None and hasattr(mx, study + '_study'):
                run_study = getattr(mx, study + '_study')
            receipt['results'] = run_study(sv, cols, params, stage)
            raise StopIteration
        if experiment == 'value_synth':
            # VALUE-17: synthetic worlds with known truth; no sealed data is read (the run is public and reproducible)
            import importlib.util
            spec = importlib.util.spec_from_file_location('brl_synth', ROOT / 'tools' / 'brl_synth.py')
            sy = importlib.util.module_from_spec(spec); spec.loader.exec_module(sy)
            receipt['results'] = sy.main(params, stage)['results']
            raise StopIteration
        if experiment == 'umpires':
            # UMP-01: the plate umpire's zone (public officials per game, cached on the data branch; offsets and persistence are the only outputs)
            import importlib.util
            spec = importlib.util.spec_from_file_location('brl_umpires', ROOT / 'tools' / 'brl_umpires.py')
            um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
            tables_ = []; ump_tables = {}
            for year in params.get('seasons', (2025, 2026)):
                stage(f'load {year}')
                raw = read_blob(repo, token, study_path(int(year)), branch)
                if raw is None:
                    continue
                doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(int(year)))))
                Ty = pitch_table(doc, int(year)); del doc, raw
                if int(year) == 2026:
                    Ty = take(Ty, Ty['day'] < date(2026, 8, 1).toordinal())
                ump_tables[int(year)] = um.load_or_build(repo, token, branch, int(year), np.unique(Ty['game']).tolist(), stage, read_blob, put_bytes, int(params.get('workers', 6)))
                tables_.append(Ty)
            T = concat(tables_); del tables_
            receipt['seasons_rows'] = {int(s): int((T['season'] == s).sum()) for s in np.unique(T['season'])}
            receipt['results'] = um.study(T, ump_tables, params, stage)
            if params.get('publish'):
                # the umpires' table for the page (model outputs about public figures, aggregated over thousands of calls)
                put_bytes(repo, token, 'public/reports/umpires.json', json.dumps(receipt['results']['umpires_table'], separators=(',', ':')).encode(), branch, 'BRL: plate umpire tendencies')
            raise StopIteration
        if experiment in ('challenges', 'scarcity'):
            stage('fetch the 2026 play-by-play')
            cdoc = challenge_study(int(params.get('season', 2026)), int(params.get('workers', 6)))
            T = pitch_table(cdoc, int(params.get('season', 2026))); X = challenge_columns(cdoc); del cdoc
            receipt['seasons_rows'] = {int(params.get('season', 2026)): int(len(T['season']))}
            assert len(X['role']) == len(T['season'])
            receipt['results'] = (challenges if experiment == 'challenges' else scarcity)(T, X, params, stage)
            raise StopIteration
        tables = []
        for year in params.get('seasons', (2023, 2024, 2025, 2026)):
            stage(f'load {year}')
            raw = read_blob(repo, token, study_path(int(year)), branch)
            if raw is None:
                continue
            doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(int(year)))))
            if doc.get('schema') != STUDY_SCHEMA:
                raise ValueError('study schema mismatch')
            tables.append(pitch_table(doc, int(year), tuple(params.get('game_types', ('R',))))); del doc, raw
        T = concat(tables); del tables
        untouched = (T['season'] == 2026) & (T['day'] >= date(2026, 8, 1).toordinal())
        if params.get('final_eval'):
            receipt['final_eval'] = {'frozen_commit': params.get('frozen_commit'), 'untouched_rows': int(untouched.sum())}
        elif params.get('product_through') and params.get('experiment') == 'scout':
            # product export only (SCOUT-03): the August-September regular-season months, scored already, are fitted;
            # postseason pitches never enter (only regular-season rows up to the date are kept)
            pt = date.fromisoformat(str(params['product_through'])).toordinal()
            keep_p = ~untouched | ((T['post'] == 0) & (T['day'] <= pt))
            receipt['product_export'] = {'through': str(params['product_through']), 'august_on_rows': int((untouched & keep_p).sum())}
            T = take(T, keep_p)
        else:
            T = take(T, ~untouched)                       # the matchup program's untouched set never enters development runs
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
        elif experiment == 'horizon11':
            receipt['results'] = horizon11(T, params, stage)
        elif experiment == 'horizon12':
            receipt['results'] = horizon12(T, params, stage)
        elif experiment == 'horizon13':
            receipt['results'] = horizon13(T, params, stage)
        elif experiment == 'horizon14':
            receipt['results'] = horizon14(T, params, stage)
        elif experiment == 'zone_audit':
            receipt['results'] = zone_audit(T, params, stage)
        elif experiment == 'matchup_swing':
            positions = None
            if params.get('body_relative'):
                import importlib.util
                spec = importlib.util.spec_from_file_location('brl_matchup', ROOT / 'tools' / 'brl_matchup.py')
                mx = importlib.util.module_from_spec(spec); spec.loader.exec_module(mx)
                pos_seasons = [int(v) for v in params.get('position_seasons', (2025,))]
                got = []
                for year in pos_seasons:
                    stage(f'load savant {year}')
                    got.extend(load_savant(repo, token, branch, key, int(year)))
                S = mx.merge(got); del got
                keep_s = S['day'] < date(2026, 8, 1).toordinal()             # the untouched months stay out
                S = {k: (v[keep_s] if not k.endswith('__vocab') else v) for k, v in S.items()}
                pos, pos_check = _batter_positions(S, pos_seasons, int(params.get('min_swings', 150)), float(params.get('grid_lo', -3.0)))
                receipt['position_check'] = pos_check; del S
                positions = {}
                for (b_, s_), v_ in sorted(pos.items(), key=lambda kv: kv[0][1]):
                    positions[b_] = v_['off_plate_in']                       # the latest position season wins
                stage(f'positions {len(positions)}')
            receipt['results'] = matchup_swing(T, params, stage, positions)
        elif experiment == 'bench_swing':
            receipt['results'] = bench_swing(T, params, stage)
        elif experiment == 'fatigue':
            receipt['results'] = fatigue(T, params, stage)
        elif experiment == 'surprise':
            receipt['results'] = surprise_study(T, params, stage)
        elif experiment == 'seq2':
            receipt['results'] = seq2_study(T, params, stage)
        elif experiment == 'plan':
            receipt['results'] = plan_study(T, params, stage)
        elif experiment == 'matchup_pa':
            receipt['results'] = matchup_pa(T, params, stage)
        elif experiment == 'exposure':
            receipt['results'] = exposure_study(T, params, stage)
        elif experiment == 'matchup_whiff':
            receipt['results'] = matchup_whiff(T, params, stage)
        elif experiment == 'engine_pa':
            receipt['results'] = engine_pa(T, params, stage)
        elif experiment == 'discipline':
            receipt['results'] = discipline_study(T, params, stage)
        elif experiment == 'exploit':
            receipt['results'] = exploit_study(T, params, stage)
        elif experiment == 'exploit2':
            receipt['results'] = exploit2_study(T, params, stage)
        elif experiment == 'value_whiff':
            receipt['results'] = value_whiff_study(T, params, stage)
        elif experiment == 'abs':
            receipt['results'] = abs_study(T, params, stage)
        elif experiment == 'abs2':
            receipt['results'] = abs2_study(T, params, stage)
        elif experiment == 'map_stability':
            receipt['results'] = map_stability_study(T, params, stage)
        elif experiment == 'series':
            receipt['results'] = series_study(T, params, stage)
        elif experiment == 'pressure':
            stage('load plate-appearance states')
            H = load_pa_states(repo, token, os.environ['BRL_PA_PACKAGE_KEY'])
            H = H[H['date_key'].astype(str).str[:10] < '2026-08-01']
            receipt['results'] = pressure_study(T, H, params, stage); del H
        elif experiment == 'framing':
            stage('load plate-appearance states')
            H = load_pa_states(repo, token, os.environ['BRL_PA_PACKAGE_KEY'])
            if not params.get('final_eval'):
                H = H[H['date_key'].astype(str).str[:10] < '2026-08-01']
            receipt['results'] = framing_study(T, H, params, stage); del H
        elif experiment == 'command2':
            receipt['results'] = command2_study(T, params, stage)
        elif experiment == 'swingmap':
            import importlib.util
            spec = importlib.util.spec_from_file_location('brl_matchup', ROOT / 'tools' / 'brl_matchup.py')
            mx = importlib.util.module_from_spec(spec); spec.loader.exec_module(mx)
            got = []
            for year in (2025, 2026):
                stage(f'load savant {year}')
                got.extend(load_savant(repo, token, branch, key, int(year)))
            S = mx.merge(got); del got
            keep_s = S['day'] < date(2026, 8, 1).toordinal()             # the untouched months stay out
            S = {k: (v[keep_s] if not k.endswith('__vocab') else v) for k, v in S.items()}
            receipt['savant_rows'] = int(keep_s.sum())
            receipt['results'] = swingmap_study(T, S, params, stage); del S
        elif experiment == 'warmup':
            stage('load plate-appearance states')
            H = load_pa_states(repo, token, os.environ['BRL_PA_PACKAGE_KEY'])
            H = H[H['date_key'].astype(str).str[:10] < '2026-08-01']
            receipt['results'] = warmup_study(T, H, params, stage); del H
        elif experiment == 'change':
            stage('load plate-appearance states')
            H = load_pa_states(repo, token, os.environ['BRL_PA_PACKAGE_KEY'])
            if not params.get('final_eval'):
                H = H[H['date_key'].astype(str).str[:10] < '2026-08-01']
            receipt['results'] = change_study(T, H, params, stage); del H
        elif experiment == 'scout':
            receipt['results'] = scout_export(T, params, stage)
        elif experiment == 'value':
            receipt['results'] = value_study(T, params, stage)
        elif experiment == 'value2':
            receipt['results'] = value2_study(T, params, stage)
        elif experiment == 'adapt':
            receipt['results'] = adapt_study(T, params, stage)
        elif experiment == 'value3':
            receipt['results'] = value3_study(T, params, stage)
        elif experiment == 'family':
            receipt['results'] = matchup_family(T, params, stage)
        elif experiment == 'damage':
            receipt['results'] = damage_study(T, params, stage)
        elif experiment == 'steer':
            receipt['results'] = steer_profile(T, params, stage)
        elif experiment == 'matchup_final':
            receipt['results'] = matchup_final(T, params, stage)
        elif experiment == 'drift':
            receipt['results'] = drift_study(T, params, stage)
        elif experiment == 'matchup_table':
            from cloud.security import seal

            def store(target, doc):
                path = f'private/matchup/table-{int(target)}.enc'
                put_bytes(repo, token, path, seal(gzip.compress(json.dumps(doc).encode()), key, f'matchup-table-{int(target)}'), branch,
                          f'BRL: decision-moment matchup table for {int(target)}')
                return path
            receipt['results'] = matchup_table(T, params, stage, store)
        receipt['status'] = 'completed'
    except StopIteration:
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
