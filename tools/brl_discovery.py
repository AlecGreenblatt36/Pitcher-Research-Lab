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
          'v0', 'v1', 'spin', 'pfx_x', 'pfx_z', 'px', 'pz', 'x0', 'z0', 'ext', 'last_in_pa', 'bunt_pa', 'ab', 'pitch_no', 'la', 'ls', 'cs', 'zone', 'out7', 'half')
OUT7 = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')
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
                cols['cs'].append(1 if code == 'C' else 0)
                try:
                    cols['zone'].append(int(_zone) if _zone is not None else -1)
                except (TypeError, ValueError):
                    cols['zone'].append(-1)
                o_ = str(row.get('o') or '')
                cols['out7'].append(OUT7.index(o_) if o_ in OUT7 else -1)
                cols['half'].append(0 if str(row.get('half') or 'top') == 'top' else 1)
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


def matchup_swing(T: dict, params: dict, stage) -> dict:
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
        out[name] = {'shrinkage_chosen': best, 'validation_logloss': {str(k): round(v, 5) for k, v in val_ll.items()},
                     'test_logloss_league': round(float(logloss_vec(p_league, yt).mean()), 5), 'test_logloss_hitter_maps': round(float(logloss_vec(p_hit, yt).mean()), 5),
                     'test_share_with_hitter_map': round(float(covered.mean()), 4)}
        stage('representation ' + name)
    lt, ht = [logloss_vec(p, yt) for p in preds['true']]
    lp, hp = [logloss_vec(p, yt) for p in preds['percept']]
    cc = lambda d: [round(v * 1000, 3) for v in clustered_ci(d, games)]
    res['representations'] = out
    res['gain_nats_per_1000_decisions'] = {'percept_over_true_league': cc(lt - lp), 'percept_over_true_hitter_maps': cc(ht - hp),
                                           'hitter_maps_over_league_true': cc(lt - ht), 'hitter_maps_over_league_percept': cc(lp - hp)}
    # matchups: chases by hitter-pitcher pair on pitches outside the zone, observed against the league-plus-additive model
    xt, zt = reps['true']
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


# ---------------------------------------------------------------- MATCHUP-02: do decision-moment matchups move plate-appearance outcomes
def matchup_pa(T: dict, params: dict, stage) -> dict:
    """Hitter maps and the league map at the decision moment as in MATCHUP-01 (fitted on 2023-2024, shrinkage 10). For
    every 2025 hitter-pitcher pair, the pitcher's 2024 pitches (his arsenal as the hitter could have known it) are run
    through the hitter's map and the league map: the matchup's expected extra swing rate on pitches outside the zone
    (chase deviation) and inside it (zone-swing deviation), beyond the additive terms. Do these predict the pair's 2025
    strikeouts and walks beyond both players' earlier strikeout and walk rates? Coefficients and out-of-sample log loss
    by game-parity cross-fitting within 2025."""
    res = {}
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)))
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
    T = take(T, np.isin(T['season'], (2023, 2024, 2025)))
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
    # run value per plate appearance on the same features (least squares), coefficients with game-clustered intervals
    rv = LW7[y7]
    names = [f'pc{i}' for i in range(pc.shape[1])] + ['tto2', 'tto3plus'] + [f'b{c}' for c in range(7)] + [f'p{c}' for c in range(7)] + ['platoon',
             'drift_v', 'drift_z', 'drift_spin100', 'drift_ext', 'has_drift', 'start_vs_norm_v', 'has_start']
    X = designs['M2_plus_drift_and_start']
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
    keep_n = ('tto2', 'tto3plus', 'drift_v', 'drift_z', 'drift_spin100', 'drift_ext', 'start_vs_norm_v')
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
            receipt['results'] = mx.contact_study(sv, cols, params, stage)
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
            receipt['results'] = matchup_swing(T, params, stage)
        elif experiment == 'fatigue':
            receipt['results'] = fatigue(T, params, stage)
        elif experiment == 'surprise':
            receipt['results'] = surprise_study(T, params, stage)
        elif experiment == 'matchup_pa':
            receipt['results'] = matchup_pa(T, params, stage)
        elif experiment == 'exposure':
            receipt['results'] = exposure_study(T, params, stage)
        elif experiment == 'matchup_whiff':
            receipt['results'] = matchup_whiff(T, params, stage)
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
