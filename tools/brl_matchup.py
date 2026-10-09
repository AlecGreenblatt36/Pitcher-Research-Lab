"""Matchup physics experiments on Savant pitch data (discovery/MATCHUP_PROGRAM.md), run inside Actions.

contact_study (CONTACT-01): does the angle between a hitter's swing plane and the pitch's approach, scaled by his
timing spread, predict misses beyond the pitch's own quality and the hitter's own rates?

    theta = AA - |VAA|,   F = |sin(theta)| * sigma_y   (inches),

AA the hitter's usual attack angle for a pitch at that height (earlier swings only), VAA the pitch's approach angle at
the front of the plate from its fitted flight, sigma_y the spread of his contact depth (earlier contact swings only).

Only metrics leave the runner. Pitches from August 1, 2026 on are the program's untouched set and are dropped here
unless params['final_eval'] names the frozen commit that is being scored.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np

UNTOUCHED_FROM = date(2026, 8, 1).toordinal()
WHIFF = ('swinging_strike', 'swinging_strike_blocked')
CONTACT = ('foul', 'foul_tip', 'hit_into_play')
GROUPS = {'FF': 0, 'FA': 0, 'SI': 1, 'FT': 1, 'FC': 2, 'SL': 3, 'ST': 3, 'SV': 3, 'CU': 4, 'KC': 4, 'CS': 4, 'CH': 5, 'FS': 5, 'FO': 5, 'SC': 5}


def merge(parts: list) -> dict:
    """Concatenate packed Savant column sets (each with its own category vocabularies) into one. A column missing from
    a part (bat tracking before it was recorded, for instance) is filled with NaN (numbers) or -1 (integers)."""
    parts = [p for p in parts if p]
    keys = set()
    for p in parts:
        keys |= set(p)
    out = {}
    for k in sorted(keys):
        if k.endswith('__vocab'):
            continue
        if k + '__vocab' in keys:
            vocab = np.unique(np.concatenate([p[k + '__vocab'] for p in parts if k + '__vocab' in p] + [np.array([''])]))
            chunks = []
            for p in parts:
                n = len(p['day'])
                chunks.append(np.searchsorted(vocab, p[k + '__vocab'][p[k]]) if k in p else np.full(n, int(np.searchsorted(vocab, ''))))
            out[k] = np.concatenate(chunks).astype(np.int16); out[k + '__vocab'] = vocab
        else:
            ref = next(p[k] for p in parts if k in p)
            fill = np.nan if ref.dtype.kind == 'f' else -1
            out[k] = np.concatenate([p[k] if k in p else np.full(len(p['day']), fill, dtype=ref.dtype) for p in parts])
    return out


def label(cols, name):
    return cols[name + '__vocab'][cols[name]]


def guard(cols: dict, params: dict) -> dict:
    """Drop the untouched set unless this run is the registered final evaluation of a frozen candidate."""
    if params.get('final_eval'):
        return cols
    keep = cols['day'] < UNTOUCHED_FROM
    return {k: (v[keep] if not k.endswith('__vocab') else v) for k, v in cols.items()}


def geometry(sv, cols: dict) -> dict:
    """Front-of-plate crossing and approach angles from each pitch's fitted flight, every season on one reference."""
    year = np.asarray([date.fromordinal(int(d)).year for d in cols['day']])
    x = np.full(len(year), np.nan); z = np.full(len(year), np.nan); vaa = np.full(len(year), np.nan); haa = np.full(len(year), np.nan)
    for y in np.unique(year):
        m = year == y
        c = {k: cols[k][m].astype(np.float64) for k in ('vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'plate_x', 'plate_z')}
        sv.anchor(c, int(y))
        xf, zf, _, v = sv.at(c, sv.FRONT)
        x[m], z[m] = xf, zf
        vaa[m] = np.degrees(np.arctan2(v['vz'], -v['vy'])); haa[m] = np.degrees(np.arctan2(v['vx'], -v['vy']))
    return {'x': x, 'z': z, 'vaa': vaa, 'haa': haa, 'year': year}


def prior_stats(key: np.ndarray, day: np.ndarray, value: np.ndarray, use: np.ndarray):
    """For every row: count, sum and sum of squares of `value` over rows of the same key on earlier days where `use`."""
    n = len(key)
    order = np.lexsort((day, key))
    k, d = key[order], day[order]
    v = np.where(use[order], np.nan_to_num(value[order]), 0.0)
    u = use[order].astype(np.float64)
    cn, cs, cq = np.cumsum(u), np.cumsum(v), np.cumsum(v * v)
    # first row of each (key, day): everything before it is strictly earlier
    first = np.ones(n, bool); first[1:] = (k[1:] != k[:-1]) | (d[1:] != d[:-1])
    start_key = np.ones(n, bool); start_key[1:] = k[1:] != k[:-1]
    idx_first = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    idx_key = np.maximum.accumulate(np.where(start_key, np.arange(n), 0))
    def before(c):
        at_first = np.where(idx_first > 0, c[idx_first - 1], 0.0)
        at_key = np.where(idx_key > 0, c[idx_key - 1], 0.0)
        return at_first - at_key
    out_n, out_s, out_q = np.empty(n), np.empty(n), np.empty(n)
    out_n[order], out_s[order], out_q[order] = before(cn), before(cs), before(cq)
    return out_n, out_s, out_q


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clustered(diff, clusters, reps=300, seed=7):
    uc, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uc)); cnt = np.bincount(inv, minlength=len(uc))
    rng = np.random.default_rng(seed); draws = np.empty(reps)
    for r in range(reps):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        draws[r] = (w * sums).sum() / max((w * cnt).sum(), 1)
    return [round(float(diff.mean()) * 1000, 3), round(float(np.percentile(draws, 2.5)) * 1000, 3), round(float(np.percentile(draws, 97.5)) * 1000, 3)]


def offset_fit(base_logit, X, y, l2=1e-4, iters=50):
    """Logistic regression of y on X with base_logit as offset (Newton steps); returns coefficients (with intercept)."""
    A = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(A.shape[1])
    for _ in range(iters):
        eta = base_logit + A @ b
        p = 1 / (1 + np.exp(-eta))
        g = A.T @ (y - p) - l2 * b
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.eye(A.shape[1])
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def contact_study(sv, cols: dict, params: dict, stage) -> dict:
    res = {}
    cols = guard(cols, params)
    desc = label(cols, 'description')
    G = geometry(sv, cols)
    stage('geometry')
    swing = np.isin(desc, WHIFF + CONTACT)
    whiff = np.isin(desc, WHIFF)
    contact = np.isin(desc, CONTACT)
    batter = cols['batter'].astype(np.int64); day = cols['day'].astype(np.int64)
    aa = cols.get('attack_angle'); iy = cols.get('intercept_ball_minus_batter_pos_y_inches')
    if aa is None or iy is None:
        return {'error': 'bat tracking fields missing', 'columns': sorted(k for k in cols if not k.endswith('__vocab'))}
    has_aa = np.isfinite(aa) & swing
    has_iy = np.isfinite(iy) & swing
    res['coverage'] = {
        'swings': int(swing.sum()),
        'attack_angle_on_whiffs': round(float(has_aa[whiff].mean()), 4) if whiff.any() else None,
        'attack_angle_on_contact': round(float(has_aa[contact].mean()), 4) if contact.any() else None,
        'intercept_on_whiffs': round(float(has_iy[whiff].mean()), 4) if whiff.any() else None,
        'intercept_on_contact': round(float(has_iy[contact].mean()), 4) if contact.any() else None,
        'by_year_aa_on_swings': {int(y): round(float(has_aa[swing & (G['year'] == y)].mean()), 4) for y in np.unique(G['year'])},
        'vaa_mean_by_group': {}}
    ptype = label(cols, 'pitch_type'); grp = np.asarray([GROUPS.get(t, 6) for t in ptype])
    for g in range(6):
        m = (grp == g) & np.isfinite(G['vaa'])
        if m.any():
            res['coverage']['vaa_mean_by_group'][g] = round(float(np.nanmean(G['vaa'][m])), 3)
    # league attack-angle slope on pitch height (earlier seasons only, to keep the test season out)
    test_year = int(params.get('test_year', 2026))
    train_rows = has_aa & (G['year'] < test_year) & np.isfinite(G['z'])
    zc = G['z'] - 2.5
    slope = float(np.polyfit(zc[train_rows], aa[train_rows], 1)[0]) if train_rows.sum() > 1000 else 0.0
    res['league_aa_slope_per_ft'] = round(slope, 3)
    # hitter priors from earlier days: attack angle (height-adjusted), contact-depth spread, whiff rate, bat speed
    aa_adj = aa - slope * zc
    n_a, s_a, _ = prior_stats(batter, day, aa_adj, has_aa & np.isfinite(zc))
    league_aa = float(np.nanmean(aa_adj[train_rows])) if train_rows.any() else 10.0
    k_aa = float(params.get('k_aa', 50.0))
    mu_aa = (s_a + k_aa * league_aa) / (n_a + k_aa)
    ci = contact & np.isfinite(iy)
    n_y, s_y, q_y = prior_stats(batter, day, iy, ci)
    lv = float(np.nanvar(iy[ci & (G['year'] < test_year)])) if (ci & (G['year'] < test_year)).any() else 36.0
    k_y = float(params.get('k_y', 50.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        var_h = np.where(n_y > 1, (q_y - s_y * s_y / np.maximum(n_y, 1)) / np.maximum(n_y - 1, 1), lv)
    sig_y = np.sqrt((np.maximum(n_y - 1, 0) * var_h + k_y * lv) / (np.maximum(n_y - 1, 0) + k_y))
    n_w, s_w, _ = prior_stats(batter, day, whiff.astype(float), swing)
    lw = float(whiff[swing & (G['year'] < test_year)].mean())
    wr = (s_w + 200 * lw) / (n_w + 200)
    bs = cols.get('bat_speed')
    n_b, s_b, _ = prior_stats(batter, day, bs if bs is not None else np.zeros(len(day)), swing & np.isfinite(bs) if bs is not None else np.zeros(len(day), bool))
    lb = float(np.nanmean(bs[swing & (G['year'] < test_year)])) if bs is not None else 70.0
    bat_speed = (s_b + 50 * lb) / (n_b + 50)
    stage('hitter priors')
    theta = (mu_aa + slope * zc) - np.abs(G['vaa'])                 # degrees; the hitter's usual angle at this height
    F = np.abs(np.sin(np.radians(theta))) * sig_y                   # inches
    # competitor: the same construction with height in place of the pitch's approach (league approach at that height)
    ok = swing & np.isfinite(F) & np.isfinite(G['x']) & np.isfinite(G['z']) & np.isfinite(cols['release_speed'])
    league_vaa_by_z = np.polyfit(zc[ok & (G['year'] < test_year)], G['vaa'][ok & (G['year'] < test_year)], 2)
    theta_h = (mu_aa + slope * zc) - np.abs(np.polyval(league_vaa_by_z, zc))
    F_h = np.abs(np.sin(np.radians(theta_h))) * sig_y
    stand_r = label(cols, 'stand') == 'R'; throw_r = label(cols, 'p_throws') == 'R'
    u = np.where(stand_r, G['x'], -G['x'])
    X_pitch = np.column_stack([cols['release_speed'], cols['pfx_x'] * np.where(throw_r, 1, -1), cols['pfx_z'], u, G['z'], G['vaa'], G['haa'] * np.where(stand_r, 1, -1),
                               cols['release_spin_rate'], cols['release_extension'], cols['release_pos_z'], cols['balls'], cols['strikes'], grp,
                               (stand_r == throw_r).astype(float)] + ([cols['arm_angle']] if 'arm_angle' in cols else []))
    tr = ok & (G['year'] < test_year) & (n_y >= float(params.get('min_prior_contacts', 30)))
    te = ok & (G['year'] == test_year) & (n_y >= float(params.get('min_prior_contacts', 30)))
    res['rows'] = {'train_swings': int(tr.sum()), 'test_swings': int(te.sum()), 'test_whiff_rate': round(float(whiff[te].mean()), 4) if te.any() else None,
                   'sigma_y_league_in': round(math.sqrt(lv), 3), 'F_quantiles_in': [round(float(np.nanpercentile(F[te], q)), 3) for q in (10, 25, 50, 75, 90)] if te.any() else None}
    if tr.sum() < 20000 or te.sum() < 5000:
        res['error'] = 'too few swings with priors'; return res
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = whiff.astype(float)
    hp = dict(max_iter=int(params.get('gbm_iter', 300)), learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0, random_state=11)
    X_rate = np.column_stack([np.log(wr / (1 - wr)), bat_speed])          # the hitter's own rates, no swing geometry
    X_geo = np.column_stack([mu_aa, sig_y])                               # his swing geometry
    designs = {'G0_pitch': X_pitch, 'G1_pitch_rates': np.column_stack([X_pitch, X_rate]),
               'G2_pitch_rates_geometry': np.column_stack([X_pitch, X_rate, X_geo]),
               'G3_G2_plus_F': np.column_stack([X_pitch, X_rate, X_geo, F])}
    fits = {}
    for name, X in designs.items():
        m = HistGradientBoostingClassifier(**hp).fit(X[tr], y[tr])
        fits[name] = m.predict_proba(X[te])[:, 1]
        stage('fit ' + name)
    games = cols['game_pk'][te]
    yt = y[te]
    ll = {k: logloss(v, yt) for k, v in fits.items()}
    res['test_logloss_per_swing'] = {k: round(float(v.mean()), 5) for k, v in ll.items()}
    res['gain_nats_per_1000'] = {'rates_over_pitch': clustered(ll['G0_pitch'] - ll['G1_pitch_rates'], games),
                                 'geometry_over_rates_flexible': clustered(ll['G1_pitch_rates'] - ll['G2_pitch_rates_geometry'], games),
                                 'F_on_top_of_flexible_geometry': clustered(ll['G2_pitch_rates_geometry'] - ll['G3_G2_plus_F'], games)}
    # the one-parameter mechanism on top of the rates model: the coefficient on F is fitted on the training swings'
    # cross-fitted logits (games split by parity, each half predicted by a model fitted on the other) and applied to the
    # test year's logits from the rates model fitted on all training swings
    XB = designs['G1_pitch_rates']
    oof = np.full(len(y), np.nan)
    par = (cols['game_pk'] % 2 == 0)
    for side in (True, False):
        fit_rows, pred_rows = tr & (par == side), tr & (par != side)
        mdl = HistGradientBoostingClassifier(**hp).fit(XB[fit_rows], y[fit_rows])
        oof[pred_rows] = mdl.predict_proba(XB[pred_rows])[:, 1]
    stage('cross-fitted logits')
    lo_tr = np.log(np.clip(oof[tr], 1e-6, 1 - 1e-6) / np.clip(1 - oof[tr], 1e-6, 1))
    p_te = fits['G1_pitch_rates']; lo_te = np.log(p_te / (1 - p_te))
    flex = ll['G1_pitch_rates'] - ll['G2_pitch_rates_geometry']
    for name, f in (('F_vaa', F), ('F_height', F_h)):
        b = offset_fit(lo_tr, f[tr][:, None], y[tr])
        pt = 1 / (1 + np.exp(-(lo_te + b[0] + b[1] * f[te])))
        g = ll['G1_pitch_rates'] - logloss(pt, yt)
        res.setdefault('one_parameter_mechanism', {})[name] = {
            'coef_per_in': round(float(b[1]), 4), 'intercept': round(float(b[0]), 4),
            'gain_over_rates_nats_per_1000': clustered(g, games),
            'share_of_flexible_geometry_gain': round(float(g.mean() / flex.mean()), 3) if flex.mean() > 0 else None}
    # does the spread of timing modulate the angle (P4): |sin theta| split by the hitter's sigma_y
    s_med = float(np.median(sig_y[te]))
    lo_all = lo_te
    out = {}
    for name, m in (('tight_timing', sig_y[te] <= s_med), ('loose_timing', sig_y[te] > s_med)):
        st = np.abs(np.sin(np.radians(theta[te])))[m]
        b = offset_fit(lo_all[m], st[:, None], yt[m])
        out[name] = {'coef_per_unit_sin': round(float(b[1]), 4), 'swings': int(m.sum()), 'sigma_y_mean_in': round(float(sig_y[te][m].mean()), 3)}
    res['angle_effect_by_timing_spread'] = out
    # whiff rate by F quintile, observed against the flexible model without F
    qs = np.nanpercentile(F[te], [20, 40, 60, 80])
    qi = np.searchsorted(qs, F[te])
    res['by_F_quintile'] = [{'F_in': round(float(F[te][qi == k].mean()), 3), 'swings': int((qi == k).sum()), 'observed': round(float(yt[qi == k].mean()), 4),
                             'predicted_rates_model': round(float(p_te[qi == k].mean()), 4),
                             'predicted_flexible_geometry': round(float(fits['G2_pitch_rates_geometry'][qi == k].mean()), 4)} for k in range(5)]
    # realized swings (mechanism check where the angle exists on the swing itself)
    real = te & has_aa
    if real.sum() > 5000:
        th_r = aa[real] - np.abs(G['vaa'][real])
        fr = np.abs(np.sin(np.radians(th_r))) * sig_y[real]
        lo_r = np.log(fits['G1_pitch_rates'][has_aa[te]] / (1 - fits['G1_pitch_rates'][has_aa[te]]))
        b = offset_fit(lo_r, fr[:, None], y[real])
        res['realized_angle_check'] = {'swings': int(real.sum()), 'whiff_share': round(float(y[real].mean()), 4), 'coef_per_in': round(float(b[1]), 4),
                                       'theta_sd_deg': round(float(np.std(th_r)), 3)}
    stage('contact study')
    return res


def contact_decompose(sv, cols: dict, params: dict, stage) -> dict:
    """CONTACT-02: which part of a hitter's swing geometry carries the miss information CONTACT-01 found (1.04 nats per
    1,000 swings in the flexible model). Parametric interaction terms on top of the rates model's cross-fitted logits,
    all together and each one left out; and the flexible model with more geometry (the hitter's own attack-angle
    spread, swing path tilt, attack direction). Terms: timing spread (sigma_y) alone, with speed, and with the late
    change in approach angle (the plane the hitter sees at his steering limit against the plane the ball arrives on);
    usual attack angle with height, with approach angle, with vertical movement; the plane formula F; attack-angle
    spread alone and with height."""
    res = {}
    cols = guard(cols, params)
    desc = label(cols, 'description')
    G = geometry(sv, cols)
    swing = np.isin(desc, WHIFF + CONTACT); whiff = np.isin(desc, WHIFF); contact = np.isin(desc, CONTACT)
    batter = cols['batter'].astype(np.int64); day = cols['day'].astype(np.int64)
    aa = cols['attack_angle']; iy = cols['intercept_ball_minus_batter_pos_y_inches']
    has_aa = np.isfinite(aa) & swing
    test_year = int(params.get('test_year', 2026))
    train_rows = has_aa & (G['year'] < test_year) & np.isfinite(G['z'])
    zc = G['z'] - 2.5
    slope = float(np.polyfit(zc[train_rows], aa[train_rows], 1)[0])
    aa_adj = aa - slope * zc
    n_a, s_a, q_a = prior_stats(batter, day, aa_adj, has_aa & np.isfinite(zc))
    league_aa = float(np.nanmean(aa_adj[train_rows])); la_var = float(np.nanvar(aa_adj[train_rows]))
    mu_aa = (s_a + 50 * league_aa) / (n_a + 50)
    with np.errstate(invalid='ignore', divide='ignore'):
        v_aa = np.where(n_a > 1, (q_a - s_a * s_a / np.maximum(n_a, 1)) / np.maximum(n_a - 1, 1), la_var)
    aa_sd = np.sqrt((np.maximum(n_a - 1, 0) * v_aa + 50 * la_var) / (np.maximum(n_a - 1, 0) + 50))
    ci = contact & np.isfinite(iy)
    n_y, s_y, q_y = prior_stats(batter, day, iy, ci)
    lv = float(np.nanvar(iy[ci & (G['year'] < test_year)]))
    with np.errstate(invalid='ignore', divide='ignore'):
        var_h = np.where(n_y > 1, (q_y - s_y * s_y / np.maximum(n_y, 1)) / np.maximum(n_y - 1, 1), lv)
    sig_y = np.sqrt((np.maximum(n_y - 1, 0) * var_h + 50 * lv) / (np.maximum(n_y - 1, 0) + 50))
    def prior_mean(v, k=50.0):
        ok_ = swing & np.isfinite(v)
        nn, ss, _ = prior_stats(batter, day, v, ok_)
        lg = float(np.nanmean(v[ok_ & (G['year'] < test_year)]))
        return (ss + k * lg) / (nn + k)
    stand_r = label(cols, 'stand') == 'R'; throw_r = label(cols, 'p_throws') == 'R'
    tilt_h = prior_mean(cols['swing_path_tilt'])
    dir_h = prior_mean(np.where(stand_r, 1.0, -1.0) * cols['attack_direction'])
    n_w, s_w, _ = prior_stats(batter, day, whiff.astype(float), swing)
    lw = float(whiff[swing & (G['year'] < test_year)].mean()); wr = (s_w + 200 * lw) / (n_w + 200)
    bat_speed = prior_mean(cols['bat_speed'])
    stage('hitter priors')
    # late change in approach angle: the ball's vertical speed at the plate against a gravity-only flight from the steering limit
    c = {k: cols[k].astype(np.float64) for k in ('vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'plate_x', 'plate_z')}
    _, _, _, vel = sv.at({**c, '_x50': np.zeros(len(day)), '_z50': np.zeros(len(day))}, sv.FRONT)
    dvaa = {t: np.degrees(((cols['az'].astype(np.float64) + 32.174) * t) / np.abs(vel['vy'])) for t in (0.11, 0.26)}
    theta = (mu_aa + slope * zc) - np.abs(G['vaa'])
    F = np.abs(np.sin(np.radians(theta))) * sig_y
    v = cols['release_speed'].astype(np.float64)
    terms = {'sigma_y': sig_y / 10, 'sigma_y_x_speed': sig_y / 10 * (v - 90) / 5, 'sigma_y_x_late_plane': sig_y / 10 * np.abs(dvaa[0.11]),
             'aa_x_height': mu_aa / 10 * zc, 'aa_x_vaa': mu_aa / 10 * (G['vaa'] + 6), 'aa_x_ivb': mu_aa / 10 * cols['pfx_z'] / 10,
             'F': F, 'aa_spread': aa_sd / 10, 'aa_spread_x_height': aa_sd / 10 * zc}
    u = np.where(stand_r, G['x'], -G['x'])
    X_pitch = np.column_stack([v, cols['pfx_x'] * np.where(throw_r, 1, -1), cols['pfx_z'], u, G['z'], G['vaa'], G['haa'] * np.where(stand_r, 1, -1),
                               cols['release_spin_rate'], cols['release_extension'], cols['release_pos_z'], cols['balls'], cols['strikes'],
                               np.asarray([GROUPS.get(t, 6) for t in label(cols, 'pitch_type')]), (stand_r == throw_r).astype(float), cols['arm_angle']])
    X_rate = np.column_stack([np.log(wr / (1 - wr)), bat_speed])
    ok = swing & np.isfinite(F) & np.isfinite(G['z']) & np.isfinite(v) & np.all(np.isfinite(np.column_stack(list(terms.values()))), axis=1)
    tr = ok & (G['year'] < test_year) & (n_y >= 30); te = ok & (G['year'] == test_year) & (n_y >= 30)
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = whiff.astype(float)
    hp = dict(max_iter=int(params.get('gbm_iter', 300)), learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0, random_state=11)
    designs = {'G1': np.column_stack([X_pitch, X_rate]), 'G2': np.column_stack([X_pitch, X_rate, mu_aa, sig_y]),
               'G2_plus': np.column_stack([X_pitch, X_rate, mu_aa, sig_y, aa_sd, tilt_h, dir_h])}
    preds = {k: HistGradientBoostingClassifier(**hp).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] for k, X in designs.items()}
    stage('flexible')
    yt = y[te]; games = cols['game_pk'][te]
    ll = {k: logloss(p_, yt) for k, p_ in preds.items()}
    res['flexible_gain_nats_per_1000'] = {'G2_over_G1': clustered(ll['G1'] - ll['G2'], games), 'G2plus_over_G2': clustered(ll['G2'] - ll['G2_plus'], games)}
    # training noise: the same comparisons refitted on bootstrap resamples of training games with feature subsampling
    tg = np.unique(cols['game_pk'][tr]); gi = np.searchsorted(tg, cols['game_pk'][tr]); tri = np.flatnonzero(tr)
    reps = {'G2_over_G1': [], 'G2plus_over_G2': []}
    for k in range(int(params.get('refits', 3))):
        rng = np.random.default_rng(100 + k)
        w = np.bincount(rng.integers(0, len(tg), len(tg)), minlength=len(tg))[gi]
        rows = np.repeat(tri, w)
        hk = dict(hp, random_state=100 + k, max_features=0.85)
        lk = {d: logloss(HistGradientBoostingClassifier(**hk).fit(X[rows], y[rows]).predict_proba(X[te])[:, 1], yt).mean() for d, X in designs.items()}
        reps['G2_over_G1'].append((lk['G1'] - lk['G2']) * 1000); reps['G2plus_over_G2'].append((lk['G2'] - lk['G2_plus']) * 1000)
        stage(f'refit {k}')
    res['flexible_gain_refits'] = {k: {'values': [round(v, 3) for v in vs], 'mean': round(float(np.mean(vs)), 3), 'sd': round(float(np.std(vs, ddof=1)), 3) if len(vs) > 1 else None}
                                   for k, vs in reps.items()}
    # parametric terms on the rates model's cross-fitted logits
    XB = designs['G1']; oof = np.full(len(y), np.nan); par = cols['game_pk'] % 2 == 0
    for side in (True, False):
        fr, pr = tr & (par == side), tr & (par != side)
        oof[pr] = HistGradientBoostingClassifier(**hp).fit(XB[fr], y[fr]).predict_proba(XB[pr])[:, 1]
    lo_tr = np.log(np.clip(oof[tr], 1e-6, 1 - 1e-6) / np.clip(1 - oof[tr], 1e-6, 1))
    p1 = preds['G1']; lo_te = np.log(p1 / (1 - p1))
    names = list(terms)
    Ttr = np.column_stack([terms[k][tr] for k in names]); Tte = np.column_stack([terms[k][te] for k in names])
    def gain(cols_idx):
        b = offset_fit(lo_tr, Ttr[:, cols_idx], y[tr])
        pt = 1 / (1 + np.exp(-(lo_te + b[0] + Tte[:, cols_idx] @ b[1:])))
        return ll['G1'] - logloss(pt, yt), b
    g_all, b_all = gain(list(range(len(names))))
    res['parametric_all'] = {'gain': clustered(g_all, games), 'share_of_G2_gain': round(float(g_all.mean() / (ll['G1'] - ll['G2']).mean()), 3),
                             'coefs': {n_: round(float(c_), 4) for n_, c_ in zip(names, b_all[1:])}}
    drops = {}
    for j, n_ in enumerate(names):
        g_j, _ = gain([k for k in range(len(names)) if k != j])
        drops[n_] = round(float((g_all - g_j).mean()) * 1000, 3)
    res['drop_one_loss_nats_per_1000'] = dict(sorted(drops.items(), key=lambda kv: -kv[1]))
    singles = {}
    for j, n_ in enumerate(names):
        g_j, b_j = gain([j])
        singles[n_] = {'gain': round(float(g_j.mean()) * 1000, 3), 'coef': round(float(b_j[1]), 4)}
    res['single_term'] = singles
    res['rows'] = {'train': int(tr.sum()), 'test': int(te.sum()), 'late_plane_change_deg_sd': round(float(np.nanstd(dvaa[0.11][te])), 3)}
    stage('parametric')
    return res


# ---------------------------------------------------------------- TIMING-01: timing carried from one pitch to the next
def flight_time(sv, cols: dict) -> np.ndarray:
    """Seconds from release (60.5 ft minus extension) to the front of the plate along each pitch's fitted flight."""
    c = {k: cols[k].astype(np.float64) for k in ('vy0', 'ay')}
    return sv._time_to(c, sv.FRONT) - sv._time_to(c, 60.5 - cols['release_extension'].astype(np.float64))


def _previous(cols: dict, lag: int = 1) -> np.ndarray:
    """Index of the pitch `lag` pitches earlier in the same plate appearance (-1 when there is none)."""
    n = len(cols['day'])
    o = np.lexsort((cols['pitch_number'], cols['at_bat_number'], cols['game_pk']))
    g, ab, pn = cols['game_pk'][o], cols['at_bat_number'][o], cols['pitch_number'][o]
    prev = np.full(n, -1, np.int64)
    if n > lag:
        same = (g[lag:] == g[:-lag]) & (ab[lag:] == ab[:-lag]) & (pn[lag:] == pn[:-lag] + lag)
        prev[o[lag:]] = np.where(same, o[:-lag], -1)
    return prev


def _ols(X, y, clusters, reps=200, seed=7):
    """Least squares with a bootstrap over clusters: (coefficients, 2.5 and 97.5 percentiles)."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    uc, inv = np.unique(clusters, return_inverse=True)
    p = X.shape[1]; XtX = np.zeros((len(uc), p, p))
    for j in range(p):
        for k in range(j, p):
            v = np.bincount(inv, weights=X[:, j] * X[:, k], minlength=len(uc)); XtX[:, j, k] = v; XtX[:, k, j] = v
    Xty = np.column_stack([np.bincount(inv, weights=X[:, j] * y, minlength=len(uc)) for j in range(p)])
    rng = np.random.default_rng(seed); draws = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc)).astype(float)
        draws.append(np.linalg.solve(np.tensordot(w, XtX, 1) + 1e-9 * np.eye(X.shape[1]), w @ Xty))
    d = np.asarray(draws)
    return b, np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)


def timing_study(sv, cols: dict, params: dict, stage) -> dict:
    """TIMING-01. A hitter times his swing to the flight he expects. If part of that expectation is carried over from
    the previous pitch of the plate appearance, a pitch slower than the one before (flight time longer by dt) finds him
    early by a share alpha of dt, and contact is made farther in front of him by about u * alpha * dt, u the speed at
    which the meeting point moves along the flight per unit of timing error (bat and ball speeds v_s, v_b:
    u = v_s * v_b / (v_s + v_b), about 56 ft/s, 6.8 inches per 10 ms). Measured on contact swings from Savant's
    contact depth (intercept, inches toward the pitcher) against dt, holding the pitch's own flight time, type,
    location, count and platoon fixed and the hitter's usual depth removed: the slope per 10 ms over 6.8 is the share
    of the previous pitch's timing carried into this swing. Then: after swings against after takes, two pitches back,
    each hitter's own carryover (its reliability across halves of a season), and whether carryover and the contact
    window (the spread of his contact depth, CONTACT-02) move misses on speed changes beyond a flexible model that
    already sees the speed change. Fitted on 2025, checked on 2026 through July."""
    res = {}
    cols = guard(cols, params)
    desc = label(cols, 'description')
    swing = np.isin(desc, WHIFF + CONTACT); whiff = np.isin(desc, WHIFF); contact = np.isin(desc, CONTACT)
    batter = cols['batter'].astype(np.int64); day = cols['day'].astype(np.int64)
    G = geometry(sv, cols)
    year = G['year']
    ft = flight_time(sv, cols) * 1000.0                                   # ms
    p1, p2 = _previous(cols, 1), _previous(cols, 2)
    has1, has2 = p1 >= 0, p2 >= 0
    dt1 = np.where(has1, ft - ft[np.maximum(p1, 0)], np.nan); dt2 = np.where(has2, ft - ft[np.maximum(p2, 0)], np.nan)
    prev_swing = np.where(has1, swing[np.maximum(p1, 0)], False)
    ptype = label(cols, 'pitch_type'); grp = np.asarray([GROUPS.get(t, 6) for t in ptype])
    prev_grp = np.where(has1, grp[np.maximum(p1, 0)], -1)
    stand_r = label(cols, 'stand') == 'R'; throw_r = label(cols, 'p_throws') == 'R'
    u = np.where(stand_r, G['x'], -G['x']); z = G['z']
    iy = cols['intercept_ball_minus_batter_pos_y_inches'].astype(np.float64)
    test_year = int(params.get('test_year', 2026)); train_year = int(params.get('train_year', 2025))
    ci = contact & np.isfinite(iy)
    n_y, s_y, q_y = prior_stats(batter, day, iy, ci)
    lm = float(np.nanmean(iy[ci & (year == train_year)])); lv = float(np.nanvar(iy[ci & (year == train_year)]))
    mu_y = (s_y + 50 * lm) / (n_y + 50)
    with np.errstate(invalid='ignore', divide='ignore'):
        var_h = np.where(n_y > 1, (q_y - s_y * s_y / np.maximum(n_y, 1)) / np.maximum(n_y - 1, 1), lv)
    sig_y = np.sqrt((np.maximum(n_y - 1, 0) * var_h + 50 * lv) / (np.maximum(n_y - 1, 0) + 50))
    stage('sequence and priors')
    ok = ci & has1 & np.isfinite(dt1) & np.isfinite(ft) & np.isfinite(u) & np.isfinite(z) & (n_y >= 20)
    def design(rows, extra):
        gd = np.column_stack([(grp[rows] == g_).astype(float) for g_ in range(1, 7)])
        base = np.column_stack([np.ones(rows.sum()), (ft[rows] - 400.0) / 10.0, gd, u[rows], np.abs(u[rows]), z[rows] - 2.5, (z[rows] - 2.5) ** 2,
                                cols['balls'][rows], cols['strikes'][rows], (stand_r == throw_r)[rows].astype(float)])
        return np.column_stack([base] + extra)
    U_IN_PER_10MS = float(params.get('u_ft_s', 56.0)) * 12.0 * 0.01
    depth = {}
    for yr in (train_year, test_year):
        r = ok & (year == yr)
        if r.sum() < 5000:
            continue
        y = iy[r] - mu_y[r]
        X = design(r, [dt1[r][:, None] / 10.0])
        b, lo, hi = _ols(X, y, cols['game_pk'][r])
        Xs = design(r, [dt1[r][:, None] / 10.0, (dt1[r] * prev_swing[r])[:, None] / 10.0])
        bs, los, his = _ols(Xs, y, cols['game_pk'][r])
        r2 = r & has2 & np.isfinite(dt2)
        X2 = design(r2, [dt1[r2][:, None] / 10.0, (dt2[r2] - dt1[r2])[:, None] / 10.0])
        b2, lo2, hi2 = _ols(X2, iy[r2] - mu_y[r2], cols['game_pk'][r2])
        depth[yr] = {'contacts': int(r.sum()), 'dt_sd_ms': round(float(np.std(dt1[r])), 2),
                     'slope_in_per_10ms': [round(float(b[-1]), 3), round(float(lo[-1]), 3), round(float(hi[-1]), 3)],
                     'carryover_share': [round(float(b[-1] / U_IN_PER_10MS), 4), round(float(lo[-1] / U_IN_PER_10MS), 4), round(float(hi[-1] / U_IN_PER_10MS), 4)],
                     'own_flight_time_slope_in_per_10ms': round(float(b[1]), 3),
                     'after_take_slope': [round(float(bs[-2]), 3), round(float(los[-2]), 3), round(float(his[-2]), 3)],
                     'after_swing_extra_slope': [round(float(bs[-1]), 3), round(float(los[-1]), 3), round(float(his[-1]), 3)],
                     'two_back': {'contacts': int(r2.sum()), 'slope_previous': round(float(b2[-2]), 3),
                                  'slope_two_back_beyond_previous': [round(float(b2[-1]), 3), round(float(lo2[-1]), 3), round(float(hi2[-1]), 3)]}}
        stage(f'depth {yr}')
    res['contact_depth'] = depth
    if train_year not in depth:
        res['error'] = 'too few contact swings'; return res
    # each hitter's own carryover: shrunk slope of his depth residuals on dt (pooled controls), reliability across halves
    r = ok & (year == train_year)
    X = design(r, [dt1[r][:, None] / 10.0]); y = iy[r] - mu_y[r]
    b = np.linalg.lstsq(X, y, rcond=None)[0]; resid = y - X @ b; x = dt1[r] / 10.0
    s2 = float(np.var(resid))
    hit = batter[r]
    mid = date(train_year, 7, 1).toordinal(); half = day[r] >= mid
    def slopes(sel):
        uh, ih = np.unique(hit[sel], return_inverse=True)
        sxx = np.bincount(ih, weights=x[sel] ** 2, minlength=len(uh)); sxr = np.bincount(ih, weights=x[sel] * resid[sel], minlength=len(uh))
        nn = np.bincount(ih, minlength=len(uh))
        return uh, sxx, sxr, nn
    uh, sxx, sxr, nn = slopes(np.ones(len(x), bool))
    big = nn >= 100
    raw = sxr[big] / sxx[big]; se2 = s2 / sxx[big]
    tau2 = max(float(np.var(raw) - np.mean(se2)), 1e-4)
    lam = s2 / tau2
    beta_h = {int(h): float(b[-1] + sxr[i] / (sxx[i] + lam)) for i, h in enumerate(uh)}
    rel = {}
    a_, b_ = slopes(~half), slopes(half)
    da = {int(h): (a_[2][i] / (a_[1][i] + lam), a_[3][i]) for i, h in enumerate(a_[0])}
    db = {int(h): (b_[2][i] / (b_[1][i] + lam), b_[3][i]) for i, h in enumerate(b_[0])}
    common = [h for h in da if h in db and da[h][1] >= 60 and db[h][1] >= 60]
    if len(common) > 20:
        va = np.asarray([da[h][0] for h in common]); vb = np.asarray([db[h][0] for h in common])
        rel = {'hitters': len(common), 'corr_halves': round(float(np.corrcoef(va, vb)[0, 1]), 3)}
    res['hitter_carryover'] = {'hitters_with_100_contacts': int(big.sum()), 'between_hitter_sd_in_per_10ms': round(float(np.sqrt(tau2)), 3),
                               'shrinkage_lambda': round(float(lam), 1), 'reliability': rel}
    stage('hitter carryover')
    # misses on speed changes: flexible base that sees the speed change, then carryover and window interactions
    sw_ok = swing & has1 & np.isfinite(dt1) & np.isfinite(ft) & np.isfinite(u) & np.isfinite(z) & np.isfinite(cols['release_speed']) & (n_y >= 20)
    n_w, s_w, _ = prior_stats(batter, day, whiff.astype(float), swing)
    lw = float(whiff[swing & (year == train_year)].mean()); wr = (s_w + 200 * lw) / (n_w + 200)
    X_pitch = np.column_stack([cols['release_speed'], cols['pfx_x'] * np.where(throw_r, 1, -1), cols['pfx_z'], u, z, G['vaa'], G['haa'] * np.where(stand_r, 1, -1),
                               cols['release_spin_rate'], cols['release_extension'], cols['release_pos_z'], cols['balls'], cols['strikes'], grp,
                               (stand_r == throw_r).astype(float), ft, dt1, np.abs(dt1), prev_grp, prev_swing.astype(float), np.log(wr / (1 - wr))])
    bh = np.asarray([beta_h.get(int(h_), np.nan) for h_ in batter])
    tr = sw_ok & (year == train_year) & np.isfinite(bh); te = sw_ok & (year == test_year) & np.isfinite(bh)
    res['swings'] = {'train': int(tr.sum()), 'test': int(te.sum())}
    if tr.sum() < 20000 or te.sum() < 5000:
        res['error'] = 'too few swings'; return res
    from sklearn.ensemble import HistGradientBoostingClassifier
    yv = whiff.astype(float)
    hp = dict(max_iter=int(params.get('gbm_iter', 300)), learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0, random_state=11)
    oof = np.full(len(yv), np.nan); par = cols['game_pk'] % 2 == 0
    for side in (True, False):
        fr, pr = tr & (par == side), tr & (par != side)
        oof[pr] = HistGradientBoostingClassifier(**hp).fit(X_pitch[fr], yv[fr]).predict_proba(X_pitch[pr])[:, 1]
    p_te = HistGradientBoostingClassifier(**hp).fit(X_pitch[tr], yv[tr]).predict_proba(X_pitch[te])[:, 1]
    lo_tr = np.log(np.clip(oof[tr], 1e-6, 1 - 1e-6) / np.clip(1 - oof[tr], 1e-6, 1)); lo_te = np.log(np.clip(p_te, 1e-6, 1 - 1e-6) / np.clip(1 - p_te, 1e-6, 1))
    stage('whiff base')
    adt = np.abs(dt1) / 10.0
    # the base model sees neither the hitter's carryover nor his window: each enters with its own main term and its
    # product with the size of the speed change
    terms = {'carryover': bh - b[-1], 'carryover_x_speed_change': (bh - b[-1]) * adt,
             'window': (sig_y - math.sqrt(lv)) / 10.0, 'window_x_speed_change': (sig_y - math.sqrt(lv)) / 10.0 * adt}
    yt = yv[te]; games = cols['game_pk'][te]; base_ll = logloss(p_te, yt)
    out = {}
    for name, cols_ in (('carryover', ['carryover', 'carryover_x_speed_change']), ('window', ['window', 'window_x_speed_change']), ('both', list(terms))):
        Ttr = np.column_stack([terms[k][tr] for k in cols_]); Tte = np.column_stack([terms[k][te] for k in cols_])
        bb = offset_fit(lo_tr, Ttr, yv[tr])
        pt = 1 / (1 + np.exp(-(lo_te + bb[0] + Tte @ bb[1:])))
        # coefficient interval: refit on bootstrap resamples of training games
        tg = np.unique(cols['game_pk'][tr]); gi = np.searchsorted(tg, cols['game_pk'][tr]); bsd = []
        rng = np.random.default_rng(5)
        for _ in range(int(params.get('reps', 60))):
            w = np.bincount(rng.integers(0, len(tg), len(tg)), minlength=len(tg))[gi]; sel = np.repeat(np.arange(int(tr.sum())), w)
            bsd.append(offset_fit(lo_tr[sel], Ttr[sel], yv[tr][sel], iters=20)[1:])
        bsd = np.asarray(bsd)
        out[name] = {'coefs': {k: [round(float(bb[1 + j]), 4), round(float(np.percentile(bsd[:, j], 2.5)), 4), round(float(np.percentile(bsd[:, j], 97.5)), 4)] for j, k in enumerate(cols_)},
                     'gain_nats_per_1000_swings': clustered(base_ll - logloss(pt, yt), games)}
    res['whiffs_on_speed_changes'] = out
    # observed whiff rate by speed change and the hitter's carryover (top against bottom third), against the base model
    qb = np.nanpercentile(bh[te], [33.3, 66.7]); hi_c = bh[te] >= qb[1]; lo_c = bh[te] <= qb[0]
    big_dt = np.abs(dt1[te]) >= 30
    res['table_big_speed_change'] = {nm: {'swings': int((m & big_dt).sum()), 'observed': round(float(yt[m & big_dt].mean()), 4), 'base_model': round(float(p_te[m & big_dt].mean()), 4)}
                                     for nm, m in (('high_carryover', hi_c), ('low_carryover', lo_c))}
    stage('whiffs')
    return res


# ---------------------------------------------------------------- CONTACT-03: the frozen swing-geometry miss model, scored once
def contact_final_study(sv, cols: dict, params: dict, stage) -> dict:
    """CONTACT-03. CONTACT-02's frozen recipe: the rates model (gradient boosting on the pitch and the hitter's whiff rate
    and bat speed) plus one term, the hitter's contact-depth spread from earlier contact swings (shrunk, k = 50), its
    coefficient fitted on the rates model's cross-fitted logits. Refitted on all development swings (2025 and 2026
    before August 1) and scored once on every swing from August 1, 2026 (the program's untouched set). Also reported:
    the flexible model with the usual attack angle and the spread (CONTACT-01's G2) on the same swings."""
    if not params.get('final_eval') or not params.get('frozen_commit'):
        raise ValueError('contact_final runs only as the registered final evaluation of a frozen commit')
    res = {}
    desc = label(cols, 'description')
    G = geometry(sv, cols)
    swing = np.isin(desc, WHIFF + CONTACT); whiff = np.isin(desc, WHIFF); contact = np.isin(desc, CONTACT)
    batter = cols['batter'].astype(np.int64); day = cols['day'].astype(np.int64)
    dev = day < UNTOUCHED_FROM; unt = ~dev
    aa = cols['attack_angle']; iy = cols['intercept_ball_minus_batter_pos_y_inches']
    has_aa = np.isfinite(aa) & swing
    zc = G['z'] - 2.5
    rows_aa = has_aa & dev & np.isfinite(zc)
    slope = float(np.polyfit(zc[rows_aa], aa[rows_aa], 1)[0])
    aa_adj = aa - slope * zc
    n_a, s_a, _ = prior_stats(batter, day, aa_adj, has_aa & np.isfinite(zc))
    mu_aa = (s_a + 50 * float(np.nanmean(aa_adj[rows_aa]))) / (n_a + 50)
    ci = contact & np.isfinite(iy)
    n_y, s_y, q_y = prior_stats(batter, day, iy, ci)
    lv = float(np.nanvar(iy[ci & dev]))
    with np.errstate(invalid='ignore', divide='ignore'):
        var_h = np.where(n_y > 1, (q_y - s_y * s_y / np.maximum(n_y, 1)) / np.maximum(n_y - 1, 1), lv)
    sig_y = np.sqrt((np.maximum(n_y - 1, 0) * var_h + 50 * lv) / (np.maximum(n_y - 1, 0) + 50))
    n_w, s_w, _ = prior_stats(batter, day, whiff.astype(float), swing)
    lw = float(whiff[swing & dev].mean()); wr = (s_w + 200 * lw) / (n_w + 200)
    bs = cols['bat_speed']; okb = swing & np.isfinite(bs)
    n_b, s_b, _ = prior_stats(batter, day, bs, okb)
    bat_speed = (s_b + 50 * float(np.nanmean(bs[okb & dev]))) / (n_b + 50)
    stage('hitter priors')
    stand_r = label(cols, 'stand') == 'R'; throw_r = label(cols, 'p_throws') == 'R'
    u = np.where(stand_r, G['x'], -G['x']); v = cols['release_speed'].astype(np.float64)
    X_pitch = np.column_stack([v, cols['pfx_x'] * np.where(throw_r, 1, -1), cols['pfx_z'], u, G['z'], G['vaa'], G['haa'] * np.where(stand_r, 1, -1),
                               cols['release_spin_rate'], cols['release_extension'], cols['release_pos_z'], cols['balls'], cols['strikes'],
                               np.asarray([GROUPS.get(t, 6) for t in label(cols, 'pitch_type')]), (stand_r == throw_r).astype(float), cols['arm_angle']])
    X_rate = np.column_stack([np.log(wr / (1 - wr)), bat_speed])
    ok = swing & np.isfinite(sig_y) & np.isfinite(G['z']) & np.isfinite(v) & (n_y >= 30)
    tr, te = ok & dev, ok & unt
    res['rows'] = {'development_swings': int(tr.sum()), 'untouched_swings': int(te.sum()), 'untouched_whiff_rate': round(float(whiff[te].mean()), 4) if te.any() else None,
                   'untouched_days': [str(date.fromordinal(int(day[te].min()))), str(date.fromordinal(int(day[te].max())))] if te.any() else None}
    if tr.sum() < 20000 or te.sum() < 5000:
        res['error'] = 'too few swings'; return res
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = whiff.astype(float)
    hp = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0, random_state=11)
    XB = np.column_stack([X_pitch, X_rate]); XG = np.column_stack([X_pitch, X_rate, mu_aa, sig_y])
    p1 = HistGradientBoostingClassifier(**hp).fit(XB[tr], y[tr]).predict_proba(XB[te])[:, 1]
    p2 = HistGradientBoostingClassifier(**hp).fit(XG[tr], y[tr]).predict_proba(XG[te])[:, 1]
    stage('flexible')
    oof = np.full(len(y), np.nan); par = cols['game_pk'] % 2 == 0
    for side in (True, False):
        fr, pr = tr & (par == side), tr & (par != side)
        oof[pr] = HistGradientBoostingClassifier(**hp).fit(XB[fr], y[fr]).predict_proba(XB[pr])[:, 1]
    lo_tr = np.log(np.clip(oof[tr], 1e-6, 1 - 1e-6) / np.clip(1 - oof[tr], 1e-6, 1)); lo_te = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1))
    term_tr = (sig_y[tr] / 10.0)[:, None]; term_te = (sig_y[te] / 10.0)[:, None]
    b = offset_fit(lo_tr, term_tr, y[tr])
    tg = np.unique(cols['game_pk'][tr]); gi = np.searchsorted(tg, cols['game_pk'][tr]); rng = np.random.default_rng(5); bsd = []
    for _ in range(int(params.get('reps', 60))):
        w = np.bincount(rng.integers(0, len(tg), len(tg)), minlength=len(tg))[gi]; sel = np.repeat(np.arange(int(tr.sum())), w)
        bsd.append(offset_fit(lo_tr[sel], term_tr[sel], y[tr][sel], iters=20)[1])
    pt = 1 / (1 + np.exp(-(lo_te + b[0] + term_te @ b[1:])))
    yt = y[te]; games = cols['game_pk'][te]
    l1, l2, lt = logloss(p1, yt), logloss(p2, yt), logloss(pt, yt)
    res['frozen_term'] = {'coef_per_10_in': [round(float(b[1]), 4), round(float(np.percentile(bsd, 2.5)), 4), round(float(np.percentile(bsd, 97.5)), 4)],
                          'gain_over_rates_nats_per_1000_swings': clustered(l1 - lt, games)}
    res['flexible_geometry_gain_nats_per_1000_swings'] = clustered(l1 - l2, games)
    res['test_logloss_per_swing'] = {'rates': round(float(l1.mean()), 5), 'rates_plus_spread': round(float(lt.mean()), 5), 'flexible_geometry': round(float(l2.mean()), 5)}
    qs = np.percentile(sig_y[te], [20, 40, 60, 80]); qi = np.searchsorted(qs, sig_y[te])
    res['by_spread_quintile'] = [{'spread_in': round(float(sig_y[te][qi == k].mean()), 3), 'swings': int((qi == k).sum()), 'observed': round(float(yt[qi == k].mean()), 4),
                                  'rates_model': round(float(p1[qi == k].mean()), 4), 'frozen_model': round(float(pt[qi == k].mean()), 4)} for k in range(5)]
    stage('scored')
    return res


# ---------------------------------------------------------------- TIMING-02: carried timing within each pitcher's pitch type
def _demean(v: np.ndarray, gi: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    return v - (np.bincount(gi, weights=v, minlength=len(cnt)) / np.maximum(cnt, 1))[gi]


def _hitter_slopes(hit, x, r, s2, min_n=100):
    """Shrunk per-hitter slopes of residuals r on x (deviations from the pooled slope), the shrinkage set by the
    between-hitter variance of raw slopes beyond their sampling variance."""
    uh, ih = np.unique(hit, return_inverse=True)
    sxx = np.bincount(ih, weights=x * x, minlength=len(uh)); sxr = np.bincount(ih, weights=x * r, minlength=len(uh)); nn = np.bincount(ih, minlength=len(uh))
    big = (nn >= min_n) & (sxx > 0)
    if big.sum() < 20:
        return {}, None, None
    raw = sxr[big] / sxx[big]; se2 = s2 / sxx[big]
    tau2 = max(float(np.var(raw) - np.mean(se2)), 1e-6)
    lam = s2 / tau2
    return {int(h): float(sxr[i] / (sxx[i] + lam)) for i, h in enumerate(uh)}, float(np.sqrt(tau2)), float(lam)


def timing2_study(sv, cols: dict, params: dict, stage) -> dict:
    """TIMING-02. TIMING-01 measured carried timing against the speed change with the pitch's own flight time entering
    linearly beside its type; a pitcher-level effect on contact depth (a pitcher's pitches share his speed level) or a
    hitter's own way of following pitch speed can then be read as carried timing. Here (a) carried timing is measured
    within one pitcher's one pitch type in one season (fixed effects; the flight times centered within it): the same
    pitch preceded by different pitches, after a take, a foul or a miss, and two pitches back; (b) two hitter traits
    from 2025 contact swings, each a shrunk slope: speed-follow, how far his contact moves out front per 10 ms of
    slower flight across one pitcher's pitches (0 for perfect re-timing, 6.7 inches for none), and carried, his own
    after-swing slope on the previous pitch within pitcher-types; their reliability across the halves of 2025; (c) their
    value for misses in 2026 through July on top of a flexible whiff model that sees the speed change."""
    res = {}
    cols = guard(cols, params)
    desc = label(cols, 'description')
    swing = np.isin(desc, WHIFF + CONTACT); whiff = np.isin(desc, WHIFF); contact = np.isin(desc, CONTACT)
    batter = cols['batter'].astype(np.int64); day = cols['day'].astype(np.int64)
    G = geometry(sv, cols); year = G['year']
    ft = flight_time(sv, cols) * 1000.0
    p1, p2 = _previous(cols, 1), _previous(cols, 2)
    has1, has2 = p1 >= 0, p2 >= 0
    f1 = np.where(has1, ft[np.maximum(p1, 0)], np.nan); f2 = np.where(has2, ft[np.maximum(p2, 0)], np.nan)
    sw1 = np.where(has1, swing[np.maximum(p1, 0)], False); wh1 = np.where(has1, whiff[np.maximum(p1, 0)], False)
    sw2 = np.where(has2, swing[np.maximum(p2, 0)], False)
    ptype = label(cols, 'pitch_type'); grp = np.asarray([GROUPS.get(t, 6) for t in ptype])
    tcode = np.unique(ptype, return_inverse=True)[1]
    stand_r = label(cols, 'stand') == 'R'; throw_r = label(cols, 'p_throws') == 'R'
    u = np.where(stand_r, G['x'], -G['x']); z = G['z']
    iy = cols['intercept_ball_minus_batter_pos_y_inches'].astype(np.float64)
    train_year, test_year = int(params.get('train_year', 2025)), int(params.get('test_year', 2026))
    ci = contact & np.isfinite(iy)
    n_y, s_y, q_y = prior_stats(batter, day, iy, ci)
    lm = float(np.nanmean(iy[ci & (year == train_year)])); lv = float(np.nanvar(iy[ci & (year == train_year)]))
    mu_y = (s_y + 50 * lm) / (n_y + 50)
    with np.errstate(invalid='ignore', divide='ignore'):
        var_h = np.where(n_y > 1, (q_y - s_y * s_y / np.maximum(n_y, 1)) / np.maximum(n_y - 1, 1), lv)
    sig_y = np.sqrt((np.maximum(n_y - 1, 0) * var_h + 50 * lv) / (np.maximum(n_y - 1, 0) + 50))
    pkey = cols['pitcher'].astype(np.int64) * 100 + (year - 2000)                       # pitcher-season
    gkey = pkey * 1000 + tcode.astype(np.int64)                                          # pitcher-season-type
    def center(v, key, rows):
        ug, gi = np.unique(key[rows], return_inverse=True); cnt = np.bincount(gi)
        out = np.full(len(v), np.nan); out[rows] = _demean(v[rows], gi, cnt); return out
    base_ok = ci & np.isfinite(ft) & np.isfinite(u) & np.isfinite(z) & (n_y >= 20)
    ok = base_ok & has1 & np.isfinite(f1)
    ftc = center(ft, gkey, base_ok) / 10.0                                               # own flight, within pitcher-type
    f1c = center(np.where(ok, f1, 0.0), gkey, ok) / 10.0                                 # previous flight, within pitcher-type
    stage('sequence and priors')
    U10 = float(params.get('u_ft_s', 56.0)) * 12.0 * 0.01

    def within(rows, key, cols_):
        ug, gi = np.unique(key[rows], return_inverse=True); cnt = np.bincount(gi)
        keep = cnt[gi] >= 2
        names = list(cols_)
        X = np.column_stack([_demean(np.asarray(cols_[k], dtype=np.float64), gi, cnt) for k in names])[keep]
        y = _demean(iy[rows] - mu_y[rows], gi, cnt)[keep]
        return names, X, y, keep

    def controls(r):
        return {'side': u[r], 'side_abs': np.abs(u[r]), 'height': z[r] - 2.5, 'height_sq': (z[r] - 2.5) ** 2, 'balls': cols['balls'][r].astype(float),
                'strikes': cols['strikes'][r].astype(float), 'platoon': (stand_r == throw_r)[r].astype(float)}
    depth = {}
    for yr in (train_year, test_year):
        r = ok & (year == yr)
        if r.sum() < 5000:
            continue
        c_ = {'own_flight': ftc[r], 'prev_flight': f1c[r], 'prev_flight_x_swung': f1c[r] * sw1[r], 'prev_flight_x_missed': f1c[r] * wh1[r],
              'prev_swung': sw1[r].astype(float), 'prev_missed': wh1[r].astype(float), **controls(r)}
        names, X, y, keep = within(r, gkey, c_)
        b, lo, hi = _ols(X, y, cols['game_pk'][r][keep]); j = {n_: k for k, n_ in enumerate(names)}
        ci_ = lambda n_: [round(float(b[j[n_]]), 4), round(float(lo[j[n_]]), 4), round(float(hi[j[n_]]), 4)]
        r2 = r & has2 & np.isfinite(f2)
        f2c = center(np.where(r2, f2, 0.0), gkey, r2) / 10.0
        c2 = {'own_flight': ftc[r2], 'prev_flight': f1c[r2], 'prev_flight_x_swung': f1c[r2] * sw1[r2], 'two_back': f2c[r2], 'two_back_x_swung': f2c[r2] * sw2[r2],
              'prev_swung': sw1[r2].astype(float), 'two_back_swung': sw2[r2].astype(float), **controls(r2)}
        names2, X2, y2, keep2 = within(r2, gkey, c2)
        b2, lo2, hi2 = _ols(X2, y2, cols['game_pk'][r2][keep2]); j2 = {n_: k for k, n_ in enumerate(names2)}
        ci2 = lambda n_: [round(float(b2[j2[n_]]), 4), round(float(lo2[j2[n_]]), 4), round(float(hi2[j2[n_]]), 4)]
        # speed-follow, pooled: within pitcher-season (across his pitch types), own flight centered there
        rs = base_ok & (year == yr)
        cs_ = {'own_flight_pitcher': center(ft, pkey, rs)[rs] / 10.0, **controls(rs)}
        names3, X3, y3, keep3 = within(rs, pkey, cs_)
        b3, lo3, hi3 = _ols(X3, y3, cols['game_pk'][rs][keep3])
        depth[yr] = {'contacts': int(keep.sum()), 'pitcher_types': int(len(np.unique(gkey[r]))),
                     'own_flight_within_type_in_per_10ms': ci_('own_flight'),
                     'prev_flight_after_take': ci_('prev_flight'), 'prev_flight_extra_after_swing': ci_('prev_flight_x_swung'),
                     'prev_flight_extra_after_miss': ci_('prev_flight_x_missed'),
                     'carried_share_after_take': round(float(-b[j['prev_flight']] / U10), 4),
                     'carried_share_after_foul': round(float(-(b[j['prev_flight']] + b[j['prev_flight_x_swung']]) / U10), 4),
                     'carried_share_after_miss': round(float(-(b[j['prev_flight']] + b[j['prev_flight_x_swung']] + b[j['prev_flight_x_missed']]) / U10), 4),
                     'two_back': {'contacts': int(keep2.sum()), 'prev_after_take': ci2('prev_flight'), 'prev_extra_after_swing': ci2('prev_flight_x_swung'),
                                  'two_back_after_take': ci2('two_back'), 'two_back_extra_after_swing': ci2('two_back_x_swung')},
                     'speed_follow_within_pitcher': {'contacts': int(keep3.sum()), 'in_per_10ms': [round(float(b3[0]), 4), round(float(lo3[0]), 4), round(float(hi3[0]), 4)],
                                                     'share_not_retimed': round(float(b3[0] / U10), 4)}}
        stage(f'depth {yr}')
    res['contact_depth'] = depth
    if train_year not in depth:
        res['error'] = 'too few contact swings'; return res
    # hitter traits on 2025
    rel, traits = {}, {}
    rs = base_ok & (year == train_year)
    cs_ = {'own_flight_pitcher': center(ft, pkey, rs)[rs] / 10.0, **controls(rs)}
    names3, X3, y3, keep3 = within(rs, pkey, cs_)
    b3 = np.linalg.lstsq(X3, y3, rcond=None)[0]; res3 = y3 - X3 @ b3
    r = ok & (year == train_year)
    c_ = {'own_flight': ftc[r], 'prev_flight': f1c[r], 'prev_flight_x_swung': f1c[r] * sw1[r], 'prev_swung': sw1[r].astype(float), **controls(r)}
    names, X, y, keep = within(r, gkey, c_)
    b = np.linalg.lstsq(X, y, rcond=None)[0]; resid = y - X @ b
    for nm, hit, x, rr, dd in (('speed_follow', batter[rs][keep3], X3[:, 0], res3, day[rs][keep3]),
                               ('carried', batter[r][keep], (f1c[r] * sw1[r])[keep], resid, day[r][keep])):
        s2 = float(np.var(rr)); half = dd >= date(train_year, 7, 1).toordinal()
        full, sd, lam = _hitter_slopes(hit, x, rr, s2)
        a_, _, _ = _hitter_slopes(hit[~half], x[~half], rr[~half], s2, 60)
        b_, _, _ = _hitter_slopes(hit[half], x[half], rr[half], s2, 60)
        common = [h for h in a_ if h in b_]
        rel[nm] = {'between_hitter_sd_in_per_10ms': round(sd, 4) if sd else None, 'lambda': round(lam, 1) if lam else None, 'hitters': len(full),
                   'hitters_both_halves': len(common),
                   'corr_halves': round(float(np.corrcoef([a_[h] for h in common], [b_[h] for h in common])[0, 1]), 3) if len(common) > 20 else None}
        traits[nm] = full
    res['hitter_traits'] = rel
    stage('hitter traits')
    if not traits['speed_follow'] or not traits['carried']:
        res['error'] = 'traits not estimable'; return res
    sw_ok = swing & has1 & np.isfinite(f1) & np.isfinite(ft) & np.isfinite(u) & np.isfinite(z) & np.isfinite(cols['release_speed']) & (n_y >= 20)
    n_w, s_w, _ = prior_stats(batter, day, whiff.astype(float), swing)
    lw = float(whiff[swing & (year == train_year)].mean()); wr = (s_w + 200 * lw) / (n_w + 200)
    dt1 = ft - f1
    prev_grp = np.where(has1, grp[np.maximum(p1, 0)], -1)
    allp = np.isfinite(ft)
    slow = center(ft, pkey, allp) / 10.0                                                   # this pitch's flight against the pitcher's average
    X_pitch = np.column_stack([cols['release_speed'], cols['pfx_x'] * np.where(throw_r, 1, -1), cols['pfx_z'], u, z, G['vaa'], G['haa'] * np.where(stand_r, 1, -1),
                               cols['release_spin_rate'], cols['release_extension'], cols['release_pos_z'], cols['balls'], cols['strikes'], grp,
                               (stand_r == throw_r).astype(float), ft, slow, dt1, np.abs(dt1), prev_grp, sw1.astype(float), np.log(wr / (1 - wr))])
    t_sf = np.asarray([traits['speed_follow'].get(int(h_), np.nan) for h_ in batter]); t_car = np.asarray([traits['carried'].get(int(h_), np.nan) for h_ in batter])
    tr = sw_ok & (year == train_year) & np.isfinite(t_sf) & np.isfinite(t_car) & np.isfinite(slow)
    te = sw_ok & (year == test_year) & np.isfinite(t_sf) & np.isfinite(t_car) & np.isfinite(slow)
    res['swings'] = {'train': int(tr.sum()), 'test': int(te.sum())}
    if tr.sum() < 20000 or te.sum() < 5000:
        res['error'] = 'too few swings'; return res
    from sklearn.ensemble import HistGradientBoostingClassifier
    yv = whiff.astype(float)
    hp = dict(max_iter=int(params.get('gbm_iter', 300)), learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0, random_state=11)
    oof = np.full(len(yv), np.nan); par = cols['game_pk'] % 2 == 0
    for side in (True, False):
        fr, pr = tr & (par == side), tr & (par != side)
        oof[pr] = HistGradientBoostingClassifier(**hp).fit(X_pitch[fr], yv[fr]).predict_proba(X_pitch[pr])[:, 1]
    p_te = HistGradientBoostingClassifier(**hp).fit(X_pitch[tr], yv[tr]).predict_proba(X_pitch[te])[:, 1]
    lo_tr = np.log(np.clip(oof[tr], 1e-6, 1 - 1e-6) / np.clip(1 - oof[tr], 1e-6, 1)); lo_te = np.log(np.clip(p_te, 1e-6, 1 - 1e-6) / np.clip(1 - p_te, 1e-6, 1))
    stage('whiff base')
    adt = np.abs(dt1) / 10.0
    terms = {'speed_follow': t_sf, 'speed_follow_x_slow': t_sf * slow, 'speed_follow_x_speed_change': t_sf * adt,
             'carried': t_car, 'carried_x_speed_change_after_swing': t_car * adt * sw1}
    yt = yv[te]; games = cols['game_pk'][te]; base_ll = logloss(p_te, yt)
    tg = np.unique(cols['game_pk'][tr]); gi_ = np.searchsorted(tg, cols['game_pk'][tr])
    out = {}
    for name, cl in (('speed_follow', ['speed_follow', 'speed_follow_x_slow', 'speed_follow_x_speed_change']),
                     ('carried', ['carried', 'carried_x_speed_change_after_swing']),
                     ('both', ['speed_follow', 'speed_follow_x_slow', 'speed_follow_x_speed_change', 'carried', 'carried_x_speed_change_after_swing'])):
        Ttr = np.column_stack([terms[k][tr] for k in cl]); Tte = np.column_stack([terms[k][te] for k in cl])
        bb = offset_fit(lo_tr, Ttr, yv[tr])
        pt = 1 / (1 + np.exp(-(lo_te + bb[0] + Tte @ bb[1:])))
        rng = np.random.default_rng(5); bsd = []
        for _ in range(int(params.get('reps', 60))):
            w = np.bincount(rng.integers(0, len(tg), len(tg)), minlength=len(tg))[gi_]; sel = np.repeat(np.arange(int(tr.sum())), w)
            bsd.append(offset_fit(lo_tr[sel], Ttr[sel], yv[tr][sel], iters=20)[1:])
        bsd = np.asarray(bsd)
        out[name] = {'coefs': {k: [round(float(bb[1 + q]), 4), round(float(np.percentile(bsd[:, q], 2.5)), 4), round(float(np.percentile(bsd[:, q], 97.5)), 4)] for q, k in enumerate(cl)},
                     'gain_nats_per_1000_swings': clustered(base_ll - logloss(pt, yt), games)}
    res['whiffs'] = out
    res['trait_sd_on_test_swings'] = {'speed_follow': round(float(np.std(t_sf[te])), 4), 'carried': round(float(np.std(t_car[te])), 4),
                                      'corr': round(float(np.corrcoef(t_sf[te], t_car[te])[0, 1]), 3)}
    # observed misses on slow pitches (at least 25 ms slower than the pitcher's average) by speed-follow thirds
    q = np.nanpercentile(t_sf[te], [33.3, 66.7]); sl = slow[te] >= 2.5
    res['slow_pitches_by_speed_follow'] = {nm: {'swings': int((m & sl).sum()), 'observed': round(float(yt[m & sl].mean()), 4), 'base_model': round(float(p_te[m & sl].mean()), 4)}
                                           for nm, m in (('follows_most', t_sf[te] >= q[1]), ('follows_least', t_sf[te] <= q[0]))}
    stage('whiffs')
    return res


# ---------------------------------------------------------------- TIMING-03: the sequence effects with fuller controls
def timing3_study(sv, cols: dict, params: dict, stage) -> dict:
    """TIMING-03 (development). TIMING-02 found contact met earlier after a slower pitch was taken (hitters expecting the
    other speed) and later after a slower pitch was missed (the swing's timing persisting). Here, within pitcher-type-
    seasons as before, with the count as twelve categories, the pitch number, the previous pitch's location (side,
    height, in or out of the zone) and its outcome in four kinds (taken for a ball, taken for a strike, fouled, missed),
    each with its own slope on the previous pitch's flight time; then the same with the previous pitch's type family
    (fastball, breaking, offspeed) added, to see whether the speed or the type is what hitters key on; and alternations
    (previous pitch of another family) against repeats. Misses: speed-follow (TIMING-02) against the contact window
    (CONTACT-03), alone and together, on 2026 through July; the full-season slope 'speed_follow' computed as in TIMING-02."""
    res = {}
    cols = guard(cols, params)
    desc = label(cols, 'description')
    swing = np.isin(desc, WHIFF + CONTACT); whiff = np.isin(desc, WHIFF); contact = np.isin(desc, CONTACT)
    batter = cols['batter'].astype(np.int64); day = cols['day'].astype(np.int64)
    G = geometry(sv, cols); year = G['year']
    ft = flight_time(sv, cols) * 1000.0
    p1 = _previous(cols, 1); has1 = p1 >= 0; q1 = np.maximum(p1, 0)
    f1 = np.where(has1, ft[q1], np.nan)
    called = desc == 'called_strike'; ball = np.isin(desc, ('ball', 'blocked_ball'))
    k_ball, k_cs, k_foul, k_miss = (has1 & ball[q1]), (has1 & called[q1]), (has1 & contact[q1]), (has1 & whiff[q1])
    ptype = label(cols, 'pitch_type'); grp = np.asarray([GROUPS.get(t, 6) for t in ptype])
    fam = np.where(grp <= 2, 0, np.where(grp <= 4, 1, np.where(grp == 5, 2, 3)))
    fam1 = np.where(has1, fam[q1], -1)
    tcode = np.unique(ptype, return_inverse=True)[1]
    stand_r = label(cols, 'stand') == 'R'; throw_r = label(cols, 'p_throws') == 'R'
    u = np.where(stand_r, G['x'], -G['x']); z = G['z']
    u1 = np.where(has1, u[q1], np.nan); z1 = np.where(has1, z[q1], np.nan)
    inz1 = (np.abs(u1) <= 0.83) & (z1 >= 1.5) & (z1 <= 3.5)
    iy = cols['intercept_ball_minus_batter_pos_y_inches'].astype(np.float64)
    train_year, test_year = int(params.get('train_year', 2025)), int(params.get('test_year', 2026))
    ci = contact & np.isfinite(iy)
    n_y, s_y, q_y = prior_stats(batter, day, iy, ci)
    lm = float(np.nanmean(iy[ci & (year == train_year)])); lv = float(np.nanvar(iy[ci & (year == train_year)]))
    mu_y = (s_y + 50 * lm) / (n_y + 50)
    with np.errstate(invalid='ignore', divide='ignore'):
        var_h = np.where(n_y > 1, (q_y - s_y * s_y / np.maximum(n_y, 1)) / np.maximum(n_y - 1, 1), lv)
    sig_y = np.sqrt((np.maximum(n_y - 1, 0) * var_h + 50 * lv) / (np.maximum(n_y - 1, 0) + 50))
    pkey = cols['pitcher'].astype(np.int64) * 100 + (year - 2000); gkey = pkey * 1000 + tcode.astype(np.int64)
    def center(v, key, rows):
        ug, gi = np.unique(key[rows], return_inverse=True); cnt = np.bincount(gi)
        out = np.full(len(v), np.nan); out[rows] = _demean(v[rows], gi, cnt); return out
    base_ok = ci & np.isfinite(ft) & np.isfinite(u) & np.isfinite(z) & (n_y >= 20)
    ok = base_ok & has1 & np.isfinite(f1) & np.isfinite(u1) & np.isfinite(z1) & (k_ball | k_cs | k_foul | k_miss)
    ftc = center(ft, gkey, base_ok) / 10.0
    f1c = center(np.where(ok, f1, 0.0), gkey, ok) / 10.0
    cnt_code = np.clip(cols['balls'], 0, 3) * 3 + np.clip(cols['strikes'], 0, 2)
    pno = np.clip(cols['pitch_number'], 1, 8).astype(float)
    stage('sequence and priors')

    kind = np.where(k_ball, 0, np.where(k_cs, 1, np.where(k_foul, 2, 3)))

    def within(rows, cell, cols_):
        # fixed effects for every cell (pitcher-type-season by the previous pitch's outcome kind, and more when asked):
        # the previous pitch's flight is compared only among pitches of the same type, same pitcher and season, after
        # the same kind of previous outcome
        ug, gi = np.unique(cell[rows], return_inverse=True); cnt = np.bincount(gi); keep = cnt[gi] >= 2
        names = list(cols_)
        X = np.column_stack([_demean(np.asarray(cols_[k], dtype=np.float64), gi, cnt) for k in names])[keep]
        return names, X, _demean(iy[rows] - mu_y[rows], gi, cnt)[keep], keep

    def design(r):
        c_ = {'own_flight': ftc[r]}
        for q, nm in enumerate(('after_ball', 'after_called_strike', 'after_foul', 'after_miss')):
            c_['prev_flight_' + nm] = np.where(kind[r] == q, f1[r] / 10.0, 0.0)
        for cc in range(1, 12):
            c_[f'count_{cc}'] = (cnt_code[r] == cc).astype(float)
        c_.update({'pitch_no': pno[r], 'prev_side': u1[r], 'prev_height': z1[r] - 2.5, 'prev_in_zone': inz1[r].astype(float),
                   'side': u[r], 'side_abs': np.abs(u[r]), 'height': z[r] - 2.5, 'height_sq': (z[r] - 2.5) ** 2, 'platoon': (stand_r == throw_r)[r].astype(float)})
        return c_
    keys = ['prev_flight_after_ball', 'prev_flight_after_called_strike', 'prev_flight_after_foul', 'prev_flight_after_miss', 'own_flight']
    cell_kind = gkey * 10 + kind
    cell_fam = cell_kind * 10 + (fam1 + 1)
    depth = {}
    for yr in (train_year, test_year):
        r0 = ok & (year == yr)
        if r0.sum() < 5000:
            continue
        out = {}
        for vname, rows, cell in (('full_controls', r0, cell_kind), ('within_previous_family', r0, cell_fam),
                                  ('alternations', r0 & (fam1 != fam), cell_kind), ('repeats', r0 & (fam1 == fam), cell_kind)):
            names, X, y, keep = within(rows, cell, design(rows))
            b, lo, hi = _ols(X, y, cols['game_pk'][rows][keep]); j = {n_: q for q, n_ in enumerate(names)}
            out[vname] = {'contacts': int(keep.sum()), **{k: [round(float(b[j[k]]), 4), round(float(lo[j[k]]), 4), round(float(hi[j[k]]), 4)] for k in keys}}
        depth[yr] = out
        stage(f'depth {yr}')
    res['contact_depth'] = depth
    # speed-follow (as TIMING-02) against the contact window, on misses
    rs = base_ok & (year == train_year)
    ug, gi = np.unique(pkey[rs], return_inverse=True); cnt = np.bincount(gi); keep = cnt[gi] >= 2
    xo = _demean(center(ft, pkey, rs)[rs] / 10.0, gi, cnt)
    C = np.column_stack([_demean(v, gi, cnt) for v in (u[rs], np.abs(u[rs]), z[rs] - 2.5, (z[rs] - 2.5) ** 2, cols['balls'][rs].astype(float), cols['strikes'][rs].astype(float),
                                                       (stand_r == throw_r)[rs].astype(float))])
    X3 = np.column_stack([xo, C])[keep]; y3 = _demean(iy[rs] - mu_y[rs], gi, cnt)[keep]
    b3 = np.linalg.lstsq(X3, y3, rcond=None)[0]; r3 = y3 - X3 @ b3
    sf, sd, lam = _hitter_slopes(batter[rs][keep], X3[:, 0], r3, float(np.var(r3)))
    t_sf = np.asarray([sf.get(int(h_), np.nan) for h_ in batter])
    sw_ok = swing & np.isfinite(ft) & np.isfinite(u) & np.isfinite(z) & np.isfinite(cols['release_speed']) & (n_y >= 30) & np.isfinite(t_sf)
    n_w, s_w, _ = prior_stats(batter, day, whiff.astype(float), swing)
    lw = float(whiff[swing & (year == train_year)].mean()); wr = (s_w + 200 * lw) / (n_w + 200)
    bs_ = cols['bat_speed']; okb = swing & np.isfinite(bs_)
    n_b, s_b, _ = prior_stats(batter, day, bs_, okb); bat_speed = (s_b + 50 * float(np.nanmean(bs_[okb & (year == train_year)]))) / (n_b + 50)
    X_pitch = np.column_stack([cols['release_speed'], cols['pfx_x'] * np.where(throw_r, 1, -1), cols['pfx_z'], u, z, G['vaa'], G['haa'] * np.where(stand_r, 1, -1),
                               cols['release_spin_rate'], cols['release_extension'], cols['release_pos_z'], cols['balls'], cols['strikes'], grp,
                               (stand_r == throw_r).astype(float), cols['arm_angle'], np.log(wr / (1 - wr)), bat_speed])
    tr = sw_ok & (year == train_year); te = sw_ok & (year == test_year)
    res['swings'] = {'train': int(tr.sum()), 'test': int(te.sum()), 'corr_speed_follow_window': round(float(np.corrcoef(t_sf[te], sig_y[te])[0, 1]), 3) if te.any() else None}
    if tr.sum() < 20000 or te.sum() < 5000:
        res['error'] = 'too few swings'; return res
    from sklearn.ensemble import HistGradientBoostingClassifier
    yv = whiff.astype(float)
    hp = dict(max_iter=int(params.get('gbm_iter', 300)), learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0, random_state=11)
    oof = np.full(len(yv), np.nan); par = cols['game_pk'] % 2 == 0
    for side in (True, False):
        fr, pr = tr & (par == side), tr & (par != side)
        oof[pr] = HistGradientBoostingClassifier(**hp).fit(X_pitch[fr], yv[fr]).predict_proba(X_pitch[pr])[:, 1]
    p_te = HistGradientBoostingClassifier(**hp).fit(X_pitch[tr], yv[tr]).predict_proba(X_pitch[te])[:, 1]
    lo_tr = np.log(np.clip(oof[tr], 1e-6, 1 - 1e-6) / np.clip(1 - oof[tr], 1e-6, 1)); lo_te = np.log(np.clip(p_te, 1e-6, 1 - 1e-6) / np.clip(1 - p_te, 1e-6, 1))
    terms = {'speed_follow': t_sf, 'window': (sig_y - math.sqrt(lv)) / 10.0}
    yt = yv[te]; games = cols['game_pk'][te]; base_ll = logloss(p_te, yt)
    tg = np.unique(cols['game_pk'][tr]); gi_ = np.searchsorted(tg, cols['game_pk'][tr])
    wh_out = {}
    for name, cl in (('window', ['window']), ('speed_follow', ['speed_follow']), ('both', ['window', 'speed_follow'])):
        Ttr = np.column_stack([terms[k][tr] for k in cl]); Tte = np.column_stack([terms[k][te] for k in cl])
        bb = offset_fit(lo_tr, Ttr, yv[tr]); pt = 1 / (1 + np.exp(-(lo_te + bb[0] + Tte @ bb[1:])))
        rng = np.random.default_rng(5); bsd = []
        for _ in range(int(params.get('reps', 60))):
            w = np.bincount(rng.integers(0, len(tg), len(tg)), minlength=len(tg))[gi_]; sel = np.repeat(np.arange(int(tr.sum())), w)
            bsd.append(offset_fit(lo_tr[sel], Ttr[sel], yv[tr][sel], iters=20)[1:])
        bsd = np.asarray(bsd)
        wh_out[name] = {'coefs': {k: [round(float(bb[1 + q]), 4), round(float(np.percentile(bsd[:, q], 2.5)), 4), round(float(np.percentile(bsd[:, q], 97.5)), 4)] for q, k in enumerate(cl)},
                        'gain_nats_per_1000_swings': clustered(base_ll - logloss(pt, yt), games)}
    res['whiffs'] = wh_out
    stage('whiffs')
    return res
