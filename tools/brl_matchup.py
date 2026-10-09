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
