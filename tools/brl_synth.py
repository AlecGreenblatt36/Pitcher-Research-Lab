"""Synthetic plate appearances for stress-testing the run-value estimator (VALUE-17): worlds with known truth, generated
pitch by pitch with count progression, hitter-specific swing shifts and miss holes, and pitchers who may target or avoid
a hitter's spots. Runs without sealed data (public, reproducible).

World kinds:
  null         hitters swing alike (no shifts, no level differences, no holes); maps fitted anyway are noise, so the
               planted deviation is zero and any coefficient found is a false positive
  level        hitters differ only in how much they swing at everything (no location-specific shape)
  effect       hitters have their own swing shifts (chase spots), levels and miss holes; pitchers do not target them
  small        as effect, with the shifts, levels and holes at 40% of the size (nearer the real own-part spread)
  target       as effect, but every pitcher aims 30% of his pitches toward the hitter's true chase spot
  opposing     as effect, but half the pitchers target the spot and the other half pitch away from it
  coincident   as effect, but each hitter's miss hole sits on his chase spot (chasing and missing coincide)
  noisy        as effect, with three times as many hitters sharing the same pitches (each seen a third as often)

Truth for the outside coefficient (runs per point of a hitter's own swing part): at each outside pitch of the test
season, 0.01 times (the value if he swings minus the value if he takes) under the generator's own probabilities and
its own count values, averaged over outside pitches (simple mean, and weighted by the squared true deviation of the
hitter's swing chance from the league's, the weighting a regression on the true deviation would use). The estimator
must recover that from the data alone.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('brl_discovery', ROOT / 'tools' / 'brl_discovery.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
LW7 = D.LW7
KINDS = ('null', 'level', 'effect', 'small', 'target', 'opposing', 'coincident', 'noisy')
Q_CUTS = (0.35, 1.0, 1.5)                     # damage rule: q below 0.35 an out, then a single, a double or triple, a home run
Q_SD = 0.9


def _bip_value(q_mean: np.ndarray) -> np.ndarray:
    """Expected ball-in-play value when the damage draw is q ~ N(q_mean, Q_SD)."""
    c = [(ndtr((cut - q_mean) / Q_SD)) for cut in Q_CUTS]
    p_out = c[0]; p_1b = c[1] - c[0]; p_2b = c[2] - c[1]; p_hr = 1 - c[2]
    return p_out * LW7[0] + p_1b * LW7[3] + p_2b * LW7[4] + p_hr * LW7[5]


def _chase_spot(su: np.ndarray, sz: np.ndarray):
    """The point just outside the zone in the direction of the hitter's swing-region shift (u is away-positive, z in feet):
    where his swing chance most exceeds the league's. A hitter with no shift gets the away edge."""
    du = -su; dz = -sz
    nrm = np.sqrt(du ** 2 + dz ** 2)
    du = np.where(nrm > 1e-6, du / np.maximum(nrm, 1e-6), 1.0); dz = np.where(nrm > 1e-6, dz / np.maximum(nrm, 1e-6), 0.0)
    with np.errstate(divide='ignore'):
        t_edge = np.minimum(np.where(np.abs(du) > 1e-9, 0.83 / np.abs(du), np.inf), np.where(np.abs(dz) > 1e-9, 1.0 / np.abs(dz), np.inf))
    t_ = t_edge + 0.25
    return du * t_, 2.5 + dz * t_


ADJ = np.array([0.0, 0.35, 0.15])            # the hitter reads a height above the crossing for the two breaking types (late drop he cannot see)
SPEEDS = np.array([94.0, 85.0, 86.0]); GRP_OF = np.array([0, 3, 5]); TYPE_OF = {0: 0, 3: 1, 5: 2}


def probs(G: dict, b, p, sr, px, pz, t_, k_):
    """The generator's probabilities at these pitches: swing (own and league-average hitter), whiff on a swing, called
    strike on a take, foul on contact and the expected ball-in-play value."""
    za = pz + ADJ[t_]; u = np.where(sr == 1, px, -px)
    inz = (np.abs(px) < 0.83) & (pz > 1.5) & (pz < 3.5)
    su, sz, lv, pwr = G['sh_u'][b], G['sh_z'][b], G['lvl'][b], G['pw'][b]
    hd, hu, hz = G['hole_d'][b], G['hole_u'][b], G['hole_z'][b]
    dist = lambda su_, sz_: np.sqrt((u + su_) ** 2 + ((za - 2.5 + sz_) / 1.1) ** 2)
    base_lo = 1.4 + 0.45 * k_ - 0.6 * ((t_ == 1) & (za > 3.2))
    p_sw = 1 / (1 + np.exp(-(base_lo - 2.0 * dist(su, sz) + lv)))
    p_sw_league = 1 / (1 + np.exp(-(base_lo - 2.0 * dist(0.0, 0.0))))
    w_lo = -1.4 + 0.3 * lv + G['plv'][p] + 0.5 * (pz - 2.5) + 0.35 * (t_ != 0) + 0.3 * k_ + hd * np.exp(-((u - hu) ** 2 + (pz - hz) ** 2) / 0.3)
    p_w = 1 / (1 + np.exp(-w_lo))
    p_c = np.where(inz, 0.88, 0.08); p_f = np.full(len(px), 0.42)
    q_mean = -0.3 + 0.25 * (np.abs(u) < 0.5) + 0.2 * ((pz > 1.8) & (pz < 3.2)) + 2.0 * pwr
    return {'inz': inz, 'p_sw': p_sw, 'p_sw_league': p_sw_league, 'p_w': p_w, 'p_c': p_c, 'p_f': p_f, 'q_mean': q_mean, 'vb': _bip_value(q_mean)}


def swing_minus_take(G: dict, pr: dict, bl, k_):
    """Value of the plate appearance if the hitter swings minus if he takes, under the generator's count values."""
    cv = G['cv']; bl = np.asarray(bl, int); k_ = np.asarray(k_, int)
    v_next_strike = np.where(k_ == 2, LW7[1], cv[np.clip(bl * 3 + k_ + 1, 0, 11)])
    v_foul = np.where(k_ == 2, cv[bl * 3 + 2], cv[np.clip(bl * 3 + k_ + 1, 0, 11)])
    v_ball = np.where(bl == 3, LW7[2], cv[np.clip((bl + 1) * 3 + k_, 0, 11)])
    v_swing = pr['p_w'] * v_next_strike + (1 - pr['p_w']) * (pr['p_f'] * v_foul + (1 - pr['p_f']) * pr['vb'])
    v_take = pr['p_c'] * v_next_strike + (1 - pr['p_c']) * v_ball
    return v_swing - v_take


def world(kind: str, n_pa: int = 200000, seed: int = 5, H: int = 240, P: int = 120):
    if kind not in KINDS:
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    shape = kind not in ('null', 'level'); scale = 0.4 if kind == 'small' else 1.0
    H_all = H * 3 if kind == 'noisy' else H
    sh_u = rng.normal(0, 0.35 * scale, H_all) * shape; sh_z = rng.normal(0, 0.35 * scale, H_all) * shape
    hole_u = rng.normal(0, 0.5, H_all); hole_z = rng.normal(2.5, 0.5, H_all); hole_d = rng.normal(0, 0.7 * scale, H_all) * shape
    spot_u, spot_z = _chase_spot(sh_u, sh_z)
    if kind == 'coincident':
        hole_u, hole_z = spot_u.copy(), spot_z.copy()
    lvl = rng.normal(0, 0.25 * scale, H_all) * (kind != 'null'); pw = rng.normal(0, 0.12, H_all); plv = rng.normal(0, 0.25, P)
    G = {'sh_u': sh_u, 'sh_z': sh_z, 'hole_u': hole_u, 'hole_z': hole_z, 'hole_d': hole_d, 'lvl': lvl, 'pw': pw, 'plv': plv, 'kind': kind}
    pc_x = rng.normal(0, 0.5, (P, 3)); pc_z = rng.normal(2.4, 0.4, (P, 3))
    aim_mode = np.zeros(P)
    if kind == 'target':
        aim_mode[:] = 1
    elif kind == 'opposing':
        aim_mode[: P // 2] = 1; aim_mode[P // 2:] = -1
    i = np.arange(n_pa)
    season = np.array([2023, 2024, 2025])[i % 3]
    day = np.array([date(s, 4, 1).toordinal() for s in (2023, 2024, 2025)])[i % 3] + (i // 3000) % 150
    b = rng.integers(0, H_all, n_pa); p = rng.integers(0, P, n_pa); sr = (b % 2).astype(np.int64)
    balls = np.zeros(n_pa, np.int64); strikes = np.zeros(n_pa, np.int64); alive = np.ones(n_pa, bool)
    fin = np.full(n_pa, -1, np.int64)
    chunks = []; truth_rows = []
    for pno in range(13):
        a = np.flatnonzero(alive)
        if not len(a):
            break
        na = len(a); ba, pa_, sra = b[a], p[a], sr[a]; k_ = strikes[a]; bl = balls[a]
        t_ = rng.choice(3, size=na, p=[0.55, 0.3, 0.15])
        v = SPEEDS[t_] + rng.normal(0, 1.5, na)
        px = pc_x[pa_, t_] + rng.normal(0, 0.7, na); pz = pc_z[pa_, t_] + rng.normal(0, 0.7, na)
        aimed = (aim_mode[pa_] != 0) & (rng.random(na) < 0.30)
        if aimed.any():
            sgn = aim_mode[pa_[aimed]]
            tu = np.where(sgn > 0, spot_u[ba[aimed]], -spot_u[ba[aimed]]); tz = np.where(sgn > 0, spot_z[ba[aimed]], 5.0 - spot_z[ba[aimed]])
            px[aimed] = np.where(sra[aimed] == 1, tu, -tu) + rng.normal(0, 0.45, int(aimed.sum())); pz[aimed] = tz + rng.normal(0, 0.45, int(aimed.sum()))
        pr = probs(G, ba, pa_, sra, px, pz, t_, k_)
        inz, p_sw, p_w, p_c, p_f, q_mean = pr['inz'], pr['p_sw'], pr['p_w'], pr['p_c'], pr['p_f'], pr['q_mean']
        truth_rows.append(np.column_stack([season[a], inz, bl, k_, p_sw, pr['p_sw_league'], p_w, p_f, p_c, pr['vb']]))
        swing = rng.random(na) < p_sw
        whiff = swing & (rng.random(na) < p_w)
        contact = swing & ~whiff
        foul = contact & (rng.random(na) < p_f)
        bip = contact & ~foul
        cs_ = ~swing & (rng.random(na) < p_c)
        ball = ~swing & ~cs_
        call = np.where(swing, np.where(whiff, 2, 1), 0)
        k1 = k_ + (whiff | cs_) + (foul & (k_ < 2)); b1 = bl + ball
        out7 = np.full(na, -1, np.int64); ls = np.full(na, np.nan)
        if bip.any():
            q = q_mean[bip] + rng.normal(0, Q_SD, int(bip.sum()))
            out7[bip] = np.where(q < Q_CUTS[0], 0, np.where(q < Q_CUTS[1], 3, np.where(q < Q_CUTS[2], 4, 5)))
            ls[bip] = 85 + 8 * q + rng.normal(0, 6, int(bip.sum()))
        out7[k1 >= 3] = 1; out7[b1 >= 4] = 2
        last = (out7 >= 0) | (pno == 12)
        chunks.append({'pa': a, 'pno': np.full(na, pno), 'balls': bl, 'strikes': k_, 'call': call, 'v0': v, 'px': px, 'pz': pz, 'grp': GRP_OF[t_],
                       'last': last, 'ls': ls, 'cs': cs_.astype(np.int64)})
        fin[a[out7 >= 0]] = out7[out7 >= 0]
        balls[a] = np.minimum(b1, 3); strikes[a] = np.minimum(k1, 2); alive[a[last]] = False
    cat = lambda k: np.concatenate([c[k] for c in chunks])
    pa_ix = cat('pa'); n = len(pa_ix)
    T = {'season': season[pa_ix], 'day': day[pa_ix], 'game': pa_ix // 70, 'pitcher': 1000 + p[pa_ix], 'batter': 5000 + b[pa_ix], 'stand_r': sr[pa_ix],
         'throw_r': p[pa_ix] % 2, 'inning': np.ones(n, np.int64), 'group': cat('grp'), 'sub': np.zeros(n, np.int64), 'balls': cat('balls'), 'strikes': cat('strikes'),
         'call': cat('call'), 'v0': cat('v0'), 'v1': cat('v0') - 8.0, 'spin': np.full(n, 2300.0), 'pfx_x': np.zeros(n), 'pfx_z': np.zeros(n), 'px': cat('px'), 'pz': cat('pz'),
         'x0': np.full(n, -1.5), 'z0': np.full(n, 5.8), 'ext': np.full(n, 6.3), 'last_in_pa': cat('last').astype(np.int64), 'bunt_pa': np.zeros(n, np.int64),
         'ab': pa_ix % 70, 'pitch_no': cat('pno'), 'la': np.full(n, np.nan), 'ls': cat('ls'), 'cs': cat('cs'), 'zone': np.ones(n, np.int64),
         'out7': fin[pa_ix], 'half': np.zeros(n, np.int64), 'post': np.zeros(n, np.int64)}
    T['pfx_z'] = np.where(T['group'] == 3, -8.0, np.where(T['group'] == 5, -4.0, 12.0))   # late drop, so the decision-moment projection differs from the crossing
    order = np.lexsort((T['pitch_no'], T['ab'], T['game']))
    T = {k: np.asarray(v[order], dtype=np.float32 if k in ('v0', 'v1', 'spin', 'pfx_x', 'pfx_z', 'px', 'pz', 'x0', 'z0', 'ext', 'la', 'ls') else np.int64) for k, v in T.items()}
    # the truth under the generator's own count values (mean plate-appearance value by count over every pitch row with an outcome)
    R = np.vstack(truth_rows)
    ssn, inz_r, bl_r, k_r, p_sw, p_swl, p_w, p_f, p_c, vb = (R[:, j] for j in range(10))
    fin_rows = fin[pa_ix]
    ok_rows = fin_rows >= 0
    ci_rows = cat('balls') * 3 + cat('strikes')
    cv = np.array([LW7[fin_rows[ok_rows & (ci_rows == c_)]].mean() if (ok_rows & (ci_rows == c_)).any() else 0.0 for c_ in range(12)])
    G['cv'] = cv
    tau = 0.01 * swing_minus_take(G, {'p_w': p_w, 'p_f': p_f, 'p_c': p_c, 'vb': vb}, bl_r, k_r)
    wts = ((p_sw - p_swl) * 100) ** 2
    G['val_true'] = ((p_sw - p_swl) * tau * 100)[order]     # the hitter-specific part of each pitch's expected value, aligned with the table's rows
    G['dev_true'] = ((p_sw - p_swl) * 100)[order]
    G['tau_true'] = tau[order]
    truth = {}
    for zn, m in (('outside', (ssn == 2025) & (inz_r == 0)), ('inside', (ssn == 2025) & (inz_r == 1))):
        truth[f'runs_per_point_{zn}'] = round(float(tau[m].mean()), 6)
        truth[f'runs_per_point_{zn}_deviation_weighted'] = round(float((tau[m] * wts[m]).sum() / max(wts[m].sum(), 1e-9)), 6)
        truth[f'{zn}_test_pitches'] = int(m.sum())
        truth[f'sd_true_deviation_points_{zn}'] = round(float(np.sqrt(wts[m].mean())), 3)
    truth['count_values'] = {f'{c_ // 3}-{c_ % 3}': round(float(cv[c_]), 4) for c_ in range(12)}
    truth['rows'] = int(n); truth['hitters'] = int(H_all); truth['planted_deviation'] = bool(kind != 'null')
    if kind == 'null':
        # no hitter deviates, so the coefficient's truth is zero; the run value a deviation would have had is kept for reference
        for zn in ('outside', 'inside'):
            truth[f'runs_per_point_{zn}_if_deviation_existed'] = truth[f'runs_per_point_{zn}']
            truth[f'runs_per_point_{zn}'] = 0.0; truth[f'runs_per_point_{zn}_deviation_weighted'] = 0.0
    return T, truth, G


class TruthPricer:
    """Prices the estimator's aim choices with the generator: the hitter-specific part of the expected plate-appearance
    value, (own swing chance minus the league's) times (value if swing minus value if take), at each scattered pitch;
    the gain of the chosen third over the pitcher's whole pool, by thirds of the league swing chance as the estimator
    does it; and the same gain had the pitcher chosen by the truth itself (the oracle)."""

    def __init__(self, G: dict):
        self.G = G; self.acc = {}; self.pairs = {}; self.diag = {}; self.structural = {}

    def regression_pieces(self, ix, D, outside, y, Xd, b, tau_struct=None):
        # the same regression with the generator's hitter-specific truth as the outcome: the coefficient a perfect outcome would give
        vt = self.G['val_true'][ix]
        if tau_struct is not None:
            # the engine's structural per-point value against the generator's at the test pitches
            tt = self.G['tau_true'][ix]
            self.structural = {'mean_structural_outside': round(float(tau_struct[outside].mean()), 6), 'mean_true_outside': round(float(tt[outside].mean()), 6),
                               'mean_structural_inside': round(float(tau_struct[~outside].mean()), 6), 'mean_true_inside': round(float(tt[~outside].mean()), 6),
                               'corr_outside': round(float(np.corrcoef(tau_struct[outside], tt[outside])[0, 1]), 4) if tt[outside].std() > 0 else None,
                               'slope_true_on_structural_outside': round(float(np.polyfit(tau_struct[outside], tt[outside], 1)[0]), 4) if tau_struct[outside].std() > 0 else None}
        bt = np.linalg.lstsq(Xd, vt, rcond=None)[0]
        # and the truth regressed on the fitted part alone (no controls), outside and inside
        sl = lambda m: float(np.dot(D[m] - D[m].mean(), vt[m] - vt[m].mean()) / max(np.dot(D[m] - D[m].mean(), D[m] - D[m].mean()), 1e-12))
        dv = self.G['dev_true'][ix]
        rel = lambda m: float(np.dot(D[m] - D[m].mean(), dv[m] - dv[m].mean()) / max(np.dot(D[m] - D[m].mean(), D[m] - D[m].mean()), 1e-12))
        self.diag = {'coef_outside': round(float(b[-2]), 6), 'coef_inside': round(float(b[-1]), 6),
                     'coef_outside_with_true_outcome': round(float(bt[-2]), 6), 'coef_inside_with_true_outcome': round(float(bt[-1]), 6),
                     'slope_true_value_on_fitted_part_outside': round(sl(outside), 6), 'slope_true_value_on_fitted_part_inside': round(sl(~outside), 6),
                     'slope_true_deviation_on_fitted_part_outside': round(rel(outside), 4), 'slope_true_deviation_on_fitted_part_inside': round(rel(~outside), 4),
                     'corr_fitted_true_deviation_outside': round(float(np.corrcoef(D[outside], dv[outside])[0, 1]), 4) if dv[outside].std() > 0 and D[outside].std() > 0 else None,
                     'sd_fitted_outside': round(float(D[outside].std()), 3), 'sd_true_outside': round(float(dv[outside].std()), 3)}

    def __call__(self, sg, zone, hitter, pitcher, rows, K, x_true, z_true, balls, strikes, group, chosen, graded, third, weight, dev=None, coef=None):
        b = np.full(len(x_true), hitter - 5000); p = np.full(len(x_true), pitcher - 1000); sr = np.full(len(x_true), (hitter - 5000) % 2)
        t_ = np.vectorize(TYPE_OF.get)(group)
        pr = probs(self.G, b, p, sr, np.asarray(x_true, float), np.asarray(z_true, float), t_, np.asarray(strikes, int))
        val = ((pr['p_sw'] - pr['p_sw_league']) * swing_minus_take(self.G, pr, balls, strikes)).reshape(len(rows), K).mean(1)
        gains = {nm: [] for nm in chosen}; oracle = []
        for t3 in range(3):
            sel = graded & (third == t3)
            if not sel.any():
                continue
            k3 = max(1, int(sel.sum()) // 3)
            for nm, ch in chosen.items():
                gains[nm].append(float(val[sel & ch].mean() - val[sel].mean()))
            oracle.append(float(np.sort(val[sel])[:k3].mean() - val[sel].mean()))
        if oracle:
            a_ = self.acc.setdefault((str(sg), zone), {'oracle': 0.0, 'weight': 0.0})
            a_['oracle'] += weight * float(np.mean(oracle)); a_['weight'] += weight
            for nm, g_ in gains.items():
                a_[nm] = a_.get(nm, 0.0) + weight * float(np.mean(g_))
        if dev is not None:
            # the fitted part against the truth at the aim points (a sample of pairs, for the slope at the menu rather than at the test pitches)
            pr_ = self.pairs.setdefault((str(sg), zone), [])
            if len(pr_) < 400:
                pr_.append(np.column_stack([dev[graded], val[graded], np.full(int(graded.sum()), weight)]))

    def result(self, sg: str, zone: str, pitches_per_pa: float) -> dict:
        a_ = self.acc.get((sg, zone))
        if not a_ or a_['weight'] == 0:
            return {}
        w = a_['weight']
        out = {'oracle_runs_per_pitch': round(a_['oracle'] / w, 6), 'oracle_runs_per_6200': round(a_['oracle'] / w * pitches_per_pa * 6200, 1)}
        for nm in ('points', 'side', 'bands', 'structural'):
            if nm in a_:
                out[f'true_runs_per_pitch_{nm}'] = round(a_[nm] / w, 6); out[f'true_runs_per_6200_{nm}'] = round(a_[nm] / w * pitches_per_pa * 6200, 1)
        pr_ = self.pairs.get((sg, zone))
        if pr_:
            M = np.vstack(pr_); d_, v_, w_ = M[:, 0], M[:, 1], M[:, 2]
            dm = np.average(d_, weights=w_); vm = np.average(v_, weights=w_)
            out['slope_true_value_on_fitted_part_at_aims'] = round(float(np.sum(w_ * (d_ - dm) * (v_ - vm)) / max(np.sum(w_ * (d_ - dm) ** 2), 1e-12)), 6)
            out['aim_pairs'] = int(len(d_))
        return out


def run_reps(kind: str, reps: int, seed0: int, params: dict, stage):
    out = []
    ests = tuple(params.get('estimands', ('pa', 'pitch')))
    for r in range(reps):
        t0 = time.time()
        T, truth, G = world(kind, int(params.get('n_pa', 200000)), seed0 + r, int(params.get('hitters', 240)), int(params.get('pitchers', 120)))
        row = {'rep': r, 'seed': seed0 + r, 'truth': truth, 'world_seconds': round(time.time() - t0, 1)}
        for est in ests:
            t1 = time.time(); pricer = TruthPricer(G)
            sigmas = tuple(float(v) for v in params.get('sigmas', (0.6,)))
            pp = {'sigmas': sigmas, 'within_type': True, 'checks': True, 'specific': True, 'reps': int(params.get('boot_reps', 60)),
                  'estimand': est, 'pool_from_train': bool(params.get('pool_from_train', True)), 'reprice_boundary': bool(params.get('reprice_boundary', True)),
                  'reprice_bands': bool(params.get('reprice_bands', True)), 'reprice_structural': bool(params.get('reprice_structural', False)) and est == ests[-1],
                  'map_draws': int(params.get('map_draws', 0)),
                  'placebo_draws': int(params.get('placebo_draws', 2)), 'aim_hook': pricer, 'diag_hook': pricer.regression_pieces}
            res = D.value2_study(T, pp, lambda s: None)
            c = res['coefficient_checks']; co = res['coefficients']
            row[est] = {'own': c['own_map']['outside'], 'own_interval': c['own_map']['outside_interval'],
                        'own_inside': c['own_map']['inside'], 'own_inside_interval': c['own_map']['inside_interval'],
                        'within': c['within_hitter_group_and_zone_side']['outside'], 'within_interval': c['within_hitter_group_and_zone_side']['outside_interval'],
                        'within_held': c['within_hitter_holding_pa_others']['outside'], 'within_held_interval': c['within_hitter_holding_pa_others']['outside_interval'],
                        'own_held': c['own_map_holding_pa_others']['outside'], 'own_held_interval': c['own_map_holding_pa_others']['outside_interval'],
                        'placebo': c['placebo_other_hitter']['outside'], 'placebo_interval': c['placebo_other_hitter']['outside_interval'],
                        'within_pa_correlation_outside': c.get('within_pa_correlation_outside'),
                        'sd_points_own_part_outside': c['own_map'].get('sd_points_outside'),
                        'test_pitches': co['test_pitches'], 'outside_pitches_per_pa': co['outside_pitches_per_pa'],
                        'runs_6200_repriced': (res.get('repriced_by_side') or {}).get(str(sigmas[-1]), {}).get('runs_per_6200_outside_only'),
                        'runs_6200_structural': (res.get('repriced_structural') or {}).get(str(sigmas[-1]), {}).get('runs_per_6200_outside_only'),
                        'runs_6200_structural_inside': (res.get('repriced_structural') or {}).get(str(sigmas[-1]), {}).get('runs_per_6200_inside_only'),
                        'structural_map_draws': (res.get('repriced_structural') or {}).get(str(sigmas[-1]), {}).get('map_draws'),
                        'runs_6200_points_inside': res['by_command_sd_ft'][str(sigmas[-1])]['runs_per_6200_inside_only'],
                        'runs_6200_structural_interval': (res.get('repriced_structural') or {}).get(str(sigmas[-1]), {}).get('runs_per_6200_outside_only_interval'),
                        'own_part_calibration': res.get('own_part_calibration'),
                        'structural_vs_truth': pricer.structural,
                        'runs_6200_bands': (res.get('repriced_by_band') or {}).get(str(sigmas[-1]), {}).get('runs_per_6200_outside_only'),
                        'runs_6200_bands_interval': (res.get('repriced_by_band') or {}).get(str(sigmas[-1]), {}).get('runs_per_6200_outside_only_interval'),
                        'coefficients_by_band': res.get('coefficients_by_band'),
                        'runs_6200_points': res['by_command_sd_ft'][str(sigmas[-1])]['runs_per_6200_outside_only'],
                        'runs_6200_points_interval': res['by_command_sd_ft'][str(sigmas[-1])]['runs_per_6200_outside_only_interval'],
                        'outside_gain_points': res['by_command_sd_ft'][str(sigmas[-1])]['outside_gain_points'],
                        'truth_policy_outside': pricer.result(str(sigmas[-1]), 'outside', co['outside_pitches_per_pa']),
                        'truth_policy_inside': pricer.result(str(sigmas[-1]), 'inside', co['inside_pitches_per_pa']),
                        'by_sigma': {str(sg): {'runs_6200_points': res['by_command_sd_ft'][str(sg)]['runs_per_6200_outside_only'],
                                               'runs_6200_repriced': (res.get('repriced_by_side') or {}).get(str(sg), {}).get('runs_per_6200_outside_only'),
                                               'runs_6200_bands': (res.get('repriced_by_band') or {}).get(str(sg), {}).get('runs_per_6200_outside_only'),
                                               'runs_6200_structural': (res.get('repriced_structural') or {}).get(str(sg), {}).get('runs_per_6200_outside_only'),
                                               'truth_policy_outside': pricer.result(str(sg), 'outside', co['outside_pitches_per_pa'])} for sg in sigmas},
                        'regression_against_truth': pricer.diag,
                        'seconds': round(time.time() - t1, 1)}
        out.append(row)
        stage(f'{kind} rep {r} ({round(time.time() - t0)} s): truth {truth["runs_per_point_outside"]:.6f} ' + ' '.join(
            f'{e_} own {row[e_]["own"]} within {row[e_]["within"]} runs {row[e_]["runs_6200_points"]}/{row[e_]["runs_6200_repriced"]}/{row[e_]["runs_6200_bands"]}/{row[e_]["runs_6200_structural"]} '
            f'true {row[e_]["truth_policy_outside"].get("true_runs_per_6200_points")}/{row[e_]["truth_policy_outside"].get("true_runs_per_6200_side")}/{row[e_]["truth_policy_outside"].get("true_runs_per_6200_bands")}/{row[e_]["truth_policy_outside"].get("true_runs_per_6200_structural")}' for e_ in ests))
    return out


def summarize(rows: list, ests=('pa', 'pitch')) -> dict:
    summ = {}
    for est in ests:
        if not rows or est not in rows[0]:
            continue
        for nm in ('own', 'own_held', 'within', 'within_held', 'placebo'):
            vals = np.array([r[est][nm] for r in rows]); lo = np.array([r[est][nm + '_interval'][0] for r in rows]); hi = np.array([r[est][nm + '_interval'][1] for r in rows])
            tr = np.array([r['truth']['runs_per_point_outside'] for r in rows]) if nm != 'placebo' else np.zeros(len(rows))
            trw = np.array([r['truth']['runs_per_point_outside_deviation_weighted'] for r in rows]) if nm != 'placebo' else np.zeros(len(rows))
            # per-point coefficients are not comparable across scales (a shrunk map has fewer points per true point), so the value of one
            # standard deviation of the part is compared too: coefficient times the fitted part's spread against the truth times the true spread
            sd_fit = np.array([r[est]['sd_points_own_part_outside'] or 0.0 for r in rows]); sd_true = np.array([r['truth']['sd_true_deviation_points_outside'] for r in rows])
            summ[f'{est}_{nm}'] = {'mean': round(float(vals.mean()), 6), 'sd_across_reps': round(float(vals.std()), 6), 'truth_mean': round(float(tr.mean()), 6),
                                   'truth_weighted_mean': round(float(trw.mean()), 6), 'bias': round(float((vals - tr).mean()), 6),
                                   'bias_vs_weighted': round(float((vals - trw).mean()), 6),
                                   'coverage': round(float(np.mean((lo <= tr) & (tr <= hi))), 3), 'coverage_weighted': round(float(np.mean((lo <= trw) & (trw <= hi))), 3),
                                   'excludes_zero': round(float(np.mean((lo > 0) | (hi < 0))), 3),
                                   'one_sd_value_mean': round(float((vals * sd_fit).mean()), 5), 'one_sd_value_truth': round(float((trw * sd_true).mean()), 5),
                                   'reps': int(len(rows))}
        pol = [r[est]['truth_policy_outside'] for r in rows]
        orc = np.array([q.get('oracle_runs_per_6200', np.nan) for q in pol], float)
        blk = {'oracle_mean': round(float(np.nanmean(orc)), 1)}
        for nm, key, ikey in (('points', 'runs_6200_points', 'runs_6200_points_interval'), ('side', 'runs_6200_repriced', None), ('bands', 'runs_6200_bands', 'runs_6200_bands_interval'),
                              ('structural', 'runs_6200_structural', 'runs_6200_structural_interval')):
            est_v = np.array([r[est].get(key) if r[est].get(key) is not None else np.nan for r in rows], float)
            tp = np.array([q.get(f'true_runs_per_6200_{nm}', np.nan) for q in pol], float)
            d_ = {'estimate_mean': round(float(np.nanmean(est_v)), 1), 'true_value_of_choices_mean': round(float(np.nanmean(tp)), 1),
                  'estimate_minus_true_mean': round(float(np.nanmean(est_v - tp)), 1),
                  'ratio_estimate_to_true': round(float(np.nanmean(est_v) / np.nanmean(tp)), 3) if np.nanmean(tp) != 0 else None}
            if ikey is not None:
                lo = np.array([r[est][ikey][0] if r[est].get(ikey) else np.nan for r in rows], float); hi = np.array([r[est][ikey][1] if r[est].get(ikey) else np.nan for r in rows], float)
                d_['interval_covers_true'] = round(float(np.nanmean((lo <= tp) & (tp <= hi))), 3); d_['interval_excludes_zero'] = round(float(np.nanmean((lo > 0) | (hi < 0))), 3)
            blk[nm] = d_
        md_ = [r[est].get('structural_map_draws') for r in rows]
        if any(md_):
            tp_s = np.array([r[est]['truth_policy_outside'].get('true_runs_per_6200_structural', np.nan) for r in rows], float)
            lo_m = np.array([m['runs_per_6200_outside_only_interval_with_maps'][0] if m else np.nan for m in md_], float); hi_m = np.array([m['runs_per_6200_outside_only_interval_with_maps'][1] if m else np.nan for m in md_], float)
            blk['structural']['interval_with_maps_covers_true'] = round(float(np.nanmean((lo_m <= tp_s) & (tp_s <= hi_m))), 3)
            blk['structural']['sd_from_maps_mean'] = round(float(np.nanmean([m['sd_from_maps'] for m in md_ if m])), 1)
            blk['structural']['sd_from_hitters_mean'] = round(float(np.nanmean([m['sd_from_hitters_and_calibration'] for m in md_ if m])), 1)
        summ[f'{est}_runs_6200'] = blk
        # the inside part (VALUE-18I): the structural estimate against the truth of its inside choices
        pin = [r[est]['truth_policy_inside'] for r in rows]
        est_i = np.array([r[est].get('runs_6200_structural_inside') if r[est].get('runs_6200_structural_inside') is not None else np.nan for r in rows], float)
        tp_i = np.array([q.get('true_runs_per_6200_structural', np.nan) for q in pin], float)
        summ[f'{est}_runs_6200_inside'] = {'structural_estimate_mean': round(float(np.nanmean(est_i)), 1), 'true_value_of_structural_choices_mean': round(float(np.nanmean(tp_i)), 1),
                                           'ratio_estimate_to_true': round(float(np.nanmean(est_i) / np.nanmean(tp_i)), 3) if np.nanmean(tp_i) != 0 else None,
                                           'points_choice_true_mean': round(float(np.nanmean([q.get('true_runs_per_6200_points', np.nan) for q in pin])), 1),
                                           'oracle_mean': round(float(np.nanmean([q.get('oracle_runs_per_6200', np.nan) for q in pin])), 1),
                                           'structural_beats_points': int(sum(1 for q in pin if q.get('true_runs_per_6200_structural', 0) < q.get('true_runs_per_6200_points', 0)))}
    return summ


def main(params: dict | None = None, stage=None):
    if params is None:
        params = json.loads((ROOT / 'tools' / 'discovery_params.json').read_text())
    kinds = list(params.get('kinds', ('null', 'effect'))); reps = int(params.get('reps', 10)); seed0 = int(params.get('seed', 100))
    receipt = {'schema': 'brl.synth-receipt.v1', 'params': params, 'stages': [], 'results': {}}
    t0 = time.time()
    if stage is None:
        def stage(name):
            receipt['stages'].append({'stage': name, 'at_seconds': round(time.time() - t0, 1)}); print(name, flush=True)
    ests = tuple(params.get('estimands', ('pa', 'pitch')))
    for kind in kinds:
        rows = run_reps(kind, reps, seed0, params, stage)
        receipt['results'][kind] = {'summary': summarize(rows, ests), 'rows': rows}
    receipt['seconds'] = round(time.time() - t0, 1)
    return receipt


if __name__ == '__main__':
    rec = main()
    print(json.dumps({k: v['summary'] for k, v in rec['results'].items()}, indent=1))


# ---------------------------------------------------------------- PLAN-02S: the planner priced by the generator
def true_probs_for_pool(G: dict, M, P: dict, h: int, p: int):
    """The generator's probabilities for every pool pitch of pitcher p against hitter h at every count: {(b, k): (s, w, c, f, v, fam)}."""
    T = M.T; rows = P['rows']; b_ = np.full(len(rows), h - 5000); p_ = np.full(len(rows), p - 1000); sr = np.full(len(rows), (h - 5000) % 2)
    px, pz = T['px'][rows].astype(np.float64), T['pz'][rows].astype(np.float64)
    t_ = np.vectorize(TYPE_OF.get)(T['group'][rows]); fam = np.where(t_ == 0, 0, np.where(t_ == 1, 1, 2))
    out = {}
    for b in range(4):
        for k in range(3):
            pr = probs(G, b_, p_, sr, px, pz, t_, np.full(len(rows), k))
            out[(b, k)] = (pr['p_sw'], pr['p_w'], pr['p_c'], pr['p_f'], pr['vb'], fam)
    return out


def policy_value(pr: dict, act, cg=None, optimal=False):
    """Value iteration over counts and expectation states with the probabilities pr; the policy is act[(b, k, e)] (a pool index),
    or the pool mix of the count group (cg weights) when act is None, or the best action when optimal. Returns V[0, 0, 0] and
    the value table."""
    nE = 13; V = np.zeros((4, 3, nE)); best = {}
    for total in range(5, -1, -1):
        for b in range(4):
            k = total - b
            if k < 0 or k > 2:
                continue
            s, w, c, f, v, fam = pr[(b, k)]
            c3 = 2 if k == 2 else (1 if b > k else 0)
            wts = None
            if act is None and not optimal:
                wts = (cg == c3).astype(np.float64); wts = wts / max(wts.sum(), 1e-9)
            for _ in range(40 if k == 2 else 1):
                for e in range(nE):
                    idx_e = lambda kind: 1 + fam * 4 + kind
                    q = s * (1 - w) * (1 - f) * v
                    q = q + s * w * (LW7[1] if k == 2 else V[b, k + 1][idx_e(3)])
                    q = q + s * (1 - w) * f * (V[b, 2][idx_e(2)] if k == 2 else V[b, k + 1][idx_e(2)])
                    q = q + (1 - s) * c * (LW7[1] if k == 2 else V[b, k + 1][idx_e(1)])
                    q = q + (1 - s) * (1 - c) * (LW7[2] if b == 3 else V[b + 1, k][idx_e(0)])
                    if optimal:
                        j = int(np.argmin(q)); V[b, k, e] = q[j]; best[(b, k, e)] = j
                    elif act is None:
                        V[b, k, e] = float(np.dot(wts, q))
                    else:
                        V[b, k, e] = float(q[act[(b, k, e)]])
    return float(V[0, 0, 0]), V, best


def planner_truth(G: dict, M, P: dict, h: int, p: int, act_h, act_l):
    """True values (runs per plate appearance from 0-0, the hitter's perspective, lower is better for the pitcher) of the actual mix,
    the league plan, the hitter plan and the oracle, under the generator."""
    pr = true_probs_for_pool(G, M, P, h, p)
    v_mix, _, _ = policy_value(pr, None, P['cg'])
    v_l, _, _ = policy_value(pr, act_l)
    v_h, _, _ = policy_value(pr, act_h)
    v_o, _, _ = policy_value(pr, None, None, optimal=True)
    return {'actual_mix': v_mix, 'league_plan': v_l, 'hitter_plan': v_h, 'oracle': v_o}
