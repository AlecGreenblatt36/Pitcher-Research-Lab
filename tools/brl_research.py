"""Plate-appearance model research inside Actions, where the sealed data and the key live.

Experiment: does pitch quality and contact quality (from the sealed pitch-physics seasons)
improve the locked plate-appearance model? The locked features are rebuilt exactly as the
engine builds them; the candidate adds time-valid player features computed from prior dates
only (a batter's exit velocity, hard-hit and barrel rates, whiff, chase and swing rates; a
pitcher's fastball velocity, its change since his previous outing, spin, movement, whiff,
chase, zone and called-strike-plus-whiff rates, contact quality and ground-ball share
allowed). Both models are fitted on 2023-2024, tuned and calibrated on 2025 exactly like the
locked run, and scored on the 2025 blend dates and on 2026, paired with a game-clustered
bootstrap. Only metrics and coefficients are written: research/<experiment>-<run>.json on
the ledger branch. No player rows leave the runner.
"""
from __future__ import annotations
import base64, gzip, importlib.util, json, os, sys, time, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
from cloud.security import unseal, key_bytes  # noqa: E402
from brl_live.bookkeeping_season import STUDY_SCHEMA, study_path, study_purpose, physics_path, physics_purpose  # noqa: E402

LABELS = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')
from research_lab.pa_model import physics as phys  # noqa: E402

# The feature code is shared with the live provider (research_lab/pa_model/physics.py) so an
# experiment measures exactly what production would compute.
EXTRA = phys.BASE_FEATURES
XVALUE_COLS = phys.XVALUE_FEATURES
RECENT_COLS = phys.RECENT_FEATURES
PITCH_SUMS, BIP_SUMS, PS, BS = phys.PITCH_SUMS, phys.BIP_SUMS, phys.PS, phys.BS
SWING, WHIFF, CALLED, FASTBALL = phys.SWING, phys.WHIFF, phys.CALLED, phys.FASTBALL
columns_for = phys.feature_names
per_pa_physics = phys.table_from_study
build_extras = phys.build_features
DEFAULT_PARAMS = phys.DEFAULT_PARAMS


def api(url, token, method='GET', payload=None):
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-research/1.0'}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode(); headers['Content-Type'] = 'application/json'
    with urlopen(Request(url, headers=headers, data=body, method=method), timeout=60) as r:
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
            time.sleep(2 + 3 * attempt)



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
            time.sleep(2 + 3 * attempt)


class _Done(Exception):
    """Raised to leave the experiment body early with the receipt already filled."""


def gbm_experiment(features, locked_columns, extras, y, partitions, parts, config, probs_locked, configs=None) -> dict:
    """Gradient-boosted classifiers on the v2 feature set: fitted on the training years, the number of
    iterations chosen on the tuning dates, temperature-calibrated on the calibration dates, scored like
    the linear models and paired against the locked model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from research_lab.pa_model.model import fit_logit_calibration, apply_logit_calibration
    from research_lab.pa_model.evaluation import probability_metrics, clustered_log_loss_difference_ci
    cols = list(locked_columns) + list(extras.columns)
    frame = features.copy()
    for c in extras.columns:
        frame[c] = extras[c].to_numpy()
    train_mask = frame['season'].isin(config.train_years).to_numpy()
    tune_mask, cal_mask = partitions['tune'], partitions['calibration']
    X = frame[cols].to_numpy(np.float32)
    configs = configs or {'gbm_d6': dict(max_depth=6, learning_rate=0.05, l2_regularization=1.0, checkpoints=(200, 400, 700, 1000, 1500)),
                          'gbm_d4': dict(max_depth=4, learning_rate=0.08, l2_regularization=1.0, checkpoints=(200, 400, 700, 1000, 1500))}
    out_all = {}
    for name, kw in configs.items():
        kw = dict(kw); checkpoints = kw.pop('checkpoints')
        clf = HistGradientBoostingClassifier(early_stopping=False, random_state=36, warm_start=True, max_iter=checkpoints[0], **kw)
        best, best_iter = None, checkpoints[0]
        for n_iter in checkpoints:
            clf.set_params(max_iter=n_iter)
            clf.fit(X[train_mask], y[train_mask])
            ll = probability_metrics(y[tune_mask], clf.predict_proba(X[tune_mask])).log_loss
            if best is None or ll < best - 1e-5:
                best, best_iter = ll, n_iter
            else:
                break
        clf = HistGradientBoostingClassifier(early_stopping=False, random_state=36, max_iter=best_iter, **kw)
        clf.fit(X[train_mask | tune_mask], y[train_mask | tune_mask])
        raw_cal = clf.predict_proba(X[cal_mask])
        temperature, biases, cal_audit = fit_logit_calibration(y[cal_mask], raw_cal, config.calibration_l2)
        out = {'feature_count': len(cols), 'iterations': int(best_iter), 'tune_log_loss': float(best), 'settings': {k: v for k, v in kw.items()},
               'calibration': {'temperature': float(temperature), 'pre': cal_audit['pre_calibration_log_loss'], 'post': cal_audit['post_calibration_log_loss']}}
        for part, mask in parts:
            p = apply_logit_calibration(clf.predict_proba(X[mask]), temperature, biases)
            out[part] = probability_metrics(y[mask], p).to_dict()
            games = frame.loc[mask, 'game_pk'].to_numpy()
            out[part + '_minus_locked'] = clustered_log_loss_difference_ci(y[mask], p, probs_locked[part], games, replicates=600)
        out_all[name] = out
    return out_all


# Linear weights (runs per plate appearance outcome, average out = about -0.27) for residual summaries.
RV_WEIGHTS = np.array([-0.26, -0.28, 0.32, 0.47, 0.80, 1.40, 0.45])
# Experiments that also score team offsets and residual bins on every variant's predictions.
TEAM_EVAL = {'aging', 'stage2', 'role', 'window'}
TEAM_GRID = [(k, hl, sides) for k in (2000.0, 4000.0, 8000.0) for hl in (90.0, 180.0, None) for sides in (('fld',), ('bat', 'fld'))]


def team_offsets(P, Y, dates, bat, fld, n_teams, k, half_life, sides):
    """Each date's probabilities times exp(batting team offset + fielding team offset), offsets = log((observed + k q) /
    (expected + k q)) per class from strictly earlier dates (q league share; expected includes the other side's offset),
    decayed by half_life days when given. Returns the adjusted probabilities (same order as P)."""
    league = Y.mean(0)
    O = {s: np.zeros((n_teams, 7)) for s in ('bat', 'fld')}; E = {s: np.zeros((n_teams, 7)) for s in ('bat', 'fld')}
    Q = P.copy(); last = None
    order = np.argsort(dates, kind='mergesort'); ds = dates[order]
    bounds = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1], True])
    idx_t = {'bat': bat, 'fld': fld}
    for a, b in zip(bounds[:-1], bounds[1:]):
        ii = order[a:b]; day = pd.Timestamp(str(ds[a])[:10]).toordinal()
        if half_life and last is not None:
            f = 0.5 ** ((day - last) / half_life)
            for s in O:
                O[s] *= f; E[s] *= f
        off = {s: (np.log((O[s] + k * league) / (E[s] + k * league)) if s in sides else np.zeros((n_teams, 7))) for s in ('bat', 'fld')}
        L = np.log(np.clip(P[ii], 1e-12, 1)) + off['bat'][bat[ii]] + off['fld'][fld[ii]]
        L -= L.max(1, keepdims=True); q = np.exp(L); q /= q.sum(1, keepdims=True); Q[ii] = q
        for s, other in (('bat', 'fld'), ('fld', 'bat')):
            pe = P[ii] * np.exp(off[other][idx_t[other][ii]]); pe /= pe.sum(1, keepdims=True)
            np.add.at(O[s], idx_t[s][ii], Y[ii]); np.add.at(E[s], idx_t[s][ii], pe)
        last = day
    return Q


def team_eval(P, y, frame, bat, fld, n_teams, parts_all):
    """Team offsets on one variant's 2025-2026 predictions: log loss change per PA on each holdout part, game-clustered."""
    from research_lab.pa_model.evaluation import clustered_log_loss_difference_ci
    Y = np.eye(7)[y]
    dates = frame['date_key'].astype(str).to_numpy()
    out = {}
    for k, hl, sides in TEAM_GRID:
        Q = team_offsets(P, Y, dates, bat, fld, n_teams, k, hl, sides)
        key = f'k{int(k)}_hl{int(hl) if hl else 0}_{"+".join(sides)}'
        out[key] = {}
        for part, mask in parts_all:
            out[key][part] = clustered_log_loss_difference_ci(y[mask], Q[mask], P[mask], frame['game_pk'].to_numpy()[mask], replicates=300)
    return out


def residual_bins(P, y, frame, mask):
    """Observed minus predicted runs per 1,000 plate appearances by batter and pitcher age and prior plate appearances."""
    obs = RV_WEIGHTS[y[mask]]; pred = P[mask] @ RV_WEIGHTS
    res = {}
    for name, col, bins in (('batter_age', 'age_bat', [0, 23, 25, 27, 29, 31, 33, 35, 60]), ('pitcher_age', 'age_pit', [0, 23, 25, 27, 29, 31, 33, 35, 60]),
                            ('batter_history', 'batter_history_pa', [-1, 0, 150, 600, 1200, 1e9]), ('pitcher_history', 'pitcher_history_pa', [-1, 0, 150, 600, 1200, 1e9])):
        if col not in frame.columns:
            continue
        b = pd.cut(frame.loc[mask, col].to_numpy(float), bins)
        g = pd.DataFrame({'b': b, 'o': obs, 'p': pred}).groupby('b', observed=True)
        res[name] = {str(k): {'pa': int(len(v)), 'resid_x1000': round(float((v.o.mean() - v.p.mean()) * 1000), 2),
                              'se_x1000': round(float(v.o.std() / np.sqrt(len(v)) * 1000), 2)} for k, v in g}
    return res


def role_aggregates(P, y, meta):
    """Per (date, role, times through the order, batting side, inning bucket): plate appearances, observed counts, the
    model's predicted sums and the predicted sums after the production context offsets. Group-level only."""
    from brl_live.provider_adjust import load_offsets
    ctx = load_offsets()['table']
    bucket = np.where(meta['inning'] <= 1, '1st', np.where(meta['inning'] <= 8, 'mid', 'late'))
    side = meta['bat_home'].astype(int).astype(str)
    mult = np.stack([ctx[s_ + '_' + b_] for s_, b_ in zip(side, bucket)])
    Q = P * mult; Q = Q / Q.sum(1, keepdims=True)
    Y = np.eye(P.shape[1])[y]
    df = pd.DataFrame({'d': meta['date'], 'r': meta['role'], 't': meta['tto'], 's': side, 'b': bucket})
    rows = []
    for (d, r, t, s_, b_), ii in df.groupby(['d', 'r', 't', 's', 'b']).indices.items():
        rows.append({'date': d, 'role': r, 'tto': int(t), 'side': s_, 'bucket': b_, 'n': int(len(ii)), 'obs': [int(v) for v in Y[ii].sum(0)],
                     'pred': [round(float(v), 4) for v in P[ii].sum(0)], 'pred_ctx': [round(float(v), 4) for v in Q[ii].sum(0)]})
    summary = {}
    for season in sorted(set(str(x)[:4] for x in meta['date'])):
        m = np.array([str(x)[:4] == season for x in meta['date']])
        for r in ('starter', 'reliever'):
            mm = m & (meta['role'] == r)
            if mm.any():
                o, q_ = Y[mm].sum(0), Q[mm].sum(0)
                summary[season + '_' + r] = {'pa': int(mm.sum()), 'obs_over_pred_ctx': [round(float(a / b), 4) for a, b in zip(o, q_)]}
        for t in (1, 2, 3):
            mm = m & (meta['role'] == 'starter') & (meta['tto'] == t)
            if mm.any():
                o, q_ = Y[mm].sum(0), Q[mm].sum(0)
                summary[season + '_starter_tto' + str(t)] = {'pa': int(mm.sum()), 'obs_over_pred_ctx': [round(float(a / b), 4) for a, b in zip(o, q_)]}
    return rows, summary


def team_aggregates(P, y, frame, bat_names, fld_names):
    """Per (date, team, side): plate appearances, observed counts and predicted sums per class (team-level only)."""
    Y = np.eye(7)[y]
    df = pd.DataFrame({'d': frame['date_key'].astype(str).to_numpy()})
    rows = []
    for side, names in (('bat', bat_names), ('fld', fld_names)):
        df['t'] = names
        for (d, team), ii in df.groupby(['d', 't']).indices.items():
            rows.append({'date': d, 'team': str(team), 'side': side, 'n': int(len(ii)),
                         'obs': [int(v) for v in Y[ii].sum(0)], 'pred': [round(float(v), 4) for v in P[ii].sum(0)]})
    return rows


def offset_fit(P, Y, Z, l2=1.0, w0=None):
    """Multinomial correction on top of fixed probabilities P: log-odds log P + Z W (columns of W sum to zero), L-BFGS."""
    from scipy.optimize import minimize
    n, F = Z.shape; K = Y.shape[1]; base = np.log(np.clip(P, 1e-12, 1))
    def f(w):
        W = w.reshape(F, K); W = W - W.mean(1, keepdims=True)
        L = base + Z @ W; L -= L.max(1, keepdims=True); lse = np.log(np.exp(L).sum(1))
        ll = -(Y * (L - lse[:, None])).sum() / n
        Q = np.exp(L - lse[:, None]); G = Z.T @ (Q - Y) / n; G = G - G.mean(1, keepdims=True)
        return ll + 0.5 * l2 / n * (W ** 2).sum(), (G + l2 / n * W).ravel()
    r = minimize(f, np.zeros(F * K) if w0 is None else w0, jac=True, method='L-BFGS-B', options={'maxiter': 500})
    W = r.x.reshape(F, K); return W - W.mean(1, keepdims=True)


def offset_apply(P, Z, W):
    L = np.log(np.clip(P, 1e-12, 1)) + Z @ W; L -= L.max(1, keepdims=True); Q = np.exp(L); return Q / Q.sum(1, keepdims=True)


def stage2_eval(P, y, A, frame, cols, bat_i, fld_i, n_teams):
    """Aging layer on top of a fitted model's 2025-2026 predictions: fitted on earlier rows, scored on later ones;
    then team offsets on top of it. A: the aging and recency columns (NaN filled by column means)."""
    from research_lab.pa_model.evaluation import clustered_log_loss_difference_ci
    Y = np.eye(7)[y]; dates = frame['date_key'].astype(str).to_numpy(); games = frame['game_pk'].to_numpy()
    A = np.where(np.isnan(A), np.nanmean(A, axis=0), A)
    out = {}
    splits = {'fit_2025_score_2026': (dates < '2026-01-01', dates >= '2026-01-01'),
              'fit_through_june_2026_score_july_on': (dates < '2026-06-28', dates >= '2026-06-28'),
              'fit_2025_first_part_score_2025_blend': (dates < '2025-08-14', (dates >= '2025-08-14') & (dates < '2026-01-01'))}
    coefs = {}
    for name, (tr, te) in splits.items():
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
        W = offset_fit(P[tr], Y[tr], (A[tr] - mu) / sd)
        Q = offset_apply(P[te], (A[te] - mu) / sd, W)
        out[name] = clustered_log_loss_difference_ci(y[te], Q, P[te], games[te], replicates=300)
        if name == 'fit_2025_score_2026':
            coefs['fit_2025'] = {'mean': mu.tolist(), 'sd': sd.tolist(), 'coef': np.round(W, 5).tolist()}
    for label, tr in (('fit_2026', dates >= '2026-01-01'), ('fit_all', np.ones(len(dates), bool))):
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
        W = offset_fit(P[tr], Y[tr], (A[tr] - mu) / sd)
        coefs[label] = {'mean': mu.tolist(), 'sd': sd.tolist(), 'coef': np.round(W, 5).tolist()}
    out['coefficients'] = {'columns': list(cols), 'labels': list(LABELS), **coefs}
    # Team offsets on top of the aging layer fitted on 2025 (2026 scored) and on 2025's first part (2025 blend scored)
    c25 = coefs['fit_2025']; Z = (A - np.array(c25['mean'])) / np.array(c25['sd'])
    Q_all = offset_apply(P, Z, np.array(c25['coef']))
    test26, blend25 = dates >= '2026-01-01', (dates >= '2025-08-14') & (dates < '2026-01-01')
    for k, hl in ((4000.0, 180.0), (8000.0, 180.0)):
        Qt = team_offsets(Q_all, Y, dates, bat_i, fld_i, n_teams, k, hl, ('bat', 'fld'))
        out[f'aging_layer_plus_team_k{int(k)}_hl{int(hl)}'] = {
            'test_2026_vs_base': clustered_log_loss_difference_ci(y[test26], Qt[test26], P[test26], games[test26], replicates=300),
            'test_2026_vs_aging_layer': clustered_log_loss_difference_ci(y[test26], Qt[test26], Q_all[test26], games[test26], replicates=300)}
    out['residual_bins_2026_after_aging_layer'] = residual_bins(Q_all, y, frame, test26)
    return out, Q_all


def retrain_experiment(pa, physics, features, locked_columns, y, config, put, run_id) -> dict:
    """Does retraining on the latest seasons beat the frozen 2023-2024 model? The frozen protocol (train 2023-2024 plus
    the first half of 2025, calibrate on 2025) against a retrained one (train 2023-2025 plus the first half of 2026,
    calibrate on mid-2026), both scored on the same final block of 2026 (mid-August on), with and without the aging and
    recency features, then team offsets on top. Writes team residual rows of the best retrained model."""
    from research_lab.pa_model.config import PAConfig
    from research_lab.pa_model.model import fit_frozen_model, validation_partitions
    from research_lab.pa_model.evaluation import probability_metrics, clustered_log_loss_difference_ci
    new_cfg = PAConfig(train_years=(2023, 2024, 2025), validation_years=(2026,), test_years=(2027,), evaluation_mode='locked_final')
    parts_new, audit_new = validation_partitions(features, new_cfg)
    final = parts_new['blend']
    out = {'final_block': {'rows': int(final.sum()), 'first_date': str(features.loc[final, 'date_key'].min()), 'last_date': str(features.loc[final, 'date_key'].max())},
           'retrained_partitions': {k: int(v.sum()) for k, v in parts_new.items()}}
    params = {'xvalue': True, 'recent_days': 30, 'aging': True, 'decay_days': 365}
    extras, audit = build_extras(pa, physics, params)
    for c in extras.columns:
        features[c] = extras[c].to_numpy()
    v2_cols = phys.feature_names({'xvalue': True, 'recent_days': 30})
    age_cols = phys.AGING_FEATURES + phys.DECAY_FEATURES
    games = features.loc[final, 'game_pk'].to_numpy()
    probs, fitted_models = {}, {}
    for name, cfg, cols in (('v2_frozen', config, v2_cols), ('v2_retrained', new_cfg, v2_cols),
                            ('aging_frozen', config, v2_cols + age_cols), ('aging_retrained', new_cfg, v2_cols + age_cols)):
        fitted, tuning = fit_frozen_model(features, list(locked_columns) + cols, cfg)
        P = fitted.predict_proba(features.loc[final])
        probs[name] = P; fitted_models[name] = fitted
        out[name] = {'features': len(locked_columns) + len(cols), 'c': tuning['best_regularization_c'], 'final_block': probability_metrics(y[final], P).to_dict()}
        if name != 'v2_frozen':
            out[name]['minus_v2_frozen'] = clustered_log_loss_difference_ci(y[final], P, probs['v2_frozen'], games, replicates=600)
        put(name)
    # team offsets on top, from each model's own residuals on 2025-2026 rows before each date
    ordered_pa = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    top = ordered_pa['inning_topbot'].astype(str).str.lower().str.startswith('top').to_numpy()
    bat_names = np.where(top, ordered_pa['away_team'].astype(str), ordered_pa['home_team'].astype(str))
    fld_names = np.where(top, ordered_pa['home_team'].astype(str), ordered_pa['away_team'].astype(str))
    teams = sorted(set(bat_names) | set(fld_names)); tix = {tm: i for i, tm in enumerate(teams)}
    bat_i = np.array([tix[tm] for tm in bat_names]); fld_i = np.array([tix[tm] for tm in fld_names])
    span = features['season'].isin((2025, 2026)).to_numpy()
    Y = np.eye(7)[y[span]]; dates = features.loc[span, 'date_key'].astype(str).to_numpy()
    final_in_span = final[span]
    for name in ('v2_frozen', 'v2_retrained', 'aging_retrained'):
        P_span = fitted_models[name].predict_proba(features.loc[span])
        for k, hl in ((4000.0, 180.0), (8000.0, 180.0)):
            Q = team_offsets(P_span, Y, dates, bat_i[span], fld_i[span], len(teams), k, hl, ('bat', 'fld'))
            out[name][f'team_k{int(k)}_hl{int(hl)}_minus_v2_frozen'] = clustered_log_loss_difference_ci(y[final], Q[final_in_span], probs['v2_frozen'], games, replicates=600)
            out[name][f'team_k{int(k)}_hl{int(hl)}_minus_self'] = clustered_log_loss_difference_ci(y[final], Q[final_in_span], probs[name], games, replicates=300)
        meta = features.loc[span, ['date_key', 'game_pk', 'batter_history_pa', 'pitcher_history_pa']].copy()
        meta['age_bat'] = ordered_pa.loc[span, 'age_bat'].to_numpy(float); meta['age_pit'] = ordered_pa.loc[span, 'age_pit'].to_numpy(float)
        out[name]['residual_bins_final_block'] = residual_bins(P_span, y[span], meta, final_in_span)
        if name == 'aging_retrained':
            aggregates = team_aggregates(P_span, y[span], meta, bat_names[span], fld_names[span])
            out['team_aggregates'] = {'variant': name, 'rows': len(aggregates)}
            out['_aggregates'] = aggregates
    return out


def entrypoint_module():
    spec = importlib.util.spec_from_file_location('brl_entrypoint', ROOT / 'brl_engine' / 'entrypoint.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# Experiment sets: each variant is a parameter dict for build_extras; the locked model is the shared reference.
SETS = {
    'physics': {'physics': {}},
    'physics2': {'base': {}, 'k_low': {'k_rate': 60.0, 'k_bip': 30.0}, 'k_high': {'k_rate': 300.0, 'k_bip': 120.0},
                 'xvalue': {'xvalue': True}, 'recent30': {'recent_days': 30}, 'xvalue_recent30': {'xvalue': True, 'recent_days': 30}},
    'physics3': {'v2': {'xvalue': True, 'recent_days': 30}, 'types': {'xvalue': True, 'recent_days': 30, 'pitch_types': True},
                 'types_k30': {'xvalue': True, 'recent_days': 30, 'pitch_types': True, 'k_type': 30.0},
                 'recent45': {'xvalue': True, 'recent_days': 45}, 'recent20': {'xvalue': True, 'recent_days': 20}},
    'environment': {'v2': {'xvalue': True, 'recent_days': 30}, 'env30': {'xvalue': True, 'recent_days': 30, 'environment': True},
                    'env14': {'xvalue': True, 'recent_days': 30, 'environment': True, 'env_days': 14, 'k_env': 1000.0},
                    'env60': {'xvalue': True, 'recent_days': 30, 'environment': True, 'env_days': 60, 'k_env': 4000.0}},
    'matchup': {'v2': {'xvalue': True, 'recent_days': 30}, 'matchup': {'xvalue': True, 'recent_days': 30, 'matchup': True},
                'matchup_k30': {'xvalue': True, 'recent_days': 30, 'matchup': True, 'k_type': 30.0},
                'matchup_k150': {'xvalue': True, 'recent_days': 30, 'matchup': True, 'k_type': 150.0}},
    'workload': {'v2': {'xvalue': True, 'recent_days': 30}, 'workload': {'xvalue': True, 'recent_days': 30, 'workload': True},
                 'workload_matchup': {'xvalue': True, 'recent_days': 30, 'workload': True, 'matchup': True}},
    'aging': {'v2': {'xvalue': True, 'recent_days': 30}, 'aging': {'xvalue': True, 'recent_days': 30, 'aging': True},
              'dec365': {'xvalue': True, 'recent_days': 30, 'decay_days': 365},
              'aging_dec365': {'xvalue': True, 'recent_days': 30, 'aging': True, 'decay_days': 365}},
    # stage2: one feature build with the aging and recency columns; v2 fitted without them, aging_dec365 with them; an
    # aging layer fitted on each model's own earlier predictions (TEAM_EVAL path, stage2_eval).
    'stage2': {'v2': {'xvalue': True, 'recent_days': 30, 'aging': True, 'decay_days': 365, '_model_excludes_aging': True},
               'aging_dec365': {'xvalue': True, 'recent_days': 30, 'aging': True, 'decay_days': 365}},
    # role: the production model's residuals by pitcher role (starter or reliever), times through the order and inning bucket,
    # after the production context offsets (ROLE-01)
    'role': {'v2': {'xvalue': True, 'recent_days': 30}},
    # window: the physics features limited to the last 548 days (RETRAIN-03) against all history; team residual rows for both
    'window': {'v2': {'xvalue': True, 'recent_days': 30}, 'w548': {'xvalue': True, 'recent_days': 30, 'window_days': 548}},
    'defense': {'v2': {'xvalue': True, 'recent_days': 30}, 'defense': {'xvalue': True, 'recent_days': 30, 'defense': True},
                'defense_k200': {'xvalue': True, 'recent_days': 30, 'defense': True, 'k_def': 200.0},
                'defense_730': {'xvalue': True, 'recent_days': 30, 'defense': True, 'defense_days': 730}},
}


def postseason_usage(repo, token, key, branch, seasons=(2023, 2024, 2025, 2026)) -> dict:
    """How starters and bullpens are used in postseason games compared with the regular season (from the sealed study seasons)."""
    import collections
    starts, relievers = [], []
    for year in seasons:
        raw = read_blob(repo, token, study_path(year), branch)
        if raw is None:
            continue
        doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(year))))
        for pk, game in doc['games'].items():
            gt = game.get('game_type', 'R'); rows = sorted(game['rows'], key=lambda r: r['i'])
            by_half = {'top': [], 'bottom': []}
            for r in rows:
                by_half[r.get('half', 'top')].append(r)
            for half, hr in by_half.items():
                if not hr:
                    continue
                counts = collections.Counter(r['p'] for r in hr)
                starter = hr[0]['p']
                starts.append({'season': year, 'game_type': gt, 'pitcher': starter, 'bf': counts[starter], 'date': game.get('date')})
                relievers.append({'season': year, 'game_type': gt, 'n': len(counts) - 1})
        del doc
    st = pd.DataFrame(starts); rl = pd.DataFrame(relievers)
    out = {'starters_by_game_type': {}, 'relievers_per_team_game': {}, 'same_pitcher': {}}
    for gt, g in st.groupby('game_type'):
        out['starters_by_game_type'][gt] = {'starts': int(len(g)), 'mean_bf': round(float(g.bf.mean()), 2), 'median_bf': float(g.bf.median()),
                                            'q25': float(g.bf.quantile(0.25)), 'q75': float(g.bf.quantile(0.75)), 'share_under_16_bf': round(float((g.bf < 16).mean()), 3)}
    for gt, g in rl.groupby('game_type'):
        out['relievers_per_team_game'][gt] = round(float(g.n.mean()), 2)
    reg = st[st.game_type == 'R'].groupby(['pitcher', 'season']).bf.median().rename('reg_median')
    post = st[st.game_type.isin(['F', 'D', 'L', 'W'])].join(reg, on=['pitcher', 'season']).dropna()
    for gt, g in post.groupby('game_type'):
        out['same_pitcher'][gt] = {'starts': int(len(g)), 'mean_bf': round(float(g.bf.mean()), 2), 'own_regular_median': round(float(g.reg_median.mean()), 2),
                                   'ratio_mean': round(float((g.bf / g.reg_median).mean()), 3), 'ratio_median': round(float((g.bf / g.reg_median).median()), 3)}
    allpost = post
    out['same_pitcher']['all_postseason'] = {'starts': int(len(allpost)), 'ratio_mean': round(float((allpost.bf / allpost.reg_median).mean()), 3) if len(allpost) else None,
                                            'mean_bf': round(float(allpost.bf.mean()), 2) if len(allpost) else None, 'own_regular_median': round(float(allpost.reg_median.mean()), 2) if len(allpost) else None}
    return out


def postseason_tiers(repo, token, key, branch, seasons=(2023, 2024, 2025, 2026), boot=2000, seed=20261011) -> dict:
    """POST-03 (LEDGER): do postseason managers keep their best starters in longer? Each postseason start's batters faced
    over the starter's own regular-season median (same season), by his regular-season strikeouts minus walks per plate
    appearance (terciles of the postseason starts) and by his own regular-season median (workhorses against the rest);
    intervals from resampling starts. Aggregates only."""
    import collections
    starts, lines = [], collections.defaultdict(lambda: [0, 0, 0])
    for year in seasons:
        raw = read_blob(repo, token, study_path(year), branch)
        if raw is None:
            continue
        doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(year))))
        for pk, game in doc['games'].items():
            gt = game.get('game_type', 'R'); rows = sorted(game['rows'], key=lambda r: r['i'])
            by_half = {'top': [], 'bottom': []}
            for r in rows:
                by_half[r.get('half', 'top')].append(r)
                if gt == 'R':
                    ln = lines[(int(r['p']), year)]; ln[0] += 1; ln[1] += r.get('o') == 'K'; ln[2] += r.get('o') == 'BB_HBP'
            for half, hr in by_half.items():
                if hr:
                    counts = collections.Counter(r['p'] for r in hr)
                    starts.append({'season': year, 'game_type': gt, 'pitcher': int(hr[0]['p']), 'bf': counts[hr[0]['p']]})
        del doc
    st = pd.DataFrame(starts)
    reg = st[st.game_type == 'R'].groupby(['pitcher', 'season']).bf.median().rename('reg_median')
    post = st[st.game_type.isin(['F', 'D', 'L'])].join(reg, on=['pitcher', 'season']).dropna().copy()
    post['ratio'] = post.bf / post.reg_median
    post['kbb'] = [((lines[(p, y)][1] - lines[(p, y)][2]) / lines[(p, y)][0]) if lines[(p, y)][0] >= 200 else np.nan for p, y in zip(post.pitcher, post.season)]
    post = post.dropna(subset=['kbb'])
    rng = np.random.default_rng(seed)
    def summary(mask):
        x = post.ratio[mask].to_numpy()
        if not len(x):
            return None
        b = x[rng.integers(0, len(x), size=(boot, len(x)))].mean(1)
        return {'starts': int(len(x)), 'ratio_mean': round(float(x.mean()), 3), 'ci': [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)],
                'mean_bf': round(float(post.bf[mask].mean()), 2), 'own_regular_median': round(float(post.reg_median[mask].mean()), 2)}
    q1, q2 = np.percentile(post.kbb, [100 / 3, 200 / 3])
    tiers = {'top': post.kbb >= q2, 'middle': (post.kbb >= q1) & (post.kbb < q2), 'bottom': post.kbb < q1}
    out = {'starts': int(len(post)), 'kbb_cuts': [round(float(q1), 4), round(float(q2), 4)], 'by_kbb': {k: summary(m) for k, m in tiers.items()},
           'by_regular_median': {'24_or_more': summary(post.reg_median >= 24), '21_to_23': summary((post.reg_median >= 21) & (post.reg_median < 24)), 'under_21': summary(post.reg_median < 21)}}
    a, b_ = post.ratio[tiers['top']].to_numpy(), post.ratio[tiers['bottom']].to_numpy()
    d = (a[rng.integers(0, len(a), size=(boot, len(a)))].mean(1) - b_[rng.integers(0, len(b_), size=(boot, len(b_)))].mean(1))
    out['top_minus_bottom'] = {'diff': round(float(a.mean() - b_.mean()), 3), 'ci': [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]}
    return out


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key_hex = os.environ['BRL_PA_PACKAGE_KEY']; key = key_bytes(key_hex)
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    experiment = (os.environ.get('BRL_EXPERIMENT') or 'physics').strip()
    if experiment in ('postseason_usage', 'postseason_tiers'):
        run_id = os.environ.get('GITHUB_RUN_ID', 'local')
        receipt = {'schema': 'brl.research-receipt.v1', 'experiment': experiment, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat()}
        try:
            receipt['results'] = (postseason_tiers if experiment == 'postseason_tiers' else postseason_usage)(repo, token, key, branch); receipt['status'] = 'completed'
        except Exception as exc:
            receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
            receipt['where'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in traceback.extract_tb(exc.__traceback__)[-5:]]
        put_text(repo, token, f'research/{experiment}-{run_id}.json', json.dumps(receipt, indent=1, default=float), branch, 'BRL: research ' + experiment)
        print(json.dumps({k: receipt.get(k) for k in ('status', 'error')}))
        return
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    work = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'brl-research'
    receipt = {'schema': 'brl.research-receipt.v1', 'experiment': experiment, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat(), 'stages': []}
    t0 = time.time()
    def stage(name):
        receipt['stages'].append({'stage': name, 'at_seconds': round(time.time() - t0, 1)}); print(name, round(time.time() - t0), 's', flush=True)
    try:
        stage('restore data package')
        ep = entrypoint_module()
        manifest = json.loads((ROOT / 'brl_engine' / 'data_package.json').read_text())
        data_root = work / 'data'
        if data_root.exists():
            import shutil; shutil.rmtree(data_root)
        data_root.mkdir(parents=True)
        ep.restore_package(repo, token, key_hex, manifest, data_root)
        pa_path = next(data_root.rglob('plate_appearances.csv.gz'))
        from research_lab.pa_model.config import PAConfig
        from research_lab.pa_model.features import build_time_valid_features
        from research_lab.pa_model.model import fit_frozen_model, validation_partitions
        from research_lab.pa_model.evaluation import probability_metrics, clustered_log_loss_difference_ci
        config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final')
        stage('load plate appearances')
        pa = pd.read_csv(pa_path, low_memory=False)
        pa['date_key'] = pa['date_key'].astype(str)
        receipt['pa_rows'] = int(len(pa)); receipt['seasons'] = sorted(int(s) for s in pa['season'].unique())
        stage('load pitch physics seasons')
        physics = []; seasons_loaded = []
        for year in receipt['seasons']:
            raw = read_blob(repo, token, physics_path(year), branch)
            if raw is not None:
                import io
                table = pd.read_csv(io.BytesIO(gzip.decompress(unseal(raw, key, physics_purpose(year)))))
                physics.append(table); seasons_loaded.append(year)
                continue
            raw = read_blob(repo, token, study_path(year), branch)
            if raw is None:
                continue
            doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(year))))
            if doc.get('schema') != STUDY_SCHEMA:
                raise ValueError('study schema mismatch for ' + str(year))
            physics.append(per_pa_physics(doc)); seasons_loaded.append(year); del doc
        receipt['physics_seasons'] = seasons_loaded
        if not physics:
            raise ValueError('No pitch-physics seasons are sealed yet')
        physics = pd.concat(physics, ignore_index=True)
        stage('build locked features')
        features, locked_columns = build_time_valid_features(pa, config)
        label_to_index = {label: i for i, label in enumerate(LABELS)}
        y = features['outcome'].map(label_to_index).to_numpy(int)
        partitions, _ = validation_partitions(features, config)
        test_mask = features['season'].isin(config.test_years).to_numpy(); blend_mask = partitions['blend']
        parts = (('validation_blend_2025', blend_mask), ('test_2026', test_mask))
        results = {}
        probs = {}

        def fit_and_score(name, cols, extra_cols):
            fitted, tuning = fit_frozen_model(features, cols, config)
            out = {'feature_count': len(cols), 'best_regularization_c': tuning['best_regularization_c'], 'best_tune_log_loss': tuning['best_tune_log_loss'],
                   'calibration': {k: tuning['calibration'][k] for k in ('temperature', 'pre_calibration_log_loss', 'post_calibration_log_loss')}}
            probs[name] = {'_fitted': fitted} if experiment in TEAM_EVAL else {}
            for part, mask in parts:
                p = fitted.predict_proba(features.loc[mask]); probs[name][part] = p
                out[part] = probability_metrics(y[mask], p).to_dict()
                months = pd.to_datetime(features.loc[mask, 'date_key']).dt.month.to_numpy()
                out[part + '_by_month'] = {int(m): float(probability_metrics(y[mask][months == m], p[months == m]).log_loss) for m in np.unique(months)}
            if extra_cols:
                coef = fitted.estimator.named_steps['model'].coef_
                idx = [cols.index(c) for c in extra_cols]
                out['extra_coefficients_by_class'] = {LABELS[k]: {c: round(float(coef[k, i]), 4) for c, i in zip(extra_cols, idx)} for k in range(coef.shape[0])}
            return out

        if experiment == 'retrain':
            stage('retrain experiment')
            res = retrain_experiment(pa, physics, features, locked_columns, y, config, stage, run_id)
            aggregates = res.pop('_aggregates', None)
            if aggregates:
                raw = gzip.compress(json.dumps({'schema': 'brl.team-residuals.v1', 'variant': 'aging_retrained', 'labels': list(LABELS), 'rows': aggregates}).encode(), mtime=0)
                put_bytes(repo, token, f'research/team-resid-aging_retrained-{run_id}.json.gz', raw, branch, 'BRL: team residuals, retrained model')
                res['team_aggregates']['file'] = f'research/team-resid-aging_retrained-{run_id}.json.gz'
            results['retrain'] = res
            receipt['results'] = results; receipt['status'] = 'completed'
            raise _Done()
        stage('fit locked')
        results['locked'] = fit_and_score('locked', list(locked_columns), [])
        if experiment in ('gbm', 'gbm_slow'):
            extras, audit = build_extras(pa, physics, {'xvalue': True, 'recent_days': 30})
            results.setdefault('physics_join', {})['gbm'] = audit
            configs = None
            if experiment == 'gbm_slow':
                configs = {'gbm_slow_d5': dict(max_depth=5, learning_rate=0.02, l2_regularization=2.0, min_samples_leaf=200, checkpoints=(300, 600, 900, 1200, 1600, 2000)),
                           'gbm_slow_d3': dict(max_depth=3, learning_rate=0.03, l2_regularization=2.0, min_samples_leaf=200, checkpoints=(300, 600, 900, 1200, 1600, 2000))}
            results.update(gbm_experiment(features, list(locked_columns), extras, y, partitions, parts, config, probs['locked'], configs=configs))
            receipt['results'] = results; receipt['status'] = 'completed'
            raise _Done()
        variants = SETS.get(experiment) or {experiment: {}}
        team_eval_on = experiment in TEAM_EVAL
        if team_eval_on:
            ordered_pa = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
            if not (ordered_pa['game_pk'].to_numpy() == features['game_pk'].to_numpy()).all():
                raise ValueError('PA order does not match the feature frame')
            top = ordered_pa['inning_topbot'].astype(str).str.lower().str.startswith('top').to_numpy()
            bat_names = np.where(top, ordered_pa['away_team'].astype(str), ordered_pa['home_team'].astype(str))
            fld_names = np.where(top, ordered_pa['home_team'].astype(str), ordered_pa['away_team'].astype(str))
            teams = sorted(set(bat_names) | set(fld_names)); tix = {tm: i for i, tm in enumerate(teams)}
            bat_i = np.array([tix[tm] for tm in bat_names]); fld_i = np.array([tix[tm] for tm in fld_names])
            for col in ('age_bat', 'age_pit'):
                features[col + '_meta'] = ordered_pa[col].to_numpy(float)
            eval_mask = features['season'].isin((2025, 2026)).to_numpy()
            sub = features.loc[eval_mask].reset_index(drop=True)
            if experiment == 'role':
                first_p = ordered_pa.groupby([ordered_pa['game_pk'].to_numpy(), fld_names])['pitcher'].transform('first').to_numpy()
                tto = pd.to_numeric(ordered_pa['n_thruorder_pitcher'], errors='coerce').fillna(1).clip(1, 3).astype(int).to_numpy() if 'n_thruorder_pitcher' in ordered_pa.columns else np.ones(len(ordered_pa), int)
                role_meta = {'date': ordered_pa['date_key'].astype(str).str[:10].to_numpy()[eval_mask],
                             'role': np.where(ordered_pa['pitcher'].to_numpy() == first_p, 'starter', 'reliever')[eval_mask],
                             'tto': np.where(ordered_pa['pitcher'].to_numpy() == first_p, tto, 0)[eval_mask],
                             'inning': pd.to_numeric(ordered_pa['inning'], errors='coerce').fillna(1).astype(int).to_numpy()[eval_mask],
                             'bat_home': (~top)[eval_mask]}
            parts_eval = (('validation_blend_2025', blend_mask[eval_mask]), ('test_2026', test_mask[eval_mask]))
        for vname, params in variants.items():
            stage('build physics features ' + vname)
            params = dict(params); excl = params.pop('_model_excludes_aging', False)
            extras, audit = build_extras(pa, physics, params)
            results.setdefault('physics_join', {})[vname] = audit
            cols = list(extras.columns)
            for c in cols:
                features[c] = extras[c].to_numpy()
            age_cols = [c for c in cols if c in phys.AGING_FEATURES + phys.DECAY_FEATURES]
            model_cols = [c for c in cols if not (excl and c in age_cols)]
            stage('fit ' + vname)
            results[vname] = fit_and_score(vname, list(locked_columns) + model_cols, model_cols)
            for part, mask in parts:
                games = features.loc[mask, 'game_pk'].to_numpy()
                results[vname][part + '_minus_locked'] = clustered_log_loss_difference_ci(y[mask], probs[vname][part], probs['locked'][part], games, replicates=600)
            if team_eval_on:
                stage('team offsets ' + vname)
                fitted_v = probs[vname].pop('_fitted')
                P_all = fitted_v.predict_proba(features.loc[eval_mask])
                y_sub = y[eval_mask]
                sub_meta = sub[['date_key', 'game_pk']].copy()
                sub_meta['age_bat'] = features.loc[eval_mask, 'age_bat_meta'].to_numpy(); sub_meta['age_pit'] = features.loc[eval_mask, 'age_pit_meta'].to_numpy()
                for col in ('batter_history_pa', 'pitcher_history_pa'):
                    sub_meta[col] = features.loc[eval_mask, col].to_numpy(float)
                results[vname]['team_offsets'] = team_eval(P_all, y_sub, sub_meta, bat_i[eval_mask], fld_i[eval_mask], len(teams), parts_eval)
                results[vname]['residual_bins_2026'] = residual_bins(P_all, y_sub, sub_meta, parts_eval[1][1])
                aggregates = team_aggregates(P_all, y_sub, sub_meta, bat_names[eval_mask], fld_names[eval_mask])
                if experiment == 'role':
                    stage('role residuals ' + vname)
                    r_rows, r_summary = role_aggregates(P_all, y_sub, role_meta)
                    results[vname]['role_summary'] = r_summary
                    raw_r = gzip.compress(json.dumps({'schema': 'brl.role-residuals.v1', 'variant': vname, 'labels': list(LABELS), 'rows': r_rows}).encode(), mtime=0)
                    put_bytes(repo, token, f'research/role-resid-{vname}-{run_id}.json.gz', raw_r, branch, 'BRL: role residuals ' + vname)
                    results[vname]['role_aggregates_file'] = f'research/role-resid-{vname}-{run_id}.json.gz'
                if experiment == 'stage2' and age_cols:
                    stage('aging layer ' + vname)
                    s2, Q_aged = stage2_eval(P_all, y_sub, features.loc[eval_mask, age_cols].to_numpy(float), sub_meta, age_cols,
                                             bat_i[eval_mask], fld_i[eval_mask], len(teams))
                    results[vname]['aging_layer'] = s2
                    aged = team_aggregates(Q_aged, y_sub, sub_meta, bat_names[eval_mask], fld_names[eval_mask])
                    raw2 = gzip.compress(json.dumps({'schema': 'brl.team-residuals.v1', 'variant': vname + '+aging-layer-fit2025', 'params': params,
                                                     'labels': list(LABELS), 'rows': aged}).encode(), mtime=0)
                    put_bytes(repo, token, f'research/team-resid-{vname}-aged-{run_id}.json.gz', raw2, branch, 'BRL: team residuals after the aging layer ' + vname)
                raw = gzip.compress(json.dumps({'schema': 'brl.team-residuals.v1', 'variant': vname, 'params': params, 'labels': list(LABELS),
                                                'rows': aggregates}).encode(), mtime=0)
                put_bytes(repo, token, f'research/team-resid-{vname}-{run_id}.json.gz', raw, branch, 'BRL: team residuals ' + vname)
                results[vname]['team_aggregates_file'] = f'research/team-resid-{vname}-{run_id}.json.gz'
                del P_all
            features.drop(columns=cols, inplace=True)
            del extras
        receipt['results'] = results
        receipt['status'] = 'completed'
    except _Done:
        pass
    except Exception as exc:
        receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        frames = traceback.extract_tb(exc.__traceback__)
        receipt['where'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in frames[-6:]]
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat(); receipt['seconds'] = round(time.time() - t0, 1)
    text = json.dumps(receipt, indent=1, default=float)
    for secret in (key_hex, token):
        text = text.replace(secret, '[redacted]')
    put_text(repo, token, f'research/{experiment}-{run_id}.json', text, branch, 'BRL: research ' + experiment)
    print(json.dumps({k: receipt.get(k) for k in ('status', 'error', 'seconds')}))
    if receipt['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
