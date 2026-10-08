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
TEAM_EVAL = {'aging'}
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


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key_hex = os.environ['BRL_PA_PACKAGE_KEY']; key = key_bytes(key_hex)
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    experiment = (os.environ.get('BRL_EXPERIMENT') or 'physics').strip()
    if experiment == 'postseason_usage':
        run_id = os.environ.get('GITHUB_RUN_ID', 'local')
        receipt = {'schema': 'brl.research-receipt.v1', 'experiment': experiment, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat()}
        try:
            receipt['results'] = postseason_usage(repo, token, key, branch); receipt['status'] = 'completed'
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
            parts_eval = (('validation_blend_2025', blend_mask[eval_mask]), ('test_2026', test_mask[eval_mask]))
        for vname, params in variants.items():
            stage('build physics features ' + vname)
            extras, audit = build_extras(pa, physics, params)
            results.setdefault('physics_join', {})[vname] = audit
            cols = list(extras.columns)
            for c in cols:
                features[c] = extras[c].to_numpy()
            stage('fit ' + vname)
            results[vname] = fit_and_score(vname, list(locked_columns) + cols, cols)
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
