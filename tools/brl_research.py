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
            probs[name] = {}
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
        variants = SETS.get(experiment) or {experiment: {}}
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
            features.drop(columns=cols, inplace=True)
            del extras
        receipt['results'] = results
        receipt['status'] = 'completed'
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
