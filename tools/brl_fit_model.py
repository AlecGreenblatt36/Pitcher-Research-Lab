"""Fit the next plate-appearance model inside Actions and seal it on the ledger branch.

The locked benchmark (research_lab.pa_model.pipeline.run_benchmark: 2023-2024 train, 2025
tune/calibrate/blend, 2026 locked holdout) is run on the locked features plus the pitch-physics
features (research_lab.pa_model.physics) built from the sealed per-PA physics tables. The
resulting bundle (talent_plus_context and talent_only models, empirical-Bayes blend, physics
parameters and feature list, a name) is sealed as private/models/<name>.enc, and a public
manifest with hashes, parameters and the benchmark's metrics is committed to main at
brl_engine/models/<name>.json. The live runtime selects a model through brl_engine/model.json.
No player rows leave the runner.
"""
from __future__ import annotations
import base64, gzip, hashlib, importlib.util, io, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
from cloud.security import seal, unseal, key_bytes  # noqa: E402
from brl_live.bookkeeping_season import physics_path, physics_purpose  # noqa: E402
from research_lab.pa_model import physics as phys  # noqa: E402

DEFAULT_NAME = 'pa-2026-v2-physics'


def api(url, token, method='GET', payload=None):
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-fit/1.0'}
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


def entrypoint_module():
    spec = importlib.util.spec_from_file_location('brl_entrypoint', ROOT / 'brl_engine' / 'entrypoint.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def model_path(name: str) -> str:
    return f'private/models/{name}.enc'


def model_purpose(name: str) -> str:
    return f'model:{name}'


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key_hex = os.environ['BRL_PA_PACKAGE_KEY']; key = key_bytes(key_hex)
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    settings = {}
    settings_file = ROOT / 'tools' / 'fit_model_params.json'
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
    name = str(settings.get('name') or DEFAULT_NAME)
    params = dict(settings.get('physics_params') or {})
    work = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'brl-fit'
    receipt = {'schema': 'brl.model-fit-receipt.v1', 'name': name, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat(), 'stages': [], 'physics_params': params}
    t0 = time.time()
    def stage(label):
        receipt['stages'].append({'stage': label, 'at_seconds': round(time.time() - t0, 1)}); print(label, round(time.time() - t0), 's', flush=True)
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
        from research_lab.pa_model.pipeline import run_benchmark
        # Training window from tools/fit_model_params.json (default: the frozen 2023-2024 protocol). A retrained model
        # (train 2023-2025, validate on 2026, no test season yet) reports its metrics on 2026's final block (RETRAIN-01).
        # TOTALS-07: the park counter's prior (plate appearances) can be set from the settings; the default is the frozen 1,000.
        priors = {k: float(settings[k]) for k in ('park_prior_pa', 'player_prior_pa', 'split_prior_pa', 'recent_prior_pa') if settings.get(k) is not None}
        config = PAConfig(train_years=tuple(settings.get('train_years') or (2023, 2024)), validation_years=tuple(settings.get('validation_years') or (2025,)),
                          test_years=tuple(settings.get('test_years') or (2026,)), evaluation_mode='locked_final', **priors)
        receipt['config'] = {'train_years': list(config.train_years), 'validation_years': list(config.validation_years), 'test_years': list(config.test_years), **priors}
        stage('load plate appearances')
        pa = pd.read_csv(pa_path, low_memory=False)
        pa['date_key'] = pa['date_key'].astype(str)
        receipt['pa_rows'] = int(len(pa)); receipt['seasons'] = sorted(int(s) for s in pa['season'].unique())
        receipt['history_sha256'] = hashlib.sha256(pa_path.read_bytes()).hexdigest()
        stage('load physics tables')
        tables = []
        for year in receipt['seasons']:
            raw = read_blob(repo, token, physics_path(year), branch)
            if raw is None:
                raise ValueError(f'physics table for {year} is not sealed yet')
            tables.append(pd.read_csv(io.BytesIO(gzip.decompress(unseal(raw, key, physics_purpose(year))))))
        table = pd.concat(tables, ignore_index=True)
        receipt['physics_rows'] = int(len(table))
        stage('build physics features')
        extras, audit = phys.build_features(pa, table, params)
        receipt['physics_join'] = audit
        stage('run the locked benchmark with physics features')
        out = work / 'model'
        bundle_extra = {'physics_params': dict(phys.DEFAULT_PARAMS, **params), 'physics_features': list(extras.columns), 'name': name}
        report = run_benchmark(pa, out, config, extra_features=extras, bundle_extra=bundle_extra, blend_as_test=bool(settings.get('blend_as_test')))
        stage('seal the model')
        raw_model = (out / 'pa_model.joblib').read_bytes()
        cipher = seal(raw_model, key, model_purpose(name))
        put_bytes(repo, token, model_path(name), cipher, branch, f'BRL: model {name}')
        public_report = {k: v for k, v in report.items() if k in ('models', 'baselines', 'incremental_value', 'promotion', 'claim_status', 'interpretation_boundary', 'artifacts')}
        manifest_doc = {'schema': 'brl.model-manifest.v1', 'name': name, 'path': model_path(name), 'purpose': model_purpose(name), 'branch': branch,
                        'format': 'AES-256-GCM; BRLAESG1 header; 12-byte nonce; AAD BRL:' + model_purpose(name),
                        'model_sha256': hashlib.sha256(raw_model).hexdigest(), 'cipher_sha256': hashlib.sha256(cipher).hexdigest(), 'model_bytes': len(raw_model),
                        'history_sha256': receipt['history_sha256'], 'physics_params': bundle_extra['physics_params'], 'physics_features': bundle_extra['physics_features'],
                        'physics_seasons': receipt['seasons'], 'physics_rows': receipt['physics_rows'], 'physics_join': audit,
                        'config': config.to_dict(), 'benchmark': json.loads(json.dumps(public_report, default=float)), 'fitted_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_id}
        text = json.dumps(manifest_doc, indent=1, default=float)
        for secret in (key_hex, token):
            text = text.replace(secret, '[redacted]')
        put_bytes(repo, token, f'brl_engine/models/{name}.json', text.encode(), 'main', f'BRL: model manifest {name}')
        receipt['manifest'] = f'brl_engine/models/{name}.json'; receipt['model_sha256'] = manifest_doc['model_sha256']
        receipt['test_2026'] = json.loads(json.dumps({k: v.get('metrics', {}).get('log_loss') for k, v in report.get('models', {}).items()}, default=float))
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        frames = traceback.extract_tb(exc.__traceback__)
        receipt['where'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in frames[-6:]]
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat(); receipt['seconds'] = round(time.time() - t0, 1)
    text = json.dumps(receipt, indent=1, default=float)
    for secret in (key_hex, token):
        text = text.replace(secret, '[redacted]')
    put_bytes(repo, token, f'research/fit-{name}-{run_id}.json', text.encode(), branch, 'BRL: model fit receipt ' + name)
    print(json.dumps({k: receipt.get(k) for k in ('status', 'error', 'seconds', 'model_sha256')}))
    if receipt['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
