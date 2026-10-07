"""Own-runtime launcher: public engine and pipeline code, encrypted data only.

The inherited release asset is used purely as the sealed container of the data files (seed
history, locked model, starter hazard, names, team results). Its private code is never
imported. Everything else runs from this repository.
"""
from __future__ import annotations
import argparse, base64, json, os, re, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from brl_live.bootstrap import api, fetch, restore, setup_page, TAG, ASSET  # noqa: E402

DATA_DIRS = ('pa_model_reference', 'model_runs', 'data')
DATA_FILES = ('requirements-cloud.txt', 'RUNTIME_MANIFEST.json')


def restore_package(repo, token, key_hex, manifest, destination):
    """Fetch private/data-package.enc from the ledger branch, verify, unseal and unzip into destination."""
    import hashlib, io, zipfile
    sys.path.insert(0, str(REPO_ROOT / 'brl_engine' / 'runtime'))
    from cloud.security import unseal, key_bytes
    branch, path = manifest['branch'], manifest['path']
    meta = api(f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}', token)
    if meta.get('encoding') == 'base64' and isinstance(meta.get('content'), str) and meta.get('content'):
        cipher = base64.b64decode(''.join(meta['content'].split()))
    else:
        blob = api(f'https://api.github.com/repos/{repo}/git/blobs/{meta["sha"]}', token)
        cipher = base64.b64decode(''.join(blob['content'].split()))
    if hashlib.sha256(cipher).hexdigest() != manifest['cipher_sha256']:
        raise ValueError('Data package hash mismatch')
    plain = unseal(cipher, key_bytes(key_hex), manifest['purpose'])
    if hashlib.sha256(plain).hexdigest() != manifest['plaintext_sha256']:
        raise ValueError('Data package plaintext hash mismatch')
    destination = Path(destination).resolve()
    with zipfile.ZipFile(io.BytesIO(plain)) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination) or '\\' in item.filename:
                raise ValueError('Unsafe package path')
        z.extractall(destination)
    return destination


def select_model(repo, token, key_hex, data_root):
    """brl_engine/model.json names the PA model: the locked 2026 model inside the data package (default)
    or a sealed model from the fit job (private/models/<name>.enc, public manifest brl_engine/models/<name>.json),
    which is verified, unsealed into the data root and announced to the runtime through the environment."""
    selection_path = REPO_ROOT / 'brl_engine' / 'model.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    name = str(selection.get('model') or 'locked-pa-2026-v1')
    for var in ('BRL_MODEL_PATH', 'BRL_MODEL_SHA256', 'BRL_MODEL_NAME'):
        os.environ.pop(var, None)
    if name == 'locked-pa-2026-v1':
        return {'name': name, 'source': 'data package'}
    import hashlib
    sys.path.insert(0, str(REPO_ROOT / 'brl_engine' / 'runtime'))
    from cloud.security import unseal, key_bytes
    manifest = json.loads((REPO_ROOT / selection['manifest']).read_text())
    if manifest.get('name') != name:
        raise ValueError('Model manifest name mismatch')
    branch, path = manifest['branch'], manifest['path']
    meta = api(f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}', token)
    if meta.get('encoding') == 'base64' and isinstance(meta.get('content'), str) and meta.get('content'):
        cipher = base64.b64decode(''.join(meta['content'].split()))
    else:
        blob = api(f'https://api.github.com/repos/{repo}/git/blobs/{meta["sha"]}', token)
        cipher = base64.b64decode(''.join(blob['content'].split()))
    if hashlib.sha256(cipher).hexdigest() != manifest['cipher_sha256']:
        raise ValueError('Model cipher hash mismatch')
    raw = unseal(cipher, key_bytes(key_hex), manifest['purpose'])
    if hashlib.sha256(raw).hexdigest() != manifest['model_sha256']:
        raise ValueError('Model plaintext hash mismatch')
    target = Path(data_root) / 'pa_model_reference' / 'model_runs' / name / 'artifacts' / 'pa_model.joblib'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    os.environ['BRL_MODEL_PATH'] = str(target)
    os.environ['BRL_MODEL_SHA256'] = manifest['model_sha256']
    os.environ['BRL_MODEL_NAME'] = name
    return {'name': name, 'source': path, 'sha256': manifest['model_sha256'], 'fitted_at': manifest.get('fitted_at'), 'physics_features': len(manifest.get('physics_features') or [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', required=True); parser.add_argument('--runtime', required=True)
    parser.add_argument('--config', default='brl_live/runtime.json')
    parser.add_argument('--shadow', action='store_true', help='write to the shadow ledger branch and never publish the page')
    args = parser.parse_args()
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; now = datetime.now(timezone.utc)
    if args.shadow:
        os.environ['BRL_LEDGER_BRANCH'] = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data-v2')
    receipt = {'created_at': now.isoformat(), 'status': 'starting', 'runtime': 'brl_engine (public code, sealed data)',
               'ledger_branch': os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data'), 'shadow': bool(args.shadow),
               'live_forecasts_created': 0, 'raw_data_published': False, 'stage': 'source_setup', 'secret_present': False}
    games = []
    try:
        day = now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
        schedule = api('https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=' + day, None)
        games = [{'game_pk': g['gamePk'], 'scheduled_start': g['gameDate'], 'away': g['teams']['away']['team']['name'], 'home': g['teams']['home']['team']['name']}
                 for d in schedule.get('dates', []) for g in d.get('games', [])]
        receipt['scheduled_games'] = games
        receipt['stage'] = 'data_restore'
        key = os.environ.get('BRL_PA_PACKAGE_KEY', '')
        receipt['secret_present'] = bool(key)
        if not key:
            raise ValueError('Activation needed: BRL_PA_PACKAGE_KEY secret missing.')
        data_root = Path(args.runtime) / 'data'
        if data_root.exists():
            shutil.rmtree(data_root)
        data_root.mkdir(parents=True)
        manifest_path = REPO_ROOT / 'brl_engine' / 'data_package.json'
        receipt['data_source'] = None
        if manifest_path.exists():
            # The project's own sealed package on the ledger branch (made by the data-package workflow).
            try:
                manifest = json.loads(manifest_path.read_text())
                restore_package(repo, token, key, manifest, data_root)
                receipt['data_source'] = manifest['version']
            except Exception as exc:
                receipt['data_package_error'] = type(exc).__name__ + ': ' + str(exc)[:160]
        if receipt['data_source'] is None:
            config = json.loads(Path(args.config).read_text())
            container = Path(args.runtime) / 'container'
            restore(fetch(f'https://github.com/{repo}/releases/download/{TAG}/{ASSET}'), key, config, container)
            for name in DATA_DIRS:
                if (container / name).exists():
                    shutil.copytree(container / name, data_root / name)
            for name in DATA_FILES:
                if (container / name).exists():
                    shutil.copy2(container / name, data_root / name)
            shutil.rmtree(container)   # the inherited code is never imported
            receipt['data_source'] = 'inherited release asset ' + TAG
        os.environ['BRL_DATA_ROOT'] = str(data_root)
        os.environ['BRL_WORK_ROOT'] = str(Path(args.runtime) / 'work')
        receipt['pa_model_selected'] = select_model(repo, token, key, data_root)
        receipt['stage'] = 'dependencies'
        env = {k: v for k, v in os.environ.items() if k not in ('BRL_PA_PACKAGE_KEY', 'GH_TOKEN', 'GITHUB_TOKEN')}
        req = data_root / 'requirements-cloud.txt'
        if req.exists():
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--quiet', '-r', str(req)], env=env, check=True)
        sys.path.insert(0, str(REPO_ROOT / 'brl_engine' / 'runtime'))
        from brl_live.box_runner import main as run_iteration
        receipt['stage'] = 'history_and_forecast_iteration'
        receipt.update(run_iteration(args.site)); receipt['status'] = 'iteration_completed'; receipt['stage'] = 'complete'
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__ + ': ' + str(exc)[:200]
        receipt.update(status='blocked', reason=reason, error_type=type(exc).__name__)
        frames = traceback.extract_tb(exc.__traceback__)
        receipt['error_trace'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in frames[-6:]]
        preserved = False
        try:
            branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
            prior = {}
            for name in ('index.html', 'predictions.json', '.nojekyll'):
                value = api(f'https://api.github.com/repos/{repo}/contents/public/{name}?ref={branch}', token)
                if value.get('encoding') != 'base64':
                    raise ValueError('Prior public page unavailable')
                prior[name] = base64.b64decode(value['content'], validate=False)
            Path(args.site).mkdir(parents=True, exist_ok=True)
            for name, raw in prior.items():
                (Path(args.site) / name).write_bytes(raw)
            try:
                listing = api(f'https://api.github.com/repos/{repo}/contents/public/days?ref={branch}', token)
                (Path(args.site) / 'days').mkdir(exist_ok=True)
                for item in listing if isinstance(listing, list) else []:
                    if item.get('type') == 'file' and item['name'].endswith('.json'):
                        value = api(item['url'], token)
                        if value.get('encoding') == 'base64':
                            (Path(args.site) / 'days' / item['name']).write_bytes(base64.b64decode(value['content'], validate=False))
            except Exception:
                pass
            # The static season files come from the repository, not the ledger branch.
            (Path(args.site) / 'days').mkdir(exist_ok=True)
            for src in sorted((REPO_ROOT / 'brl_live' / 'archive').glob('*.json')):
                if re.fullmatch(r'(season-\d{4}|win-expectancy)\.json', src.name):
                    (Path(args.site) / 'days' / src.name).write_bytes(src.read_bytes())
            page = Path(args.site) / 'index.html'
            page.write_text(page.read_text().replace('<body>', '<body data-refresh-blocked="true">', 1))
            preserved = True
        except Exception:
            setup_page(args.site, reason, games)
        receipt['preserved_previous_forecasts'] = preserved
        print('::warning::BRL v2 worker blocked; see receipt.')
    public = Path(args.site)
    files = {p.name for p in public.iterdir() if p.is_file()}
    dirs = {p.name for p in public.iterdir() if p.is_dir()}
    if files != {'index.html', 'predictions.json', '.nojekyll'} or not dirs <= {'days'}:
        raise RuntimeError('Public output file allowlist failed')
    if 'days' in dirs:
        for p in (public / 'days').iterdir():
            if not p.is_file() or not re.fullmatch(r'(\d{4}-\d{2}-\d{2}|index|season-\d{4}|win-expectancy)\.json', p.name):
                raise RuntimeError('Public day archive allowlist failed')
    Path('brl_v2_run_receipt.json').write_text(json.dumps(receipt, indent=2))
    # The receipt (no secrets, no data) also goes to the ledger branch so it can be read through git.
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    url = f'https://api.github.com/repos/{repo}/contents/diagnostics/v2_receipt.json'
    for attempt in range(6):
        # Other jobs write the ledger branch too; a 409 means it moved between the sha read and the write.
        try:
            sha = None
            try:
                sha = api(url + '?ref=' + branch, token).get('sha')
            except Exception:
                sha = None
            payload = {'message': 'BRL v2: run receipt', 'content': base64.b64encode(json.dumps(receipt, indent=1).encode()).decode(), 'branch': branch}
            if sha:
                payload['sha'] = sha
            api(url, token, 'PUT', payload)
            break
        except Exception as exc:
            print('::warning::receipt not saved to the ledger branch (attempt ' + str(attempt + 1) + '): ' + type(exc).__name__)
            time.sleep(3 + 4 * attempt)
    if os.environ.get('GITHUB_OUTPUT'):
        # A blocked run republishes the previous page; when even that could not be restored, the
        # last deployed page stays up rather than being replaced by the setup notice.
        pages_ready = (not args.shadow) and (receipt.get('status') == 'iteration_completed' or bool(receipt.get('preserved_previous_forecasts')))
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            stream.write('completed=' + str(receipt.get('status') == 'iteration_completed').lower() + '\n')
            stream.write('pages_ready=' + str(pages_ready).lower() + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'scheduled_games'}, indent=2))
    return receipt


if __name__ == '__main__':
    main()
