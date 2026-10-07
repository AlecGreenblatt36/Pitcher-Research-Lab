"""Own-runtime launcher: public engine and pipeline code, encrypted data only.

The inherited release asset is used purely as the sealed container of the data files (seed
history, locked model, starter hazard, names, team results). Its private code is never
imported. Everything else runs from this repository.
"""
from __future__ import annotations
import argparse, base64, json, os, shutil, subprocess, sys, traceback
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
            page = Path(args.site) / 'index.html'
            page.write_text(page.read_text().replace('<body>', '<body data-refresh-blocked="true">', 1))
            preserved = True
        except Exception:
            setup_page(args.site, reason, games)
        receipt['preserved_previous_forecasts'] = preserved
        print('::warning::BRL v2 worker blocked; see receipt.')
    public = Path(args.site)
    if {p.name for p in public.iterdir() if p.is_file()} != {'index.html', 'predictions.json', '.nojekyll'} or any(p.is_dir() for p in public.iterdir()):
        raise RuntimeError('Public output file allowlist failed')
    Path('brl_v2_run_receipt.json').write_text(json.dumps(receipt, indent=2))
    # The receipt (no secrets, no data) also goes to the ledger branch so it can be read through git.
    try:
        branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
        url = f'https://api.github.com/repos/{repo}/contents/diagnostics/v2_receipt.json'
        sha = None
        try:
            sha = api(url + '?ref=' + branch, token).get('sha')
        except Exception:
            sha = None
        payload = {'message': 'BRL v2: run receipt', 'content': base64.b64encode(json.dumps(receipt, indent=1).encode()).decode(), 'branch': branch}
        if sha:
            payload['sha'] = sha
        api(url, token, 'PUT', payload)
    except Exception:
        pass
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            stream.write('completed=' + str(receipt.get('status') == 'iteration_completed').lower() + '\n')
            stream.write('pages_ready=' + ('false' if args.shadow else 'true') + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'scheduled_games'}, indent=2))
    return receipt


if __name__ == '__main__':
    main()
