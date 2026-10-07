"""Seal the data files into the project's own encrypted package on the ledger branch.

Runs in GitHub Actions with the repository secret. Reads the data files from the inherited
release asset (and any files already in the current package), zips them, seals the zip with
AES-256-GCM (purpose data-package) and commits it to private/data-package.enc on the ledger
branch with a public manifest (hashes, sizes, file list) at brl_engine/data_package.json on
main. Plaintext never leaves the runner; the key is never printed.
"""
from __future__ import annotations
import base64, hashlib, io, json, os, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'brl_engine' / 'runtime'))
from brl_live.bootstrap import api, fetch, restore, TAG, ASSET
from cloud.security import seal, key_bytes, sha

DATA_DIRS = ('pa_model_reference', 'model_runs', 'data')
DATA_FILES = ('requirements-cloud.txt', 'RUNTIME_MANIFEST.json')
PURPOSE = 'data-package'


def put(repo, token, path, raw, branch, message):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    sha_old = None
    try:
        sha_old = api(url + '?ref=' + branch, token).get('sha')
    except HTTPError as exc:
        if exc.code != 404:
            raise
    payload = {'message': message, 'content': base64.b64encode(raw).decode(), 'branch': branch}
    if sha_old:
        payload['sha'] = sha_old
    return api(url, token, 'PUT', payload)


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key_hex = os.environ['BRL_PA_PACKAGE_KEY']
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    work = Path(os.environ['RUNNER_TEMP']) / 'brl-seal'
    config = json.loads(Path('brl_live/runtime.json').read_text())
    container = restore(fetch(f'https://github.com/{repo}/releases/download/{TAG}/{ASSET}'), key_hex, config, work / 'container')
    extra = Path(os.environ.get('BRL_EXTRA_DATA', '')) if os.environ.get('BRL_EXTRA_DATA') else None
    buffer = io.BytesIO(); listing = []
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for base in ([container] + ([extra] if extra and extra.exists() else [])):
            for p in sorted(base.rglob('*')):
                if not p.is_file():
                    continue
                rel = p.relative_to(base).as_posix()
                if not (rel.split('/')[0] in DATA_DIRS or rel in DATA_FILES):
                    continue
                if any(item['path'] == rel for item in listing):
                    listing = [item for item in listing if item['path'] != rel]
                raw = p.read_bytes()
                z.writestr(rel, raw)
                listing.append({'path': rel, 'size': len(raw), 'sha256': sha(raw)})
    plain = buffer.getvalue()
    cipher = seal(plain, key_bytes(key_hex), PURPOSE)
    version = datetime.now(timezone.utc).strftime('brl-data-%Y%m%d%H%M')
    put(repo, token, 'private/data-package.enc', cipher, branch, 'BRL: data package ' + version)
    manifest = {'schema': 'brl.data-package.v1', 'version': version, 'branch': branch, 'path': 'private/data-package.enc',
                'purpose': PURPOSE, 'format': 'AES-256-GCM; BRLAESG1 header; 12-byte nonce; AAD BRL:' + PURPOSE,
                'cipher_sha256': sha(cipher), 'plaintext_sha256': sha(plain), 'cipher_bytes': len(cipher), 'plaintext_bytes': len(plain),
                'files': listing, 'sealed_at': datetime.now(timezone.utc).isoformat()}
    put(repo, token, 'brl_engine/data_package.json', json.dumps(manifest, indent=1).encode(), 'main', 'BRL: data package manifest ' + version)
    print(json.dumps({'version': version, 'files': len(listing), 'plaintext_bytes': len(plain)}))


if __name__ == '__main__':
    main()
