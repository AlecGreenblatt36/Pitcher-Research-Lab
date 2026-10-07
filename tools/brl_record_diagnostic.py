"""Save a text file (last 400 lines) to the ledger branch so a workflow's output can be read from outside.

Usage: python tools/brl_record_diagnostic.py <path on ledger branch> <local file>
Never used for private data: only test and job output that contains no player rows or secrets.
"""
from __future__ import annotations
import base64, json, os, sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def api(url, token, method='GET', payload=None):
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-diagnostic/1.0'}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode(); headers['Content-Type'] = 'application/json'
    with urlopen(Request(url, headers=headers, data=body, method=method), timeout=40) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def main():
    target, source = sys.argv[1], sys.argv[2]
    repo, token = os.environ['GITHUB_REPOSITORY'], os.environ['GH_TOKEN']
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    lines = Path(source).read_text(errors='replace').splitlines()[-400:] if Path(source).exists() else ['(no output file)']
    text = '\n'.join(lines) + '\n'
    for secret in (os.environ.get('BRL_PA_PACKAGE_KEY'), token):
        if secret:
            text = text.replace(secret, '[redacted]')
    url = f'https://api.github.com/repos/{repo}/contents/{target}'
    payload = {'message': 'BRL: diagnostic ' + target, 'content': base64.b64encode(text.encode()).decode(), 'branch': branch}
    try:
        payload['sha'] = api(url + '?ref=' + branch, token)['sha']
    except HTTPError as exc:
        if exc.code != 404:
            raise
    api(url, token, 'PUT', payload)
    print('recorded', target, len(lines), 'lines')


if __name__ == '__main__':
    main()
