"""Start the next live run after this one, paced by a wait, unless one is already queued or waiting.

GitHub's cron starts only a small share of this repository's scheduled runs, so each run
dispatches its successor. The repository's own GITHUB_TOKEN can do this: workflow_dispatch is
one of the two events a GITHUB_TOKEN-made request is allowed to start. A fine-grained token in
BRL_DISPATCH_TOKEN is used instead when present. Idempotent: when another run of the workflow
is already queued, waiting or in progress (other than this one), nothing is dispatched, so an
extra push never doubles the chain.
"""
from __future__ import annotations
import json, os, sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

WORKFLOW = 'brl-live.yml'


def api(url, token, method='GET', payload=None):
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-chain/1.0'}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode(); headers['Content-Type'] = 'application/json'
    with urlopen(Request(url, headers=headers, data=body, method=method), timeout=40) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def main():
    repo = os.environ['GITHUB_REPOSITORY']
    token = os.environ.get('BRL_DISPATCH_TOKEN') or os.environ['GH_TOKEN']
    wait = os.environ.get('BRL_CHAIN_WAIT_MINUTES', '12')
    me = os.environ.get('GITHUB_RUN_ID', '')
    base = f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/runs'
    others = []
    for status in ('queued', 'waiting', 'in_progress', 'requested', 'pending'):
        try:
            runs = api(f'{base}?status={status}&per_page=20', token).get('workflow_runs') or []
        except HTTPError as exc:
            print('could not list', status, exc.code); runs = []
        others += [r['id'] for r in runs if str(r['id']) != str(me)]
    if others:
        print('another run is already pending or active; not chaining:', sorted(set(others))); return
    import time
    for attempt in range(4):
        try:
            api(f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches', token, 'POST', {'ref': 'main', 'inputs': {'wait_minutes': str(wait)}})
            print('chained the next run with a', wait, 'minute wait'); return
        except HTTPError as exc:
            body = exc.read()[:200]
            print('dispatch refused:', exc.code, body)
            if exc.code in (401, 403, 404, 422):
                return
        except Exception as exc:       # network hiccup: try again, the chain must not break on one failed call
            print('dispatch error:', type(exc).__name__, str(exc)[:120])
        time.sleep(15 * (attempt + 1))


if __name__ == '__main__':
    main()
