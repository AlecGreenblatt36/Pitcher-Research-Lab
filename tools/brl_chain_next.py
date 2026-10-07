"""Start the next live run after this one, paced by a wait, unless one is already queued or waiting.

GitHub's cron starts only a small share of this repository's scheduled runs, so each run
dispatches its successor. The repository's own GITHUB_TOKEN can do this: workflow_dispatch is
one of the two events a GITHUB_TOKEN-made request is allowed to start. A fine-grained token in
BRL_DISPATCH_TOKEN is used instead when present. Idempotent: when another run of the workflow
is already queued, waiting or in progress (other than this one), nothing is dispatched, so an
extra push never doubles the chain.
"""
from __future__ import annotations
import json, os, sys, time
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


MAX_WAIT = 30 * 60   # the chain outlasts a rate-limit window or a short outage, never a broken token


def pause_for(exc, attempt, waited):
    """Seconds before retrying, or None when the refusal is final (a permission or validation answer)."""
    if isinstance(exc, HTTPError):
        headers = exc.headers or {}
        if exc.code in (403, 429):
            after = str(headers.get('Retry-After') or '').strip(); reset = str(headers.get('X-RateLimit-Reset') or '').strip()
            if after.isdigit():
                pause = float(after) + 2
            elif headers.get('X-RateLimit-Remaining') == '0' and reset.isdigit():
                pause = max(5.0, float(reset) - time.time() + 3)
            elif exc.code == 429:
                pause = 60.0 * (attempt + 1)
            else:
                return None
        elif exc.code >= 500:
            pause = 10.0 + 20.0 * attempt
        else:
            return None
    elif isinstance(exc, OSError):
        pause = 5.0 + 10.0 * attempt
    else:
        return None
    return None if waited + pause > MAX_WAIT else pause


def call(url, token, method='GET', payload=None, label=''):
    waited = 0.0
    for attempt in range(12):
        try:
            return api(url, token, method, payload)
        except (HTTPError, OSError) as exc:
            code = getattr(exc, 'code', None)
            pause = pause_for(exc, attempt, waited)
            print(label, 'failed:', type(exc).__name__, code or str(exc)[:80], '' if pause is None else f'(retry in {pause:.0f}s)')
            if pause is None:
                raise
            time.sleep(pause); waited += pause
    raise RuntimeError(label + ' unavailable')


def main():
    repo = os.environ['GITHUB_REPOSITORY']
    token = os.environ.get('BRL_DISPATCH_TOKEN') or os.environ['GH_TOKEN']
    wait = os.environ.get('BRL_CHAIN_WAIT_MINUTES', '12')
    me = os.environ.get('GITHUB_RUN_ID', '')
    base = f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/runs'
    others = []
    for status in ('queued', 'waiting', 'in_progress', 'requested', 'pending'):
        try:
            runs = call(f'{base}?status={status}&per_page=20', token, label='list ' + status).get('workflow_runs') or []
        except Exception as exc:
            print('could not list', status, type(exc).__name__); runs = []
        others += [r['id'] for r in runs if str(r['id']) != str(me)]
    if others:
        print('another run is already pending or active; not chaining:', sorted(set(others))); return
    try:
        call(f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches', token, 'POST',
             {'ref': 'main', 'inputs': {'wait_minutes': str(wait)}}, label='dispatch')
        print('chained the next run with a', wait, 'minute wait')
    except Exception as exc:
        # Printed, not failed: this step runs after the page artifact is made, and a failed job would
        # keep the page from publishing. The workflow's own cron entries restart a broken chain.
        print('::warning::the next run was not chained:', type(exc).__name__, getattr(exc, 'code', ''), str(exc)[:120])


if __name__ == '__main__':
    main()
