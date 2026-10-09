"""Start the day's matchup-report build from the live slate chain when GitHub's cron has not.

The report workflow's schedule (13:20 and 20:35 UTC) has not fired since it was added; the live slate chain runs every
quarter hour and can dispatch workflows with the repository token. Within an hour after each scheduled time, if no run of
brl-report.yml has started since that time, this dispatches one with daily=true (yesterday's and today's reports, player
cards). Idempotent: a run already started in the window, by cron or by this, means nothing is dispatched. Printed, never
failed: the live page must publish whatever happens here.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone

from brl_chain_next import call

WORKFLOW = 'brl-report.yml'
WINDOWS = ((13, 20), (20, 35))      # UTC, the report workflow's own cron times
WINDOW_MINUTES = 70


def window_start(now: datetime) -> datetime | None:
    for h, m in WINDOWS:
        start = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if start <= now < start + timedelta(minutes=WINDOW_MINUTES):
            return start
    return None


def main():
    repo = os.environ['GITHUB_REPOSITORY']
    token = os.environ.get('BRL_DISPATCH_TOKEN') or os.environ['GH_TOKEN']
    now = datetime.now(timezone.utc)
    start = window_start(now)
    if start is None:
        print('outside the daily report windows; nothing to do'); return
    base = f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/runs'
    try:
        runs = call(f'{base}?created=>={start.isoformat().replace("+00:00", "Z")}&per_page=20', token, label='list report runs').get('workflow_runs') or []
    except Exception as exc:
        print('could not list report runs:', type(exc).__name__); runs = []
    daily = [r['id'] for r in runs if r.get('event') in ('schedule', 'workflow_dispatch')]
    if daily:
        print('a daily report run already started in this window:', daily); return
    try:
        call(f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches', token, 'POST', {'ref': 'main', 'inputs': {'daily': 'true'}}, label='dispatch report')
        print('dispatched the daily report build for the', start.strftime('%H:%M'), 'UTC window')
    except Exception as exc:
        print('::warning::the daily report build was not dispatched:', type(exc).__name__, getattr(exc, 'code', ''), str(exc)[:120])


if __name__ == '__main__':
    main()
