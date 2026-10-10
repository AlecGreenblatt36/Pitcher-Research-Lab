"""Start the day's matchup-report build from the live slate chain when GitHub's cron has not, and again when the day's
games change under the plans.

The report workflow's schedule (13:20 and 20:35 UTC) fires late or not at all; the live slate chain runs every quarter
hour and can dispatch workflows with the repository token. Within an hour after each scheduled time, if no run of
brl-report.yml has started since that time, this dispatches one with daily=true (yesterday's, today's and tomorrow's
plans, player cards). Game-day refresh: when a forecast for today or tomorrow was saved after that date's plans were
built (lineups posted, a starter named), the plans are rebuilt the same way, at most once every 45 minutes. Idempotent;
printed, never failed: the live page must publish whatever happens here.
"""
from __future__ import annotations
import base64
import json
import os
from datetime import date, datetime, timedelta, timezone

from brl_chain_next import call

WORKFLOW = 'brl-report.yml'
WINDOWS = ((13, 20), (20, 35))      # UTC, the report workflow's own cron times
WINDOW_MINUTES = 70
REFRESH_GAP_MINUTES = 45


def newest_forecasts(D: dict) -> dict:
    """The latest forecast save time for today and tomorrow, from the page's own prediction file."""
    today = D.get('date')
    if not today:
        return {}
    days = (today, (date.fromisoformat(today) + timedelta(days=1)).isoformat())
    out = {}
    for f in (D.get('forecasts') or {}).values():
        d, saved = f.get('date'), f.get('saved_at')
        if d in days and saved:
            out[d] = max(out.get(d, ''), str(saved))
    return out


def stale_dates(newest: dict, built: dict) -> list:
    """Dates whose plans were built before their latest forecast (or not at all)."""
    return sorted(d for d, saved in newest.items() if not built.get(d) or str(built[d]) < saved)


def plans_built(repo: str, token: str, days) -> dict:
    out = {}
    for d in days:
        try:
            doc = call(f'https://api.github.com/repos/{repo}/contents/public/reports/{d}/index.json?ref=brl-live-data', token, label='read plan index ' + d)
            out[d] = json.loads(base64.b64decode(doc.get('content') or '')).get('built_at')
        except Exception as exc:
            print('no plan index for', d, type(exc).__name__, getattr(exc, 'code', ''))
            out[d] = None
    return out


def refresh(repo: str, token: str, now: datetime) -> None:
    site = os.path.join(os.environ.get('RUNNER_TEMP', ''), 'brl-site', 'predictions.json')
    try:
        with open(site) as fh:
            newest = newest_forecasts(json.load(fh))
    except Exception as exc:
        print('game-day refresh: no prediction file to read:', type(exc).__name__); return
    if not newest:
        print('game-day refresh: no forecasts for today or tomorrow'); return
    stale = stale_dates(newest, plans_built(repo, token, newest))
    if not stale:
        print('game-day refresh: plans are newer than every forecast'); return
    since = (now - timedelta(minutes=REFRESH_GAP_MINUTES)).isoformat().replace('+00:00', 'Z')
    base = f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/runs'
    try:
        runs = call(f'{base}?created=>={since}&per_page=20', token, label='list recent report runs').get('workflow_runs') or []
    except Exception as exc:
        print('game-day refresh: could not list report runs:', type(exc).__name__); return
    recent = [r['id'] for r in runs if r.get('event') in ('schedule', 'workflow_dispatch')]
    if recent:
        print('game-day refresh: plans for', stale, 'are older than their forecasts; a build started in the last', REFRESH_GAP_MINUTES, 'minutes:', recent); return
    try:
        call(f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches', token, 'POST', {'ref': 'main', 'inputs': {'daily': 'true'}}, label='dispatch report')
        print('game-day refresh: dispatched a plan build for', stale)
    except Exception as exc:
        print('::warning::the game-day plan build was not dispatched:', type(exc).__name__, getattr(exc, 'code', ''), str(exc)[:120])


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
        print('outside the daily report windows')
        refresh(repo, token, now)
        return
    base = f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/runs'
    try:
        runs = call(f'{base}?created=>={start.isoformat().replace("+00:00", "Z")}&per_page=20', token, label='list report runs').get('workflow_runs') or []
    except Exception as exc:
        print('could not list report runs:', type(exc).__name__); runs = []
    daily = [r['id'] for r in runs if r.get('event') in ('schedule', 'workflow_dispatch')]
    if daily:
        print('a daily report run already started in this window:', daily)
        refresh(repo, token, now)
        return
    try:
        call(f'https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches', token, 'POST', {'ref': 'main', 'inputs': {'daily': 'true'}}, label='dispatch report')
        print('dispatched the daily report build for the', start.strftime('%H:%M'), 'UTC window')
    except Exception as exc:
        print('::warning::the daily report build was not dispatched:', type(exc).__name__, getattr(exc, 'code', ''), str(exc)[:120])


if __name__ == '__main__':
    main()
