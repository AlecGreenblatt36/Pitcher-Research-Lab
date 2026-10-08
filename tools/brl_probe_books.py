"""Which sportsbooks does ESPN's odds feed carry for MLB games, today and on past dates? Counts and sample prices only.

For each date: the scoreboard's events, then the core odds endpoint for up to four events: providers (id, name),
whether each has a moneyline, open and close blocks, and a total. Writes diagnostics/books_probe.json.
"""
from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date}'
CORE = 'https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{id}/competitions/{id}/odds'


def get(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0 (BRL probe)', 'Accept': 'application/json'}), timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 + 2 * attempt)


def side(item, s):
    o = item.get(s + 'TeamOdds') or {}
    cur = (o.get('current') or {}).get('moneyLine') or {}
    return {'moneyLine': o.get('moneyLine'), 'current': cur.get('american') if isinstance(cur, dict) else cur,
            'open': ((o.get('open') or {}).get('moneyLine') or {}).get('american') if isinstance((o.get('open') or {}).get('moneyLine'), dict) else None,
            'close': ((o.get('close') or {}).get('moneyLine') or {}).get('american') if isinstance((o.get('close') or {}).get('moneyLine'), dict) else None}


def main():
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    dates = [today] + (os.environ.get('BRL_PROBE_DATES') or '20261006,20260704,20260501').split(',')
    out = {'schema': 'brl.books-probe.v1', 'at': datetime.now(timezone.utc).isoformat(), 'dates': {}}
    for d in dates:
        try:
            events = (get(SCOREBOARD.format(date=d.strip())).get('events') or [])
            games = []
            for e in events[:4]:
                try:
                    doc = get(CORE.format(id=e['id']))
                    items = doc.get('items') or []
                    games.append({'event': e.get('id'), 'name': e.get('shortName'), 'count': doc.get('count'),
                                  'books': [{'id': (it.get('provider') or {}).get('id'), 'name': (it.get('provider') or {}).get('name'),
                                             'priority': (it.get('provider') or {}).get('priority'), 'home': side(it, 'home'), 'away': side(it, 'away'),
                                             'overUnder': it.get('overUnder'), 'keys': sorted(it.keys())[:30]} for it in items]})
                except Exception as exc:
                    games.append({'event': e.get('id'), 'error': type(exc).__name__ + ': ' + str(exc)[:120]})
            out['dates'][d] = {'events': len(events), 'games': games}
        except Exception as exc:
            out['dates'][d] = {'error': type(exc).__name__ + ': ' + str(exc)[:150]}
    text = json.dumps(out, indent=1)
    print(text[:20000])
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GH_TOKEN')
    if repo and token:
        url = f'https://api.github.com/repos/{repo}/contents/diagnostics/books_probe.json'
        headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-probe/1.0', 'Content-Type': 'application/json'}
        for attempt in range(5):
            sha = None
            try:
                with urlopen(Request(url + '?ref=brl-live-data', headers=headers), timeout=30) as r:
                    sha = json.loads(r.read()).get('sha')
            except HTTPError:
                pass
            payload = {'message': 'BRL: books probe', 'content': base64.b64encode(text.encode()).decode(), 'branch': 'brl-live-data'}
            if sha:
                payload['sha'] = sha
            try:
                with urlopen(Request(url, headers=headers, data=json.dumps(payload).encode(), method='PUT'), timeout=60):
                    break
            except HTTPError as exc:
                if exc.code != 409 or attempt == 4:
                    raise
                time.sleep(3)


if __name__ == '__main__':
    main()
