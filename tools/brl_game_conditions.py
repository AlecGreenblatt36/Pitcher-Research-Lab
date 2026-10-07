"""Game conditions for the 2021 to 2026 regular seasons: weather at first pitch, roof, day or night, home plate umpire.

Source: MLB's public schedule (statsapi.mlb.com/api/v1/schedule with hydrate=weather,officials,venue(fieldInfo)),
one request per month. Public game information only, no plate-appearance data. Output on the ledger branch:
research/conditions-<run>.jsonl.gz and a receipt. Used to study the run environment (totals) with prior-date fits.
"""
from __future__ import annotations

import base64
import calendar
import gzip
import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SCHEDULE = ('https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={end}&gameType=R'
            '&hydrate=weather,officials,venue(fieldInfo)')
WIND = re.compile(r'^\s*(\d+)\s*mph\s*,?\s*(.*)$', re.I)


def get(url, tries=5):
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'BRL-research/1.0', 'Accept': 'application/json'}), timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 + 4 * attempt)


def put(repo, token, path, raw, message, branch='brl-live-data'):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-conditions/1.0', 'Content-Type': 'application/json'}
    for attempt in range(6):
        sha = None
        try:
            with urlopen(Request(url + '?ref=' + branch, headers=headers), timeout=30) as r:
                sha = json.loads(r.read()).get('sha')
        except HTTPError:
            pass
        payload = {'message': message, 'content': base64.b64encode(raw).decode(), 'branch': branch}
        if sha:
            payload['sha'] = sha
        try:
            with urlopen(Request(url, headers=headers, data=json.dumps(payload).encode(), method='PUT'), timeout=120) as r:
                return r.read()
        except HTTPError as exc:
            if exc.code != 409 or attempt == 5:
                raise
            time.sleep(3 + 3 * attempt)


def parse_game(g: dict, day: str) -> dict:
    w = g.get('weather') or {}
    temp = w.get('temp')
    try:
        temp = int(str(temp).strip()) if temp not in (None, '') else None
    except ValueError:
        temp = None
    wind_mph, wind_dir = None, None
    m = WIND.match(str(w.get('wind') or ''))
    if m:
        wind_mph, wind_dir = int(m.group(1)), m.group(2).strip() or None
    hp = next((o for o in g.get('officials') or [] if str(o.get('officialType', '')).lower().startswith('home plate')), None)
    venue = g.get('venue') or {}
    field = venue.get('fieldInfo') or {}
    return {'game_pk': int(g['gamePk']), 'date': day, 'season': int(day[:4]), 'start': g.get('gameDate'), 'day_night': g.get('dayNight'),
            'double_header': g.get('doubleHeader'), 'scheduled_innings': g.get('scheduledInnings'),
            'venue_id': venue.get('id'), 'venue': venue.get('name'), 'roof': field.get('roofType'), 'turf': field.get('turfType'),
            'temp_f': temp, 'condition': w.get('condition'), 'wind_mph': wind_mph, 'wind_dir': wind_dir,
            'hp_umpire_id': ((hp or {}).get('official') or {}).get('id'), 'hp_umpire': ((hp or {}).get('official') or {}).get('fullName')}


def main():
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    first, last = int(os.environ.get('BRL_CONDITIONS_FIRST', '2021')), int(os.environ.get('BRL_CONDITIONS_LAST', '2026'))
    t0 = time.time()
    receipt = {'schema': 'brl.conditions.v1', 'source': 'statsapi.mlb.com schedule, hydrate=weather,officials,venue(fieldInfo)',
               'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat(), 'seasons': [first, last]}
    rows, seen = [], set()
    for season in range(first, last + 1):
        for month in range(3, 11):
            end = calendar.monthrange(season, month)[1]
            doc = get(SCHEDULE.format(start=f'{season}-{month:02d}-01', end=f'{season}-{month:02d}-{end:02d}'))
            for d in doc.get('dates') or []:
                for g in d.get('games') or []:
                    if ((g.get('status') or {}).get('abstractGameState')) != 'Final' or int(g['gamePk']) in seen:
                        continue
                    seen.add(int(g['gamePk']))
                    rows.append(parse_game(g, d['date']))
    receipt.update(rows=len(rows), by_season={str(s): sum(1 for r in rows if r['season'] == s) for s in range(first, last + 1)},
                   with_temp=sum(1 for r in rows if r['temp_f'] is not None), with_wind=sum(1 for r in rows if r['wind_dir']),
                   with_umpire=sum(1 for r in rows if r['hp_umpire_id']), with_roof=sum(1 for r in rows if r['roof']),
                   wind_directions=sorted({r['wind_dir'] for r in rows if r['wind_dir']}),
                   roof_types=sorted({r['roof'] for r in rows if r['roof']}), seconds=round(time.time() - t0, 1))
    raw = gzip.compress('\n'.join(json.dumps(r) for r in rows).encode() + b'\n', mtime=0)
    path = f'research/conditions-{run_id}.jsonl.gz'; receipt['file'] = path
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GH_TOKEN')
    if repo and token:
        put(repo, token, path, raw, 'BRL: game conditions')
        put(repo, token, f'research/conditions-{run_id}.json', json.dumps(receipt, indent=1).encode(), 'BRL: game conditions receipt')
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
