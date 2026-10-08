"""Base running inputs from public sources, per player and season: Statcast sprint speed (Baseball Savant's
leaderboard) and stolen bases, caught stealing and times on first base (MLB's season stats), plus stolen bases and
caught stealing allowed by catchers and by pitchers.

Output on the ledger branch: research/running-<run>.json.gz and a receipt. Public season statistics only.
"""
from __future__ import annotations

import base64
import csv
import gzip
import io
import json
import os
import random
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
SAVANT = 'https://baseballsavant.mlb.com/leaderboard/sprint_speed?min_season={y}&max_season={y}&position=&team=&min=1&csv=true'
STATS = 'https://statsapi.mlb.com/api/v1/stats?stats=season&group={g}&season={y}&sportId=1&playerPool=ALL&limit=3000&offset={o}'


def get(url, raw=False, tries=4):
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers={'User-Agent': UA, 'Accept': '*/*'}), timeout=60) as r:
                body = r.read()
            return body if raw else json.loads(body)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 + 4 * attempt + random.random())


def put(repo, token, path, raw, message, branch='brl-live-data'):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-running/1.0', 'Content-Type': 'application/json'}
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


def sprint(year):
    text = get(SAVANT.format(y=year), raw=True).decode('utf-8-sig', 'replace')
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        pid = row.get('player_id') or row.get('﻿player_id')
        try:
            out[str(int(pid))] = {'sprint': round(float(row['sprint_speed']), 2), 'runs': int(float(row.get('competitive_runs') or 0))}
        except (TypeError, ValueError, KeyError):
            continue
    return out


def season_stats(group, year):
    rows, offset = [], 0
    while True:
        doc = get(STATS.format(g=group, y=year, o=offset))
        splits = ((doc.get('stats') or [{}])[0].get('splits')) or []
        rows += splits
        if len(splits) < 3000:
            return rows
        offset += 3000


def main():
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    seasons = [int(s) for s in (os.environ.get('BRL_RUNNING_SEASONS') or '2023,2024,2025,2026').split(',')]
    receipt = {'schema': 'brl.running-data.v1', 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat(), 'seasons': {}}
    out = {'schema': 'brl.running.v1', 'seasons': {}}
    for y in seasons:
        rec = {}
        season = {'runners': {}, 'catchers': {}, 'pitchers': {}}
        try:
            sp = sprint(y); rec['sprint_players'] = len(sp)
        except Exception as exc:
            sp = {}; rec['sprint_error'] = type(exc).__name__ + ': ' + str(exc)[:160]
        for s in season_stats('hitting', y):
            st, pid = s.get('stat') or {}, str((s.get('player') or {}).get('id'))
            h, d, t, hr = int(st.get('hits') or 0), int(st.get('doubles') or 0), int(st.get('triples') or 0), int(st.get('homeRuns') or 0)
            on1 = (h - d - t - hr) + int(st.get('baseOnBalls') or 0) + int(st.get('hitByPitch') or 0) + int(st.get('intentionalWalks') or 0) * 0
            row = {'sb': int(st.get('stolenBases') or 0), 'cs': int(st.get('caughtStealing') or 0), 'on1': on1, 'pa': int(st.get('plateAppearances') or 0)}
            if pid in sp:
                row.update(sp[pid])
            prev = season['runners'].get(pid)
            if prev:   # traded players appear once per team; add them up
                for k in ('sb', 'cs', 'on1', 'pa'):
                    prev[k] += row[k]
            else:
                season['runners'][pid] = row
        for pid, v in sp.items():
            season['runners'].setdefault(pid, {'sb': 0, 'cs': 0, 'on1': 0, 'pa': 0, **v})
        for s in season_stats('fielding', y):
            st, pid = s.get('stat') or {}, str((s.get('player') or {}).get('id'))
            pos = ((s.get('position') or {}).get('abbreviation')) or ''
            if pos != 'C':
                continue
            row = season['catchers'].setdefault(pid, {'sb': 0, 'cs': 0, 'innings': 0.0})
            row['sb'] += int(st.get('stolenBases') or 0); row['cs'] += int(st.get('caughtStealing') or 0)
            try:
                row['innings'] += float(st.get('innings') or 0)
            except ValueError:
                pass
        for s in season_stats('pitching', y):
            st, pid = s.get('stat') or {}, str((s.get('player') or {}).get('id'))
            row = season['pitchers'].setdefault(pid, {'sb': 0, 'cs': 0, 'bf': 0})
            row['sb'] += int(st.get('stolenBases') or 0); row['cs'] += int(st.get('caughtStealing') or 0); row['bf'] += int(st.get('battersFaced') or 0)
        rec.update(runners=len(season['runners']), catchers=len(season['catchers']), pitchers=len(season['pitchers']),
                   league_sb=sum(r['sb'] for r in season['runners'].values()), league_cs=sum(r['cs'] for r in season['runners'].values()),
                   league_on1=sum(r['on1'] for r in season['runners'].values()))
        out['seasons'][str(y)] = season
        receipt['seasons'][str(y)] = rec
        time.sleep(1.0)
    raw = gzip.compress(json.dumps(out).encode(), mtime=0)
    path = f'research/running-{run_id}.json.gz'
    receipt['file'] = path; receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GH_TOKEN')
    if repo and token:
        put(repo, token, path, raw, 'BRL: base running data')
        put(repo, token, f'research/running-{run_id}.json', json.dumps(receipt, indent=1).encode(), 'BRL: base running receipt')
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
