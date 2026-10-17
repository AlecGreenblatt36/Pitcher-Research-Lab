"""Daily active rosters, pitchers only (ROSTER-01): which arms each team actually had on the day of each of its games.

For one season (BRL_SEASON) the official schedule gives every completed regular-season and postseason game; for each
team and game date the active roster as MLB's public API shows it for that date is read and its pitchers kept
(position type Pitcher or Two-Way Player). Output on the ledger branch, public data only (team, date, player ids):
research/rosters-<year>.json.gz = {'season', 'teams': {team id: abbreviation}, 'days': {date: {abbreviation: [ids]}},
'games', 'requests', 'missing': [[date, abbreviation], ...]} and a receipt research/rosters-<year>.json with the counts.

The bullpen pools of the replay and of the reliever-choice fit (brl_replay.harness.bullpen, tools/brl_reliever_choice.py)
read these when given, instead of guessing the roster from the last 14 days of appearances (BULLPEN-10).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

POST_TYPES = ('F', 'D', 'L', 'W')
# MLB's abbreviations where they differ from the Statcast codes the history carries
TO_STATCAST = {'CWS': 'CWS', 'KC': 'KC', 'SD': 'SD', 'SF': 'SF', 'TB': 'TB', 'WSH': 'WSH', 'AZ': 'AZ', 'ATH': 'ATH'}


def rosters_path(year: int) -> str:
    return f'research/rosters-{int(year)}.json.gz'


def receipt_path(year: int) -> str:
    return f'research/rosters-{int(year)}.json'


def team_dates(get_json, year: int, through: str | None = None) -> tuple[dict, dict]:
    """({team id: abbreviation}, {(date, team id): game type}) for the season's completed games."""
    end = through or f'{year}-11-30'
    doc = get_json(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={year}-03-15&endDate={end}'
                   '&fields=dates,date,games,gamePk,gameType,officialDate,status,abstractGameState,teams,away,home,team,id,abbreviation')
    teams, days = {}, {}
    for d in (doc or {}).get('dates') or []:
        for g in d.get('games') or []:
            if g.get('gameType') not in ('R',) + POST_TYPES or (g.get('status') or {}).get('abstractGameState') != 'Final':
                continue
            date = str(g.get('officialDate') or d['date'])[:10]
            for side in ('away', 'home'):
                t = ((g.get('teams') or {}).get(side) or {}).get('team') or {}
                if t.get('id'):
                    teams[int(t['id'])] = str(t.get('abbreviation') or t['id'])
                    days[(date, int(t['id']))] = g['gameType']
    return teams, days


def pitchers_on(get_json, team_id: int, date: str) -> list[int] | None:
    doc = get_json(f'https://statsapi.mlb.com/api/v1/teams/{int(team_id)}/roster?rosterType=active&date={date}'
                   '&fields=roster,person,id,position,type')
    if not doc or 'roster' not in doc:
        return None
    out = []
    for e in doc.get('roster') or []:
        pid = int(((e.get('person') or {}).get('id')) or 0)
        if pid and (e.get('position') or {}).get('type') in ('Pitcher', 'Two-Way Player'):
            out.append(pid)
    return sorted(set(out))


def build(get_json, year: int, workers: int = 4, through: str | None = None, log=print) -> dict:
    teams, days = team_dates(get_json, year, through)
    keys = sorted(days)
    out = {'season': int(year), 'teams': {str(k): v for k, v in teams.items()}, 'days': {}, 'games': len(keys), 'requests': 0, 'missing': []}
    t0 = time.time()

    def one(key):
        date, tid = key
        for attempt in range(3):
            try:
                return key, pitchers_on(get_json, tid, date)
            except Exception:
                time.sleep(2 + 3 * attempt)
        return key, None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (key, pit) in enumerate(pool.map(one, keys)):
            out['requests'] += 1
            date, tid = key
            if pit is None:
                out['missing'].append([date, teams[tid]]); continue
            out['days'].setdefault(date, {})[teams[tid]] = pit
            if i % 500 == 0:
                log(json.dumps({'done': i, 'of': len(keys), 'seconds': round(time.time() - t0)}))
    out['seconds'] = round(time.time() - t0)
    return out


def summary(doc: dict) -> dict:
    sizes = [len(v) for day in doc['days'].values() for v in day.values()]
    return {'schema': 'brl.rosters-receipt.v1', 'experiment': 'ROSTER-01', 'season': doc['season'], 'teams': len(doc['teams']), 'team_dates': doc['games'],
            'read': len(sizes), 'missing': len(doc['missing']), 'pitchers_per_roster_mean': round(sum(sizes) / max(len(sizes), 1), 2),
            'pitchers_per_roster_min': min(sizes, default=0), 'pitchers_per_roster_max': max(sizes, default=0),
            'dates': [min(doc['days']), max(doc['days'])] if doc['days'] else None, 'requests': doc['requests'], 'seconds': doc.get('seconds'),
            'path': rosters_path(doc['season']), 'public_data_only': True, 'built_at': datetime.now(timezone.utc).isoformat(), 'run_id': os.environ.get('GITHUB_RUN_ID')}


def main():
    from brl_bookkeeping_backfill import get_json, put
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    year = int(os.environ['BRL_SEASON'])
    doc = build(get_json, year, workers=int(os.environ.get('BRL_WORKERS', '4') or 4))
    raw = gzip.compress(json.dumps(doc, separators=(',', ':')).encode(), mtime=0)
    put(repo, token, rosters_path(year), raw, branch, f'BRL: daily rosters {year} (ROSTER-01)')
    rec = summary(doc)
    put(repo, token, receipt_path(year), json.dumps(rec, indent=1).encode(), branch, f'BRL: daily rosters {year} receipt')
    print(json.dumps({k: rec[k] for k in ('season', 'team_dates', 'read', 'missing', 'pitchers_per_roster_mean', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
