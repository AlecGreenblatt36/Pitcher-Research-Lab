"""SPRING-01: one row per spring-training pitch, sealed per season, for the chain's early-season prior.

Runs in GitHub Actions, one season per run (BRL_SEASON). The official schedule's spring games (game type S, final)
between February 15 and April 5 are fetched play by play like the bookkeeping backfill does, flattened to the pitch
table's columns (tools/brl_pitch_table.py) with the count before each pitch, and sealed as gzip CSV at
private/statcast/spring-<year>.enc (purpose spring-pitches-season-<year>). The receipt (spring/<year>.json) carries counts
only: games, pitches, the share with a tracked location and speed (not every spring park has the tracking).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))


def spring_path(year: int) -> str:
    return f'private/statcast/spring-{int(year)}.enc'


def spring_purpose(year: int) -> str:
    return f'spring-pitches-season-{int(year)}'


def receipt_path(year: int) -> str:
    return f'spring/{int(year)}.json'


def spring_games(get_json, year: int) -> list[dict]:
    doc = get_json(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={year}-02-15&endDate={year}-04-05&gameType=S'
                   '&fields=dates,date,games,gamePk,gameType,officialDate,status,abstractGameState,detailedState')
    games = {}
    for d in (doc or {}).get('dates') or []:
        for g in d.get('games') or []:
            status = g.get('status') or {}
            if g.get('gameType') != 'S' or status.get('abstractGameState') != 'Final':
                continue
            games[int(g['gamePk'])] = {'game_pk': int(g['gamePk']), 'date': str(g.get('officialDate') or d['date'])[:10], 'game_type': 'S'}
    return sorted(games.values(), key=lambda g: (g['date'], g['game_pk']))


def main():
    import pandas as pd
    from cloud.security import seal, key_bytes, sha
    sys.path.append(str(ROOT / 'tools'))
    from brl_bookkeeping_backfill import get_json, extract
    from brl_fit_model import put_bytes
    from brl_pitch_table import rows_from_study, PITCH_COLUMNS
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key = key_bytes(os.environ['BRL_PA_PACKAGE_KEY'])
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    year = int(os.environ['BRL_SEASON'])
    t0 = time.time()
    games = spring_games(get_json, year)
    print(json.dumps({'year': year, 'spring_games': len(games)}), flush=True)

    def fetch(game):
        try:
            doc = get_json(f'https://statsapi.mlb.com/api/v1/game/{game["game_pk"]}/playByPlay')
            research: list = []
            extract(doc, research)
            return game['game_pk'], {'date': game['date'], 'game_type': 'S', 'rows': research}, None
        except Exception as exc:
            return game['game_pk'], None, type(exc).__name__ + ': ' + str(exc)[:120]

    study = {'games': {}}; failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pk, value, err in pool.map(fetch, games):
            if err:
                failures.append({'game_pk': pk, 'error': err})
            elif value['rows']:
                study['games'][str(pk)] = value
    frame = pd.DataFrame(rows_from_study(study), columns=PITCH_COLUMNS)
    frame = frame.sort_values(['date_key', 'game_pk', 'at_bat_number', 'pitch_number'], kind='mergesort').reset_index(drop=True)
    plain_csv = frame.to_csv(index=False).encode()
    cipher = seal(gzip.compress(plain_csv, mtime=0), key, spring_purpose(year))
    put_bytes(repo, token, spring_path(year), cipher, branch, f'BRL: spring pitch table {year} (SPRING-01)')
    tracked = frame['zone'].notna() & frame['speed'].notna()
    receipt = {'schema': 'brl.spring-table-receipt.v1', 'experiment': 'SPRING-01', 'year': year, 'schedule_games': len(games), 'games': int(frame['game_pk'].nunique()),
               'failures': failures[:30], 'plate_appearances': int(frame[['game_pk', 'at_bat_number']].drop_duplicates().shape[0]), 'pitches': int(len(frame)),
               'tracked_pct': round(float(tracked.mean()) * 100, 2) if len(frame) else None,
               'games_with_tracking': int(frame.loc[tracked, 'game_pk'].nunique()) if len(frame) else 0,
               'date_min': str(frame['date_key'].min()) if len(frame) else None, 'date_max': str(frame['date_key'].max()) if len(frame) else None,
               'columns': PITCH_COLUMNS, 'plaintext_csv_sha256': hashlib.sha256(plain_csv).hexdigest(), 'cipher_sha256': sha(cipher), 'cipher_bytes': len(cipher),
               'path': spring_path(year), 'format': 'AES-256-GCM; BRLAESG1 header; 12-byte nonce; AAD BRL:' + spring_purpose(year) + '; gzip CSV inside',
               'sealed_at': datetime.now(timezone.utc).isoformat(), 'seconds': round(time.time() - t0), 'run_id': os.environ.get('GITHUB_RUN_ID')}
    put_bytes(repo, token, receipt_path(year), json.dumps(receipt, indent=1).encode(), branch, f'BRL: spring pitch table {year} receipt')
    print(json.dumps({k: receipt[k] for k in ('year', 'schedule_games', 'games', 'pitches', 'tracked_pct', 'games_with_tracking', 'seconds')} | {'failures': len(failures)}), flush=True)


if __name__ == '__main__':
    main()
