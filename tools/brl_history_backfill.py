"""Extend the plate-appearance history back before 2023 (HIST-01), one season per run.

Runs in GitHub Actions. For one season (BRL_SEASON) it reads the official schedule for the season's completed games,
downloads Baseball Savant's pitch-by-pitch export for the regular season and the postseason in three-day chunks, and
collapses it to one row per plate appearance with the same builder the current history uses
(research_lab.pa_model.data.build_plate_appearances), so the columns, outcome labels and cleaning match the
2023-2026 history exactly. Postseason rows keep their game type (F, D, L, W).

The season's rows are gzipped, sealed with AES-256-GCM (purpose pa-history-season-<year>) and committed to
private/history/pa-<year>.enc on the ledger branch. A public receipt (history/pa-<year>.json) carries only counts:
games against the schedule, plate appearances by game type, the outcome mix, players and dates. Plaintext never
leaves the runner; the key is never printed.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))

POST_TYPES = ('F', 'D', 'L', 'W')
SAVANT = 'https://baseballsavant.mlb.com/statcast_search/csv'


def history_path(year: int) -> str:
    return f'private/history/pa-{int(year)}.enc'


def history_purpose(year: int) -> str:
    return f'pa-history-season-{int(year)}'


def receipt_path(year: int) -> str:
    return f'history/pa-{int(year)}.json'


def chunks(start: date, end: date, days: int = 3) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        nxt = min(end, cur + timedelta(days=days - 1))
        out.append((cur, nxt)); cur = nxt + timedelta(days=1)
    return out


def savant_url(start: date, end: date, game_types: str) -> str:
    from urllib.parse import urlencode
    params = {'all': 'true', 'hfGT': game_types, 'player_type': 'pitcher', 'game_date_gt': start.isoformat(), 'game_date_lt': end.isoformat(),
              'min_pitches': '0', 'min_results': '0', 'group_by': 'name', 'sort_col': 'pitches', 'player_event_sort': 'h_launch_speed',
              'sort_order': 'desc', 'min_abs': '0', 'type': 'details'}
    return SAVANT + '?' + urlencode(params)


def season_games(get_json, year: int) -> list[dict]:
    """Completed regular-season and postseason games of a season from the official schedule."""
    doc = get_json(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={year}-02-15&endDate={year}-11-30'
                   '&fields=dates,date,games,gamePk,gameType,officialDate,status,abstractGameState')
    games = {}
    for d in (doc or {}).get('dates') or []:
        for g in d.get('games') or []:
            if g.get('gameType') not in ('R',) + POST_TYPES or (g.get('status') or {}).get('abstractGameState') != 'Final':
                continue
            games[int(g['gamePk'])] = {'game_pk': int(g['gamePk']), 'date': str(g.get('officialDate') or d['date'])[:10], 'game_type': g['gameType']}
    return sorted(games.values(), key=lambda g: (g['date'], g['game_pk']))


def download(spans, game_types: str, cache: Path, columns, tries: int = 6, workers: int = 3) -> list[Path]:
    """Savant's export for each span (a file per span; empty spans give no file)."""
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor
    cache.mkdir(parents=True, exist_ok=True)

    def one(span):
        target = cache / f'{span[0]}_{span[1]}_{game_types.replace("|", "")}.csv.gz'
        if target.exists():
            return target
        last = None
        for attempt in range(tries):
            try:
                r = requests.get(savant_url(span[0], span[1], game_types), headers={'User-Agent': 'Mozilla/5.0 (BRL research)', 'Accept': 'text/csv,*/*'}, timeout=240)
                r.raise_for_status()
                text = r.content.decode('utf-8-sig', 'replace')
                if '<html' in text[:300].lower():
                    raise RuntimeError('html instead of csv')
                if not text.strip():
                    return None
                frame = pd.read_csv(io.StringIO(text), low_memory=False)
                if not len(frame):
                    return None
                if 'game_pk' not in frame.columns:
                    raise RuntimeError('no game_pk column')
                if len(frame) >= 24990:
                    raise RuntimeError(f'{len(frame)} rows: the export may be capped; use shorter spans')
                frame.loc[:, [c for c in columns if c in frame.columns]].to_csv(target, index=False, compression='gzip')
                return target
            except Exception as exc:          # throttled, a timeout or a capped answer: wait and retry
                last = exc
                time.sleep(min(90, 5 * 2 ** attempt))
        raise RuntimeError(f'savant {span[0]} to {span[1]} ({game_types}): {type(last).__name__}: {str(last)[:160]}')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [p for p in pool.map(one, spans) if p is not None]


def relabel_postseason(files: list[Path]) -> dict[int, str]:
    """The history builder keeps regular-season rows only, so postseason chunks are relabeled R for the build and the
    real game type is returned (game_pk -> type) to put back afterwards."""
    import pandas as pd
    types = {}
    for f in files:
        frame = pd.read_csv(f, low_memory=False)
        if 'game_type' not in frame.columns:
            continue
        for pk, gt in zip(frame['game_pk'], frame['game_type'].astype(str)):
            if gt in POST_TYPES:
                types[int(pk)] = gt
        frame['game_type'] = 'R'
        frame.to_csv(f, index=False, compression='gzip')
    return types


def season_receipt(pa, games: list[dict], year: int) -> dict:
    """Counts only: what the season's history holds against the official schedule."""
    sched = {g['game_pk']: g['game_type'] for g in games}
    have = set(int(x) for x in pa['game_pk'].unique())
    by_type = {}
    for gt in ('R',) + POST_TYPES:
        want = {pk for pk, t in sched.items() if t == gt}
        if not want:
            continue
        rows = pa[pa['game_type'] == gt]
        by_type[gt] = {'schedule_games': len(want), 'games': int(rows['game_pk'].nunique()), 'missing_games': len(want - have),
                       'plate_appearances': int(len(rows))}
    reg = pa[pa['game_type'] == 'R']
    mix = (reg['outcome'].value_counts(normalize=True) * 100).round(2).to_dict()
    return {'year': year, 'by_game_type': by_type, 'outcome_mix_pct_regular_season': mix,
            'batters': int(pa['batter'].nunique()), 'pitchers': int(pa['pitcher'].nunique()),
            'date_min': str(pa['date_key'].min())[:10], 'date_max': str(pa['date_key'].max())[:10],
            'not_in_schedule_games': len(have - set(sched))}


def main():
    import pandas as pd
    from research_lab.pa_model.data import PA_SOURCE_COLUMNS, build_plate_appearances
    from cloud.security import seal, key_bytes, sha
    sys.path.insert(0, str(ROOT / 'tools'))
    from brl_bookkeeping_backfill import get_json, put       # the backfill's GitHub and MLB helpers

    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key = key_bytes(os.environ['BRL_PA_PACKAGE_KEY'])
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    year = int(os.environ['BRL_SEASON'])
    if not 2015 <= year <= 2022:
        raise ValueError('HIST-01 covers 2015 to 2022 (Statcast); 2023 on is in the data package')
    work = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / f'brl-history-{year}'
    t0 = time.time()
    games = season_games(get_json, year)
    reg = [g for g in games if g['game_type'] == 'R']; post = [g for g in games if g['game_type'] in POST_TYPES]
    print(json.dumps({'year': year, 'regular_season_games': len(reg), 'postseason_games': len(post)}), flush=True)
    cols = list(PA_SOURCE_COLUMNS)
    span = lambda gs: chunks(date.fromisoformat(gs[0]['date']), date.fromisoformat(gs[-1]['date']))
    reg_files = download(span(reg), 'R|', work / 'reg', cols)
    post_files = download(span(post), '|'.join(POST_TYPES) + '|', work / 'post', cols) if post else []
    print(json.dumps({'downloaded_files': len(reg_files) + len(post_files), 'seconds': round(time.time() - t0)}), flush=True)
    post_types = relabel_postseason(post_files)
    pa, report = build_plate_appearances(reg_files + post_files)
    pa['game_type'] = [post_types.get(int(pk), 'R') for pk in pa['game_pk']]
    pa['date_key'] = pd.to_datetime(pa['date_key']).dt.strftime('%Y-%m-%d')
    pa['game_date'] = pd.to_datetime(pa['game_date']).dt.strftime('%Y-%m-%d')
    pa = pa[pa['season'] == year].reset_index(drop=True)
    plain_csv = pa.to_csv(index=False).encode()
    plain = gzip.compress(plain_csv, mtime=0)
    cipher = seal(plain, key, history_purpose(year))
    put(repo, token, history_path(year), cipher, branch, f'BRL: plate-appearance history {year} (HIST-01)')
    receipt = {'schema': 'brl.history-season-receipt.v1', 'experiment': 'HIST-01', **season_receipt(pa, games, year),
               'plate_appearances': int(len(pa)), 'columns': list(pa.columns),
               'build': {k: report[k] for k in ('raw_pitch_rows', 'pitch_rows', 'terminal_pa_rows', 'eligible_pa_rows', 'excluded_terminal_rows', 'duplicate_pa_rows_removed') if k in report},
               'plaintext_csv_sha256': hashlib.sha256(plain_csv).hexdigest(), 'cipher_sha256': sha(cipher), 'cipher_bytes': len(cipher),
               'path': history_path(year), 'format': 'AES-256-GCM; BRLAESG1 header; 12-byte nonce; AAD BRL:' + history_purpose(year) + '; gzip CSV inside',
               'sealed_at': datetime.now(timezone.utc).isoformat(), 'seconds': round(time.time() - t0), 'run_id': os.environ.get('GITHUB_RUN_ID')}
    put(repo, token, receipt_path(year), json.dumps(receipt, indent=1).encode(), branch, f'BRL: plate-appearance history {year} receipt')
    print(json.dumps({k: receipt[k] for k in ('year', 'plate_appearances', 'by_game_type', 'cipher_bytes', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
