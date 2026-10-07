"""Pregame betting lines for 2021-2025 from a public dataset, matched to official game ids.

Source: the dataset release of github.com/ArnavSaraogi/mlb-odds-scraper (lines collected from
SportsBookReview: opening and current moneylines per sportsbook, 2021-03-20 to 2025-08-16).
Each game is matched to its MLB game_pk by date, teams and start time (official schedule), and
the lines are turned into vig-free home win probabilities: DraftKings opening and closing, and
the average over the books present. Output on the ledger branch: research/market-sbr-<run>.jsonl.gz
and a receipt. Public odds only; used to judge and to teach the model, never as a forecast input.
"""
from __future__ import annotations
import base64, gzip, io, json, os, sys, time, zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brl_live.market import vig_free_home, _minutes_apart  # noqa: E402

REPO_SRC = 'ArnavSaraogi/mlb-odds-scraper'
ALIASES = {'Cleveland Indians': 'Cleveland Guardians', 'Oakland Athletics': 'Athletics', 'Athletics': 'Athletics'}


def http(url, token=None, accept='application/json', raw=False, tries=5):
    headers = {'User-Agent': 'BRL-research/1.0', 'Accept': accept}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as r:
                body = r.read()
            return body if raw else json.loads(body)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 + 4 * attempt)


def put(repo, token, path, raw, message, branch='brl-live-data'):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-market/1.0', 'Content-Type': 'application/json'}
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


def norm(name):
    name = str(name or '').strip()
    return ALIASES.get(name, name)


def main():
    token = os.environ.get('GH_TOKEN'); repo = os.environ.get('GITHUB_REPOSITORY'); run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    t0 = time.time()
    receipt = {'schema': 'brl.market-sbr.v1', 'source': 'https://github.com/' + REPO_SRC + ' (dataset release; lines from SportsBookReview)', 'run_id': run_id,
               'started_at': datetime.now(timezone.utc).isoformat()}
    rel = http(f'https://api.github.com/repos/{REPO_SRC}/releases/tags/dataset', token)
    assets = [{'name': a['name'], 'size': a['size'], 'url': a['browser_download_url']} for a in rel.get('assets') or []]
    receipt['assets'] = assets
    asset = max(assets, key=lambda a: a['size'])
    blob = http(asset['url'], raw=True, accept='application/octet-stream')
    if asset['name'].endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
            blob = z.read(name)
    elif asset['name'].endswith('.gz'):
        blob = gzip.decompress(blob)
    data = json.loads(blob)
    receipt['dates_in_source'] = len(data)
    # official schedule, by season, to find game ids
    sched = {}
    for season in range(2021, 2026):
        doc = http(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={season}-03-01&endDate={season}-11-15&gameType=R,F,D,L,W')
        for d in doc.get('dates') or []:
            for g in d.get('games') or []:
                if ((g.get('status') or {}).get('abstractGameState')) != 'Final':
                    continue
                t = g.get('teams') or {}
                sched.setdefault(d['date'], []).append({'game_pk': int(g['gamePk']), 'start': g.get('gameDate'), 'type': g.get('gameType'),
                                                       'home': norm((t.get('home') or {}).get('team', {}).get('name')),
                                                       'away': norm((t.get('away') or {}).get('team', {}).get('name')),
                                                       'home_score': (t.get('home') or {}).get('score'), 'away_score': (t.get('away') or {}).get('score')})
    rows, unmatched, books_seen = [], 0, defaultdict(int)
    used = set()
    for day, games in sorted(data.items()):
        for item in games or []:
            gv = item.get('gameView') or {}
            home, away = norm((gv.get('homeTeam') or {}).get('fullName')), norm((gv.get('awayTeam') or {}).get('fullName'))
            start = gv.get('startDate')
            cands = [g for g in sched.get(day, []) if g['home'] == home and g['away'] == away and g['game_pk'] not in used]
            if not cands:
                unmatched += 1; continue
            best = min(cands, key=lambda g: _minutes_apart(g['start'], start))
            if len(cands) > 1 and _minutes_apart(best['start'], start) > 240:
                unmatched += 1; continue
            used.add(best['game_pk'])
            lines = {}
            for b in (item.get('odds') or {}).get('moneyline') or []:
                book = str(b.get('sportsbook') or '').lower()
                o, c = b.get('openingLine') or {}, b.get('currentLine') or {}
                try:
                    lines[book] = {'open': [int(o['homeOdds']), int(o['awayOdds'])] if o.get('homeOdds') is not None and o.get('awayOdds') is not None else None,
                                   'close': [int(c['homeOdds']), int(c['awayOdds'])] if c.get('homeOdds') is not None and c.get('awayOdds') is not None else None}
                    books_seen[book] += 1
                except (TypeError, ValueError):
                    continue
            if not lines:
                continue
            def p(kind, book=None):
                vals = [vig_free_home(*v[kind]) for k, v in lines.items() if v.get(kind) and (book is None or k == book) and abs(v[kind][0]) >= 100 and abs(v[kind][1]) >= 100]
                return round(sum(vals) / len(vals), 5) if vals else None
            rows.append({'game_pk': best['game_pk'], 'date': day, 'season': int(day[:4]), 'game_type': best['type'], 'home': home, 'away': away,
                         'home_score': best['home_score'], 'away_score': best['away_score'], 'books': sorted(lines),
                         'p_close_dk': p('close', 'draftkings'), 'p_open_dk': p('open', 'draftkings'), 'p_close_avg': p('close'), 'p_open_avg': p('open'),
                         'dk_close': (lines.get('draftkings') or {}).get('close'), 'dk_open': (lines.get('draftkings') or {}).get('open')})
    receipt.update(rows=len(rows), unmatched=unmatched, books=dict(books_seen), by_season={str(s): sum(1 for r in rows if r['season'] == s) for s in range(2021, 2026)},
                   with_dk_close=sum(1 for r in rows if r['p_close_dk'] is not None), seconds=round(time.time() - t0, 1))
    raw = gzip.compress('\n'.join(json.dumps(r) for r in rows).encode() + b'\n', mtime=0)
    path = f'research/market-sbr-{run_id}.jsonl.gz'; receipt['file'] = path
    if repo and token:
        put(repo, token, path, raw, 'BRL: market history 2021-2025')
        put(repo, token, f'research/market-sbr-{run_id}.json', json.dumps(receipt, indent=1).encode(), 'BRL: market history 2021-2025 receipt')
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
