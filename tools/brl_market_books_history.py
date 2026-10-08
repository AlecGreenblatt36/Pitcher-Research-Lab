"""Several sportsbooks' 2026 lines from SportsBookReview's public odds pages, matched to official game ids.

For each date of the season: the moneyline page and the totals page, each book's opening and current (closing, for
finished games) line, the consensus (mean of the books' vig-free chances) and the DraftKings line. Output on the
ledger branch: research/market-books-<season>-<run>.jsonl.gz and a receipt. Public odds only; used to judge, to teach
the model and to choose the headline's market reference, never as an input to our model.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brl_live.market import vig_free_home, _minutes_apart  # noqa: E402
from brl_live.market_books import page_url, parse_page, consensus_home, consensus_total  # noqa: E402

UA = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15']
ALIASES = {'Oakland Athletics': 'Athletics', 'Sacramento Athletics': 'Athletics', "Oakland A's": 'Athletics', 'Cleveland Indians': 'Cleveland Guardians'}


def norm(name):
    name = str(name or '').strip()
    return ALIASES.get(name, name)


def fetch(url, accept='application/json', tries=4):
    for attempt in range(tries):
        try:
            req = Request(url, headers={'User-Agent': random.choice(UA), 'Accept': accept, 'Accept-Language': 'en-US,en;q=0.9'})
            with urlopen(req, timeout=40) as r:
                raw = r.read()
            return raw if accept != 'application/json' else json.loads(raw)
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


def main():
    season = int(os.environ.get('BRL_BOOKS_SEASON') or 2026)
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    t0 = time.time()
    receipt = {'schema': 'brl.market-books.v1', 'season': season, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat(),
               'source': 'sportsbookreview.com MLB odds pages (moneyline and totals), one request per date and market'}
    end = min(date.today() - timedelta(days=1), date(season, 11, 15)).isoformat()
    sched = fetch(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={season}-03-01&endDate={end}&gameType=R,F,D,L,W')
    by_date = {}
    for d in sched.get('dates') or []:
        for g in d.get('games') or []:
            if ((g.get('status') or {}).get('abstractGameState')) != 'Final':
                continue
            t = g.get('teams') or {}
            by_date.setdefault(d['date'], []).append({'game_pk': int(g['gamePk']), 'start': g.get('gameDate'), 'type': g.get('gameType'),
                                                      'home': norm((t.get('home') or {}).get('team', {}).get('name')),
                                                      'away': norm((t.get('away') or {}).get('team', {}).get('name'))})
    rows, misses, pages_failed, used = [], 0, 0, set()
    for day in sorted(by_date):
        try:
            ml = parse_page(fetch(page_url(day, 'moneyline'), accept='text/html').decode('utf-8', 'replace'), 'moneyline')
            time.sleep(1.0 + random.random())
            tot = parse_page(fetch(page_url(day, 'totals'), accept='text/html').decode('utf-8', 'replace'), 'totals')
            time.sleep(1.0 + random.random())
        except Exception:
            pages_failed += 1
            continue
        totals_by = {}
        for g in tot:
            totals_by.setdefault((norm(g['away']), norm(g['home'])), []).append(g)
        for g in ml:
            home, away = norm(g['home']), norm(g['away'])
            cands = [x for x in by_date[day] if x['home'] == home and x['away'] == away and x['game_pk'] not in used]
            if not cands:
                misses += 1
                continue
            best = min(cands, key=lambda x: _minutes_apart(x['start'], g['start']))
            used.add(best['game_pk'])
            p_close, n_close = consensus_home(g['books'], 'current')
            p_open, n_open = consensus_home(g['books'], 'open')
            dk = g['books'].get('draftkings') or {}
            row = {'game_pk': best['game_pk'], 'date': day, 'type': best['type'], 'home': home, 'away': away, 'books': sorted(g['books']),
                   'p_cons_close': None if p_close is None else round(p_close, 5), 'n_close': n_close,
                   'p_cons_open': None if p_open is None else round(p_open, 5), 'n_open': n_open,
                   'p_dk_close': round(vig_free_home(*dk['current']), 5) if dk.get('current') else None,
                   'p_dk_open': round(vig_free_home(*dk['open']), 5) if dk.get('open') else None,
                   'lines': {k: v.get('current') for k, v in g['books'].items() if v.get('current')}}
            tg = totals_by.get((away, home)) or []
            if tg:
                total, p_over, n_tot = consensus_total(tg[0]['books'], 'current')
                row.update(total_cons=total, p_over_cons=None if p_over is None else round(p_over, 5), n_total=n_tot)
            rows.append(row)
    receipt.update(dates=len(by_date), rows=len(rows), unmatched=misses, pages_failed=pages_failed,
                   books_seen=sorted({b for r in rows for b in r['books']}),
                   mean_books=round(sum(r['n_close'] for r in rows) / max(len(rows), 1), 2), seconds=round(time.time() - t0, 1))
    raw = gzip.compress('\n'.join(json.dumps(r) for r in rows).encode() + b'\n', mtime=0)
    path = f'research/market-books-{season}-{run_id}.jsonl.gz'
    receipt['file'] = path
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GH_TOKEN')
    if repo and token:
        put(repo, token, path, raw, f'BRL: market books {season}')
        put(repo, token, f'research/market-books-{season}-{run_id}.json', json.dumps(receipt, indent=1).encode(), f'BRL: market books receipt {season}')
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
