"""Pregame betting lines for a past regular season, for judging the model against the market.

ESPN's scoreboard drops the odds once a game is over, but each game's summary keeps its pick
center (DraftKings for 2026; nothing for 2025 and earlier). For every regular-season game of the
season (official MLB schedule) this reads the ESPN event of the same teams and start, then the
summary's moneylines (and the open line when given), totals and run lines, and converts them to a vig-free home win
probability. Output on the ledger branch: research/market-<season>-<run>.jsonl.gz (one line per
game: game_pk, date, teams, moneylines, p_home, provider) and a receipt. Public odds only.
"""
from __future__ import annotations
import base64, gzip, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brl_live.market import ABBR, _ml_value, vig_free_home, _minutes_apart  # noqa: E402

UA = {'User-Agent': 'Mozilla/5.0 (BRL research)', 'Accept': 'application/json'}


def get(url, tries=5):
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 + 3 * attempt)


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
            with urlopen(Request(url, headers=headers, data=json.dumps(payload).encode(), method='PUT'), timeout=60) as r:
                return r.read()
        except HTTPError as exc:
            if exc.code != 409 or attempt == 5:
                raise
            time.sleep(3 + 3 * attempt)


def lines_from_summary(doc):
    for pc in doc.get('pickcenter') or []:
        h = _ml_value((pc.get('homeTeamOdds') or {}).get('moneyLine'))
        a = _ml_value((pc.get('awayTeamOdds') or {}).get('moneyLine'))
        ml = pc.get('moneyline') if isinstance(pc.get('moneyline'), dict) else {}
        open_h = _ml_value(((ml.get('home') or {}).get('open') if isinstance(ml.get('home'), dict) else None))
        open_a = _ml_value(((ml.get('away') or {}).get('open') if isinstance(ml.get('away'), dict) else None))
        close_h = _ml_value(((ml.get('home') or {}).get('close') if isinstance(ml.get('home'), dict) else None))
        close_a = _ml_value(((ml.get('away') or {}).get('close') if isinstance(ml.get('away'), dict) else None))
        if h is None or a is None:
            h, a = close_h, close_a
        if h is None or a is None:
            continue
        out = {'home_ml': int(h), 'away_ml': int(a), 'open_home_ml': open_h, 'open_away_ml': open_a, 'close_home_ml': close_h, 'close_away_ml': close_a,
               'provider': str((pc.get('provider') or {}).get('name') or ''), 'over_under': pc.get('overUnder')}
        # Totals and run line at the same moment as the moneyline (the line and both prices), plus ESPN's
        # open/close blocks as given, for later parsing.
        over, under = _ml_value(pc.get('overOdds')), _ml_value(pc.get('underOdds'))
        out.update(over_odds=over, under_odds=under, spread=pc.get('spread'),
                   home_spread_odds=_ml_value((pc.get('homeTeamOdds') or {}).get('spreadOdds')),
                   away_spread_odds=_ml_value((pc.get('awayTeamOdds') or {}).get('spreadOdds')))
        if isinstance(pc.get('total'), dict):
            out['total_raw'] = pc['total']
        if isinstance(pc.get('pointSpread'), dict):
            out['point_spread_raw'] = pc['pointSpread']
        return out
    return None


def main():
    season = int(os.environ.get('BRL_MARKET_SEASON') or 2026)
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    t0 = time.time()
    receipt = {'schema': 'brl.market-history.v1', 'season': season, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat()}
    sched = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={season}-03-01&endDate={season}-10-01&gameType=R')
    games = []
    for d in sched.get('dates') or []:
        for g in d.get('games') or []:
            if ((g.get('status') or {}).get('abstractGameState')) != 'Final':
                continue
            t = g.get('teams') or {}
            games.append({'game_pk': int(g['gamePk']), 'date': d['date'], 'start': g.get('gameDate'),
                          'home': (t.get('home') or {}).get('team', {}).get('name'), 'away': (t.get('away') or {}).get('team', {}).get('name')})
    receipt['schedule_games'] = len(games)
    by_date = {}
    for g in games:
        by_date.setdefault(g['date'], []).append(g)

    def scoreboard(day):
        doc = get('https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=' + day.replace('-', '') + '&limit=50')
        out = []
        for e in doc.get('events') or []:
            for c in e.get('competitions') or []:
                teams = {x.get('homeAway'): (x.get('team') or {}).get('displayName') for x in c.get('competitors') or []}
                out.append({'id': e.get('id'), 'start': e.get('date'), 'home': teams.get('home'), 'away': teams.get('away')})
        return day, out

    with ThreadPoolExecutor(4) as pool:
        boards = dict(pool.map(scoreboard, sorted(by_date)))
    matched = []
    for day, gs in by_date.items():
        events = boards.get(day) or []
        for g in gs:
            cands = [e for e in events if e['home'] == g['home'] and e['away'] == g['away']]
            if not cands:
                continue
            best = min(cands, key=lambda e: _minutes_apart(e['start'], g['start']))
            if len(cands) > 1 and _minutes_apart(best['start'], g['start']) > 180:
                continue
            matched.append((g, best['id']))
    receipt['matched_events'] = len(matched)

    def summary(item):
        g, eid = item
        try:
            doc = get('https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event=' + str(eid))
            line = lines_from_summary(doc)
        except Exception as exc:
            return {'game_pk': g['game_pk'], 'error': type(exc).__name__}
        if line is None:
            return {'game_pk': g['game_pk'], 'missing': True}
        out = {'game_pk': g['game_pk'], 'date': g['date'], 'home': g['home'], 'away': g['away'], 'espn_event': str(eid), **line,
               'p_home': round(vig_free_home(line['home_ml'], line['away_ml']), 5)}
        if line.get('open_home_ml') is not None and line.get('open_away_ml') is not None:
            out['p_home_open'] = round(vig_free_home(line['open_home_ml'], line['open_away_ml']), 5)
        if line.get('over_odds') is not None and line.get('under_odds') is not None:
            out['p_over'] = round(vig_free_home(line['over_odds'], line['under_odds']), 5)
        if line.get('home_spread_odds') is not None and line.get('away_spread_odds') is not None:
            out['p_home_cover'] = round(vig_free_home(line['home_spread_odds'], line['away_spread_odds']), 5)
        return out

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(summary, matched))
    good = [r for r in rows if 'p_home' in r]
    receipt.update(lines=len(good), missing=sum(1 for r in rows if r.get('missing')), errors=sum(1 for r in rows if r.get('error')),
                   providers=sorted({r['provider'] for r in good}), with_open=sum(1 for r in good if 'p_home_open' in r),
                   with_total=sum(1 for r in good if r.get('over_under') is not None), with_total_prices=sum(1 for r in good if 'p_over' in r),
                   with_run_line=sum(1 for r in good if 'p_home_cover' in r),
                   total_raw_shape=next(({k: (sorted(v.keys()) if isinstance(v, dict) else type(v).__name__) for k, v in r['total_raw'].items()} for r in good if r.get('total_raw')), None))
    raw = gzip.compress('\n'.join(json.dumps(r) for r in good).encode() + b'\n', mtime=0)
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GH_TOKEN')
    path = f'research/market-{season}-{run_id}.jsonl.gz'
    receipt['seconds'] = round(time.time() - t0, 1); receipt['file'] = path
    if repo and token:
        put(repo, token, path, raw, f'BRL: market history {season}')
        put(repo, token, f'research/market-{season}-{run_id}.json', json.dumps(receipt, indent=1).encode(), f'BRL: market history receipt {season}')
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
