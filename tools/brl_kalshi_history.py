"""KALSHI-01: Kalshi's public MLB markets with their last pregame prices, matched to official game and player ids.

Kalshi is the market the leg book is compared with (LEDGER LEG-01): it lists player props (hits, hits + runs + RBIs, RBIs)
next to the game markets (winner, run line, total runs, a run in the first inning). For one series (BRL_KALSHI_SERIES, for
example KXMLBHIT) this job lists every settled market from Kalshi's public API (the historical endpoint before its cutoff,
the live endpoint after), reads the trades before first pitch (the last trade and the volume-weighted price of the final
hour), and matches the event to MLB's game id and, for player props, the player's id through MLB's public box score.

Output on the ledger branch: research/kalshi-<series>-<run>.jsonl.gz (one row per market: ids, the market's line, the
pregame prices, volume and the settled result) and a receipt. Public market data and public ids only; nothing here is an
input to the model. No account and no key: every endpoint used is Kalshi's unauthenticated public market data.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

API = 'https://api.elections.kalshi.com/trade-api/v2'
ET = ZoneInfo('America/New_York')
MONTHS = {m: i for i, m in enumerate(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 1)}
# a doubleheader's tickers end in G1 or G2 after the team codes (KXMLBGAME-26AUG171840STLCING2); before October 18 those
# were left unparsed
EVENT_RE = re.compile(r'^(?P<series>[A-Z0-9]+)-(?P<yy>\d\d)(?P<mon>[A-Z]{3})(?P<dd>\d\d)(?P<hhmm>\d{4})(?P<teams>[A-Z]+?)(?:G(?P<game>\d))?$')
PROP_RE = re.compile(r'^(?P<name>.+?):\s*(?P<k>\d+)\+\s*(?P<what>.+?)\?*$')
RATE = float(os.environ.get('BRL_KALSHI_RATE', '6'))     # requests per second, well under the public limit
_last = [0.0]


def kget(path: str, params: dict | None = None, tries: int = 6):
    """GET a Kalshi public endpoint, paced to RATE requests per second, with backoff on 429 and 5xx."""
    url = API + path + ('?' + urlencode({k: v for k, v in (params or {}).items() if v is not None}) if params else '')
    for attempt in range(tries):
        wait = _last[0] + 1.0 / RATE - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            with urlopen(Request(url, headers={'User-Agent': 'BaseballResearchLab/2.0 (+public research)', 'Accept': 'application/json'}), timeout=60) as r:
                return json.loads(r.read())
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 2))
        except URLError:
            if attempt == tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 2))


def event_start(event_ticker: str):
    """(first pitch in UTC, team string) from an event ticker such as KXMLBHIT-26AUG171940ATHKC (time in Eastern)."""
    m = EVENT_RE.match(event_ticker or '')
    if not m:
        return None, None
    local = datetime(2000 + int(m['yy']), MONTHS[m['mon']], int(m['dd']), int(m['hhmm'][:2]), int(m['hhmm'][2:]), tzinfo=ET)
    return local.astimezone(timezone.utc), m['teams']


def list_markets(series: str):
    """Every settled market of the series: the historical endpoint, then the live one; de-duplicated by ticker."""
    seen, out = set(), []
    for path, extra in (('/historical/markets', {}), ('/markets', {'status': 'settled'})):
        cursor = None
        while True:
            doc = kget(path, {'series_ticker': series, 'limit': 200, 'cursor': cursor, **extra}) or {}
            for mk in doc.get('markets') or []:
                if mk.get('ticker') and mk['ticker'] not in seen:
                    seen.add(mk['ticker']); mk['_source'] = path; out.append(mk)
            cursor = doc.get('cursor')
            if not cursor or not doc.get('markets'):
                break
    return out


def pregame_prices(ticker: str, start: datetime, historical: bool):
    """Last trade before first pitch and the volume-weighted yes price of the final hour (None when nothing traded)."""
    path = '/historical/trades' if historical else '/markets/trades'
    ts = int(start.timestamp())
    doc = kget(path, {'ticker': ticker, 'max_ts': ts, 'limit': 200}) or {}
    trades = doc.get('trades') or []
    if not trades:
        return {'last': None, 'last_time': None, 'vwap_hour': None, 'trades_hour': 0, 'count_hour': 0.0}
    def price(t):
        v = t.get('yes_price_dollars')
        return float(v) if v is not None else (t.get('yes_price') or 0) / 100.0
    def count(t):
        v = t.get('count_fp')
        return float(v) if v is not None else float(t.get('count') or 0)
    last = trades[0]
    hour = [t for t in trades if datetime.fromisoformat(t['created_time'].replace('Z', '+00:00')).timestamp() >= ts - 3600]
    w = sum(count(t) for t in hour)
    vwap = sum(price(t) * count(t) for t in hour) / w if w else None
    return {'last': round(price(last), 4), 'last_time': last.get('created_time'), 'vwap_hour': round(vwap, 4) if vwap is not None else None,
            'trades_hour': len(hour), 'count_hour': round(w, 2)}


def norm_name(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', s)
    return re.sub(r'[^a-z]', '', s)


class MLB:
    """Official game ids by date and time, and player ids by name within a game (public statsapi)."""

    def __init__(self, get_json):
        self.get = get_json; self._days = {}; self._box = {}

    def games(self, date: str):
        if date not in self._days:
            doc = self.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&fields=dates,games,gamePk,gameDate,gameType,teams,away,home,team,id,name') or {}
            self._days[date] = [g for d in doc.get('dates') or [] for g in d.get('games') or []]
        return self._days[date]

    def game_for(self, start: datetime):
        best, gap = None, None
        for g in self.games(start.astimezone(ET).date().isoformat()):
            t = datetime.fromisoformat(g['gameDate'].replace('Z', '+00:00'))
            d = abs((t - start).total_seconds())
            if gap is None or d < gap:
                best, gap = g, d
        return best if gap is not None and gap <= 4 * 3600 else None

    def player(self, game_pk: int, name: str):
        if game_pk not in self._box:
            doc = self.get(f'https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore') or {}
            names = {}
            for side in ('away', 'home'):
                for p in (((doc.get('teams') or {}).get(side) or {}).get('players') or {}).values():
                    person = p.get('person') or {}
                    if person.get('id'):
                        names[norm_name(person.get('fullName'))] = (int(person['id']), side)
            self._box[game_pk] = names
        return self._box[game_pk].get(norm_name(name))


def describe(mk: dict) -> dict:
    """The market's statement from its title: a player prop (name, threshold, stat) or a game market (title as given)."""
    title = mk.get('title') or ''
    m = PROP_RE.match(title)
    if m and ':' in title:
        return {'kind': 'prop', 'name': m['name'].strip(), 'at_least': int(m['k']), 'stat': m['what'].strip().lower()}
    return {'kind': 'game', 'title': title, 'yes_sub_title': mk.get('yes_sub_title'), 'no_sub_title': mk.get('no_sub_title'),
            'floor_strike': mk.get('floor_strike'), 'strike_type': mk.get('strike_type')}


def main():
    from brl_bookkeeping_backfill import get_json, put
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    series, _, window = os.environ['BRL_KALSHI_SERIES'].partition('@')      # SERIES or SERIES@YYYY-MM-DD:YYYY-MM-DD (game dates)
    lo, _, hi = window.partition(':')
    only_lines = [int(x) for x in (os.environ.get('BRL_KALSHI_PROP_LINES') or '').replace(',', ' ').split()]   # e.g. "1 2" keeps 1+ and 2+
    budget = float(os.environ.get('BRL_KALSHI_MINUTES', '320')) * 60
    t0 = time.time()
    markets = list_markets(series)
    print(json.dumps({'series': series, 'markets': len(markets), 'seconds': round(time.time() - t0)}), flush=True)
    mlb = MLB(get_json)
    rows, counts = [], {'listed': len(markets), 'priced': 0, 'no_trade': 0, 'no_game': 0, 'no_player': 0, 'skipped_line': 0, 'unparsed': 0, 'stopped_at_budget': False}
    unparsed = {}                                                # a few event tickers of each unparsed shape, for the receipt
    markets.sort(key=lambda m: m.get('close_time') or '')
    for i, mk in enumerate(markets):
        if time.time() - t0 > budget:
            counts['stopped_at_budget'] = True
            break
        start, teams = event_start(mk.get('event_ticker'))
        if start is None:
            counts['unparsed'] += 1
            shape = re.sub(r'[A-Z]', 'A', re.sub(r'\d', '9', str(mk.get('event_ticker') or '').partition('-')[2]))
            if len(unparsed.setdefault(shape, [])) < 3:
                unparsed[shape].append(mk.get('event_ticker'))
            continue
        day = start.astimezone(ET).date().isoformat()
        if (lo and day < lo) or (hi and day > hi):
            continue
        d = describe(mk)
        if d['kind'] == 'prop' and only_lines and d['at_least'] not in only_lines:
            counts['skipped_line'] += 1; continue
        g = mlb.game_for(start)
        if not g:
            counts['no_game'] += 1; continue
        row = {'series': series, 'ticker': mk['ticker'], 'event': mk.get('event_ticker'), 'game_pk': int(g['gamePk']), 'game_type': g.get('gameType'),
               'date': start.astimezone(ET).date().isoformat(), 'start': g.get('gameDate'), 'teams': teams, **d,
               'result': mk.get('result'), 'volume': mk.get('volume_fp') or mk.get('volume'), 'source': mk.get('_source')}
        if d['kind'] == 'prop':
            pid = mlb.player(int(g['gamePk']), d['name'])
            if not pid:
                counts['no_player'] += 1
            else:
                row['player_id'], row['side'] = pid
        first_pitch = datetime.fromisoformat(g['gameDate'].replace('Z', '+00:00'))
        row.update(pregame_prices(mk['ticker'], min(first_pitch, start), mk['_source'] == '/historical/markets'))
        if row['last'] is None:
            counts['no_trade'] += 1
        else:
            counts['priced'] += 1
        rows.append(row)
        if i % 2000 == 0:
            print(json.dumps({'done': i, 'of': len(markets), 'priced': counts['priced'], 'seconds': round(time.time() - t0)}), flush=True)
    run = os.environ.get('GITHUB_RUN_ID', 'local')
    tag = series + (f'-{lo}-{hi}' if window else '')
    path = f'research/kalshi-{tag}-{run}.jsonl.gz'
    raw = gzip.compress('\n'.join(json.dumps(r, separators=(',', ':')) for r in rows).encode(), mtime=0)
    put(repo, token, path, raw, branch, f'BRL: Kalshi {series} pregame prices (KALSHI-01)')
    dates = sorted(r['date'] for r in rows)
    receipt = {'schema': 'brl.kalshi-receipt.v1', 'experiment': 'KALSHI-01', 'series': series, 'rows': len(rows), 'counts': counts,
               'dates': [dates[0], dates[-1]] if dates else None, 'by_month': {}, 'path': path, 'run_id': run,
               'seconds': round(time.time() - t0), 'public_data_only': True, 'built_at': datetime.now(timezone.utc).isoformat()}
    for r in rows:
        receipt['by_month'][r['date'][:7]] = receipt['by_month'].get(r['date'][:7], 0) + 1
    receipt['window'] = [lo or None, hi or None]
    receipt['unparsed_examples'] = dict(sorted(unparsed.items(), key=lambda kv: -len(kv[1]))[:12])
    put(repo, token, f'research/kalshi-{tag}-{run}.json', json.dumps(receipt, indent=1).encode(), branch, f'BRL: Kalshi {series} receipt')
    print(json.dumps({k: receipt[k] for k in ('series', 'rows', 'counts', 'dates', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
