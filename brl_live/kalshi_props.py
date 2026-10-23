"""Kalshi's current prices for today's games, beside the board's legs (EDGE-02's capture; public market data, no account).

For each pending game the lane fetches the open markets of each MLB series on Kalshi's public API, by event ticker
(KXMLBHIT-26OCT232000TBMIL: the Eastern start time and the two team codes, away then home), matches hitter and starter
props to the box's player ids by name, and keeps a compact row per market: both sides' bid and ask, the last price and
the volume. The last capture taken 30 minutes or more before first pitch is kept as 'pre', the snapshot EDGE-02 scores.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

API = 'https://api.elections.kalshi.com/trade-api/v2'
ET = ZoneInfo('America/New_York')
MONTHS = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')
SERIES = ('KXMLBGAME', 'KXMLBTOTAL', 'KXMLBSPREAD', 'KXMLBRFI', 'KXMLBHIT', 'KXMLBHR', 'KXMLBTB', 'KXMLBKS')
PROP_RE = re.compile(r'^(?P<name>.+?):\s*(?P<k>\d+)\+\s*(?P<what>.+?)\?*$')
STATS = {'hits': 'hits', 'hit': 'hits', 'home runs': 'hr', 'home run': 'hr', 'total bases': 'tb', 'total base': 'tb',
         'strikeouts': 'k', 'strikeout': 'k'}
PRE_MINUTES = 30


def norm_name(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', s)
    return re.sub(r'[^a-z]', '', s)


def event_ticker(series: str, start_utc: str, away: str, home: str) -> str:
    t = datetime.fromisoformat(start_utc.replace('Z', '+00:00')).astimezone(ET)
    return f'{series}-{t.year % 100:02d}{MONTHS[t.month - 1]}{t.day:02d}{t.hour:02d}{t.minute:02d}{away}{home}'


def _dollars(mk: dict, key: str):
    v = mk.get(key + '_dollars')
    if v is None:
        c = mk.get(key)
        return None if c is None else float(c) / 100.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_market(mk: dict, series: str, names: dict) -> dict | None:
    """One compact row, or None for a market the page cannot use."""
    if mk.get('status') not in (None, 'active', 'open', 'initialized'):
        return None
    row = {'t': mk.get('ticker'), 'series': series, 'yb': _dollars(mk, 'yes_bid'), 'ya': _dollars(mk, 'yes_ask'),
           'nb': _dollars(mk, 'no_bid'), 'na': _dollars(mk, 'no_ask'), 'last': _dollars(mk, 'last_price'),
           'vol': float(mk.get('volume_fp') or mk.get('volume') or 0)}
    title = str(mk.get('title') or '')
    m = PROP_RE.match(title)
    if m and ':' in title and series in ('KXMLBHIT', 'KXMLBHR', 'KXMLBTB', 'KXMLBKS'):
        stat = STATS.get(m['what'].strip().lower())
        if not stat:
            return None
        row.update(k='prop', stat=stat, n=int(m['k']), name=m['name'].strip(), pid=names.get(norm_name(m['name'])))
        return row
    row.update(k='game', title=title, sub=mk.get('yes_sub_title'), line=mk.get('floor_strike'))
    if series in ('KXMLBGAME', 'KXMLBSPREAD'):
        row['team'] = str(mk.get('ticker') or '').rsplit('-', 1)[-1]
    return row


def box_names(box: dict) -> dict:
    """{normalized name: player id} for the box's hitters and pitchers."""
    out = {}
    for side in ('away', 'home'):
        for kind in ('batting', 'pitching'):
            for r in (box.get('teams') or {}).get(side, {}).get(kind) or []:
                if r.get('name'):
                    out[norm_name(r['name'])] = str(r['player_id'])
        sp = (box.get('starters') or {}).get(side) or {}
        if sp.get('name'):
            out[norm_name(sp['name'])] = str(sp['player_id'])
    return out


def capture_game(fetch_json, game: dict, names: dict, now_iso: str, prev: dict | None = None) -> dict:
    """game: {'game_pk', 'start' (UTC iso), 'away', 'home' (team codes)}. Returns the game's row for ledger['kalshi']."""
    markets, errors, events = [], {}, {}
    for series in SERIES:
        ev = event_ticker(series, game['start'], game['away'], game['home'])
        events[series] = ev
        try:
            doc, _ = fetch_json(f'{API}/markets?event_ticker={ev}&limit=200')
        except Exception as exc:
            errors[series] = type(exc).__name__
            continue
        for mk in (doc or {}).get('markets') or []:
            row = parse_market(mk, series, names)
            if row:
                markets.append(row)
    start = datetime.fromisoformat(str(game['start']).replace('Z', '+00:00'))
    now = datetime.fromisoformat(now_iso.replace('Z', '+00:00'))
    minutes = (start - now).total_seconds() / 60
    out = {'game_pk': game['game_pk'], 'captured_at': now_iso, 'minutes_to_first_pitch': round(minutes, 1), 'markets': markets,
           'events': events, 'errors': errors, 'source': 'Kalshi public API'}
    if minutes >= PRE_MINUTES or not (prev or {}).get('pre'):
        out['pre'] = {'captured_at': now_iso, 'minutes_to_first_pitch': round(minutes, 1), 'markets': markets}
    else:
        out['pre'] = prev['pre']
    return out


def capture(fetch_json, ledger: dict, date_ymd: str, now_iso: str) -> dict:
    """Today's pending games with a box: Kalshi's markets into ledger['kalshi'][game_pk]. Returns a short receipt."""
    kalshi = ledger.setdefault('kalshi', {})
    latest = {}
    for fid, f in (ledger.get('forecasts') or {}).items():
        if f.get('date') != date_ymd:
            continue
        pk = str(f.get('game_pk'))
        if pk not in latest or int(f.get('version') or 0) > int(latest[pk][1].get('version') or 0):
            latest[pk] = (fid, f)
    done = 0; skipped = 0
    for pk, (fid, f) in latest.items():
        state = str(((ledger.get('status') or {}).get(pk) or {}).get('state') or '')
        if pk in (ledger.get('actuals') or {}) or re.search(r'final|progress|live|warmup', state, re.I):
            skipped += 1; continue
        box = (ledger.get('box_scores') or {}).get(fid) or {}
        game = {'game_pk': int(pk), 'start': f['scheduled_start'], 'away': str(f['away'].get('abbr') or ''), 'home': str(f['home'].get('abbr') or '')}
        kalshi[pk] = capture_game(fetch_json, game, box_names(box), now_iso, kalshi.get(pk))
        done += 1
    for pk in list(kalshi):
        if str((kalshi[pk] or {}).get('captured_at') or '')[:10] < date_ymd and pk in (ledger.get('actuals') or {}):
            pass                                                  # kept for EDGE-02's scoring; the page drops them
    return {'at': now_iso, 'date': date_ymd, 'games': done, 'skipped': skipped,
            'markets': sum(len(v.get('markets') or []) for v in kalshi.values() if str(v.get('captured_at') or '')[:10] == date_ymd)}
