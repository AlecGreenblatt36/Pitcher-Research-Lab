"""Several sportsbooks' lines for one MLB date from SportsBookReview's public odds pages.

ESPN's feed carries one book (DraftKings). On 2022-2025 closing lines the average of the books SportsBookReview lists
(vig removed per book) was a better forecast than DraftKings alone in every season (MKT-05 in LEDGER.md), so the
market reference uses that consensus when it can be read. Parsing follows the page's embedded __NEXT_DATA__ JSON
(the same structure github.com/ArnavSaraogi/mlb-odds-scraper reads). One page per date and market type.
"""
from __future__ import annotations

import json
import re

from .market import vig_free_home

BASE = 'https://www.sportsbookreview.com/betting-odds/mlb-baseball'
NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL)


def page_url(date_ymd: str, kind: str = 'moneyline') -> str:
    if kind == 'moneyline':
        return f'{BASE}/?date={date_ymd}'
    if kind == 'totals':
        return f'{BASE}/totals/full-game/?date={date_ymd}'
    raise ValueError(kind)


def _int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def parse_page(html: str, kind: str = 'moneyline') -> list[dict]:
    """Games on the page with each book's opening and current line."""
    m = NEXT_DATA.search(html or '')
    if not m:
        return []
    data = json.loads(m.group(1))
    tables = ((data.get('props') or {}).get('pageProps') or {}).get('oddsTables') or []
    if not tables:
        return []
    out = []
    for row in ((tables[0].get('oddsTableModel') or {}).get('gameRows') or []):
        gv = row.get('gameView') or {}
        books = {}
        for ov in row.get('oddsViews') or []:
            if not ov:
                continue
            name = str(ov.get('sportsbook') or '').lower()
            entry = {}
            for phase, key in (('open', 'openingLine'), ('current', 'currentLine')):
                ln = ov.get(key) or {}
                if kind == 'moneyline':
                    h, a = _int(ln.get('homeOdds')), _int(ln.get('awayOdds'))
                    if h is not None and a is not None:
                        entry[phase] = [h, a]
                else:
                    t, o, u = _num(ln.get('total')), _int(ln.get('overOdds')), _int(ln.get('underOdds'))
                    if t is not None and o is not None and u is not None:
                        entry[phase] = [t, o, u]
            if entry:
                books[name] = entry
        out.append({'home': (gv.get('homeTeam') or {}).get('fullName'), 'away': (gv.get('awayTeam') or {}).get('fullName'),
                    'start': gv.get('startDate'), 'status': gv.get('gameStatusText'), 'books': books})
    return out


def consensus_home(books: dict, phase: str = 'current'):
    """Mean vig-free home win chance over the books with both prices, and how many books."""
    vals = []
    for entry in books.values():
        ln = entry.get(phase)
        if ln and abs(ln[0]) >= 100 and abs(ln[1]) >= 100:
            vals.append(vig_free_home(ln[0], ln[1]))
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def consensus_total(books: dict, phase: str = 'current'):
    """The most common total among the books and the mean vig-free chance of the over at that total."""
    lines = [entry[phase] for entry in books.values() if entry.get(phase)]
    if not lines:
        return None, None, 0
    totals = [ln[0] for ln in lines]
    common = max(set(totals), key=totals.count)
    at = [vig_free_home(ln[1], ln[2]) for ln in lines if ln[0] == common and abs(ln[1]) >= 100 and abs(ln[2]) >= 100]
    return common, (sum(at) / len(at) if at else None), len(at)


ALIASES = {'Oakland Athletics': 'Athletics', 'Sacramento Athletics': 'Athletics', "Oakland A's": 'Athletics', 'Cleveland Indians': 'Cleveland Guardians'}
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'


def _norm(name):
    name = str(name or '').strip()
    return ALIASES.get(name, name)


def fetch_html(url: str, timeout: int = 30) -> str:
    from urllib.request import Request, urlopen
    with urlopen(Request(url, headers={'User-Agent': UA, 'Accept': 'text/html', 'Accept-Language': 'en-US,en;q=0.9'}), timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')


def consensus_for(games: list, date_ymd: str, fetch=fetch_html) -> dict:
    """game_pk -> {'p_home_cons', 'n_books', 'books'} for schedule games (dicts with game_pk, start, home/away names)."""
    from .market import _minutes_apart
    page = parse_page(fetch(page_url(date_ymd, 'moneyline')), 'moneyline')
    out, used = {}, set()
    for g in games:
        home, away = _norm((g.get('home') or {}).get('name')), _norm((g.get('away') or {}).get('name'))
        cands = [i for i, x in enumerate(page) if i not in used and _norm(x['home']) == home and _norm(x['away']) == away]
        if not cands:
            continue
        best = min(cands, key=lambda i: _minutes_apart(page[i]['start'], g.get('start')))
        if len(cands) > 1 and _minutes_apart(page[best]['start'], g.get('start')) > 180:
            continue
        used.add(best)
        p, n = consensus_home(page[best]['books'], 'current')
        if p is not None:
            out[g['game_pk']] = {'p_home_cons': round(p, 4), 'n_books': n, 'books': sorted(page[best]['books'])}
    return out
