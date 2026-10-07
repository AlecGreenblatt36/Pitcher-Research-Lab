"""Pregame betting-market reference for the track record.

Moneylines are read from ESPN's public scoreboard (provider as named there), converted to a
vig-free home win probability (each side's implied probability divided by their sum), and
kept per game as the last capture made while the game was still pregame, with its clock. The
record page scores the market only when the kept capture precedes the observed first pitch.
This is a reference line to judge the model against, not an input to any forecast.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date}&limit=50'
MLB_SCHEDULE = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=team'
ABBR = {'ARI': 'AZ', 'CHW': 'CWS', 'OAK': 'ATH'}  # ESPN -> MLB where they differ


def implied(ml) -> float:
    ml = float(ml)
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def vig_free_home(home_ml, away_ml) -> float:
    h, a = implied(home_ml), implied(away_ml)
    return h / (h + a)


def _ml_value(v):
    """A moneyline as an int from ESPN's several encodings: -150, "+130", "EVEN", {"odds": "-150"}, {"close": {"odds": ...}}."""
    if v is None:
        return None
    if isinstance(v, dict):
        for key in ('close', 'current', 'open'):
            inner = v.get(key)
            if inner is not None:
                got = _ml_value(inner)
                if got is not None:
                    return got
        for key in ('odds', 'value', 'moneyLine', 'american'):
            if key in v:
                return _ml_value(v[key])
        return None
    s = str(v).strip().upper().replace('+', '')
    if s in ('EVEN', 'EV', 'PK'):
        return 100
    try:
        return int(float(s))
    except ValueError:
        return None


def _moneylines(competition: dict):
    for odds in competition.get('odds') or []:
        provider = str((odds.get('provider') or {}).get('name') or 'ESPN')
        ml = odds.get('moneyline') or {}
        h = _ml_value(ml.get('home')) if isinstance(ml, dict) else None
        a = _ml_value(ml.get('away')) if isinstance(ml, dict) else None
        if h is None or a is None:
            h = _ml_value((odds.get('homeTeamOdds') or {}).get('moneyLine'))
            a = _ml_value((odds.get('awayTeamOdds') or {}).get('moneyLine'))
        if h is None or a is None:
            d = str(odds.get('details') or '')      # e.g. "LAD -150"
            m = re.match(r'^([A-Z]{2,4})\s+([+-]?\d+)$', d)
            if m:
                fav_abbr, line = m.group(1), int(m.group(2))
                teams = {c.get('homeAway'): str((c.get('team') or {}).get('abbreviation') or '').upper() for c in competition.get('competitors') or []}
                if teams.get('home') == fav_abbr:
                    h, a = line, None
                elif teams.get('away') == fav_abbr:
                    a, h = line, None
        if h is not None and a is not None:
            ou = odds.get('overUnder')
            if ou is None and isinstance(odds.get('total'), dict):
                ou = _ml_value((odds['total'].get('over') or {}).get('close') if isinstance(odds['total'].get('over'), dict) else None)
            return int(h), int(a), provider, ou
    return None


def _total_line(v):
    """A total from ESPN's encodings: 7.5, "7.5", "o7.5", "u8"."""
    if v is None:
        return None
    s = str(v).strip().lower().lstrip('ou')
    try:
        return float(s)
    except ValueError:
        return None


def _totals(competition: dict):
    """(total, over price, under price) from the first odds entry that has them, else None."""
    for odds in competition.get('odds') or []:
        line = _total_line(odds.get('overUnder'))
        over, under = _ml_value(odds.get('overOdds')), _ml_value(odds.get('underOdds'))
        tot = odds.get('total') if isinstance(odds.get('total'), dict) else {}
        o = tot.get('over') if isinstance(tot.get('over'), dict) else {}
        u = tot.get('under') if isinstance(tot.get('under'), dict) else {}
        for key in ('close', 'current'):
            if line is None and isinstance(o.get(key), dict):
                line = _total_line(o[key].get('line'))
            if over is None and isinstance(o.get(key), dict):
                over = _ml_value(o[key].get('odds'))
            if under is None and isinstance(u.get(key), dict):
                under = _ml_value(u[key].get('odds'))
        if line is not None:
            return line, over, under
    return None


def parse_scoreboard(doc: dict) -> list:
    out = []
    for event in doc.get('events') or []:
        for comp in event.get('competitions') or []:
            teams = {}
            for c in comp.get('competitors') or []:
                side = c.get('homeAway')
                team = c.get('team') or {}
                if side in ('home', 'away'):
                    abbr = str(team.get('abbreviation') or '').upper()
                    teams[side] = {'name': team.get('displayName') or team.get('name'), 'abbr': ABBR.get(abbr, abbr)}
            ml = _moneylines(comp)
            if len(teams) != 2 or ml is None:
                continue
            state = ((comp.get('status') or {}).get('type') or {}).get('state')
            tot = _totals(comp)
            line = {'start': event.get('date') or comp.get('date'), 'state': state, 'away': teams['away'], 'home': teams['home'],
                    'home_ml': ml[0], 'away_ml': ml[1], 'provider': ml[2], 'over_under': ml[3] if ml[3] is not None else (tot[0] if tot else None),
                    'p_home': round(vig_free_home(ml[0], ml[1]), 4)}
            if tot and tot[1] is not None and tot[2] is not None:
                line.update(over_odds=int(tot[1]), under_odds=int(tot[2]), p_over=round(vig_free_home(tot[1], tot[2]), 4))
            out.append(line)
    return out


def schedule_games(doc: dict) -> list:
    games = []
    for day in doc.get('dates') or []:
        for g in day.get('games') or []:
            t = g.get('teams') or {}
            games.append({'game_pk': int(g['gamePk']), 'start': g.get('gameDate'),
                          'state': ((g.get('status') or {}).get('abstractGameState')),
                          'away': {'name': (t.get('away') or {}).get('team', {}).get('name'), 'abbr': (t.get('away') or {}).get('team', {}).get('abbreviation')},
                          'home': {'name': (t.get('home') or {}).get('team', {}).get('name'), 'abbr': (t.get('home') or {}).get('team', {}).get('abbreviation')}})
    return games


def _minutes_apart(a, b) -> float:
    try:
        x = datetime.fromisoformat(str(a).replace('Z', '+00:00')); y = datetime.fromisoformat(str(b).replace('Z', '+00:00'))
        return abs((x - y).total_seconds()) / 60.0
    except (TypeError, ValueError):
        return 1e9


def match(games: list, lines: list) -> dict:
    """game_pk -> market line, matched by team names (or abbreviations), nearest start for doubleheaders."""
    out = {}
    for g in games:
        candidates = [l for l in lines if (l['home']['name'] == g['home']['name'] and l['away']['name'] == g['away']['name'])
                      or (l['home']['abbr'] == str(g['home']['abbr'] or '').upper() and l['away']['abbr'] == str(g['away']['abbr'] or '').upper())]
        if not candidates:
            continue
        best = min(candidates, key=lambda l: _minutes_apart(l['start'], g['start']))
        if len(candidates) > 1 and _minutes_apart(best['start'], g['start']) > 180:
            continue
        out[g['game_pk']] = best
    return out


def capture(fetch_json, date_ymd: str, ledger: dict, now_iso: str) -> dict:
    """Store the latest pregame market line per game in ledger['market']; returns a short receipt."""
    market = ledger.setdefault('market', {})
    schedule, _ = fetch_json(MLB_SCHEDULE.format(date=date_ymd))
    games = schedule_games(schedule)
    board, _ = fetch_json(ESPN_SCOREBOARD.format(date=date_ymd.replace('-', '')))
    lines = parse_scoreboard(board)
    matched = match(games, lines)
    captured = 0
    for g in games:
        pk = str(g['game_pk'])
        line = matched.get(g['game_pk'])
        if line is None or g['state'] != 'Preview' or line['state'] not in (None, 'pre'):
            continue
        market[pk] = {'game_pk': g['game_pk'], 'date': date_ymd, 'captured_at': now_iso, 'provider': line['provider'],
                      'home_ml': line['home_ml'], 'away_ml': line['away_ml'], 'p_home': line['p_home'], 'over_under': line['over_under'],
                      'over_odds': line.get('over_odds'), 'under_odds': line.get('under_odds'), 'p_over': line.get('p_over'),
                      'source': 'ESPN public scoreboard'}
        captured += 1
    # Structure-only diagnostics (key names, no values) so a changed feed shape can be read from the ledger.
    events = board.get('events') or []
    comp0 = ((events[0].get('competitions') or [{}])[0]) if events else {}
    odds0 = (comp0.get('odds') or [{}])[0] if comp0 else {}
    return {'date': date_ymd, 'schedule_games': len(games), 'lines_seen': len(lines), 'matched': len(matched), 'captured_pregame': captured,
            'events': len(events), 'competition_keys': sorted(comp0.keys())[:40], 'odds_keys': sorted(odds0.keys())[:40],
            'odds_subkeys': {k: sorted(v.keys())[:20] for k, v in odds0.items() if isinstance(v, dict)},
            'moneyline_shape': {k: (sorted(v.keys())[:10] if isinstance(v, dict) else type(v).__name__) for k, v in (odds0.get('moneyline') or {}).items()} if isinstance(odds0.get('moneyline'), dict) else None,
            'moneyline_home_shape': {k: (sorted(v.keys())[:10] if isinstance(v, dict) else type(v).__name__) for k, v in ((odds0.get('moneyline') or {}).get('home') or {}).items()} if isinstance((odds0.get('moneyline') or {}).get('home'), dict) else None}
