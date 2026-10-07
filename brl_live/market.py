"""Pregame betting-market reference for the track record.

Moneylines are read from ESPN's public scoreboard (provider as named there), converted to a
vig-free home win probability (each side's implied probability divided by their sum), and
kept per game as the last capture made while the game was still pregame, with its clock. The
record page scores the market only when the kept capture precedes the observed first pitch.
This is a reference line to judge the model against, not an input to any forecast.
"""
from __future__ import annotations

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


def _moneylines(competition: dict):
    for odds in competition.get('odds') or []:
        h = (odds.get('homeTeamOdds') or {}).get('moneyLine')
        a = (odds.get('awayTeamOdds') or {}).get('moneyLine')
        if h is not None and a is not None:
            return int(h), int(a), str((odds.get('provider') or {}).get('name') or 'ESPN'), odds.get('overUnder')
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
            out.append({'start': event.get('date') or comp.get('date'), 'state': state, 'away': teams['away'], 'home': teams['home'],
                        'home_ml': ml[0], 'away_ml': ml[1], 'provider': ml[2], 'over_under': ml[3], 'p_home': round(vig_free_home(ml[0], ml[1]), 4)})
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
                      'source': 'ESPN public scoreboard'}
        captured += 1
    return {'date': date_ymd, 'schedule_games': len(games), 'lines_seen': len(lines), 'matched': len(matched), 'captured_pregame': captured}
