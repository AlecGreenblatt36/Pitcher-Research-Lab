"""Final result of an official game feed: runs, team ids, observed first pitch."""
from __future__ import annotations

from .common import timestamp
from .safety import Blocked


class ResultUnavailable(ValueError):
    pass


def _runs(value):
    if value is None:
        raise ResultUnavailable('Runs missing')
    return int(value)


def first_pitch_time(feed: dict):
    """Earliest pitch event time in the feed, or None."""
    times = []
    for play in (feed.get('liveData', {}).get('plays', {}).get('allPlays') or []):
        for e in play.get('playEvents') or []:
            if e.get('isPitch') is True and e.get('startTime'):
                times.append(str(e['startTime']))
                break
        if times:
            break
    if not times:
        for play in (feed.get('liveData', {}).get('plays', {}).get('allPlays') or []):
            st = (play.get('about') or {}).get('startTime')
            if st:
                times.append(str(st)); break
    return timestamp(times[0]).isoformat() if times else None


def parse_final(feed: dict, game_pk: int, fetched_at: str) -> dict:
    gd = feed['gameData']
    if feed.get('gamePk') != game_pk:
        raise Blocked('Wrong game identity')
    if gd['status']['abstractGameState'] != 'Final':
        raise ResultUnavailable('Game is not final')
    teams = feed['liveData']['linescore']['teams']
    return {'game_pk': int(game_pk), 'away': _runs((teams.get('away') or {}).get('runs')), 'home': _runs((teams.get('home') or {}).get('runs')),
            'team_ids': {'away': int(gd['teams']['away']['id']), 'home': int(gd['teams']['home']['id'])},
            'first_pitch_observed_at': first_pitch_time(feed), 'fetched_at': str(fetched_at),
            'official_date': (gd.get('datetime') or {}).get('officialDate'), 'detailed_state': gd['status'].get('detailedState')}
