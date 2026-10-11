"""The real game, read from the official feed in the same shape as a simulated one.

The page shows a simulated game as a line score, batting and pitching lines and every plate
appearance with its pitches and where the ball went. In-game and after the final it shows the
real game the same way, from the official play-by-play: the same play records, so the same
page code renders both, with the official description of each play instead of generated
wording. Nothing here feeds a forecast; it is display and record keeping only.
"""
from __future__ import annotations

import math

from .pitch_bridge import contact_of

SIDES = ('away', 'home')
BOX_OUTCOME = {'strikeout': 'strikeout', 'strikeout_double_play': 'strikeout', 'strikeout_triple_play': 'strikeout',
               'walk': 'walk', 'intent_walk': 'walk', 'hit_by_pitch': 'hit_by_pitch',
               'single': 'single', 'double': 'double', 'triple': 'triple', 'home_run': 'home_run',
               'field_error': 'other_reach', 'fielders_choice': 'other_reach', 'catcher_interf': 'other_reach'}
BATTER_OUTS = {'field_out', 'force_out', 'grounded_into_double_play', 'fielders_choice_out', 'double_play', 'triple_play',
               'sac_fly', 'sac_bunt', 'sac_fly_double_play', 'sac_bunt_double_play', 'batter_interference'}
# Official pitch result codes folded to the page's vocabulary: B ball, C called strike, S swinging strike,
# F foul, T foul tip, X in play, H hit by pitch.
PITCH_CODE = {'X': 'X', 'D': 'X', 'E': 'X', 'J': 'X', 'W': 'S', 'Q': 'S', 'L': 'F', 'M': 'F', 'R': 'F', 'O': 'T',
              '*B': 'B', 'I': 'B', 'V': 'B', 'P': 'B', 'A': 'C'}
BAT_FIELDS = {'PA': 'plateAppearances', 'AB': 'atBats', 'H': 'hits', '2B': 'doubles', '3B': 'triples', 'HR': 'homeRuns', 'R': 'runs',
              'RBI': 'rbi', 'BB': 'baseOnBalls', 'HBP': 'hitByPitch', 'K': 'strikeOuts', 'SF': 'sacFlies'}
PIT_FIELDS = {'PC': 'numberOfPitches', 'H': 'hits', 'R': 'runs', 'BB': 'baseOnBalls', 'HBP': 'hitBatsmen', 'K': 'strikeOuts',
              'HR': 'homeRuns', 'BF': 'battersFaced', 'ER': 'earnedRuns'}


def _int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _num(value, digits=2):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if v == v else None


def _pitches(play: dict) -> tuple[list, list | None]:
    """[type, mph, result, plate x, plate z] per pitch (feet, catcher's view; None when not measured) and the
    batter's strike zone [top, bottom] in feet when the feed gives it."""
    out, zone = [], None
    for e in play.get('playEvents') or []:
        if e.get('isPitch') is not True:
            continue
        d = e.get('details') or {}
        code = str(d.get('code') or '')
        ptype = str(((d.get('type') or {}).get('code')) or '') or None
        pd_ = e.get('pitchData') or {}
        co = pd_.get('coordinates') or {}
        speed = _num(pd_.get('startSpeed'), 3)
        mph = math.floor(speed + 0.5) if speed is not None else None       # half up, as the page rounds
        px, pz = _num(co.get('pX')), _num(co.get('pZ'))
        if zone is None and _num(pd_.get('strikeZoneTop')) and _num(pd_.get('strikeZoneBottom')):
            zone = [_num(pd_.get('strikeZoneTop')), _num(pd_.get('strikeZoneBottom'))]
        row = [ptype, int(mph) if mph is not None else None, PITCH_CODE.get(code, code)]
        if px is not None and pz is not None:
            row += [px, pz]
        out.append(row)
    return out, zone


def _advance(bases: dict, runners: list) -> None:
    """Apply one play's runner movements to the bases. Movements that share an event (playIndex) happen
    together, so every runner leaves his starting base before anyone is placed: the feed lists the batter
    first on a single that moves the runner from first to third, and a one-by-one pass would wipe the
    batter off first base."""
    groups: dict = {}
    for order, r in enumerate(runners):
        index = (r.get('details') or {}).get('playIndex')
        groups.setdefault(index if isinstance(index, int) else 10_000 + order, []).append(r)
    for index in sorted(groups):
        moves = []
        for r in groups[index]:
            mv = r.get('movement') or {}
            rid = ((r.get('details') or {}).get('runner') or {}).get('id')
            moves.append((mv.get('start'), mv.get('end'), str(rid) if rid is not None else None))
        for start, _, _ in moves:
            if start in bases:
                bases[start] = None
        for _, end, rid in moves:
            if end in bases and rid is not None:
                bases[end] = rid


def plays_from_feed(feed: dict) -> list[dict]:
    """Every completed play as a page play record: inning, half, batter, pitcher, outs before and after,
    runs, score after, scorers, rbi, the real pitches (type, mph, result) and where the ball went."""
    plays = []
    score = {'away': 0, 'home': 0}
    outs = 0
    key = None
    bases = {'1B': None, '2B': None, '3B': None}
    for play in ((feed.get('liveData') or {}).get('plays') or {}).get('allPlays') or []:
        about = play.get('about') or {}
        if not about.get('isComplete'):
            continue
        result = play.get('result') or {}
        m = play.get('matchup') or {}
        batter, pitcher = m.get('batter') or {}, m.get('pitcher') or {}
        inning = _int(about.get('inning'), 1)
        half = 'top' if str(about.get('halfInning') or '').lower() == 'top' else 'bottom'
        if (inning, half) != key:
            key, outs = (inning, half), 0
            bases = {'1B': None, '2B': None, '3B': None}
        bases_before = [bases['1B'], bases['2B'], bases['3B']]
        _advance(bases, play.get('runners') or [])
        event = str(result.get('eventType') or '')
        outcome = BOX_OUTCOME.get(event) or ('bip_out' if event in BATTER_OUTS else 'runner')
        after = {'away': _int(result.get('awayScore'), score['away']), 'home': _int(result.get('homeScore'), score['home'])}
        runs = max(0, after['away'] + after['home'] - score['away'] - score['home'])
        outs_after = _int((play.get('count') or {}).get('outs'), outs)
        if outs_after < outs:
            outs_after = outs
        scorers = []
        for r in play.get('runners') or []:
            if str((r.get('movement') or {}).get('end') or '') == 'score':
                rid = ((r.get('details') or {}).get('runner') or {}).get('id')
                if rid is not None and str(rid) not in scorers:
                    scorers.append(str(rid))
        pitches, zone = _pitches(play)
        if outs_after >= 3:
            bases = {'1B': None, '2B': None, '3B': None}
        contact = contact_of(play)
        if contact is not None:
            # Landing spot of the batted ball in the official field coordinates (home plate near 125, 198).
            last = [e for e in play.get('playEvents') or [] if e.get('isPitch') is True]
            co = ((last[-1].get('hitData') or {}).get('coordinates') or {}) if last else {}
            x, y = _num(co.get('coordX'), 1), _num(co.get('coordY'), 1)
            if x is not None and y is not None:
                contact['x'], contact['y'] = x, y
        plays.append({'inning': inning, 'half': half, 'batter_id': str(batter.get('id') or ''), 'batter_name': str(batter.get('fullName') or ''),
                      'pitcher_id': str(pitcher.get('id') or ''), 'pitcher_name': str(pitcher.get('fullName') or ''),
                      'box_outcome': outcome, 'event': event, 'description': str(result.get('description') or '').strip(),
                      'outs_before': outs, 'outs_after': min(outs_after, 3), 'runs_scored': runs,
                      'away_score': after['away'], 'home_score': after['home'],
                      'bases_before': bases_before, 'bases_after': [bases['1B'], bases['2B'], bases['3B']], 'scoring_players': scorers,
                      'rbi': _int(result.get('rbi'), 0), 'estimated_pitches': len(pitches), 'pitches': pitches, 'zone': zone,
                      'contact': contact, 'official': True})
        score, outs = after, min(outs_after, 3)
    return plays


def box_from_feed(feed: dict) -> dict:
    """Batting and pitching lines and the line score so far, read leniently (an in-progress game has
    partial lines); the final's strict version with its cross-checks is boxscore.parse_actual_box."""
    live = feed.get('liveData') or {}
    ls = live.get('linescore') or {}
    out = {'score': {s: _int(((ls.get('teams') or {}).get(s) or {}).get('runs'), 0) for s in SIDES},
           'batting': {}, 'pitching': {}, 'innings': {s: {} for s in SIDES}}
    for item in ls.get('innings') or []:
        for s in SIDES:
            cell = item.get(s) or {}
            if 'runs' in cell and item.get('num') is not None:
                out['innings'][s][str(item['num'])] = {'R': _int(cell.get('runs')), 'H': _int(cell.get('hits'))}
    teams = (live.get('boxscore') or {}).get('teams') or {}
    for s in SIDES:
        team = teams.get(s) or {}
        players = team.get('players') or {}
        bat, pit = [], []
        for pid in team.get('batters') or []:
            p = players.get('ID' + str(pid)) or {}
            stats = (p.get('stats') or {}).get('batting') or {}
            if not stats and not p.get('battingOrder'):
                continue          # listed but never in the lineup; a lineup spot not yet up shows zeros
            bat.append({'player_id': str(pid), 'name': str((p.get('person') or {}).get('fullName') or ''), 'spot': _int(p.get('battingOrder'), 0) // 100,
                        'order': _int(p.get('battingOrder'), 0),
                        **{k: _int(stats.get(v), 0) for k, v in BAT_FIELDS.items()}})
        for pid in team.get('pitchers') or []:
            p = players.get('ID' + str(pid)) or {}
            stats = (p.get('stats') or {}).get('pitching') or {}
            if not stats:
                continue
            ip = str(stats.get('inningsPitched') or '0.0').split('.')
            outs = _int(stats.get('outs'), -1)
            if outs < 0:
                outs = 3 * _int(ip[0]) + _int(ip[1] if len(ip) > 1 else 0)
            pit.append({'player_id': str(pid), 'name': str((p.get('person') or {}).get('fullName') or ''), 'outs': outs,
                        **{k: _int(stats.get(v), 0) for k, v in PIT_FIELDS.items()}})
        out['batting'][s], out['pitching'][s] = bat, pit
    return out
