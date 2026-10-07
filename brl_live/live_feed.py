"""Read an in-progress official game feed into an engine matchup and a LiveStart.

Everything here is observation of the public feed: current lineups (with substitutions),
the pitchers used, the pitcher on the mound and his line so far, runners, outs and score.
Pitcher roles, expected workloads and handedness fall back to the prior-date PA history.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from research_lab.game_sim.models import GameMatchup, PitcherProfile, PlayerProfile, TeamProfile
from .live_sim import LiveStart, PitcherSoFar

SIDES = ('away', 'home')


def _pid(x):
    return None if x is None else str(x.get('id') if isinstance(x, dict) else x)


def _player(feed, pid):
    return (feed['gameData'].get('players') or {}).get('ID' + str(pid)) or {}


def _bats(feed, pid):
    return ((_player(feed, pid).get('batSide') or {}).get('code') or 'R')[0].upper()


def _throws(feed, pid):
    return ((_player(feed, pid).get('pitchHand') or {}).get('code') or 'R')[0].upper()


def _name(feed, pid):
    return _player(feed, pid).get('fullName') or str(pid)


def appearances(history: pd.DataFrame) -> pd.DataFrame:
    """One row per game and pitcher from the PA history: date, team, batters faced, entry inning, role facts."""
    h = history
    top = h['inning_topbot'].astype(str).str.lower().str.startswith('top')
    h = h.assign(fielding_team=np.where(top, h['home_team'], h['away_team']))
    first_ab = h.groupby(['game_pk', 'fielding_team'])['at_bat_number'].transform('min')
    last_ab = h.groupby(['game_pk', 'fielding_team'])['at_bat_number'].transform('max')
    h = h.assign(_first=h.groupby(['game_pk', 'pitcher'])['at_bat_number'].transform('min') == first_ab,
                 _finish=h.groupby(['game_pk', 'pitcher'])['at_bat_number'].transform('max') == last_ab)
    return h.groupby(['game_pk', 'pitcher']).agg(
        date=('date_key', 'first'), team=('fielding_team', 'first'), bf=('at_bat_number', 'size'),
        entry_inning=('inning', 'min'), throws=('p_throws', 'first'), start=('_first', 'first'),
        finished=('_finish', 'first')).reset_index()


# Reliever availability from recent use (RELIEF-01). The manager scores candidates with 0.70 x rest and
# treats rest at or below 0.05 as unavailable, so: a rested arm 1.0; one appearance yesterday 0.7 (0.5 when
# it was a long one, seven batters or more); both of the two days before yesterday 0.85; yesterday and the
# day before (a third straight day today) unavailable, as are three straight days. Off until the 2026
# replay has measured it (tools/replay_params.json rest: true); off, every reliever is rested.
RELIEVER_REST = False
LONG_OUTING_BF = 7


def reliever_rest(app: pd.DataFrame, pid: int, cutoff: str) -> float:
    """How available a reliever is on the cutoff date, from his appearances on the three days before it."""
    day = pd.Timestamp(cutoff)
    window = (day - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    recent = app[(app['pitcher'] == pid) & (app['date'] < cutoff) & (app['date'] >= window)]
    if recent.empty:
        return 1.0
    ago = {}
    for d, bf in zip(recent['date'], recent['bf']):
        k = int((day - pd.Timestamp(str(d)[:10])).days)
        ago[k] = max(int(bf), ago.get(k, 0))
    if 1 in ago and 2 in ago:
        return 0.0
    if 1 in ago:
        return 0.5 if ago[1] >= LONG_OUTING_BF else 0.7
    if 2 in ago and 3 in ago:
        return 0.85
    return 1.0


def reliever_profile(app: pd.DataFrame, pid: int, throws: str, name: str, cutoff: str, rest: bool | None = None) -> PitcherProfile:
    year_start = (pd.Timestamp(cutoff) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    sg = app[(app['pitcher'] == pid) & (~app['start']) & (app['date'] < cutoff) & (app['date'] >= year_start)]
    late = float((sg['entry_inning'] >= 8).mean()) if len(sg) else 0.0
    ninth = float((sg['entry_inning'] >= 9).mean()) if len(sg) else 0.0
    exp_bf = int(max(3, round(sg['bf'].median()))) if len(sg) else 4
    role = 'closer' if ninth >= 0.6 else 'setup' if late >= 0.5 else 'long' if exp_bf >= 7 else 'reliever'
    use_rest = RELIEVER_REST if rest is None else rest
    return PitcherProfile(str(pid), name, throws, role=role, leverage=min(1.0, 0.3 + late),
                          rest=reliever_rest(app, pid, cutoff) if use_rest else 1.0,
                          expected_batters=exp_bf, max_batters=max(exp_bf + 3, 6))


def starter_profile(app: pd.DataFrame, pid: int, throws: str, name: str, cutoff: str) -> PitcherProfile:
    year_start = (pd.Timestamp(cutoff) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    starts = app[(app['pitcher'] == pid) & app['start'] & (app['date'] < cutoff) & (app['date'] >= year_start)]['bf']
    exp_bf = int(round(starts.median())) if len(starts) else 22
    max_bf = int(round(starts.quantile(0.95))) if len(starts) >= 5 else 27
    return PitcherProfile(str(pid), name, throws, role='starter', stamina=0.7, expected_batters=exp_bf,
                          max_batters=max(max_bf, exp_bf + 2))


def parse_live_state(feed: dict) -> dict:
    """Pure read of the linescore and boxscore: who is where, right now."""
    live = feed['liveData']; ls = live['linescore']; box = live['boxscore']['teams']
    inning = int(ls.get('currentInning') or 1)
    state = str(ls.get('inningState') or ('Top' if ls.get('isTopInning', True) else 'Bottom'))
    if state == 'Middle':
        half, outs, bases, between = 'bottom', 0, (None, None, None), True
    elif state == 'End':
        half, outs, bases, between, inning = 'top', 0, (None, None, None), True, inning + 1
    else:
        half = 'top' if state == 'Top' else 'bottom'
        outs = int(ls.get('outs') or 0); between = False
        off = ls.get('offense') or {}
        bases = (_pid(off.get('first')), _pid(off.get('second')), _pid(off.get('third')))
        if outs >= 3:
            half, outs, bases, between = ('bottom' if half == 'top' else 'top'), 0, (None, None, None), True
            if half == 'top':
                inning += 1
    teams = ls.get('teams') or {}
    score = {s: int((teams.get(s) or {}).get('runs') or 0) for s in SIDES}
    batting_side = 'away' if half == 'top' else 'home'
    orders = {s: [str(x) for x in (box[s].get('battingOrder') or [])] for s in SIDES}
    pitchers_used = {s: [str(x) for x in (box[s].get('pitchers') or [])] for s in SIDES}
    off, dfn = ls.get('offense') or {}, ls.get('defense') or {}
    due = {}
    for s in SIDES:
        src = off if s == batting_side else dfn
        batter = _pid(src.get('batter'))
        spot = int(src.get('battingOrder') or 0)
        idx = orders[s].index(batter) if batter in orders[s] else (spot - 1 if 1 <= spot <= 9 else 0)
        due[s] = idx
    current = {}
    for s in SIDES:
        fielding = (s != batting_side)
        pid = _pid(dfn.get('pitcher')) if fielding else None
        if not pid and pitchers_used[s]:
            pid = pitchers_used[s][-1]
        current[s] = pid
    lines = {}
    for s in SIDES:
        players = box[s].get('players') or {}
        for pid in pitchers_used[s]:
            st = ((players.get('ID' + pid) or {}).get('stats') or {}).get('pitching') or {}
            ip = str(st.get('inningsPitched') or '0.0').split('.')
            outs_recorded = int(st.get('outs')) if st.get('outs') is not None else int(ip[0]) * 3 + int(ip[1] if len(ip) > 1 else 0)
            lines[pid] = PitcherSoFar(pid, batters_faced=int(st.get('battersFaced') or 0), outs_recorded=outs_recorded,
                                      runs_allowed=int(st.get('runs') or 0), earned_runs=int(st.get('earnedRuns') or 0),
                                      hits_allowed=int(st.get('hits') or 0), walks_hbp=int(st.get('baseOnBalls') or 0) + int(st.get('hitByPitch') or 0),
                                      strikeouts=int(st.get('strikeOuts') or 0), home_runs=int(st.get('homeRuns') or 0),
                                      is_starter=(pid == pitchers_used[s][0]))
    status = feed['gameData']['status']
    return {'inning': inning, 'half': half, 'outs': outs, 'bases': bases, 'between_innings': between,
            'away_score': score['away'], 'home_score': score['home'], 'batting_side': batting_side,
            'lineups': orders, 'pitchers_used': pitchers_used, 'current_pitcher': current, 'due_up': due, 'lines': lines,
            'bullpen_available': {s: [str(x) for x in (box[s].get('bullpen') or [])] for s in SIDES},
            'detailed_state': status.get('detailedState'), 'abstract_state': status.get('abstractGameState'),
            'balls': int(ls.get('balls') or 0), 'strikes': int(ls.get('strikes') or 0),
            'innings': [{'num': int(i.get('num')), **{s: {'R': (i.get(s) or {}).get('runs'), 'H': (i.get(s) or {}).get('hits')} for s in SIDES}}
                        for i in (ls.get('innings') or []) if i.get('num') is not None]}


def live_matchup(feed: dict, history: pd.DataFrame, date: str, app: pd.DataFrame | None = None):
    """GameMatchup and LiveStart for an in-progress game. Returns (matchup, start, state, game)."""
    state = parse_live_state(feed)
    app = appearances(history) if app is None else app
    gd = feed['gameData']; teams = {}
    for s in SIDES:
        order = state['lineups'][s]
        if len(order) != 9:
            raise ValueError(f'{s} lineup has {len(order)} spots')
        lineup = tuple(PlayerProfile(pid, _name(feed, pid), _bats(feed, pid)) for pid in order)
        used = state['pitchers_used'][s]
        if not used:
            raise ValueError(f'{s} has no pitcher on record yet')
        sp = used[0]
        starter = starter_profile(app, int(sp), _throws(feed, sp), _name(feed, sp), date)
        pen_ids = [p for p in used[1:]] + [p for p in state['bullpen_available'][s] if p not in used]
        pen = tuple(reliever_profile(app, int(p), _throws(feed, p), _name(feed, p), date) for p in dict.fromkeys(pen_ids))
        t = gd['teams'][s]
        teams[s] = TeamProfile(str(t.get('id')), t.get('name') or t.get('abbreviation') or s, lineup, starter, pen)
    matchup = GameMatchup(away=teams['away'], home=teams['home'], venue=gd['teams']['home'].get('abbreviation') or 'Unknown Park',
                          game_type=str((gd.get('game') or {}).get('type') or 'R'))
    start = LiveStart(inning=state['inning'], half=state['half'], outs=state['outs'], bases=tuple(state['bases']),
                      away_score=state['away_score'], home_score=state['home_score'],
                      away_lineup_index=state['due_up']['away'], home_lineup_index=state['due_up']['home'],
                      current_pitcher=state['current_pitcher'], used_pitchers=state['pitchers_used'], lines=state['lines'],
                      automatic_runner_placed=bool(any(state['bases'])) and state['inning'] > 9)
    game = {'game_pk': int(feed['gamePk']), 'date': date, 'game_type': matchup.game_type,
            'away': {'abbr': gd['teams']['away'].get('abbreviation'), 'name': teams['away'].name, 'team_id': teams['away'].team_id},
            'home': {'abbr': gd['teams']['home'].get('abbreviation'), 'name': teams['home'].name, 'team_id': teams['home'].team_id}}
    return matchup, start, state, game


def matchup_parameters(game: dict, matchup: GameMatchup, config: dict) -> dict:
    return {'date': game['date'], 'park': game['home']['abbr'], 'game': game, 'matchup': asdict(matchup), 'config': config}
