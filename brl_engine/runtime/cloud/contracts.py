"""Forecast contracts: pregame inputs from the official feed, the forecast record, its audit rules.

Same record shape as the inherited runtime (brl.live-forecast.v1, the field set the public
checker allows). Inputs for a game come only from the pregame feed and the prior-date history.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.common import content_hash, timestamp
from app.safety import Blocked
from research_lab.game_sim.models import GameMatchup, PlayerProfile, PitcherProfile, TeamProfile
from brl_live.live_feed import appearances, reliever_profile, starter_profile

N = 10000
SEED = 20261006
FORECAST_FIELDS = set('schema scope game_pk date game_type scheduled_start forecast_origin saved_at version lineup_status home away '
                      'n_simulations seed home_win_probability probability_mcse projected_away_runs projected_home_runs '
                      'team_baseline_probability market_probability model history_through postseason_regular_bullpen_logic '
                      'automatic_runner github_run_id snapshot_hash'.split())


def utcnow():
    return datetime.now(timezone.utc)


def config_for(game_type: str) -> dict:
    """Regular season: automatic runner in extras. Postseason: none. Long caps so no world is left unfinished."""
    postseason = str(game_type) != 'R'
    return {'regulation_innings': 9, 'max_innings': 40 if postseason else 30, 'automatic_runner_in_extras': not postseason,
            'three_batter_minimum': True, 'max_plate_appearances': 700 if postseason else 500, 'record_events': False}


def draw_seeds(game_pk: int):
    return np.random.default_rng([SEED, int(game_pk)]).integers(0, np.iinfo(np.int32).max, size=N, dtype=np.int64)


def assert_preview(feed: dict, capture_at) -> None:
    gd = feed['gameData']
    if gd['status']['abstractGameState'] != 'Preview':
        raise Blocked('Game is not pregame')
    start = gd.get('datetime', {}).get('dateTime')
    if not start:
        raise Blocked('Scheduled start missing')
    if timestamp(capture_at) >= timestamp(start):
        raise Blocked('Input captured at or after the scheduled start')


def _player(feed, pid):
    return (feed['gameData'].get('players') or {}).get('ID' + str(pid)) or {}


def _name(feed, pid, names=None):
    return _player(feed, pid).get('fullName') or (names or {}).get(int(pid), None) or str(pid)


def _bats(feed, pid, history_stand=None):
    code = ((_player(feed, pid).get('batSide') or {}).get('code') or '')[:1].upper()
    return code or (history_stand or 'R')


def _throws(feed, pid, fallback='R'):
    return (((_player(feed, pid).get('pitchHand') or {}).get('code') or fallback)[:1]).upper()


def projected_order(feed: dict, side: str, history: pd.DataFrame, hand: str | None = None) -> tuple[list, str]:
    """The team's batting order before its lineup is posted, from the prior-date history, by the rule in force
    (brl_live/lineups.py RULE): its most recent lineup, or the lineup it uses against a starter of today's opposing
    starter's hand. Returns (order, source words)."""
    from brl_live import lineups
    abbr = str(feed['gameData']['teams'][side].get('abbreviation') or '').upper()
    if not abbr:
        raise Blocked('Team abbreviation missing')
    prior = lineups.games_from_history(history, abbr)
    if not prior:
        raise Blocked('No prior lineup for ' + abbr)
    today = pd.Timestamp(str(feed['gameData']['datetime']['officialDate'])[:10]).toordinal()
    order, source = lineups.project(prior, today, hand)
    if order is None or len(order) != 9:
        raise Blocked('Prior lineup incomplete for ' + abbr)
    return [int(x) for x in order], source


def likely_opener(app: pd.DataFrame, pool: list, date: str):
    """The pitcher in the pool most likely to open a bullpen game: most starts in the prior 60 days, then the
    longest typical outing; None when the pool is empty."""
    if not pool:
        return None
    window = (pd.Timestamp(date) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    recent = app[(app['date'] < date) & (app['date'] >= window) & app['pitcher'].isin(pool)]
    # Rested arms only: nobody who pitched in the last two days, nobody who started in the last four.
    two = (pd.Timestamp(date) - pd.Timedelta(days=2)).strftime('%Y-%m-%d'); four = (pd.Timestamp(date) - pd.Timedelta(days=4)).strftime('%Y-%m-%d')
    tired = set(recent[recent['date'] >= two]['pitcher']) | set(recent[(recent['date'] >= four) & recent['start']]['pitcher'])
    rested = recent[~recent['pitcher'].isin(tired)]
    if len(rested):
        recent = rested
    starts = recent[recent['start']].groupby('pitcher').size()
    if len(starts):
        return int(starts.sort_values(ascending=False).index[0])
    length = recent.groupby('pitcher')['bf'].median()
    if len(length):
        return int(length.sort_values(ascending=False).index[0])
    return int(pool[0])


def opener_profile(app: pd.DataFrame, starter: PitcherProfile, date: str) -> PitcherProfile:
    """An assumed opener keeps the starter role (so the fitted hazard runs) but expects an opener's length."""
    year_start = (pd.Timestamp(date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    mine = app[(app['pitcher'] == int(starter.player_id)) & (app['date'] < date) & (app['date'] >= year_start)]
    starts = mine[mine['start']]['bf']
    if len(starts) >= 3:
        exp_bf = int(round(starts.median()))
    else:
        exp_bf = int(max(6, min(12, round(mine['bf'].median() * 1.5)))) if len(mine) else 8
    return PitcherProfile(starter.player_id, starter.name, starter.throws, role='starter', stamina=0.5,
                          expected_batters=exp_bf, max_batters=max(exp_bf + 3, 9))


def live_inputs(feed: dict, receipt: dict, history: pd.DataFrame, names: dict | None = None, defense=None):
    """(game, matchup, notes, statuses, fingerprint) for a pregame feed.

    defense: a research_lab.pa_model.physics.DefenseState (built from the history through the day before)
    when the selected PA model reads the fielding team's defense; each team's value goes into its profile."""
    assert_preview(feed, receipt['finished_at'])
    gd = feed['gameData']
    box = (feed.get('liveData') or {}).get('boxscore', {}).get('teams') or {}
    date = gd['datetime']['officialDate']
    hist = history[history['date_key'].astype(str).str[:10] < date]
    app = appearances(hist)
    stands = hist.groupby('batter')['stand'].agg(lambda s: 'S' if s.nunique() > 1 else s.iloc[0]).to_dict()
    throws_hist = hist.groupby('pitcher')['p_throws'].first().to_dict()
    teams, game_teams, statuses, notes = {}, {}, {}, {'history_through_used': str(hist['date_key'].max()) if len(hist) else None,
                                                     'lineup_source': {}, 'bullpen_source': {}}
    probable = gd.get('probablePitchers') or {}
    sps, assumed_by = {}, {}
    for side in ('away', 'home'):
        t = gd['teams'][side]
        sp = (probable.get(side) or {}).get('id')
        assumed = None
        if sp is None:
            # No probable starter (a bullpen game, or not announced yet): open with the most likely opener and
            # say so. The starter is part of the snapshot hash, so an announcement produces a new version.
            pool = [int(x) for x in ((box.get(side) or {}).get('bullpen') or [])]
            abbr0 = str(t.get('abbreviation') or '').upper()
            if len(pool) < 4:
                window = (pd.Timestamp(date) - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
                team_app = app[app['team'] == abbr0]
                pool = [int(p) for p in team_app[team_app['date'] >= window]['pitcher'].drop_duplicates().tolist()]
                if len(pool) < 4:
                    last_games = team_app.sort_values('date')['game_pk'].drop_duplicates().tail(10)
                    pool = [int(p) for p in team_app[team_app['game_pk'].isin(last_games)]['pitcher'].drop_duplicates().tolist()]
            sp = likely_opener(app, pool, date)
            if sp is None:
                raise Blocked(f'Probable starter not announced for the {side} team and no bullpen to open with')
            assumed = sp
            statuses[side + '_starter'] = 'assumed'
            notes.setdefault('starter_source', {})[side] = 'no probable starter announced; bullpen game assumed, opened by the most likely opener'
        sps[side] = int(sp); assumed_by[side] = assumed
    for side in ('away', 'home'):
        t = gd['teams'][side]
        sp = sps[side]; assumed = assumed_by[side]
        opp = sps['home' if side == 'away' else 'away']
        order = [int(x) for x in ((box.get(side) or {}).get('battingOrder') or [])]
        if len(order) == 9 and len(set(order)) == 9:
            statuses[side] = 'official'; notes['lineup_source'][side] = 'official pregame lineup'
        else:
            order, source = projected_order(feed, side, hist, _throws(feed, opp, throws_hist.get(opp, 'R'))); statuses[side] = 'projected'
            notes['lineup_source'][side] = {'last game': "team's most recent lineup in the prior-date history"}.get(source, "team's " + source + ' in the prior-date history')
        lineup = tuple(PlayerProfile(str(pid), _name(feed, pid, names), _bats(feed, pid, stands.get(pid))) for pid in order)
        starter = starter_profile(app, sp, _throws(feed, sp, throws_hist.get(sp, 'R')), _name(feed, sp, names), date)
        if assumed is not None:
            starter = opener_profile(app, starter, date)
        pen_ids = [int(x) for x in ((box.get(side) or {}).get('bullpen') or []) if int(x) != sp]
        if len(pen_ids) >= 4:
            notes['bullpen_source'][side] = 'official pregame bullpen'
        else:
            abbr = str(t.get('abbreviation') or '').upper()
            window = (pd.Timestamp(date) - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
            recent = app[(app['team'] == abbr) & (app['date'] >= window) & (~app['start']) & (app['pitcher'] != sp)]
            if recent['pitcher'].nunique() < 5:
                last_games = app[app['team'] == abbr].sort_values('date')['game_pk'].drop_duplicates().tail(10)
                recent = app[app['game_pk'].isin(last_games) & (app['team'] == abbr) & (~app['start']) & (app['pitcher'] != sp)]
            pen_ids = [int(p) for p in recent['pitcher'].drop_duplicates().tolist()]
            notes['bullpen_source'][side] = 'relievers used in the prior 14 days (history)'
        if not pen_ids:
            raise Blocked(f'No bullpen available for the {side} team')
        pen = tuple(reliever_profile(app, pid, _throws(feed, pid, throws_hist.get(pid, 'R')), _name(feed, pid, names), date) for pid in dict.fromkeys(pen_ids))
        team_defense = float(defense.value(str(t.get('abbreviation') or '').upper())) if defense is not None else 0.0
        teams[side] = TeamProfile(str(t['id']), str(t.get('name') or t.get('abbreviation')), lineup, starter, pen, defense=team_defense)
        if defense is not None:
            notes.setdefault('team_defense', {})[side] = round(team_defense, 5)
        game_teams[side] = {'abbr': str(t.get('abbreviation') or ''), 'name': str(t.get('name') or ''), 'team_id': int(t['id']),
                            'starter': {'player_id': str(sp), 'name': starter.name}}
    game_type = str((gd.get('game') or {}).get('type') or 'R')
    matchup = GameMatchup(away=teams['away'], home=teams['home'], venue=game_teams['home']['abbr'] or 'Unknown Park', game_type=game_type)
    game = {'game_pk': int(feed['gamePk']), 'date': date, 'game_type': game_type, 'scheduled_start': gd['datetime']['dateTime'],
            'away': game_teams['away'], 'home': game_teams['home'], 'venue': (gd.get('venue') or {}).get('name')}
    fingerprint = content_hash({'lineups': {s: [p.player_id for p in teams[s].lineup] for s in ('away', 'home')},
                                'starters': {s: teams[s].starter.player_id for s in ('away', 'home')},
                                'bullpens': {s: sorted(p.player_id for p in teams[s].bullpen) for s in ('away', 'home')},
                                'statuses': statuses, 'game_type': game_type})
    return game, matchup, notes, statuses, fingerprint


def verify_completed(results) -> None:
    for r in results:
        if r.winner not in ('away', 'home') or getattr(r, 'ended_by_plate_appearance_cap', False):
            raise Blocked('Unfinished simulated world')


def assert_finished_before_start(game: dict, guard_feed: dict, finished_at) -> None:
    gd = guard_feed['gameData']
    if guard_feed.get('gamePk') != game['game_pk']:
        raise Blocked('Guard feed is another game')
    if gd['status']['abstractGameState'] != 'Preview':
        raise Blocked('Game started before the forecast finished')
    if timestamp(finished_at) >= timestamp(gd['datetime']['dateTime']):
        raise Blocked('Forecast finished after the scheduled start')


def summarize(game, statuses, results, origin, finished, baseline, notes, snapshot_hash, version, run_id) -> dict:
    verify_completed(results)
    n = len(results)
    if n != N:
        raise Blocked('Forecast needs %d worlds' % N)
    home = sum(1 for r in results if r.winner == 'home')
    p = home / n
    cfg = config_for(game['game_type'])
    return {'schema': 'brl.live-forecast.v1', 'scope': 'saved_before_game_pending_publication_and_actual_first_pitch_audit',
            'game_pk': int(game['game_pk']), 'date': game['date'], 'game_type': game['game_type'], 'scheduled_start': game['scheduled_start'],
            'forecast_origin': str(origin), 'saved_at': str(finished), 'version': int(version), 'lineup_status': dict(statuses),
            'home': {k: game['home'][k] for k in ('abbr', 'name', 'team_id')}, 'away': {k: game['away'][k] for k in ('abbr', 'name', 'team_id')},
            'n_simulations': n, 'seed': SEED, 'home_win_probability': round(p, 4), 'probability_mcse': math.sqrt(p * (1 - p) / n),
            'projected_away_runs': round(float(np.mean([r.away_score for r in results])), 4),
            'projected_home_runs': round(float(np.mean([r.home_score for r in results])), 4),
            'team_baseline_probability': None if baseline is None or baseline.get('probability') is None else float(baseline['probability']), 'market_probability': None,
            'model': getattr(results[0], 'provider_name', 'locked-pa-2026-v1'), 'history_through': notes.get('history_through_used'),
            'postseason_regular_bullpen_logic': game['game_type'] != 'R', 'automatic_runner': bool(cfg['automatic_runner_in_extras']),
            'github_run_id': str(run_id), 'snapshot_hash': snapshot_hash}


ESSENTIAL_FIELDS = {'game_pk', 'date', 'home_win_probability', 'saved_at', 'scheduled_start', 'away', 'home', 'version'}


def check_public(f: dict) -> None:
    """Only public forecast fields may reach the page; the essentials must be present."""
    if not set(f) <= FORECAST_FIELDS or not ESSENTIAL_FIELDS <= set(f):
        raise Blocked('Forecast fields differ from the public record')
    p = f['home_win_probability']
    if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise Blocked('Invalid public probability')
    if 'n_simulations' in f and f['n_simulations'] != N:
        raise Blocked('Not a full forecast')
    if 'forecast_origin' in f and not timestamp(f['forecast_origin']) <= timestamp(f['saved_at']) < timestamp(f['scheduled_start']):
        raise Blocked('Forecast timing invalid')


def score_versions(forecasts: dict, publications: dict, actuals: dict) -> dict:
    """Last version published before the observed first pitch, per final game: Brier for the model, team baseline and market."""
    by_game = {}
    for ident, f in forecasts.items():
        by_game.setdefault(str(f['game_pk']), []).append((ident, f))
    rows = []
    for pk, actual in actuals.items():
        fp = actual.get('first_pitch_observed_at')
        eligible = []
        for ident, f in by_game.get(str(pk), []):
            pub = publications.get(ident)
            if not pub or not fp:
                continue
            if timestamp(pub['published_at']) < timestamp(fp):
                eligible.append((timestamp(pub['published_at']), ident, f))
        if not eligible:
            continue
        _, ident, f = max(eligible, key=lambda x: x[0])
        y = 1.0 if actual['home'] > actual['away'] else 0.0
        rows.append({'game_pk': int(pk), 'forecast_id': ident, 'version': f.get('version'), 'home_won': bool(y),
                     'model': (f['home_win_probability'] - y) ** 2,
                     'team': None if f.get('team_baseline_probability') is None else (f['team_baseline_probability'] - y) ** 2,
                     'market': None if f.get('market_probability') is None else (f['market_probability'] - y) ** 2})

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None
    return {'n_games': len(rows), 'model_brier': mean('model'), 'team_brier': mean('team'), 'market_brier': mean('market'),
            'version_scores': rows, 'policy': 'last version published before the observed first pitch; nothing backfilled'}
