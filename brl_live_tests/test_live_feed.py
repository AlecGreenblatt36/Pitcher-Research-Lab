import copy
import pandas as pd
import pytest
from brl_live.live_feed import parse_live_state, live_matchup, appearances, matchup_parameters


def players(ids, bats='R', throws='R'):
    return {'ID' + str(i): {'id': i, 'fullName': f'Player {i}', 'batSide': {'code': bats}, 'pitchHand': {'code': throws}} for i in ids}


def feed(inning_state='Top', outs=1):
    away_order = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    home_order = [201, 202, 203, 204, 205, 206, 207, 208, 209]
    ps = {}
    ps.update(players(away_order + home_order))
    ps.update(players([301, 302, 303, 401, 402, 403], throws='L'))
    def pitching(bf, outs, runs, k, bb, hits=3, hr=0):
        return {'battersFaced': bf, 'outs': outs, 'inningsPitched': f'{outs // 3}.{outs % 3}', 'runs': runs, 'earnedRuns': runs, 'hits': hits,
                'baseOnBalls': bb, 'hitByPitch': 0, 'strikeOuts': k, 'homeRuns': hr, 'numberOfPitches': bf * 4}
    return {
        'gamePk': 777, 'gameData': {
            'status': {'abstractGameState': 'Live', 'detailedState': 'In Progress'},
            'teams': {'away': {'id': 1, 'name': 'Away Nine', 'abbreviation': 'AWY'}, 'home': {'id': 2, 'name': 'Home Nine', 'abbreviation': 'HOM'}},
            'players': ps, 'game': {'type': 'D'}, 'datetime': {'officialDate': '2026-10-07'}},
        'liveData': {
            'linescore': {'currentInning': 6, 'inningState': inning_state, 'isTopInning': inning_state == 'Top', 'outs': outs, 'balls': 2, 'strikes': 1,
                          'teams': {'away': {'runs': 3, 'hits': 7}, 'home': {'runs': 2, 'hits': 5}},
                          'innings': [{'num': i, 'away': {'runs': 1 if i == 1 else 0, 'hits': 1}, 'home': {'runs': 0, 'hits': 1}} for i in range(1, 6)],
                          'offense': {'batter': {'id': 104}, 'battingOrder': 4, 'first': {'id': 103}, 'third': {'id': 102}, 'team': {'id': 1}},
                          'defense': {'pitcher': {'id': 402}, 'batter': {'id': 207}, 'battingOrder': 7, 'team': {'id': 2}}},
            'boxscore': {'teams': {
                'away': {'battingOrder': away_order, 'pitchers': [301], 'bullpen': [302, 303], 'bench': [],
                         'players': {'ID301': {'stats': {'pitching': pitching(22, 15, 2, 6, 1)}}}},
                'home': {'battingOrder': home_order, 'pitchers': [401, 402], 'bullpen': [403], 'bench': [],
                         'players': {'ID401': {'stats': {'pitching': pitching(20, 14, 3, 4, 2, hr=1)}},
                                     'ID402': {'stats': {'pitching': pitching(2, 1, 0, 1, 0, hits=0)}}}}}}}}


def history():
    rows = []
    for g in range(1, 13):
        date = f'2026-09-{g:02d}'
        for ab, (pitcher, inning, top) in enumerate([(301, 1, 'Bot'), (301, 2, 'Bot'), (302, 8, 'Bot'), (303, 9, 'Bot'), (401, 1, 'Top'), (401, 2, 'Top'), (402, 7, 'Top'), (403, 9, 'Top')]):
            rows.append({'game_pk': g, 'date_key': date, 'at_bat_number': ab + 1, 'pitcher': pitcher, 'inning': inning, 'inning_topbot': top,
                         'home_team': 'AWY' if top == 'Bot' else 'HOM', 'away_team': 'HOM' if top == 'Bot' else 'AWY', 'p_throws': 'L',
                         'outcome': 'K', 'batter': 100 + ab, 'stand': 'R'})
    return pd.DataFrame(rows)


def test_state_mid_inning():
    s = parse_live_state(feed())
    assert s['inning'] == 6 and s['half'] == 'top' and s['outs'] == 1 and s['bases'] == ('103', None, '102')
    assert s['away_score'] == 3 and s['home_score'] == 2 and s['batting_side'] == 'away' and not s['between_innings']
    assert s['due_up'] == {'away': 3, 'home': 6}
    assert s['current_pitcher'] == {'away': '301', 'home': '402'}
    assert s['pitchers_used'] == {'away': ['301'], 'home': ['401', '402']}
    assert s['lines']['401'].outs_recorded == 14 and s['lines']['401'].walks_hbp == 2 and s['lines']['401'].home_runs == 1
    assert s['lines']['402'].batters_faced == 2 and s['lines']['301'].is_starter
    assert s['bullpen_available'] == {'away': ['302', '303'], 'home': ['403']}
    assert len(s['innings']) == 5 and s['innings'][0]['away']['R'] == 1


def test_between_innings_moves_to_the_next_half():
    s = parse_live_state(feed('Middle', outs=3))
    assert s['half'] == 'bottom' and s['inning'] == 6 and s['outs'] == 0 and s['bases'] == (None, None, None) and s['between_innings']
    s = parse_live_state(feed('End', outs=3))
    assert s['half'] == 'top' and s['inning'] == 7 and s['outs'] == 0


def test_live_matchup_and_start():
    m, start, state, game = live_matchup(feed(), history(), '2026-10-07')
    assert [p.player_id for p in m.away.lineup] == [str(i) for i in range(101, 110)]
    assert m.home.starter.player_id == '401' and {p.player_id for p in m.home.bullpen} == {'402', '403'}
    assert {p.player_id for p in m.away.bullpen} == {'302', '303'} and m.away.starter.throws == 'L'
    roles = {p.player_id: p.role for p in m.away.bullpen}
    assert roles['303'] == 'closer' and roles['302'] == 'setup'
    assert start.current_pitcher == {'away': '301', 'home': '402'} and start.outs == 1 and start.bases == ('103', None, '102')
    assert start.lines['401'].outs_recorded == 14
    assert game['home']['abbr'] == 'HOM' and game['game_type'] == 'D' and game['game_pk'] == 777
    p = matchup_parameters(game, m, {'automatic_runner_in_extras': False})
    assert p['park'] == 'HOM' and p['matchup']['away']['starter']['player_id'] == '301'
    start.validate(m)


def test_bad_lineup_is_rejected():
    f = feed(); f['liveData']['boxscore']['teams']['home']['battingOrder'] = [201, 202]
    with pytest.raises(ValueError):
        live_matchup(f, history(), '2026-10-07')


def test_run_live_update_end_to_end():
    from research_lab.game_sim.engine import GameSimulator
    from research_lab.game_sim.models import SimulationConfig
    from brl_live.live_update import run_live_update

    class Provider:
        name = 'test_only'; validation_status = 'synthetic'
        def probabilities(self, c):
            return dict(zip(('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach'), (.42, .22, .09, .15, .06, .04, .02)))
    m, start, state, game = live_matchup(feed(), history(), '2026-10-07')
    engine = GameSimulator(Provider(), SimulationConfig(max_innings=100, max_plate_appearances=4000))
    out = run_live_update(engine, m, start, state, game, None, n_worlds=300, updated_at='2026-10-07T01:00:00+00:00')
    assert out['n_worlds'] == 300 and 0.5 < out['home_win_probability'] < 0.95 or 0.05 < out['home_win_probability'] < 0.5
    assert out['projected_final']['away'] >= 3 and out['projected_final']['home'] >= 2
    assert out['state']['inning'] == 6 and out['on_mound'] == {'away': '301', 'home': '402'}
    assert out['due_up'] == {'away': 'Player 104', 'home': 'Player 207'}
    assert out['adjustments'] == ['context offsets through 2026-09-27']
    again = run_live_update(engine, m, start, state, game, None, n_worlds=300, updated_at='x')
    assert again['home_win_probability'] == out['home_win_probability']
