"""The real game from the official feed, in the page's own play and box shapes, and the win table from simulated games."""
import numpy as np
from brl_live.real_game import plays_from_feed, box_from_feed
from brl_live.win_table import WinTable, lookup, state_index, SHAPE


def play(inning, half, idx, batter, pitcher, event, desc, away, home, outs, pitches, runners=(), complete=True, hit=None):
    events = []
    for k, (ptype, mph, code) in enumerate(pitches):
        e = {'isPitch': True, 'details': {'code': code, 'type': {'code': ptype}}, 'pitchData': {'startSpeed': mph, 'strikeZoneTop': 3.4, 'strikeZoneBottom': 1.6,
             'coordinates': {'pX': -0.5 + 0.25 * k, 'pZ': 2.0 + 0.3 * k}}}
        events.append(e)
    if hit and events:
        events[-1]['hitData'] = hit
    return {'about': {'atBatIndex': idx, 'inning': inning, 'halfInning': half, 'isComplete': complete},
            'result': {'eventType': event, 'description': desc, 'awayScore': away, 'homeScore': home, 'rbi': 1 if event == 'home_run' else 0},
            'matchup': {'batter': {'id': batter, 'fullName': 'Batter ' + str(batter)}, 'pitcher': {'id': pitcher, 'fullName': 'Pitcher ' + str(pitcher)}},
            'count': {'balls': 0, 'strikes': 0, 'outs': outs}, 'playEvents': events,
            'runners': [{'movement': {'start': (str(rid).split('@') + [None])[1], 'end': end}, 'details': {'runner': {'id': str(rid).split('@')[0]}}} for rid, end in runners]}


def feed():
    plays = [play(1, 'top', 0, 1, 50, 'strikeout', 'Batter 1 strikes out swinging.', 0, 0, 1, [('FF', 95.4, 'C'), ('SL', 86.1, 'S'), ('FF', 96.0, 'W')]),
             play(1, 'top', 1, 2, 50, 'single', 'Batter 2 singles on a line drive to center fielder X.', 0, 0, 1, [('CH', 84.2, 'D')],
                  runners=[(2, '1B')], hit={'launchSpeed': 101.3, 'launchAngle': 12.0, 'totalDistance': 250.0, 'trajectory': 'line_drive', 'location': '8', 'coordinates': {'coordX': 126.0, 'coordY': 95.5}}),
             play(1, 'top', 2, 3, 50, 'home_run', 'Batter 3 homers (1) on a fly ball to left field. Batter 2 scores.', 2, 0, 1, [('FF', 94.0, 'E')],
                  runners=[('2@1B', 'score'), (3, 'score')], hit={'launchSpeed': 105.0, 'launchAngle': 28.0, 'totalDistance': 402.0, 'trajectory': 'fly_ball', 'location': '7'}),
             play(1, 'top', 3, 4, 50, 'caught_stealing_2b', 'Batter 4 ... caught stealing 2nd base.', 2, 0, 3, [('FF', 93.0, 'B')]),
             play(1, 'bottom', 4, 11, 60, 'field_out', 'Batter 11 grounds out to shortstop.', 2, 0, 1, [('SI', 92.0, '*B'), ('SI', 92.5, 'X')],
                  hit={'launchSpeed': 88.0, 'launchAngle': -5.0, 'trajectory': 'ground_ball', 'location': '6'}),
             play(1, 'bottom', 5, 12, 60, 'walk', 'Batter 12 walks.', 2, 0, 1, [('FF', 93.0, 'B')] * 4, complete=False)]
    box = {'teams': {'away': {'batters': [1, 2], 'pitchers': [60], 'players': {
                'ID1': {'person': {'fullName': 'Batter 1'}, 'battingOrder': '100', 'stats': {'batting': {'plateAppearances': 1, 'atBats': 1, 'strikeOuts': 1}}},
                'ID2': {'person': {'fullName': 'Batter 2'}, 'battingOrder': '200', 'stats': {'batting': {'plateAppearances': 1, 'atBats': 1, 'hits': 1, 'runs': 1}}},
                'ID60': {'person': {'fullName': 'Pitcher 60'}, 'stats': {'pitching': {'inningsPitched': '0.1', 'numberOfPitches': 6, 'battersFaced': 2}}}}},
                     'home': {'batters': [], 'pitchers': [50], 'players': {'ID50': {'person': {'fullName': 'Pitcher 50'}, 'stats': {'pitching': {'inningsPitched': '1.0', 'outs': 3, 'hits': 2, 'runs': 2, 'homeRuns': 1, 'strikeOuts': 1, 'numberOfPitches': 6, 'battersFaced': 4}}}}}}}
    linescore = {'teams': {'away': {'runs': 2}, 'home': {'runs': 0}}, 'innings': [{'num': 1, 'away': {'runs': 2, 'hits': 2}, 'home': {}}]}
    return {'gamePk': 7, 'gameData': {'status': {'abstractGameState': 'Live'}}, 'liveData': {'plays': {'allPlays': plays}, 'boxscore': box, 'linescore': linescore}}


def test_plays_follow_the_official_feed():
    plays = plays_from_feed(feed())
    assert [p['box_outcome'] for p in plays] == ['strikeout', 'single', 'home_run', 'runner', 'bip_out']   # the open plate appearance is not listed
    k, single, hr, cs, out = plays
    assert k['pitches'] == [['FF', 95, 'C', -0.5, 2.0], ['SL', 86, 'S', -0.25, 2.3], ['FF', 96, 'S', 0.0, 2.6]] and k['estimated_pitches'] == 3 and k['official'] is True
    assert k['zone'] == [3.4, 1.6]
    assert single['contact'] == {'t': 'L', 'loc': 8, 'dist': 250, 'ev': 101, 'x': 126.0, 'y': 95.5} and single['pitches'][0][:3] == ['CH', 84, 'X']
    assert 'x' not in hr['contact']     # no coordinates on that one
    assert hr['runs_scored'] == 2 and hr['away_score'] == 2 and hr['scoring_players'] == ['2', '3'] and hr['rbi'] == 1
    assert (k['outs_before'], k['outs_after'], single['outs_before'], cs['outs_before'], cs['outs_after']) == (0, 1, 1, 1, 3)
    assert out['inning'] == 1 and out['half'] == 'bottom' and out['outs_before'] == 0 and out['pitches'][0][2] == 'B'
    # a pitch without measured coordinates keeps the three-field form
    bare = plays_from_feed({'liveData': {'plays': {'allPlays': [{'about': {'inning': 1, 'halfInning': 'top', 'isComplete': True}, 'result': {'eventType': 'walk'},
                                                                 'matchup': {}, 'playEvents': [{'isPitch': True, 'details': {'code': 'B'}}]}]}}})
    assert bare[0]['pitches'] == [[None, None, 'B']] and bare[0]['zone'] is None
    assert out['description'].startswith('Batter 11 grounds out') and out['contact']['loc'] == 6 and out['contact']['t'] == 'G'
    # runners: the single put batter 2 on first, the homer cleared the bases, the caught stealing ended the inning
    assert single['bases_before'] == [None, None, None] and single['bases_after'] == ['2', None, None]
    assert hr['bases_before'] == ['2', None, None] and hr['bases_after'] == [None, None, None]
    assert cs['bases_after'] == [None, None, None] and out['bases_before'] == [None, None, None]


def test_box_lines_are_read_leniently_in_progress():
    box = box_from_feed(feed())
    assert box['score'] == {'away': 2, 'home': 0} and box['innings']['away'] == {'1': {'R': 2, 'H': 2}} and box['innings']['home'] == {}
    assert [b['player_id'] for b in box['batting']['away']] == ['1', '2'] and box['batting']['away'][1]['H'] == 1 and box['batting']['away'][1]['HR'] == 0
    assert box['pitching']['home'][0]['outs'] == 3 and box['pitching']['home'][0]['PC'] == 6
    assert box['pitching']['away'][0]['outs'] == 1 and box['pitching']['away'][0]['spot' if False else 'BF'] == 2


def simulated_box(home_won, lead_path):
    """A simulated game whose plays walk through the given (inning, half, outs, bases, home lead) states."""
    plays = []
    for inning, half, outs, bases, lead in lead_path:
        plays.append({'inning': inning, 'half': half, 'outs_before': outs, 'bases_before': bases, 'runs_scored': 0,
                      'home_score': max(lead, 0), 'away_score': max(-lead, 0)})
    return {'score': {'home': 1 if home_won else 0, 'away': 0 if home_won else 1}, 'plays': plays}


def test_win_table_counts_states_and_shrinks_sparse_ones():
    t = WinTable()
    path = [(1, 'top', 0, [None] * 3, 0), (5, 'bottom', 1, ['x', None, None], 2), (9, 'top', 2, [None, 'y', 'z'], -1)]
    for i in range(60):
        t.add(simulated_box(home_won=i % 3 != 0, lead_path=path))                 # home wins two thirds
    for i in range(40):
        t.add(simulated_box(home_won=i % 2 == 0, lead_path=[(1, 'top', 0, [None] * 3, 0)]))   # a coin flip from the start
    table = t.finish()
    assert table['shape'] == list(SHAPE) and len(table['cells']) == int(np.prod(SHAPE)) and table['n_games'] == 100
    assert table['states_reached'] == 3 and abs(table['pregame_home'] - 0.6) < 1e-9
    # a well-visited state is near its own share (60 games at 2/3 shrunk slightly toward the pool)
    assert 0.62 <= lookup(table, 5, 'bottom', 1, ['x', None, None], 2) <= 0.68
    assert 0.55 <= lookup(table, 1, 'top', 0, [None] * 3, 0) <= 0.62
    # an unvisited state falls back to its pooled neighbours, never to zero
    assert 0.3 < lookup(table, 7, 'top', 1, [None] * 3, 0) < 0.8
    assert lookup(table, 12, 'bottom', 2, [None, 'y', 'z'], -9) == lookup(table, 9, 'bottom', 2, [None, 'y', 'z'], -6)
    assert state_index(12, 'bottom', 5, ['a', 'b', 'c'], 9) == (8, 1, 2, 7, 12)


def test_live_update_keeps_the_real_game_even_when_the_continuation_fails(tmp_path):
    from cloud.runner import LocalStore
    from cloud.security import key_bytes
    from brl_live.box_runner import BoxRunner

    class Sim:
        def update_live(self, feed, date, now):
            raise RuntimeError('no continuation today')

    class Net:
        def json(self, url):
            raise AssertionError('not used')

    store = LocalStore(tmp_path, key_bytes('ab' * 32))
    runner = BoxRunner(Net(), store, Sim(), run_id='t')
    runner.live_update(7, feed(), '2026-10-07')
    snap = store.ledger['live']['7']
    assert snap['error'].startswith('RuntimeError') and len(snap['plays']) == 5 and snap['box']['score'] == {'away': 2, 'home': 0}
    assert snap['plays'][2]['box_outcome'] == 'home_run' and 'plays_error' not in snap


def test_runners_who_move_together_leave_their_bases_before_anyone_is_placed():
    from brl_live.real_game import _advance
    bases = {'1B': '5', '2B': None, '3B': None}
    # the feed lists the batter first: single, runner from first to third, same event
    _advance(bases, [{'movement': {'start': None, 'end': '1B'}, 'details': {'runner': {'id': 9}, 'playIndex': 3}},
                     {'movement': {'start': '1B', 'end': '3B'}, 'details': {'runner': {'id': 5}, 'playIndex': 3}}])
    assert bases == {'1B': '9', '2B': None, '3B': '5'}
    # a steal earlier in the plate appearance, then a double that scores him and puts the batter on second
    bases = {'1B': '5', '2B': None, '3B': None}
    _advance(bases, [{'movement': {'start': None, 'end': '2B'}, 'details': {'runner': {'id': 9}, 'playIndex': 4}},
                     {'movement': {'start': '2B', 'end': 'score'}, 'details': {'runner': {'id': 5}, 'playIndex': 4}},
                     {'movement': {'start': '1B', 'end': '2B'}, 'details': {'runner': {'id': 5}, 'playIndex': 1}}])
    assert bases == {'1B': None, '2B': '9', '3B': None}
    # a force at second: the runner from first is out, the batter reaches first
    bases = {'1B': '5', '2B': None, '3B': None}
    _advance(bases, [{'movement': {'start': None, 'end': '1B'}, 'details': {'runner': {'id': 9}, 'playIndex': 2}},
                     {'movement': {'start': '1B', 'end': None, 'isOut': True}, 'details': {'runner': {'id': 5}, 'playIndex': 2}}])
    assert bases == {'1B': '9', '2B': None, '3B': None}


def test_live_box_shows_lineup_spots_not_up_yet():
    f = feed()
    f['liveData']['boxscore']['teams']['away']['batters'].append(3)
    f['liveData']['boxscore']['teams']['away']['players']['ID3'] = {'person': {'fullName': 'Batter 3'}, 'battingOrder': '300', 'stats': {'batting': {}}}
    f['liveData']['boxscore']['teams']['away']['batters'].append(44)
    f['liveData']['boxscore']['teams']['away']['players']['ID44'] = {'person': {'fullName': 'Bench'}, 'stats': {'batting': {}}}
    rows = box_from_feed(f)['batting']['away']
    assert [r['player_id'] for r in rows] == ['1', '2', '3'] and rows[2]['AB'] == 0 and rows[2]['spot'] == 3
