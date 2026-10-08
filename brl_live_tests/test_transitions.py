"""Base-running transitions read from the official play-by-play (tools/brl_transitions.py): the state at contact and
where each runner ended up, with running events earlier in the plate appearance kept apart."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('brl_transitions', Path(__file__).resolve().parents[1] / 'tools' / 'brl_transitions.py')
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)


def mv(rid, start, end, index, out=False, etype='single'):
    return {'movement': {'start': start, 'end': end, 'isOut': out}, 'details': {'runner': {'id': rid}, 'playIndex': index, 'eventType': etype}}


def play(inning, top, batter, event, outs_after, runners, n_pitches=1):
    events = [{'isPitch': True, 'index': i} for i in range(n_pitches)]
    return {'about': {'inning': inning, 'isTopInning': top, 'isComplete': True}, 'result': {'eventType': event},
            'matchup': {'batter': {'id': batter}}, 'count': {'outs': outs_after}, 'playEvents': events, 'runners': runners}


def test_contact_state_and_destinations():
    plays = [
        play(1, True, 1, 'single', 0, [mv(1, None, '1B', 0)]),
        # runner 1 steals second on the second pitch, then the batter doubles him home on the fourth
        play(1, True, 2, 'double', 0, [mv(1, '1B', '2B', 1, etype='stolen_base_2b'), mv(2, None, '2B', 3, etype='double'), mv(1, '2B', 'score', 3, etype='double')], n_pitches=4),
        # single: the runner from second holds at third
        play(1, True, 3, 'single', 0, [mv(3, None, '1B', 0), mv(2, '2B', '3B', 0)]),
        # ground ball double play: runner from first and the batter out, the runner from third scores
        play(1, True, 4, 'grounded_into_double_play', 2, [mv(4, None, None, 0, out=True, etype='grounded_into_double_play'), mv(3, '1B', None, 0, out=True, etype='grounded_into_double_play'),
                                                         mv(2, '3B', 'score', 0, etype='grounded_into_double_play')]),
        play(1, True, 5, 'strikeout', 3, [mv(5, None, None, 0, out=True, etype='strikeout')]),
        play(1, False, 11, 'walk', 0, [mv(11, None, '1B', 0, etype='walk')]),
        # caught stealing ends nothing here: a running play without a plate-appearance result
        play(1, False, 12, 'caught_stealing_2b', 1, [mv(11, '1B', None, 0, out=True, etype='caught_stealing_2b')]),
        play(1, False, 12, 'field_out', 2, [mv(12, None, None, 0, out=True, etype='field_out')]),
    ]
    rows = tr.game_rows({'allPlays': plays})
    c = [(r[0], r[2], r[3], r[4], r[5]) for r in rows['contact']]
    assert c == [('1B', 0, 0, '---1', 0),           # empty bases, batter to first
                 ('2B_3B', 2, 0, '-H-2', 0),         # at contact the runner was on second (stole it), scored; batter on second
                 ('1B', 2, 0, '-3-1', 0),            # runner on second held at third
                 ('BIP_OUT', 5, 0, 'X-HX', 2),       # first and third: double play, run scores
                 ('K', 0, 2, '---X', 1),
                 ('BB_HBP', 0, 0, '---1', 0),
                 ('BIP_OUT', 0, 1, '---X', 1)]       # after the caught stealing: bases empty, one out
    assert rows['mismatch'] == 0 and rows['plays'] == 7
    # the steal is recorded as a running event before contact: runner on first, no outs -> runner on second
    assert rows['running'][1] == (1, 0, ('stolen_base_2b',), 2, 0)
    assert all(r[2] == () for i, r in enumerate(rows['running']) if i != 1)


def test_decisions_by_speed_bucket():
    speed = {'7': 0.75, '8': 0.2}
    row = ('1B', 'single', 3, 1, '3H-1', 0, ('7', '8', None), '9')        # runner from 2nd (slow) scores, from 1st... none: 1st occupied with 2nd
    d = tr.decisions(row, speed)
    assert ('single_from_2nd', 1, '0', 'H') in d and not any(x[0] == 'single_from_1st' for x in d)
    row = ('BIP_OUT', 'grounded_into_double_play', 1, 0, 'X--X', 2, ('7', None, None), '8')
    d = tr.decisions(row, speed)
    assert ('double_play_runner', 0, '5', 'dp') in d and ('double_play_batter', 0, '0', 'dp') in d
    assert tr.bucket(None) == 'na' and tr.bucket(0.5) == '3' and tr.bucket(0.29) == '0'


def test_pinch_runners_and_the_extra_inning_runner_are_followed_by_base():
    plays = [
        play(1, True, 1, 'single', 0, [mv(1, None, '1B', 0)]),
        # a pinch runner (id 99) replaced runner 1 between plays; the double scores him from first
        play(1, True, 2, 'double', 0, [mv(2, None, '2B', 0, etype='double'), mv(99, '1B', 'score', 0, etype='double')]),
        # tenth inning: the automatic runner (id 77) starts on second; a single moves him to third
        play(10, True, 3, 'single', 0, [mv(3, None, '1B', 0), mv(77, '2B', '3B', 0)]),
        play(10, True, 4, 'sac_fly', 1, [mv(4, None, None, 0, out=True, etype='sac_fly'), mv(77, '3B', 'score', 0, etype='sac_fly')]),
    ]
    rows = tr.game_rows({'allPlays': plays})
    c = [(r[0], r[2], r[3], r[4], r[5]) for r in rows['contact']]
    assert c == [('1B', 0, 0, '---1', 0), ('2B_3B', 1, 0, 'H--2', 0), ('1B', 2, 0, '-3-1', 0), ('BIP_OUT', 5, 0, '1-HX', 1)]
    assert rows['mismatch'] == 0
    # the official runners after a play replace the tracked ones when the feed lists them
    p = play(1, True, 5, 'walk', 0, [mv(5, None, '1B', 0, etype='walk')])
    p['matchup'].update({'postOnFirst': {'id': 5}, 'postOnSecond': {'id': 42}})
    rows = tr.game_rows({'allPlays': [p, play(1, True, 6, 'field_out', 1, [mv(6, None, None, 0, out=True, etype='field_out')])]})
    assert rows['contact'][1][2] == 3 and rows['contact'][1][6][:2] == ('5', '42')
