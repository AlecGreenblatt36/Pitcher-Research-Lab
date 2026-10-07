import numpy as np
import pandas as pd
import pytest
from brl_live.pitch_bridge import PitchBridge, pitch_list, sequences_from_feed, count_group
from brl_live.boxscore import BookkeepingFit


def play(event, codes, pitcher=1, stand='R', types=None, speeds=None):
    events, b, s = [], 0, 0
    for i, c in enumerate(codes):
        events.append({'isPitch': True, 'details': {'code': c, 'type': {'code': (types or ['FF'] * len(codes))[i]}},
                       'pitchData': {'startSpeed': (speeds or [95.0] * len(codes))[i]}, 'count': {}})
        if c == 'B': b += 1
        elif c in ('C', 'S') or (c == 'F' and s < 2): s += 1
        events[-1]['count'] = {'balls': b, 'strikes': s}
    return {'result': {'eventType': event}, 'about': {'isComplete': True, 'atBatIndex': 0},
            'matchup': {'pitcher': {'id': pitcher}, 'batSide': {'code': stand}}, 'playEvents': events}


def mapper(ev):
    return {'strikeout': 'K', 'walk': 'BB_HBP', 'hit_by_pitch': 'BB_HBP', 'single': '1B', 'field_out': 'BIP_OUT', 'home_run': 'HR', 'double': '2B_3B'}.get(ev)


def feed(plays):
    return {'liveData': {'plays': {'allPlays': plays}}}


def test_pitch_list_reads_counts_and_codes():
    p = pitch_list(play('strikeout', ['C', 'B', 'F', 'S']))
    assert [x['r'] for x in p] == ['C', 'B', 'F', 'S'] and (p[3]['b'], p[3]['s']) == (1, 2)
    assert pitch_list({'playEvents': [{'isPitch': True, 'details': {'code': '??'}}]}) is None


def test_bridge_keeps_only_consistent_paths_and_draws_by_outcome():
    recs = sequences_from_feed(feed([
        play('strikeout', ['C', 'S', 'S']), play('strikeout', ['B', 'C', 'F', 'F', 'S']), play('strikeout', ['B', 'B']),   # last is inconsistent
        play('walk', ['B', 'B', 'B', 'B']), play('walk', ['B', 'C', 'B', 'B', 'B']), play('hit_by_pitch', ['B', 'H']),
        play('single', ['F', 'X']), play('field_out', ['X']), play('home_run', ['B', 'B', 'X'])]), mapper)
    br = PitchBridge(recs)
    assert br.n_sequences == 8
    rng = np.random.default_rng(1)
    for _ in range(20):
        k = br.draw('99', 'strikeout', 'R', False, rng); assert k[-1][2] in ('C', 'S', 'T') and len(k) >= 3
        w = br.draw('99', 'bb_hbp', 'R', False, rng); assert w[-1][2] == 'B' and len(w) >= 4
        h = br.draw('99', 'bb_hbp', 'R', True, rng); assert h[-1][2] == 'H'
        x = br.draw('99', 'single', 'L', False, rng); assert x[-1][2] == 'X'
    assert br.draw('99', 'double_triple', 'R', False, rng) is None     # no prior double/triple paths at all
    assert all(p[0] == 'FF' and 90 <= p[1] <= 100 for p in k)


def test_pitcher_own_mix_and_speed_when_enough_pitches():
    base = [play('field_out', ['B', 'C', 'X'], pitcher=7, types=['SL', 'SL', 'SL'], speeds=[84, 84, 84]) for _ in range(12)]
    others = [play('field_out', ['X'], pitcher=8, types=['FF'], speeds=[97]) for _ in range(40)]
    br = PitchBridge(sequences_from_feed(feed(base + others), mapper))
    rng = np.random.default_rng(2)
    own = [br.draw('7', 'bip_out', 'R', False, rng) for _ in range(30)]
    assert all(p[0] == 'SL' and 80 <= p[1] <= 88 for seq in own for p in seq)
    league = [br.draw('123', 'bip_out', 'R', False, rng) for _ in range(30)]
    assert any(p[0] == 'FF' for seq in league for p in seq)


def test_count_groups():
    assert count_group(0, 0) == 'first' and count_group(3, 2) == 'full' and count_group(0, 2) == 'two_strikes'
    assert count_group(3, 0) == 'three_balls' and count_group(1, 2) == 'two_strikes' and count_group(2, 0) == 'behind' and count_group(1, 1) == 'even'


def test_bookkeeping_uses_sequences_when_present():
    rows = []
    for i in range(6):
        for ev, codes in [('strikeout', ['C', 'S', 'S']), ('walk', ['B', 'B', 'B', 'B']), ('single', ['X']), ('field_out', ['F', 'X']), ('home_run', ['B', 'X']), ('hit_by_pitch', ['H']), ('double', ['C', 'X'])]:
            for hand in ('R', 'L'):
                p = play(ev, codes, stand=hand)
                rows.append({'date_key': '2026-10-01', 'outcome': mapper(ev), 'pitcher': 1, 'stand': hand, 'terminal_event': ev,
                             'pitch_number': len(codes), 'pitches': pitch_list(p)})
    h = pd.DataFrame(rows)
    fit = BookkeepingFit(h, '2026-10-06')
    assert fit.bridge.n_sequences == len(rows)
    rng = np.random.default_rng(0)
    hbp, pc, pitches = fit.draw({'pitcher_id': '1', 'outcome': 'strikeout', 'batter_hand': 'R'}, rng)
    assert pc == 3 and pitches[-1][2] == 'S'
    hbp, pc, pitches = fit.draw({'pitcher_id': '1', 'outcome': 'double_triple', 'batter_hand': 'R'}, rng)
    assert pc == 2 and pitches[-1][2] == 'X'
    h2 = h[h.outcome != '2B_3B'].copy()
    fit2 = BookkeepingFit(h2, '2026-10-06')
    with pytest.raises(Exception):
        fit2.draw({'pitcher_id': '1', 'outcome': 'double_triple', 'batter_hand': 'R'}, rng)   # no path and no count pool
