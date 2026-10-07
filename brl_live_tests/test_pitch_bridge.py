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
    return {'strikeout': 'K', 'walk': 'BB_HBP', 'hit_by_pitch': 'BB_HBP', 'single': '1B', 'field_out': 'BIP_OUT', 'home_run': 'HR', 'double': '2B_3B',
            'triple': '2B_3B', 'grounded_into_double_play': 'BIP_OUT', 'double_play': 'BIP_OUT', 'sac_fly': 'BIP_OUT', 'force_out': 'BIP_OUT',
            'field_error': 'OTHER_REACH', 'fielders_choice': 'OTHER_REACH'}.get(ev)


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


from brl_live.pitch_bridge import contact_of, MIN_BATTER_CONTACT


def inplay(event, trajectory=None, location=None, description='', credits=None, batter=5, stand='R', dist=None, ev=None):
    p = play(event, ['X'], stand=stand)
    p['matchup']['batter'] = {'id': batter}
    p['result']['description'] = description
    hit = {}
    if trajectory: hit['trajectory'] = trajectory
    if location is not None: hit['location'] = str(location)
    if dist: hit['totalDistance'] = dist
    if ev: hit['launchSpeed'] = ev
    if hit: p['playEvents'][-1]['hitData'] = hit
    if credits: p['runners'] = [{'credits': [{'position': {'code': str(c)}} for c in credits]}]
    return p


def test_contact_reads_statcast_then_credits_then_text():
    assert contact_of(inplay('field_out', 'ground_ball', 6, dist=12.5, ev=88.2)) == {'t': 'G', 'loc': 6, 'dist': 12, 'ev': 88}
    assert contact_of(inplay('field_out', description='Mookie Betts flies out to center fielder Michael Harris II.', credits=[8])) == {'t': 'F', 'loc': 8}
    assert contact_of(inplay('single', description='Ohtani singles on a line drive to left fielder Kyle Schwarber.')) == {'t': 'L', 'loc': 7}
    assert contact_of(inplay('double', description='Freeman hits a ground-rule double on a fly ball to right fielder Nick Castellanos.')) == {'t': 'F', 'loc': 9}
    assert contact_of(inplay('field_error', description='Smith reaches on a fielding error by shortstop Trea Turner.')) == {'t': None, 'loc': 6}
    assert contact_of(inplay('sac_fly', description='Harper hits a sacrifice fly to center fielder Pham. Turner scores.')) == {'t': 'F', 'loc': 8}
    assert contact_of(inplay('field_out', 'popup', None, description='Marsh pops out to second baseman Stott.')) == {'t': 'P', 'loc': 4}
    assert contact_of(inplay('strikeout', description='Bohm strikes out swinging.')) is None
    assert contact_of(inplay('field_out', location='X')) is None


def test_contact_pools_by_kind_hand_and_batter():
    plays = []
    for _ in range(MIN_BATTER_CONTACT):
        plays.append(inplay('single', 'ground_ball', 7, batter=11, stand='L'))            # batter 11 pulls grounders to left (odd on purpose)
    for _ in range(10):
        plays.append(inplay('single', 'line_drive', 9, batter=12, stand='L'))             # the league lefty single
        plays.append(inplay('single', 'fly_ball', 8, batter=13, stand='R'))
        plays.append(inplay('home_run', 'fly_ball', 7, batter=13, stand='R', dist=405, ev=104))
        plays.append(inplay('grounded_into_double_play', 'ground_ball', 6, batter=13, stand='R'))
        plays.append(inplay('double_play', 'line_drive', 4, batter=13, stand='R'))         # lineout double play: never used for engine DPs
        plays.append(inplay('sac_fly', 'fly_ball', 8, batter=13, stand='R'))
        plays.append(inplay('field_out', 'popup', 6, batter=13, stand='R'))
        plays.append(inplay('force_out', 'ground_ball', 4, batter=13, stand='R'))          # runner out, batter safe: not an engine out
    br = PitchBridge(sequences_from_feed(feed(plays), mapper))
    assert br.n_contacts == MIN_BATTER_CONTACT + 60
    rng = np.random.default_rng(3)
    own = [br.draw_contact('11', 'L', 'single', rng) for _ in range(20)]
    assert all(c == {'t': 'G', 'loc': 7} for c in own)
    lefty = [br.draw_contact('999', 'L', 'single', rng) for _ in range(20)]
    assert all(c['loc'] in (7, 9) for c in lefty) and any(c == {'t': 'L', 'loc': 9} for c in lefty)
    assert all(br.draw_contact('999', 'R', 'dp', rng) == {'t': 'G', 'loc': 6} for _ in range(10))
    assert all(br.draw_contact('999', 'L', 'sf', rng) == {'t': 'F', 'loc': 8} for _ in range(10))      # league fallback across hands
    hr = br.draw_contact('999', 'R', 'home_run', rng)
    assert hr['dist'] == 405 and hr['ev'] == 104
    assert br.draw_contact('999', 'R', 'out', rng) == {'t': 'P', 'loc': 6}
    assert br.draw_contact('999', 'R', 'triple', rng) is None
    assert br.contact_fallbacks['b'] == 20 and br.contact_fallbacks['h'] > 0


def test_bookkeeping_contact_kind_and_draw():
    rows = []
    for ev, traj, loc in [('single', 'ground_ball', 4), ('field_out', 'fly_ball', 8), ('home_run', 'fly_ball', 7),
                          ('grounded_into_double_play', 'ground_ball', 6), ('sac_fly', 'fly_ball', 9), ('double', 'line_drive', 7),
                          ('field_error', 'ground_ball', 5), ('fielders_choice', 'ground_ball', 6), ('triple', 'line_drive', 9)]:
        for hand in ('R', 'L'):
            for _ in range(3):
                p = inplay(ev, traj, loc, stand=hand, batter=1)
                rows.append({'date_key': '2026-10-01', 'outcome': mapper(ev), 'pitcher': 1, 'batter': 1, 'stand': hand,
                             'terminal_event': ev, 'pitch_number': 1, 'pitches': pitch_list(p), 'contact': contact_of(p)})
    for hand in ('R', 'L'):
        w = play('walk', ['B', 'B', 'B', 'B'], stand=hand)
        rows.append({'date_key': '2026-10-01', 'outcome': 'BB_HBP', 'pitcher': 1, 'batter': None, 'stand': hand, 'terminal_event': 'walk',
                     'pitch_number': 4, 'pitches': pitch_list(w), 'contact': None})
    fit = BookkeepingFit(pd.DataFrame(rows), '2026-10-06')
    rng = np.random.default_rng(0)
    K = BookkeepingFit.contact_kind
    assert K({'outcome': 'bip_out', 'description': 'A grounded into a double play.'}) == 'dp'
    assert K({'outcome': 'bip_out', 'description': 'A drove in a run on a sacrifice fly.'}) == 'sf'
    assert K({'outcome': 'bip_out', 'description': 'A put the ball in play for an out.'}) == 'out'
    assert K({'outcome': 'double_triple', 'description': 'A tripled.'}) == 'triple' and K({'outcome': 'double_triple', 'description': 'A doubled.'}) == 'double'
    assert K({'outcome': 'other_reach', 'description': "A reached on a fielder's choice."}) == 'fc'
    assert K({'outcome': 'other_reach', 'description': 'A reached on an error or other play.'}) == 'error'
    assert K({'outcome': 'strikeout', 'description': 'A struck out.'}) is None and K({'outcome': 'bb_hbp', 'description': 'A walked.'}) is None
    assert fit.contact({'outcome': 'bip_out', 'description': 'A grounded into a double play.', 'batter_id': '1', 'batter_hand': 'R'}, rng) == {'t': 'G', 'loc': 6}
    assert fit.contact({'outcome': 'home_run', 'description': 'A homered.', 'batter_id': '2', 'batter_hand': 'L'}, rng) == {'t': 'F', 'loc': 7}
    assert fit.contact({'outcome': 'strikeout', 'description': 'A struck out.', 'batter_id': '2', 'batter_hand': 'L'}, rng) is None
    assert fit.contact({'outcome': 'other_reach', 'description': 'A reached on an error or other play.', 'batter_id': '2', 'batter_hand': 'L'}, rng) == {'t': 'G', 'loc': 5}
