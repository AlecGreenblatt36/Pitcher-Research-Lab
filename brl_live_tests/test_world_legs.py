"""LEG-02: the per-world leg bits behind same-game slips. Each leg's bits count the worlds it hit, two legs' AND counts the
worlds both hit, the families and lines are the page's, and a box carries them."""
import json
import re
import sys
from pathlib import Path

import numpy as np

from brl_live.boxscore import BoxAccumulator, WorldLog, unpack_bits, WORLD_LINES
from research_lab.game_sim.models import PlayerProfile, PitcherProfile, TeamProfile, GameMatchup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import brl_leg_book as lb  # noqa: E402


def matchup():
    def team(side):
        return TeamProfile(side, side, tuple(PlayerProfile(f'{side}{i}', f'{side} Hitter {i}') for i in range(9)),
                           PitcherProfile(side + 'SP', side + ' Starter', role='starter', expected_batters=18), (PitcherProfile(side + 'RP', side + ' Relief'),))
    return GameMatchup(team('away'), team('home'))


def z(pid, **k):
    return {**{x: 0 for x in ('PA', 'AB', 'H', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'HBP', 'K', 'SF', 'SB', 'CS')}, 'player_id': pid, **k}


def world(i):
    """Three kinds of world: a home win 5-2 with the home starter striking out 8 over 18 outs and away0 going 2 for 4 with a
    double; a 1-1 tie after five then a road win 3-1; a 0-0... no, a road blowout 9-0 with the away starter perfect."""
    k = i % 3
    if k == 0:
        return {'seed': i, 'score': {'away': 2, 'home': 5}, 'innings': {'away': {'3': {'R': 2, 'H': 3}}, 'home': {'1': {'R': 1, 'H': 2}, '6': {'R': 4, 'H': 4}}},
                'batting': {'away': [z('away0', PA=4, AB=4, H=2, **{'2B': 1}), z('away3', PA=4, AB=3, BB=1, K=2)], 'home': [z('home0', PA=4, AB=4, H=1, HR=1)]},
                'pitching': {'away': [{'player_id': 'awaySP', 'K': 3, 'outs': 15, 'H': 7, 'BB': 2, 'HBP': 0, 'BF': 24, 'R': 5, 'HR': 1, 'PC': 90}],
                             'home': [{'player_id': 'homeSP', 'K': 8, 'outs': 18, 'H': 4, 'BB': 1, 'HBP': 0, 'BF': 23, 'R': 2, 'HR': 0, 'PC': 95}]}}
    if k == 1:
        return {'seed': i, 'score': {'away': 3, 'home': 1}, 'innings': {'away': {'2': {'R': 1, 'H': 1}, '8': {'R': 2, 'H': 2}}, 'home': {'4': {'R': 1, 'H': 1}}},
                'batting': {'away': [z('away0', PA=4, AB=4, H=1)], 'home': [z('home0', PA=4, AB=4, K=1)]},
                'pitching': {'away': [{'player_id': 'awaySP', 'K': 6, 'outs': 21, 'H': 3, 'BB': 0, 'HBP': 0, 'BF': 24, 'R': 1, 'HR': 0, 'PC': 88}],
                             'home': [{'player_id': 'homeSP', 'K': 5, 'outs': 16, 'H': 5, 'BB': 3, 'HBP': 0, 'BF': 25, 'R': 3, 'HR': 0, 'PC': 101}]}}
    return {'seed': i, 'score': {'away': 9, 'home': 0}, 'innings': {'away': {'1': {'R': 3, 'H': 3}, '5': {'R': 6, 'H': 5}}, 'home': {}},
            'batting': {'away': [z('away0', PA=5, AB=5, H=3, HR=1, **{'3B': 1})], 'home': [z('home0', PA=3, AB=3, K=3)]},
            'pitching': {'away': [{'player_id': 'awaySP', 'K': 12, 'outs': 27, 'H': 0, 'BB': 0, 'HBP': 0, 'BF': 27, 'R': 0, 'HR': 0, 'PC': 99}],
                         'home': [{'player_id': 'homeRP', 'K': 2, 'outs': 27, 'H': 11, 'BB': 4, 'HBP': 0, 'BF': 40, 'R': 9, 'HR': 1, 'PC': 150}]}}


def test_bits_count_the_worlds_each_leg_hit_and_the_worlds_two_legs_share():
    log = WorldLog(matchup())
    early = {0: (0, 1, 2, 1), 1: (0, 0, 1, 1), 2: (3, 0, 9, 0)}
    for i in range(300):
        log.add(world(i), early[i % 3])
    legs = {(f, pid): v for f, pid, v in log.legs()}
    n0 = n1 = n2 = 100
    assert legs[('win:home', '')].sum() == n0 and legs[('runline:away-1.5', '')].sum() == n1 + n2
    assert legs[('total:over6.5', '')].sum() == n0 + n2 and legs[('team_total:home:over0.5', '')].sum() == n0 + n1
    assert legs[('first_inning:run_scored', '')].sum() == n0 + n2 and legs[('f5_total:over3.5', '')].sum() == n2
    assert legs[('hitter:hits2+', 'away0')].sum() == n0 + n2 and legs[('hitter:tb2+', 'away0')].sum() == n0 + n2 and legs[('hitter:hr', 'away0')].sum() == n2
    assert legs[('hitter:k1+', 'home0')].sum() == n1 + n2 and legs[('hitter:bb1+', 'away3')].sum() == n0
    assert legs[('starter:k:over7.5', 'homeSP')].sum() == n0 and legs[('starter:outs:over17.5', 'awaySP')].sum() == n1 + n2
    assert legs[('starter:k:over7.5', 'homeSP')].sum() == 100 and legs[('starter:h:over6.5', 'homeSP')].sum() == 0     # homeSP never pitched in kind 2
    # a same-game slip: home wins AND home starter 8+ strikeouts AND away0 under 2 hits happens in no world (away0 had 2 in kind 0)
    both = legs[('win:home', '')] & legs[('starter:k:over7.5', 'homeSP')] & ~legs[('hitter:hits2+', 'away0')]
    assert both.sum() == 0
    # ... while home wins AND away0 2+ hits is every kind-0 world, where multiplying would say 100 x 200 / 300
    assert (legs[('win:home', '')] & legs[('hitter:hits2+', 'away0')]).sum() == n0
    payload = log.payload()
    assert payload['n'] == 300 and len(payload['legs']) == len(payload['bits'])
    for (f, pid, hits), text in zip(payload['legs'], payload['bits']):
        bits = unpack_bits(text, 300)
        assert bits.sum() == hits and np.array_equal(bits, legs[(f, pid)])


def test_families_and_lines_are_the_pages_and_the_leg_books():
    page = Path(ROOT / 'brl_live/page/template.html').read_text(encoding='utf-8')
    js = re.search(r'var LEG_LINES = (\{.*?\});', page, re.S).group(1)
    lines = json.loads(re.sub(r'(\w+):', r'"\1":', js))
    for key, want in lines.items():
        assert list(WORLD_LINES[key]) == want, key
    assert tuple(WORLD_LINES['total']) == lb.TOTAL_LINES and tuple(WORLD_LINES['team']) == lb.TEAM_LINES and tuple(WORLD_LINES['k']) == lb.K_LINES
    assert tuple(WORLD_LINES['outs']) == lb.OUTS_LINES and tuple(WORLD_LINES['h']) == lb.H_LINES and tuple(WORLD_LINES['bb']) == lb.BB_LINES
    fams = {f for f, _, _ in WorldLog(matchup()).legs()} if False else None
    log = WorldLog(matchup()); log.add(world(0), (0, 1, 2, 1))
    fams = {f for f, _, _ in log.legs()}
    for must in ('win:home', 'runline:home-1.5', 'total:over8.5', 'team_total:away:over2.5', 'first_inning:run_scored', 'f5_total:over4.5',
                 'hitter:hits1+', 'hitter:hr', 'hitter:tb2+', 'starter:k:over5.5', 'starter:outs:over15.5', 'starter:h:over4.5', 'starter:bb:over1.5'):
        assert must in fams, must


def test_box_carries_the_world_legs():
    a = BoxAccumulator(matchup())
    early = {0: (0, 1, 2, 1), 1: (0, 0, 1, 1), 2: (3, 0, 9, 0)}
    for i in range(10000):
        w = world(i)
        a.add(w)
    a.features = []
    r = a.finish()
    wl = r['world_legs']
    assert wl['n'] == 10000 and wl['schema'] == 'brl.world-legs.v1'
    hits = {(f, pid): h for f, pid, h in wl['legs']}
    assert hits[('win:home', '')] == 3334 and hits[('hitter:hr', 'away0')] == 3333
    assert len(wl['bits'][0]) == 1668                                      # 10,000 bits packed, base64


def test_page_keeps_bits_for_the_slate_date_only_and_thins_a_full_slate():
    from brl_live.box_page import thin_world_legs, trim_boxes
    log = WorldLog(matchup())
    early = {0: (0, 1, 2, 1), 1: (0, 0, 1, 1), 2: (3, 0, 9, 0)}
    for i in range(400):
        log.add(world(i), early[i % 3])
    wl = log.payload()
    thin = thin_world_legs(wl, 4)
    assert thin['n'] == 100 and thin['stride'] == 4
    full = {(f, pid): unpack_bits(b, 400) for (f, pid, _), b in zip(wl['legs'], wl['bits'])}
    for (f, pid, hits), b in zip(thin['legs'], thin['bits']):
        v = unpack_bits(b, 100)
        assert hits == v.sum() and np.array_equal(v, full[(f, pid)][::4])
    boxes = {}
    forecasts = {}
    for i in range(6):
        pk = 100 + i; date = '2026-10-23' if i < 5 else '2026-10-22'
        boxes[f'f{i}'] = {'game_pk': pk, 'date': date, 'world_legs': dict(wl)}
        forecasts[f'f{i}'] = {'game_pk': pk, 'version': 1, 'saved_at': 't', 'date': date}
    public = {'date': '2026-10-23', 'box_scores': boxes, 'forecasts': forecasts}
    out = trim_boxes(public)
    assert 'world_legs' not in out['box_scores']['f5']                      # yesterday's game: no bits
    assert all(out['box_scores'][f'f{i}']['world_legs']['n'] == 100 for i in range(5))   # five games today: thinned
    public = {'date': '2026-10-23', 'box_scores': {k: v for k, v in boxes.items() if k in ('f0', 'f1')}, 'forecasts': forecasts}
    public['box_scores']['f0']['world_legs'] = dict(wl)
    out = trim_boxes(public)
    assert out['box_scores']['f0']['world_legs']['n'] == 400                 # a small slate keeps every world
