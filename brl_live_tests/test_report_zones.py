"""The standard hot-zone counts (MLB zones 1-9 and 11-14, catcher's view) carried by every hitter card."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_zones_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_zone_counts_by_zone_and_family():
    B = _report()
    #            zone group call last out7
    rows = [(5, 0, 0, 0, 3),    # fastball middle, taken; the plate appearance ends in a single later
            (5, 0, 1, 1, 3),    # fastball middle, in play for a single
            (14, 3, 2, 0, 1),   # breaking ball down and away, missed
            (14, 3, 2, 1, 1),   # again: strike three
            (11, 5, 0, 1, 2),   # changeup up and in, ball four
            (1, 0, 1, 1, 5)]    # fastball up and in, home run
    a = np.array(rows)
    T = {'zone': a[:, 0], 'group': a[:, 1], 'call': a[:, 2], 'last_in_pa': a[:, 3], 'out7': a[:, 4]}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(float); whiff = (T['call'] == 2).astype(float)
    out = B.zone_counts(T, np.arange(len(rows)), swing, whiff)
    cell = {c[0]: c[1:] for c in out['all']}
    assert [c[0] for c in out['all']] == list(B.ZONES13)
    assert cell[5] == [2, 1, 0, 1, 1, 1]          # 2 pitches, 1 swing, no miss, 1 at-bat, 1 hit, 1 base
    assert cell[14] == [2, 2, 2, 1, 0, 0]         # 2 misses, the strikeout an at-bat with no hit
    assert cell[11] == [1, 0, 0, 0, 0, 0]         # a walk is no at-bat
    assert cell[1] == [1, 1, 0, 1, 1, 4]          # the home run: four bases
    fb = {c[0]: c[1:] for c in out['fastball']}; br = {c[0]: c[1:] for c in out['breaking']}
    assert fb[5][0] == 2 and fb[14][0] == 0 and br[14][0] == 2
    assert B.zone_counts(T, [], swing, whiff) == {}


def test_zone_counts_carry_hard_hit_balls_when_speeds_are_known():
    # TAGS-08: per zone, balls in play at 95 mph or more and balls in play with a measured speed
    B = _report()
    nan = float('nan')
    #            zone group call last out7  spray  ls
    rows = [(1, 0, 1, 1, 3, -10.0, 101.0),   # up: a hard single
            (1, 0, 1, 1, 0, 20.0, 80.0),     # up: a soft out
            (1, 0, 1, 0, 0, nan, nan),       # up: a foul, not in play
            (8, 0, 1, 1, 0, 5.0, nan),       # down: in play, no speed measured
            (8, 3, 2, 1, 1, nan, nan)]       # down: strike three
    a = np.array(rows, dtype=float)
    T = {'zone': a[:, 0].astype(int), 'group': a[:, 1].astype(int), 'call': a[:, 2].astype(int), 'last_in_pa': a[:, 3].astype(int),
         'out7': a[:, 4].astype(int), 'spray': a[:, 5], 'ls': a[:, 6]}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(float); whiff = (T['call'] == 2).astype(float)
    cell = {c[0]: c[1:] for c in B.zone_counts(T, np.arange(len(rows)), swing, whiff)['all']}
    assert cell[1] == [3, 3, 0, 2, 1, 1, 1, 2]     # two balls in play measured, one hard
    assert cell[8][-2:] == [0, 0]                  # the unmeasured ball in play is not counted


def test_zone_split_by_pitcher_hand():
    B = _report()
    #            zone group call last out7 throw_r
    rows = [(5, 0, 1, 1, 3, 1),     # a single off a righty
            (5, 0, 2, 0, 1, 0),     # a miss against a lefty
            (14, 3, 2, 1, 1, 0)]    # strike three against the lefty
    a = np.array(rows)
    T = {'zone': a[:, 0], 'group': a[:, 1], 'call': a[:, 2], 'last_in_pa': a[:, 3], 'out7': a[:, 4], 'throw_r': a[:, 5]}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(float); whiff = (T['call'] == 2).astype(float)
    out = B.zone_split(T, np.arange(3), swing, whiff)
    vr = {c[0]: c[1:] for c in out['R']['all']}; vl = {c[0]: c[1:] for c in out['L']['all']}
    assert vr[5] == [1, 1, 0, 1, 1, 1] and vr[14][0] == 0
    assert vl[5] == [1, 1, 1, 0, 0, 0] and vl[14] == [1, 1, 1, 1, 0, 0]
    only_r = B.zone_split(T, [0], swing, whiff)
    assert set(only_r) == {'R'}                     # a hand he never faced is left out
    assert B.zone_split(T, [], swing, whiff) == {}


def test_zone_split_by_batter_side_for_pitchers():
    B = _report()
    #            zone group call last out7 stand_r
    rows = [(9, 3, 2, 1, 1, 1),     # a slider down and away to a righty: strike three
            (7, 3, 0, 0, 2, 0),     # the same pitch to a lefty, taken
            (7, 3, 1, 1, 0, 0)]     # then put in play for an out
    a = np.array(rows)
    T = {'zone': a[:, 0], 'group': a[:, 1], 'call': a[:, 2], 'last_in_pa': a[:, 3], 'out7': a[:, 4], 'stand_r': a[:, 5]}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(float); whiff = (T['call'] == 2).astype(float)
    out = B.zone_split(T, np.arange(3), swing, whiff, by='stand_r')
    vr = {c[0]: c[1:] for c in out['R']['breaking']}; vl = {c[0]: c[1:] for c in out['L']['breaking']}
    assert vr[9] == [1, 1, 1, 1, 0, 0] and vl[7] == [2, 1, 0, 1, 0, 0]


def test_spray_counts_by_field_third_and_trajectory():
    B = _report()
    nan = float('nan')
    #        spray  traj  ls
    rows = [(-30.0, 0, 101.0),    # a hard ground ball to the left side
            (-20.0, 2, 88.0),     # a fly ball to left
            (2.0, 1, 97.0),       # a line drive up the middle
            (25.0, 0, 70.0),      # a slow roller to the right side
            (5.0, 3, 80.0),       # a pop-up
            (nan, -1, nan)]       # a pitch with no batted ball
    a = np.array(rows, dtype=float)
    T = {'spray': a[:, 0], 'traj': a[:, 1].astype(int), 'ls': a[:, 2]}
    out = B.spray_counts(T, np.arange(len(rows)))
    assert out['gb'] == [1, 0, 1] and out['air'] == [1, 1, 0]
    assert out['popups'] == 1 and out['hard'] == [2, 5]
    assert B.spray_counts(T, [5]) is None and B.spray_counts({'zone': a[:, 0]}, [0]) is None


def test_count_groups_tendencies_and_usage():
    B = _report()
    #           balls strikes
    pairs = [(0, 0), (0, 1), (1, 1), (1, 0), (3, 1), (2, 1), (0, 2), (3, 2)]
    g = B.count_group([b for b, _ in pairs], [s for _, s in pairs]).tolist()
    assert g == [0, 1, 1, 2, 2, 2, 3, 3]
    #            balls strikes zone call group stand_r
    rows = [(0, 0, 5, 1, 0, 1),     # first pitch fastball in the zone, swung at
            (0, 1, 13, 2, 3, 1),    # 0-1 slider below the zone: a chase and a miss
            (2, 0, 12, 0, 0, 1),    # 2-0 fastball up out of the zone, taken
            (1, 2, 14, 1, 5, 1)]    # two strikes, changeup away, chased and fouled
    a = np.array(rows)
    T = {'balls': a[:, 0], 'strikes': a[:, 1], 'zone': a[:, 2], 'call': a[:, 3], 'group': a[:, 4], 'stand_r': a[:, 5]}
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(float); whiff = (T['call'] == 2).astype(float)
    ct = B.count_tend(T, np.arange(4), swing, whiff)
    assert ct['first'] == [1, 1, 0, 0, 0] and ct['ahead'] == [1, 1, 1, 1, 1]
    assert ct['behind'] == [1, 0, 0, 1, 0] and ct['two'] == [1, 1, 0, 1, 1]
    rows60 = np.resize(np.arange(4), 60)                 # enough pitches to righties for a usage line
    us = B.usage_by_count(T, rows60)
    assert set(us) == {'R'} and sum(us['R']['first']) == 15 and us['R']['ahead'] == [0, 15, 0] and us['R']['two'] == [0, 0, 15]
