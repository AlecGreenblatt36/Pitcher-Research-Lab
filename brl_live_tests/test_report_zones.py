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
