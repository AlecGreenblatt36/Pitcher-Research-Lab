"""PLAYER-07's check (tools/brl_outs_cal.py) on made-up starts: when the real outs come from the simulated chances the level,
the fifths and Brier all hold; a planted short leash (real starts a full inning shorter) fails the level and the fifths."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_outs_cal', Path(__file__).parents[1] / 'tools/brl_outs_cal.py')
oc = importlib.util.module_from_spec(spec); spec.loader.exec_module(oc)


def _games(seed, n=2500, shorter=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        st = {}
        for side in ('away', 'home'):
            mu = rng.uniform(12, 19)                                          # this starter's expected outs
            sims = np.clip(np.rint(rng.normal(mu, 3.6, 1000)), 0, 27).astype(int)
            real = int(np.clip(np.rint(rng.normal(mu, 3.6)) - shorter, 0, 27))
            st[side] = {'outs': np.bincount(sims, minlength=28).tolist(), 'actual': {'outs': real, 'bf': 20}}
        out.append({'game_pk': i, 'date': '2025-%03d' % (i // 15), 'starters': st})
    return out


def test_helpers_match_the_page():
    p = np.zeros(28); p[14] = 0.25; p[15] = 0.25; p[18] = 0.5
    assert oc.median(p) == 15                         # the page's quant(c, 0.5): running share reaches one half at 15
    assert oc.more_than(p, 14.5) == 0.75 and oc.more_than(p, 17.5) == 0.5


def test_own_chances_hold_and_a_planted_short_leash_fails():
    r = oc.season_check(_games(1), draws=300)
    assert r['starts'] == 5000 and r['level_holds'] and r['fifths_hold'] and r['brier_holds']
    assert abs(r['ladder_lines']['pred'] - r['ladder_lines']['actual']) < 0.02
    bad = oc.season_check(_games(2, shorter=3), draws=300)
    assert not bad['level_holds'] and not bad['fifths_hold']
    assert bad['lines']['16.5']['gap_ci_points'][0] > 0          # the simulator says more than 16.5 too often


def test_starts_without_a_real_line_are_left_out():
    g = _games(3, n=2)
    g[0]['starters']['away']['actual'] = None
    g[1]['starters']['home']['actual'] = {'outs': None, 'bf': 3}
    assert len(oc.starts(g)) == 2


def test_bands_add_up_and_match_when_real_outs_follow_the_chances():
    r = oc.season_check(_games(6, n=1500), draws=100)
    assert abs(sum(v['sim'] for v in r['bands'].values()) - 100) < 1e-6 and abs(sum(v['actual'] for v in r['bands'].values()) - 100) < 1e-6
    assert all(abs(v['gap_points']) < 2.0 for v in r['bands'].values())
