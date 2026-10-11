"""BULLPEN-03's check (tools/brl_bullpen_cal.py) on made-up bullpens: chances that are right hold in every bin; chances
pulled toward the middle fail the bins."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_bullpen_cal', Path(__file__).parents[1] / 'tools/brl_bullpen_cal.py')
bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)


def _games(seed, squash=1.0, n=1500):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        st = {}
        for side in ('away', 'home'):
            true = rng.beta(0.8, 1.6, 8)
            shown = 0.5 + squash * (true - 0.5)
            st[side] = {'relievers': [[j, float(s), int(rng.random() < t)] for j, (s, t) in enumerate(zip(shown, true))]}
        out.append({'game_pk': i, 'date': '2025-%03d' % (i // 15), 'starters': st})
    return out


def test_right_chances_hold_and_squashed_ones_fail():
    good = bc.season_check(_games(1), draws=200)
    assert all(good['holds'].values()) and abs(good['gap_points']) < 1.0
    bad = bc.season_check(_games(2, squash=0.6), draws=200)
    assert not bad['holds']['(2) every bin within 5']
