"""PLAYER-09's check (tools/brl_prop_ladder.py) on made-up starts: when real counts come from the simulated chances
every line holds; a planted extra hit per start fails the bound."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_prop_ladder', Path(__file__).parents[1] / 'tools/brl_prop_ladder.py')
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)


def _games(seed, extra=0, n=2500):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        st = {}
        for side in ('away', 'home'):
            bf = int(rng.integers(18, 27)); hr = rng.uniform(0.18, 0.26)
            sims = rng.binomial(bf, hr, 1000)
            st[side] = {'h': np.bincount(sims, minlength=1).tolist(), 'actual': {'h': int(rng.binomial(bf, hr)) + extra}}
        out.append({'game_pk': i, 'date': '2025-%03d' % (i // 15), 'starters': st})
    return out


def test_own_chances_hold_and_a_planted_extra_hit_fails():
    good = pl.season_check(pl.starts(_games(1), 'h'), 1.0)
    assert good['raw']['holds'] and good['raw']['max_abs_gap'] < 1.5
    bad = pl.season_check(pl.starts(_games(2, extra=1), 'h'), 1.0)
    assert not bad['raw']['holds'] and bad['raw']['lines']['4.5']['gap_points'] < -5
