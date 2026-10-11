"""PLAYER-08's check (tools/brl_k_spread.py) on made-up starts: the spread keeps the mean and adds the variance it is
asked for; when real strikeouts spread wider than the simulated ones the raw ladder misses at both ends and the spread
one closes the gaps; with the same spread c is near one."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_k_spread', Path(__file__).parents[1] / 'tools/brl_k_spread.py')
ks = importlib.util.module_from_spec(spec); spec.loader.exec_module(ks)


def test_spread_keeps_the_mean_and_adds_variance():
    p = np.zeros(ks.N); p[3:9] = [0.1, 0.2, 0.3, 0.2, 0.15, 0.05]
    m0, v0 = ks.moments(p); q = ks.spread(p, 1.1); m1, v1 = ks.moments(q)
    assert abs(m1 - m0) < 1e-9 and abs(v1 - 1.1 * v0) < 1e-9
    assert ks.spread(p, 0.9) is p


def _games(seed, wider, n=3000):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        st = {}
        for side in ('away', 'home'):
            bf = int(rng.integers(18, 27)); kr = rng.uniform(0.15, 0.32)
            sims = rng.binomial(bf, kr, 1000)
            real = int(max(0, rng.binomial(bf, kr) + (rng.choice([-1, 0, 1], p=[wider, 1 - 2 * wider, wider]) if wider else 0)))
            st[side] = {'k': np.bincount(sims, minlength=1).tolist(), 'actual': {'k': real}}
        out.append({'game_pk': i, 'date': '2025-%03d' % (i // 15), 'starters': st})
    return ks.starts(out)


def test_wider_real_counts_are_found_and_the_spread_closes_the_gaps():
    rows = _games(1, 0.2)
    c = ks.spread_factor(rows)
    assert c > 1.05
    r = ks.season_check(rows, c, draws=200)
    assert r['candidate']['max_abs_gap'] < r['reference']['max_abs_gap'] and r['brier_change'][0] < 0
    assert abs(ks.spread_factor(_games(2, 0.0)) - 1.0) < 0.06
