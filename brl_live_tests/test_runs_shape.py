"""TOTALS-05's check (tools/brl_runs_shape.py) on made-up games: the negative binomial keeps the mean and variance it is
given; when real runs spread wider than the simulated histograms the raw ranges hold the real score less often than
they claim and the inflated ones fix it; when they spread the same, c is near one."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_runs_shape', Path(__file__).parents[1] / 'tools/brl_runs_shape.py')
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)


def test_negative_binomial_keeps_mean_and_variance_and_page_quantiles():
    p = rs.nb_pmf(4.5, 10.0); m, v = rs.moments(p)
    assert abs(m - 4.5) < 1e-3 and abs(v - 10.0) < 0.02
    q = rs.nb_pmf(4.5, 4.0); m, v = rs.moments(q)
    assert abs(m - 4.5) < 1e-3 and abs(v - 4.5) < 0.02                       # not above the mean: Poisson
    x = np.array([0.1, 0.2, 0.2, 0.5]); assert rs.quant(x, .25) == 1 and rs.quant(x, .75) == 3 and rs.quant(x, .3) == 1


def _games(seed, wider, n=2400):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = {'game_pk': i, 'date': '2025-%03d' % (i // 15)}
        for side in ('home', 'away'):
            m = rng.uniform(3.5, 5.5); v = 2.0 * m
            r = m * m / (v - m); sims = rng.negative_binomial(r, r / (r + m), 1000)
            vr = wider * v; rr = m * m / (vr - m); y = int(rng.negative_binomial(rr, rr / (rr + m)))
            g[side + '_hist'] = np.bincount(sims, minlength=1).tolist(); g[side + '_runs'] = y
        out.append(g)
    return rs.team_games(out)


def test_wider_real_runs_are_found_and_fixed():
    rows = _games(1, 1.25)
    c = rs.inflation(rows)
    assert 1.12 < c < 1.4
    r = rs.season_check(rows, c, draws=200)
    assert r['reference']['team_range_gap_points'] > 2.0                    # raw ranges claim more than they hold
    assert abs(r['candidate']['team_range_gap_points']) < 1.5 and abs(r['candidate']['total_range_gap_points']) < 1.5
    assert r['team_crps_change'][0] < 0
    same = _games(2, 1.0)
    assert abs(rs.inflation(same) - 1.0) < 0.08
