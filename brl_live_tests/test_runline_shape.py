"""RUNLINE-02 (tools/brl_runline_shape.py): the margin histogram from the two teams' widened distributions is their
convolution, so given a win the chance of two or more matches a direct sum."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_runline_shape', Path(__file__).parents[1] / 'tools/brl_runline_shape.py')
rsh = importlib.util.module_from_spec(spec); spec.loader.exec_module(rsh)


def test_margin_is_the_convolution():
    g = {'home_hist': [50, 120, 180, 200, 170, 120, 80, 50, 30], 'away_hist': [70, 140, 190, 190, 160, 110, 70, 40, 30]}
    m = np.asarray(rsh.margin_hist(g, 1.1)); m = m / m.sum()
    h = rsh.rs.nb_pmf(*(lambda a, b: (a, 1.1 * b))(*rsh.rs.moments(rsh.rs.hist_pmf(g['home_hist']))))
    a = rsh.rs.nb_pmf(*(lambda x, y: (x, 1.1 * y))(*rsh.rs.moments(rsh.rs.hist_pmf(g['away_hist']))))
    direct = sum(h[i] * a[j] for i in range(len(h)) for j in range(len(a)) if i - j >= 2) / sum(h[i] * a[j] for i in range(len(h)) for j in range(len(a)) if i - j >= 1)
    k = np.arange(len(m)) - rsh.rl.MAX_MARGIN
    assert abs(m[k >= 2].sum() / m[k >= 1].sum() - direct) < 1e-6
