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


def test_a_map_fitted_on_one_season_fixes_squashed_chances_on_another():
    a, b = bc.fit_map(_games(3, squash=0.6))
    assert b > 1.3
    fixed = bc.season_check(bc.apply_map(_games(4, squash=0.6), a, b), draws=200)
    assert fixed['holds']['(2) every bin within 5'] and abs(fixed['gap_points']) < 1.5


def test_curve_fitted_on_one_season_fixes_overconfident_top_chances():
    """BULLPEN-04: a quadratic curve on the logit fitted on one season's candidates brings over-confident top chances back
    on another, and stays increasing."""
    def overconfident(seed, n=1500):
        rng = np.random.default_rng(seed); out = []
        for i in range(n):
            st = {}
            for side in ('away', 'home'):
                shown = np.clip(rng.beta(0.9, 1.6, 8), 0.005, 0.995); true = 1 / (1 + np.exp(-(0.6 * np.log(shown / (1 - shown)) - 0.2)))
                st[side] = {'relievers': [[j, float(s), int(rng.random() < t)] for j, (s, t) in enumerate(zip(shown, true))]}
            out.append({'game_pk': i, 'date': '2025-%03d' % (i // 15), 'starters': st})
        return out
    abc = bc.fit_curve(overconfident(5))
    xs = np.linspace(0.01, 0.99, 50); ys = bc.curve(xs, *abc)
    assert np.all(np.diff(ys) > 0)
    r = bc.candidates_check(overconfident(6), abc, draws=200)
    assert r['holds']['(2) every bin within 4'] and r['brier_change'][2] < 0
