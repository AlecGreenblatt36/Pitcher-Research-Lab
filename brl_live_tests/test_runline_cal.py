"""RUNLINE-01's check (tools/brl_runline_cal.py) on made-up games: when the actual margins come from the simulator's own
margin histograms, the conditional chance of winning by two or more matches what happened; when real one-run wins are
more common than the simulator says, the overstatement is found."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_runline_cal', Path(__file__).parents[1] / 'tools/brl_runline_cal.py')
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)


def _games(seed, n=2400, one_run_boost=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lam_h, lam_a = rng.uniform(3.5, 5.5), rng.uniform(3.5, 5.5)
        h, a = rng.poisson(lam_h, 2000), rng.poisson(lam_a, 2000)
        tie = h == a
        h[tie] += rng.integers(0, 2, tie.sum()); a[tie] += 1 - (h[tie] - a[tie] > 0)     # no ties: a coin settles extras
        m = np.clip(h - a, -15, 15)
        hist = np.bincount(m + 15, minlength=31)
        # the actual game: a draw from the same histogram, or a one-run win more often than the histogram says
        draw = int(rng.choice(np.arange(-15, 16), p=hist / hist.sum()))
        if one_run_boost and abs(draw) >= 2 and rng.random() < one_run_boost:
            draw = 1 if draw > 0 else -1
        out.append({'game_pk': i, 'date': '2026-%03d' % (i // 15), 'margin_hist': hist.tolist(), 'home_runs': max(draw, 0) + 3, 'away_runs': max(-draw, 0) + 3})
    return out


def test_calibrated_margins_check_out():
    r = rc.season_check(_games(1), draws=500)
    for side in ('home', 'away'):
        lo, hi = r[side]['gap_ci_points']
        assert lo < 0 < hi and abs(r[side]['gap_points']) < 2.5
        assert r[side]['runline_sim_brier'] < r[side]['runline_const_brier']


def test_too_few_one_run_wins_in_the_simulator_are_found():
    r = rc.season_check(_games(2, one_run_boost=0.15), draws=500)
    for side in ('home', 'away'):
        assert r[side]['gap_ci_points'][0] > 0 and r[side]['gap_points'] > 5      # the simulator says by 2+ too often
