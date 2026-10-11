"""PLAYER-03's scoring (tools/brl_player_cal.py) on synthetic seasons with known truth: a planted overstatement is found and
the cross-season map passes the registered rule; when the chances are right, the rule does not pass; the page's chance
(ownAtLeast) is computed from the histograms the replay records."""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_player_cal', Path(__file__).parents[1] / 'tools/brl_player_cal.py')
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)


def _season(rng, n, shift, slope=1.0, dates=180):
    p = rng.beta(6, 4, n)
    q = 1 / (1 + np.exp(-(shift + slope * np.log(p / (1 - p)))))
    y = rng.random(n) < q
    d = rng.integers(0, dates, n)
    return [{'date': '2026-%03d' % int(d[i]), 'spot': int(1 + i % 9), 'prior_pa': 400, 'p_hit': float(p[i]), 'y_hit': int(y[i])} for i in range(n)]


def test_planted_overstatement_is_found_and_the_map_passes():
    rng = np.random.default_rng(7)
    res = pc.study({'2025': _season(rng, 40000, -0.07, 0.95), '2026': _season(rng, 40000, -0.07, 0.95)}, draws=400)
    k = res['kinds']['hit']
    for s in ('2025', '2026'):
        r = k['by_season'][s]
        assert 1.0 < r['gap_points'] < 2.5                       # the chances run high by about 1.6 points
        assert abs(r['cal_gap_points']) <= 0.5 and r['brier_change_ci'][1] < 0
        a, b = r['map_from_other']
        assert -0.12 < a < -0.02 and 0.85 < b < 1.05
    assert k['passes_rule'] and len(k['map_both']) == 2


def test_right_chances_do_not_pass():
    rng = np.random.default_rng(11)
    res = pc.study({'2025': _season(rng, 40000, 0.0), '2026': _season(rng, 40000, 0.0)}, draws=400)
    k = res['kinds']['hit']
    assert not k['passes_rule']
    for s in ('2025', '2026'):
        assert abs(k['by_season'][s]['gap_points']) < 0.6


def test_rows_from_replay_files_use_the_pages_chance(tmp_path):
    g = {'game_pk': 1, 'date': '2026-05-01', 'hitters': [
        {'batter': 10, 'spot': 9, 'h_hist': [300, 400, 200, 100, 0, 0], 'hr_hist': [880, 110, 10, 0], 'bb_hist': [700, 250, 50, 0, 0],
         'k_hist': [400, 400, 200, 0, 0], 'prior_pa_365': 120, 'actual': {'pa': 4, 'h': 2, 'hr': 0, 'k': 1, 'bb': 0}},
        {'batter': 11, 'spot': 1, 'h_hist': [1000, 0, 0, 0, 0, 0], 'hr_hist': [1000, 0, 0, 0], 'bb_hist': [1000, 0, 0, 0, 0],
         'k_hist': [1000, 0, 0, 0, 0], 'prior_pa_365': 0, 'actual': None}]}
    f = tmp_path / 'x.jsonl.gz'
    with gzip.open(f, 'wt') as fh:
        fh.write(json.dumps(g) + '\n')
    rows = pc.load_rows([str(f)])
    assert len(rows) == 1                                         # a hitter without a line is left out
    r = rows[0]; keep = 0.927
    assert abs(r['p_hit'] - (1 - (300 + 400 * (1 - keep) + 200 * (1 - keep) ** 2 + 100 * (1 - keep) ** 3) / 1000)) < 1e-12
    p2 = 1 - (300 + 400 + 200 * ((1 - keep) ** 2 + 2 * keep * (1 - keep)) + 100 * ((1 - keep) ** 3 + 3 * keep * (1 - keep) ** 2)) / 1000
    assert abs(r['p_hit2'] - p2) < 1e-12
    assert (r['y_hit'], r['y_hit2'], r['y_hr'], r['y_walk'], r['y_k']) == (1, 1, 0, 0, 1)
