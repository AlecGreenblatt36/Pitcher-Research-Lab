"""MATCHUPREAD-01's study on made-up seasons: when hitters' zone weaknesses are real and pitchers have location habits,
the composite must beat the league-location baseline with a slope near one; with no zone effect the slope's interval
must reach zero."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_matchup_read_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seasons(effect, seed=7, hitters=80, pitchers=60, pa=60000):
    rng = np.random.default_rng(seed)
    zones = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])
    own = rng.normal(0, effect, (hitters, 13))                 # each hitter's own zone strengths
    loc = rng.dirichlet(np.full(13, 0.6), pitchers)             # each pitcher's location habit
    cols = {k: [] for k in ('season', 'batter', 'pitcher', 'stand_r', 'throw_r', 'zone', 'group', 'last_in_pa', 'out7', 'post')}
    for season in (2024, 2025, 2026):
        h = rng.integers(0, hitters, pa); p = rng.integers(0, pitchers, pa)
        zi = np.array([rng.choice(13, p=loc[j]) for j in p])
        hit = rng.random(pa) < np.clip(0.25 + own[h, zi], 0.02, 0.95)
        cols['season'] += [season] * pa; cols['batter'] += (1000 + h).tolist(); cols['pitcher'] += (5000 + p).tolist()
        cols['stand_r'] += [1] * pa; cols['throw_r'] += [1] * pa; cols['zone'] += zones[zi].tolist(); cols['group'] += [0] * pa
        cols['last_in_pa'] += [1] * pa; cols['out7'] += np.where(hit, 3, 0).tolist(); cols['post'] += [0] * pa
    return {k: np.array(v) for k, v in cols.items()}


def test_matchup_read_finds_a_planted_interaction():
    B = _report()
    out = B.matchup_read_study(_seasons(0.15), lambda *a: None, {'min_pitches': 200, 'min_pa': 100, 'm': 40})
    assert out['at_bats'] > 10000
    assert out['slope'][1] > 0.5 and out['mse_composite_minus_baseline'][2] < 0


def test_matchup_read_null_world():
    B = _report()
    out = B.matchup_read_study(_seasons(0.0, seed=8), lambda *a: None, {'min_pitches': 200, 'min_pa': 100, 'm': 40})
    lo, hi = out['slope'][1], out['slope'][2]
    assert lo <= 0 <= hi or abs(out['mse_composite_minus_baseline'][0]) < 1e-4
