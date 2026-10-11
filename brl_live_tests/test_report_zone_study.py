"""ZONES-01's study on a made-up league: hitters with real zone patterns of their own; shrinking toward the scaled league
shape must beat raw rates on a fresh season, and the scaled prior alone must miss the hitters' own patterns."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_zone_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _league(seed=1, hitters=120, pitches=260, own_sd=0.6):
    rng = np.random.default_rng(seed)
    zones = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])
    base_sw = np.array([.65, .7, .62, .72, .8, .7, .6, .66, .55, .3, .25, .28, .24])
    own = rng.normal(0, own_sd, (hitters, 13))                  # each hitter's own zone pattern, the same every season
    level = rng.normal(0, 0.4, hitters)
    cols = {k: [] for k in ('season', 'batter', 'stand_r', 'zone', 'call', 'last_in_pa', 'out7', 'post')}
    for season in (2023, 2024, 2025, 2026):
        for h in range(hitters):
            zi = rng.integers(0, 13, pitches)
            p = 1 / (1 + np.exp(-(np.log(base_sw[zi] / (1 - base_sw[zi])) + level[h] + own[h, zi])))
            swing = rng.random(pitches) < p
            whiff = swing & (rng.random(pitches) < 0.25)
            last = rng.random(pitches) < 0.25
            cols['season'] += [season] * pitches; cols['batter'] += [1000 + h] * pitches; cols['stand_r'] += [h % 2] * pitches
            cols['zone'] += zones[zi].tolist(); cols['call'] += np.where(whiff, 2, np.where(swing, 1, 0)).tolist()
            cols['last_in_pa'] += last.astype(int).tolist(); cols['out7'] += np.where(rng.random(pitches) < 0.25, 3, 0).tolist(); cols['post'] += [0] * pitches
    return {k: np.array(v) for k, v in cols.items()}


def test_zone_shrink_study_recovers_planted_patterns():
    B = _report()
    T = _league()
    out = B.zone_shrink_study(T, lambda *a: None, {'min_pa': 50})
    sw = out['metrics']['swing']
    assert sw['shrunk_minus_raw'][2] < 0               # shrinking beats raw rates on the fresh season
    assert sw['shrunk_minus_scaled'][2] < 0            # and the hitters' own patterns beat the league shape alone
    assert 0 < sw['m'] < 640
    assert set(out['metrics']) == {'avg', 'swing', 'miss', 'slg'} and out['hitters']['score'] == 120


def test_zone_shrink_study_null_world_leans_on_the_league_shape():
    B = _report()
    T = _league(seed=2, own_sd=0.0)                       # no zone patterns of their own: level and league shape only
    sw = B.zone_shrink_study(T, lambda *a: None, {'min_pa': 50})['metrics']['swing']
    assert sw['m'] >= 160                                 # the fit shrinks hard toward the scaled shape
    assert sw['shrunk_minus_raw'][2] < 0
    assert sw['shrunk_minus_scaled'][1] <= 0 <= sw['shrunk_minus_scaled'][2] or abs(sw['shrunk_minus_scaled'][0]) < 0.002
