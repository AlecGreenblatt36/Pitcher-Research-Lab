"""FORM-01's study on made-up seasons: when hitters' talent really drifts from game to game, the last ten games carry
into the next ten (k well above zero); when it does not, they add nothing beyond his level."""
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_form_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(seed=1, hitters=150, games=140, streak_sd=0.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.32, 0.03, hitters)
    cols = {k: [] for k in ('batter', 'season', 'day', 'game', 'last_in_pa', 'out7', 'post')}
    for season in (2023, 2024, 2025, 2026):
        drift = np.zeros(hitters)
        for g in range(games):
            drift = 0.95 * drift + rng.normal(0, streak_sd, hitters)       # a stationary wander around his base
            p = np.clip(base + drift, 0.05, 0.95)
            n = 4
            on = rng.random((hitters, n)) < p[:, None]
            kk = ~on & (rng.random((hitters, n)) < 0.3)
            out7 = np.where(on, np.where(rng.random((hitters, n)) < 0.3, 2, 3), np.where(kk, 1, 0))
            cols['batter'].append(np.repeat(np.arange(hitters), n) + 100); cols['season'].append(np.full(hitters * n, season))
            cols['day'].append(np.full(hitters * n, date(season, 4, 1).toordinal() + g)); cols['game'].append(np.full(hitters * n, season * 1000 + g))
            cols['last_in_pa'].append(np.ones(hitters * n, int)); cols['out7'].append(out7.ravel()); cols['post'].append(np.zeros(hitters * n, int))
    return {k: np.concatenate(v) for k, v in cols.items()}


def test_streaky_talent_carries_and_steady_talent_does_not():
    B = _report()
    spec = {'train': [2024, 2025], 'test': 2026, 'boot': 300}
    hot = B.form_study(_world(streak_sd=0.012), lambda *a: None, spec)['metrics']['obp']
    assert hot['k_fit'] > 0.08 and hot['k_test_ci'][0] > 0.05 and hot['relative_change'][2] < 0, hot
    flat = B.form_study(_world(seed=2), lambda *a: None, spec)['metrics']['obp']
    assert abs(flat['k_fit']) < 0.06 and flat['k_test_ci'][0] < 0.03, flat
    assert flat['windows'] > 1000 and set(B.form_study(_world(seed=3, hitters=30), lambda *a: None, spec)['metrics']) == {'obp', 'slg', 'k'}


def test_windows_count_the_last_and_next_ten_games():
    B = _report()
    T = _world(seed=4, hitters=3, games=50)
    m = (T['season'] == 2025) & (T['post'] == 0)
    w = B.form_windows(T, m, T['season'] == 2024)
    assert len(w['bat']) == 3 * 3                                           # windows at games 20, 30 and 40 of 50
    assert (w['last'][:, 0] == 40).all() and (w['next'][:, 0] == 40).all()  # ten games of four plate appearances
    assert (w['level'][w['bat'] == 100][:, 0] == np.array([80 + 200, 120 + 200, 160 + 200])).all()   # before the game, plus last season
