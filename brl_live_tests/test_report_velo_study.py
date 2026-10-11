"""VELO-01's study on made-up seasons: when a starter's speed drifts from start to start and strikeouts follow speed, a
drop in his last start carries into the next and predicts fewer strikeouts; when speed is steady noise, neither."""
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_velo_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(seed=1, pitchers=120, starts=28, drift_sd=0.0, k_per_mph=0.0):
    rng = np.random.default_rng(seed)
    B = _report()
    ff = B.D.SUBTYPES.index('FF'); sl = B.D.SUBTYPES.index('SL')
    base = rng.normal(94, 2, pitchers); kbase = rng.normal(0.22, 0.04, pitchers)
    cols = {k: [] for k in ('pitcher', 'season', 'day', 'game', 'inning', 'sub', 'v0', 'last_in_pa', 'out7', 'post')}
    g = 0
    for season in (2023, 2024, 2025, 2026):
        drift = np.zeros(pitchers)
        for s_ in range(starts):
            drift = 0.8 * drift + rng.normal(0, drift_sd, pitchers)
            velo = base + drift
            for p in range(pitchers):
                g += 1
                n = 90
                sub = np.where(np.arange(n) % 9 < 5, ff, sl)
                v0 = np.where(sub == ff, velo[p] + rng.normal(0, 0.8, n), 85 + rng.normal(0, 1, n))
                last = np.zeros(n, int); last[np.arange(3, n, 4)[:24]] = 1
                pk = np.clip(kbase[p] + k_per_mph * drift[p], 0.02, 0.6)
                out7 = np.where(last == 1, np.where(rng.random(n) < pk, 1, np.where(rng.random(n) < 0.3, 3, 0)), 0)
                cols['pitcher'].append(np.full(n, 500 + p)); cols['season'].append(np.full(n, season))
                cols['day'].append(np.full(n, date(season, 4, 1).toordinal() + 5 * s_)); cols['game'].append(np.full(n, g))
                cols['inning'].append(1 + np.arange(n) // 15); cols['sub'].append(sub); cols['v0'].append(v0.astype(np.float32))
                cols['last_in_pa'].append(last); cols['out7'].append(out7); cols['post'].append(np.zeros(n, int))
    return {k: np.concatenate(v) for k, v in cols.items()}


def test_drifting_speed_carries_and_moves_strikeouts():
    B = _report()
    out = B.velo_study(_world(drift_sd=0.6, k_per_mph=0.02), lambda *a: None, {'boot': 300})
    v, k = out['targets']['velo'], out['targets']['k']
    assert v['slope_test'] > 0.3 and v['slope_test_ci'][0] > 0.2, v
    assert k['slope_test_ci'][0] > 0 and k['after_drops']['difference'] < 0, k
    assert out['pairs']['2026'] > 1000


def test_steady_speed_neither_carries_nor_predicts():
    B = _report()
    out = B.velo_study(_world(seed=2), lambda *a: None, {'boot': 300})
    v, k = out['targets']['velo'], out['targets']['k']
    assert abs(v['slope_test']) < 0.15, v
    assert k['slope_test_ci'][0] < 0 < k['slope_test_ci'][1], k
