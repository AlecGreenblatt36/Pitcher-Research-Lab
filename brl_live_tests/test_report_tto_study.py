"""TTO-01's study on made-up seasons: a league third-time penalty is measured; a starter's own penalty carries over
when it is his, and not when it is redrawn every season."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_tto_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(seed=1, pitchers=100, starts=30, own_sd=0.06, persist=True):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.31, 0.03, pitchers); own = rng.normal(0, own_sd, pitchers)
    cols = {k: [] for k in ('pitcher', 'game', 'batter', 'ab', 'inning', 'last_in_pa', 'out7', 'season', 'post', 'day')}
    g = 0
    for season in (2023, 2024, 2025, 2026):
        if not persist:
            own = rng.normal(0, own_sd, pitchers)
        for s_ in range(starts):
            for p in range(pitchers):
                g += 1
                n = 27
                times = np.repeat([1, 2, 3], 9)
                pon = np.clip(base[p] + np.where(times == 3, 0.03 + own[p], 0.0), 0.05, 0.9)
                on = rng.random(n) < pon
                out7 = np.where(on, 3, np.where(rng.random(n) < 0.3, 1, 0))
                cols['pitcher'].append(np.full(n, 900 + p)); cols['game'].append(np.full(n, g)); cols['batter'].append(np.tile(np.arange(9), 3) + 10)
                cols['ab'].append(np.arange(n)); cols['inning'].append(1 + np.arange(n) // 5); cols['last_in_pa'].append(np.ones(n, int))
                cols['out7'].append(out7); cols['season'].append(np.full(n, season)); cols['post'].append(np.zeros(n, int)); cols['day'].append(np.full(n, 738000 + s_))
    return {k: np.concatenate(v) for k, v in cols.items()}


def test_league_penalty_and_a_starters_own_gap_that_carries():
    B = _report()
    out = B.tto_study(_world(), lambda *a: None, {'boot': 300, 'min_third': 150})
    lg = out['league']['2025']['obp']
    assert lg[2] - (lg[0] + lg[1]) / 2 > 0.02, lg                      # the planted 3-point penalty
    assert out['league']['2025']['pa'] == [27000, 27000, 27000]
    c = out['carry']['obp']
    assert c['pairs'] == 300 and c['r'] > 0.3 and c['r_ci'][0] > 0.15, c


def test_redrawn_gaps_do_not_carry():
    B = _report()
    c = B.tto_study(_world(seed=2, persist=False), lambda *a: None, {'boot': 300})['carry']['obp']
    assert abs(c['r']) < 0.15, c
