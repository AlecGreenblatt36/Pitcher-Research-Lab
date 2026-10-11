"""XSTATS-01's study on made-up seasons: when a hitter's results come from how hard he hits the ball, expected AVG from
exit speed and launch angle predicts his next season better than his AVG; when they come from something the batted ball
does not show (speed on the bases), it does not."""
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_xstats_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(seed=1, hitters=300, pa=500, contact_sd=3.0, speed_sd=0.0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(88, contact_sd, hitters); speed = rng.normal(0, speed_sd, hitters)
    cols = {k: [] for k in ('batter', 'season', 'day', 'post', 'last_in_pa', 'out7', 'call', 'ls', 'la')}
    for season in (2023, 2024, 2025, 2026):
        n = hitters * pa
        b = np.repeat(np.arange(hitters), pa)
        u = rng.random(n)
        k, bb = u < 0.22, (u >= 0.22) & (u < 0.30)
        bip = ~k & ~bb
        ls = np.where(bip, rng.normal(mu[b], 12), np.nan); la = np.where(bip, rng.normal(12, 25, n), np.nan)
        good = (la > -10) & (la < 40)
        p = 1 / (1 + np.exp(-(-1.0 + 0.07 * (np.nan_to_num(ls) - 88) + speed[b]))) * np.where(good, 1.0, 0.4)
        hit = bip & (rng.random(n) < p)
        hr = hit & (np.nan_to_num(ls) > 100) & (la > 20) & (la < 35)
        out7 = np.where(k, 1, np.where(bb, 2, np.where(hr, 5, np.where(hit, 3, 0))))
        cols['batter'].append(b + 1); cols['season'].append(np.full(n, season)); cols['day'].append(np.full(n, date(season, 5, 1).toordinal()))
        cols['post'].append(np.zeros(n, int)); cols['last_in_pa'].append(np.ones(n, int)); cols['out7'].append(out7)
        cols['call'].append(np.where(bip, 1, np.where(k, 2, 0))); cols['ls'].append(ls.astype(np.float32)); cols['la'].append(la.astype(np.float32))
    return {k: np.concatenate(v) for k, v in cols.items()}


SPEC = {'boot': 300, 'score': {'train': [2024, 2025], 'test': 2026}}


def test_expected_wins_when_contact_drives_results():
    B = _report()
    out = B.xstats_study(_world(), lambda *a: None, SPEC)
    m = out['metrics']['avg']
    assert out['hitters']['score'] == 300 and out['measured_share'] == 1.0
    assert m['expected_minus_raw'][2] < 0, m
    assert m['corr']['expected'] > m['corr']['raw'], m


def test_raw_wins_when_results_come_from_elsewhere():
    B = _report()
    m = B.xstats_study(_world(seed=2, contact_sd=0.0, speed_sd=0.5), lambda *a: None, SPEC)['metrics']['avg']
    assert m['expected_minus_raw'][1] > 0, m
    assert m['blend_weight'] < 0.3, m
