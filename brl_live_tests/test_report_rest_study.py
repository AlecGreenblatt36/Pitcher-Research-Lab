"""REST-01's study on made-up seasons: relievers who lose speed and strikeouts on no rest show it against their rested
appearances; with no such effect the differences sit at zero."""
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_rest_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(seed=1, pitchers=150, apps=55, drop=0.0, kdrop=0.0):
    rng = np.random.default_rng(seed)
    B = _report()
    ff, sl = B.D.SUBTYPES.index('FF'), B.D.SUBTYPES.index('SL')
    base = rng.normal(95, 1.5, pitchers); kb = rng.normal(0.25, 0.04, pitchers)
    cols = {k: [] for k in ('pitcher', 'season', 'day', 'game', 'inning', 'sub', 'v0', 'last_in_pa', 'out7', 'post')}
    g = 0
    for season in (2023, 2024, 2025, 2026):
        d0 = date(season, 4, 1).toordinal()
        for p in range(pitchers):
            days = np.sort(rng.choice(np.arange(180), apps, replace=False))
            prev = -9
            for dd in days:
                g += 1
                tired = dd - prev == 1
                prev = dd
                n = 20
                sub = np.where(np.arange(n) % 2 == 0, ff, sl)
                v0 = np.where(sub == ff, base[p] - (drop if tired else 0.0) + rng.normal(0, 0.7, n), 86 + rng.normal(0, 1, n))
                last = np.zeros(n, int); last[[3, 7, 11, 15, 19]] = 1
                pk = kb[p] - (kdrop if tired else 0.0)
                out7 = np.where(last == 1, np.where(rng.random(n) < pk, 1, np.where(rng.random(n) < 0.3, 3, 0)), 0)
                cols['pitcher'].append(np.full(n, 800 + p)); cols['season'].append(np.full(n, season)); cols['day'].append(np.full(n, d0 + dd))
                cols['game'].append(np.full(n, g)); cols['inning'].append(np.full(n, 7)); cols['sub'].append(sub); cols['v0'].append(v0.astype(np.float32))
                cols['last_in_pa'].append(last); cols['out7'].append(out7); cols['post'].append(np.zeros(n, int))
    return {k: np.concatenate(v) for k, v in cols.items()}


def test_no_rest_costs_speed_and_strikeouts_when_planted():
    B = _report()
    out = B.rest_study(_world(drop=0.6, kdrop=0.06), lambda *a: None, {'boot': 300})
    t = out['windows']['test']
    assert t['appearances']['no_rest'] > 500 and t['appearances']['rested'] > 2000
    assert t['velo']['minus_rested']['no_rest'][2] < -0.3, t['velo']
    assert t['k']['minus_rested']['no_rest'][2] < 0, t['k']


def test_no_effect_world_sits_at_zero():
    B = _report()
    t = B.rest_study(_world(seed=2), lambda *a: None, {'boot': 300})['windows']['test']
    assert abs(t['velo']['minus_rested']['no_rest'][0]) < 0.05 and t['velo']['minus_rested']['no_rest'][1] > -0.06, t['velo']
    assert abs(t['k']['minus_rested']['no_rest'][0]) < 0.01, t['k']
