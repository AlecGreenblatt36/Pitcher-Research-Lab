"""HCHANGE-01's study on made-up seasons: when hitters' strikeout rates really move and stay moved, the flagged changes
hold the next season; when the move reverts, they do not."""
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_change_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(seed=1, hitters=300, pa=600, persist=True):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.22, 0.04, hitters); jump = np.where(rng.random(hitters) < 0.3, rng.choice([-0.07, 0.07], hitters), 0.0)
    cols = {k: [] for k in ('batter', 'season', 'day', 'post', 'call', 'px', 'pz', 'last_in_pa', 'out7', 'ls', 'spray')}
    for season, add in ((2023, 0.0), (2024, 0.0), (2025, 1.0), (2026, 1.0 if persist else 0.0)):
        n = hitters * pa
        b = np.repeat(np.arange(hitters), pa)
        pk = np.clip(base[b] + add * jump[b], 0.02, 0.6)
        k = rng.random(n) < pk
        out7 = np.where(k, 1, np.where(rng.random(n) < 0.1, 2, np.where(rng.random(n) < 0.3, 3, 0)))
        cols['batter'].append(b + 1); cols['season'].append(np.full(n, season)); cols['day'].append(np.full(n, date(season, 5, 1).toordinal()))
        cols['post'].append(np.zeros(n, int)); cols['call'].append(np.where(k, 2, rng.integers(0, 2, n))); cols['px'].append(rng.normal(0, 0.8, n).astype(np.float32))
        cols['pz'].append(rng.normal(2.5, 0.8, n).astype(np.float32)); cols['last_in_pa'].append(np.ones(n, int)); cols['out7'].append(out7)
        cols['ls'].append(np.where(out7 == 3, rng.normal(90, 8, n), np.nan).astype(np.float32)); cols['spray'].append(np.where(out7 == 3, 0.0, np.nan).astype(np.float32))
    return {k: np.concatenate(v) for k, v in cols.items()}


def test_lasting_changes_hold_and_reverting_ones_do_not():
    B = _report()
    keep = B.change_study(_world(), lambda *a: None, {'boot': 300})['windows']['test']['k']['flagged']
    assert keep['n'] > 40 and keep['held'] > 0.8 and keep['ci'][0] > 0.6, keep
    back = B.change_study(_world(seed=2, persist=False), lambda *a: None, {'boot': 300})['windows']['test']['k']['flagged']
    assert back['n'] > 20 and back['held'] < 0.2, back
