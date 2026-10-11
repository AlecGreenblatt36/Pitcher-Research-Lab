"""LEVEL-01's check (tools/brl_level_cal.py) on made-up plate appearances: the log-likelihood change computed from per-game
sums matches the exact change plate by plate; a planted season-level drift is found and re-anchoring gains; with no drift
the re-anchoring gains nothing beyond noise; the m is picked on the other season."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_level_cal', Path(__file__).parents[1] / 'tools/brl_level_cal.py')
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)

BASE = np.array([0.45, 0.22, 0.09, 0.14, 0.05, 0.03, 0.02])


def _pas(rng, n, drift):
    p = rng.dirichlet(BASE * 60, n)                                   # each plate appearance's chances, spread around the base
    q = p * np.exp(drift); q /= q.sum(1, keepdims=True)                # the real chances carry the season's drift
    y = np.array([rng.choice(7, p=qi) for qi in q])
    return p, y


def _sums(p, y):
    return {'n': len(y), 'obs': np.bincount(y, minlength=7).tolist(), 'pred': p.sum(0).tolist(), 's2': (p.T @ p).ravel().tolist()}


def test_sums_give_the_exact_change():
    rng = np.random.default_rng(3)
    p, y = _pas(rng, 20000, np.zeros(7))
    s = _sums(p, y)
    for d in (np.array([-0.03, 0.06, 0.10, -0.05, 0.08, -0.12, 0.04]), np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]) + np.array([0, .09, -.07, 0, 0, .05, 0])):
        exact = float((d[y] - np.log((p * np.exp(d)).sum(1))).sum())
        approx = lc.ll_change(np.array(s['obs'], float), np.array(s['pred']), np.array(s['s2']).reshape(7, 7), d)
        assert abs(approx - exact) < 0.01 * abs(exact) + 0.05, (approx, exact)


def _season(seed, drift, dates=120, games=12, per_game=70):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(dates):
        for g in range(games):
            p, y = _pas(rng, per_game, drift)
            half = per_game // 2
            out.append({'game_pk': k * 100 + g, 'date': '2025-%03d' % k,
                        'real_pa': {'starter': _sums(p[:half], y[:half]), 'reliever': _sums(p[half:], y[half:])}})
    return out


def test_a_planted_drift_is_found_and_no_drift_gains_nothing():
    drift = np.array([0.0, 0.06, -0.08, 0.0, 0.05, -0.07, 0.0])
    res = {'2025': lc.season_check(_season(1, drift), draws=300), '2026': lc.season_check(_season(2, drift), draws=300)}
    for s in ('2025', '2026'):
        off = res[s]['season_offsets']
        assert abs(off['K'] - 0.06) < 0.03 and abs(off['BB_HBP'] + 0.08) < 0.04
        assert all(v['gain_per_1000'] > 0 for v in res[s]['by_m'].values())
    cross = lc.cross_season(res)
    assert all(v['holds'] for v in cross.values()) and all(v['m_from_other'] in lc.GRID for v in cross.values())
    flat = {'2025': lc.season_check(_season(3, np.zeros(7)), draws=300), '2026': lc.season_check(_season(4, np.zeros(7)), draws=300)}
    for s in ('2025', '2026'):
        assert flat[s]['by_m']['50000']['gain_ci_per_1000'][0] < 0.05      # nothing to find: no clear gain
    assert not all(v['holds'] for v in lc.cross_season(flat).values())
