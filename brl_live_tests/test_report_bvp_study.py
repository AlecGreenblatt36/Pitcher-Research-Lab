"""BVP-01's study on made-up seasons: hitters and pitchers with their own levels and a platoon edge; a planted effect of
each batter-pitcher pair on strikeouts is found at about its size and carries over, while on base (no pair effect planted)
shows none; with no pair effects anywhere the interval covers zero. The expectation leaves the pair's own meetings out."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_bvp_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(seed=1, batters=200, pitchers=90, pa_per_season=300, pair_k_sd=0.35):
    """Strikeouts and times on base for each plate appearance from logits: league + batter + pitcher + platoon + the pair's
    own strikeout effect (logit SD pair_k_sd); one plate appearance is one row (its last pitch)."""
    rng = np.random.default_rng(seed)
    bk, bo = rng.normal(0, 0.25, batters), rng.normal(0, 0.15, batters)
    pk, po = rng.normal(0, 0.25, pitchers), rng.normal(0, 0.12, pitchers)
    stand = rng.random(batters) < 0.6; throw = rng.random(pitchers) < 0.7
    pair_k = rng.normal(0, pair_k_sd, (batters, pitchers))
    cols = {k: [] for k in ('batter', 'pitcher', 'season', 'stand_r', 'throw_r', 'last_in_pa', 'out7', 'post')}
    for season in (2023, 2024, 2025, 2026):
        b = np.repeat(np.arange(batters), pa_per_season)
        p = rng.integers(0, pitchers, len(b))
        plat = np.where(stand[b] != throw[p], 0.08, -0.08)
        lk = -1.35 + bk[b] + pk[p] - 0.5 * plat + pair_k[b, p]
        k = rng.random(len(b)) < 1 / (1 + np.exp(-lk))
        lo = -0.75 + bo[b] + po[p] + plat
        on = ~k & (rng.random(len(b)) < 1 / (1 + np.exp(-lo)))     # on base among plate appearances without a strikeout
        hit = on & (rng.random(len(b)) < 0.7)
        out7 = np.where(k, 1, np.where(hit, 3, np.where(on, 2, 0)))
        cols['batter'].append(b + 1000); cols['pitcher'].append(p + 5000); cols['season'].append(np.full(len(b), season))
        cols['stand_r'].append(stand[b].astype(int)); cols['throw_r'].append(throw[p].astype(int)); cols['last_in_pa'].append(np.ones(len(b), int))
        cols['out7'].append(out7); cols['post'].append(np.zeros(len(b), int))
    T = {k: np.concatenate(v) for k, v in cols.items()}
    # the planted strikeout pair effect's variance on the probability scale, per plate appearance
    p_mean = 1 / (1 + np.exp(1.35))
    return T, (pair_k_sd * p_mean * (1 - p_mean)) ** 2


def test_planted_pair_effect_on_strikeouts_is_found_and_on_base_shows_none():
    rep = _report()
    T, tau2_k = _world()
    out = rep.bvp_study(T, lambda m: None, {'boot': 300})
    for lab in ('2023-2024 to 2025', '2023-2025 to 2026'):
        r = out['replications'][lab]
        k = r['k']
        assert k['tau2_ci'][0] > 0                                  # the strikeout pair effect is there
        assert 0.4 * tau2_k < k['tau2'] < 1.8 * tau2_k              # at about its planted size
        o = r['obp']
        assert o['tau2_ci'][0] < 0.5 * tau2_k and o['tau2'] < 0.4 * tau2_k   # none planted on base


def test_no_pair_effects_cover_zero():
    rep = _report()
    T, _ = _world(seed=3, pair_k_sd=0.0)
    out = rep.bvp_study(T, lambda m: None, {'boot': 300})
    for lab, r in out['replications'].items():
        for name in ('obp', 'k'):
            lo, hi = r[name]['tau2_ci']
            assert lo < 0 < hi or abs(r[name]['tau2']) < 2e-4, (lab, name, r[name])


def test_expected_leaves_the_pairs_own_meetings_out():
    rep = _report()
    P = {'batter': np.array([1, 1, 1, 2, 2]), 'pitcher': np.array([9, 9, 8, 9, 8]), 'season': np.full(5, 2025),
         'stand_r': np.ones(5, int), 'throw_r': np.ones(5, int), 'on': np.array([1.0, 1.0, 0.0, 0.0, 0.0])}
    e = rep.bvp_expected(P, 'on', None, 0.0)
    # batter 1 against pitcher 9: his other meeting (against 8) was an out, pitcher 9's other batter (2) was out
    assert e[0] < 1e-5 and e[1] < 1e-5
