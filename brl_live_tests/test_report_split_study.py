"""SPLITS-01's study on made-up seasons: hitters whose own split by pitcher hand is real keep it a season later and the
shrunk estimate beats both raw splits and the league's split; with no own splits the fit leans on the league's split
and the carry-over is near zero. Switch hitters are left out."""
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_split_study_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _league(seed=1, hitters=300, pitches=1400, own_sd=0.12, switch=10):
    rng = np.random.default_rng(seed)
    side = (np.arange(hitters) % 3 != 0).astype(int)                    # two righties for every lefty
    level = rng.normal(0, 0.25, hitters); clevel = rng.normal(0, 0.4, hitters)
    own = rng.normal(0, own_sd, hitters); cown = rng.normal(0, own_sd, hitters)       # his own extra split, every season
    cols = {k: [] for k in ('season', 'day', 'batter', 'stand_r', 'throw_r', 'zone', 'call', 'last_in_pa', 'out7', 'post')}
    for season in (2023, 2024, 2025, 2026):
        n = hitters * pitches
        b = np.repeat(np.arange(hitters), pitches)
        lefty_p = rng.random(n) < np.where(side[b] == 1, 0.28, 0.4)       # lefty pitchers face more lefty batters
        same = (side[b] == 1) != lefty_p                                   # a righty batter against a righty pitcher, and so on
        sgn = np.where(lefty_p, 1.0, -1.0)
        stand = side[b].copy()
        sw_h = b < switch                                                  # switch hitters bat from the side away from the pitcher
        stand[sw_h] = np.where(lefty_p[sw_h], 1, 0)
        ooz = rng.random(n) < 0.45
        zone = np.where(ooz, rng.integers(11, 15, n), rng.integers(1, 10, n))
        p_sw = 1 / (1 + np.exp(-(np.where(ooz, -0.9, 0.7) + np.where(ooz, clevel[b] + 0.15 * same + 0.5 * sgn * cown[b], 0.0))))
        swing = rng.random(n) < p_sw
        whiff = swing & (rng.random(n) < 0.24)
        last = rng.random(n) < 0.26
        p_hit = 1 / (1 + np.exp(-(-1.05 + level[b] - 0.18 * same + 0.5 * sgn * own[b])))
        hit = last & (rng.random(n) < p_hit)
        kind = rng.choice(np.array([3, 3, 3, 4, 5]), n)
        out7 = np.where(hit, kind, np.where(last & (rng.random(n) < 0.1), 2, 0))      # 2: a walk (not an at-bat)
        start = date(season, 4, 1).toordinal()
        cols['season'].append(np.full(n, season)); cols['day'].append(start + rng.integers(0, 180, n)); cols['batter'].append(5000 + b)
        cols['stand_r'].append(stand); cols['throw_r'].append((~lefty_p).astype(int)); cols['zone'].append(zone)
        cols['call'].append(np.where(whiff, 2, np.where(swing, 1, 0))); cols['last_in_pa'].append(last.astype(int)); cols['out7'].append(out7)
        cols['post'].append(np.zeros(n, int))
    return {k: np.concatenate(v) for k, v in cols.items()}


SPEC = {'min_pa': 200, 'min_cell': 60, 'boot': 300, 'score': {'train': [2024, 2025], 'test': 2026}}


def test_planted_splits_carry_over_and_shrinking_beats_both():
    B = _report()
    out = B.split_shrink_study(_league(), lambda *a: None, SPEC)
    assert out['hitters']['score'] == 290                      # ten switch hitters left out
    for name in ('avg', 'chase'):
        r = out['metrics'][name]
        assert r['shrunk_minus_raw'][2] < 0, (name, r)
        assert r['shrunk_minus_scaled'][2] < 0, (name, r)      # his own split adds to his level and his side's split
        assert r['m'] < 1e6 and r['carry']['r'] > 0.15, (name, r)
    assert set(out['metrics']) == {'avg', 'slg', 'chase', 'miss'}


def test_no_own_splits_leans_on_the_league_split():
    B = _report()
    out = B.split_shrink_study(_league(seed=2, own_sd=0.0), lambda *a: None, SPEC)
    for name in ('avg', 'chase'):
        r = out['metrics'][name]
        assert r['m'] >= 800, (name, r)
        assert r['shrunk_minus_raw'][2] < 0
        assert abs(r['carry']['r']) < 0.2, (name, r)
    # the league split is there for both sides: same-side matchups hit less
    c = out['metrics']['avg']['carry']
    assert c['league_split_lefty_batters'] < 0 < c['league_split_righty_batters']


def test_test_end_drops_late_test_pitches():
    B = _report()
    T = _league(seed=3, hitters=60, pitches=600)
    a = B.split_shrink_study(T, lambda *a: None, dict(SPEC, min_pa=50))
    b = B.split_shrink_study(T, lambda *a: None, dict(SPEC, min_pa=50, score={'train': [2024, 2025], 'test': 2026, 'test_end': '2026-06-01'}))
    assert b['metrics']['avg']['events'] < a['metrics']['avg']['events']


def _pitchers(seed=11, pitchers=260, pitches=1400, own_sd=0.12):
    """Pitchers with a level and (own_sd > 0) an own extra split against left-handed batters, on top of the league's split
    for their hand (same-side batters hit less)."""
    rng = np.random.default_rng(seed)
    rhp = (np.arange(pitchers) % 10 < 7).astype(int)
    level = rng.normal(0, 0.2, pitchers); own = rng.normal(0, own_sd, pitchers); cown = rng.normal(0, own_sd, pitchers)
    cols = {k: [] for k in ('season', 'day', 'pitcher', 'batter', 'stand_r', 'throw_r', 'zone', 'call', 'last_in_pa', 'out7', 'post')}
    for season in (2023, 2024, 2025, 2026):
        n = pitchers * pitches
        p = np.repeat(np.arange(pitchers), pitches)
        lhb = rng.random(n) < 0.4
        same = (rhp[p] == 1) != lhb
        sgn = np.where(lhb, 1.0, -1.0)
        ooz = rng.random(n) < 0.45
        zone = np.where(ooz, rng.integers(11, 15, n), rng.integers(1, 10, n))
        p_sw = 1 / (1 + np.exp(-(np.where(ooz, -0.9, 0.7) + np.where(ooz, 0.5 * sgn * cown[p] - 0.1 * same, 0.0))))
        swing = rng.random(n) < p_sw; whiff = swing & (rng.random(n) < 0.24)
        last = rng.random(n) < 0.26
        hit = last & (rng.random(n) < 1 / (1 + np.exp(-(-1.05 - level[p] - 0.18 * same + 0.5 * sgn * own[p]))))
        out7 = np.where(hit, rng.choice(np.array([3, 3, 3, 4, 5]), n), np.where(last & (rng.random(n) < 0.1), 2, 0))
        cols['season'].append(np.full(n, season)); cols['day'].append(date(season, 4, 1).toordinal() + rng.integers(0, 180, n))
        cols['pitcher'].append(7000 + p); cols['batter'].append(rng.integers(0, 500, n)); cols['stand_r'].append((~lhb).astype(int))
        cols['throw_r'].append(rhp[p]); cols['zone'].append(zone); cols['call'].append(np.where(whiff, 2, np.where(swing, 1, 0)))
        cols['last_in_pa'].append(last.astype(int)); cols['out7'].append(out7); cols['post'].append(np.zeros(n, int))
    return {k: np.concatenate(v) for k, v in cols.items()}


def test_pitcher_splits_by_batter_side():
    B = _report()
    spec = dict(SPEC, who='pitcher')
    out = B.split_shrink_study(_pitchers(), lambda *a: None, spec)
    assert out['who'] == 'pitcher' and out['hitters']['score'] == 260
    r = out['metrics']['avg']
    assert r['shrunk_minus_raw'][2] < 0 and r['shrunk_minus_scaled'][2] < 0 and r['carry']['r'] > 0.15, r
    # the league's split by the pitcher's hand: same-side batters hit less, so a righty allows more to lefties
    assert r['carry']['league_split_righty_pitchers'] > 0 > r['carry']['league_split_lefty_pitchers']
    null = B.split_shrink_study(_pitchers(seed=12, own_sd=0.0), lambda *a: None, spec)['metrics']['avg']
    assert null['m'] >= 800 and abs(null['carry']['r']) < 0.2, null
