"""PLAYER-03's scoring (tools/brl_player_cal.py) on synthetic seasons with known truth: a planted overstatement is found and
the cross-season map passes the registered rule; when the chances are right, the rule does not pass; the page's chance
(ownAtLeast) is computed from the histograms the replay records."""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_player_cal', Path(__file__).parents[1] / 'tools/brl_player_cal.py')
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)


def _season(rng, n, shift, slope=1.0, dates=180):
    p = rng.beta(6, 4, n)
    q = 1 / (1 + np.exp(-(shift + slope * np.log(p / (1 - p)))))
    y = rng.random(n) < q
    d = rng.integers(0, dates, n)
    return [{'date': '2026-%03d' % int(d[i]), 'spot': int(1 + i % 9), 'prior_pa': 400, 'p_hit': float(p[i]), 'y_hit': int(y[i])} for i in range(n)]


def test_planted_overstatement_is_found_and_the_map_passes():
    rng = np.random.default_rng(7)
    res = pc.study({'2025': _season(rng, 40000, -0.07, 0.95), '2026': _season(rng, 40000, -0.07, 0.95)}, draws=400)
    k = res['kinds']['hit']
    for s in ('2025', '2026'):
        r = k['by_season'][s]
        assert 1.0 < r['gap_points'] < 2.5                       # the chances run high by about 1.6 points
        assert abs(r['cal_gap_points']) <= 0.5 and r['brier_change_ci'][1] < 0
        a, b = r['map_from_other']
        assert -0.12 < a < -0.02 and 0.85 < b < 1.05
    assert k['passes_rule'] and len(k['map_both']) == 2


def test_right_chances_do_not_pass():
    rng = np.random.default_rng(11)
    res = pc.study({'2025': _season(rng, 40000, 0.0), '2026': _season(rng, 40000, 0.0)}, draws=400)
    k = res['kinds']['hit']
    assert not k['passes_rule']
    for s in ('2025', '2026'):
        assert abs(k['by_season'][s]['gap_points']) < 0.6


def test_rows_from_replay_files_use_the_pages_chance(tmp_path):
    g = {'game_pk': 1, 'date': '2026-05-01', 'hitters': [
        {'batter': 10, 'spot': 9, 'h_hist': [300, 400, 200, 100, 0, 0], 'hr_hist': [880, 110, 10, 0], 'bb_hist': [700, 250, 50, 0, 0],
         'k_hist': [400, 400, 200, 0, 0], 'prior_pa_365': 120, 'actual': {'pa': 4, 'h': 2, 'hr': 0, 'k': 1, 'bb': 0}},
        {'batter': 11, 'spot': 1, 'h_hist': [1000, 0, 0, 0, 0, 0], 'hr_hist': [1000, 0, 0, 0], 'bb_hist': [1000, 0, 0, 0, 0],
         'k_hist': [1000, 0, 0, 0, 0], 'prior_pa_365': 0, 'actual': None}]}
    f = tmp_path / 'x.jsonl.gz'
    with gzip.open(f, 'wt') as fh:
        fh.write(json.dumps(g) + '\n')
    rows = pc.load_rows([str(f)])
    assert len(rows) == 1                                         # a hitter without a line is left out
    r = rows[0]; keep = 0.927
    assert abs(r['p_hit'] - (1 - (300 + 400 * (1 - keep) + 200 * (1 - keep) ** 2 + 100 * (1 - keep) ** 3) / 1000)) < 1e-12
    p2 = 1 - (300 + 400 + 200 * ((1 - keep) ** 2 + 2 * keep * (1 - keep)) + 100 * ((1 - keep) ** 3 + 3 * keep * (1 - keep) ** 2)) / 1000
    assert abs(r['p_hit2'] - p2) < 1e-12
    assert (r['y_hit'], r['y_hit2'], r['y_hr'], r['y_walk'], r['y_k']) == (1, 1, 0, 0, 1)


def test_part_time_factors_are_found_and_lower_the_part_timers_chances():
    """PLAYER-05 on made-up hitter-games: part-timers (under 100 prior plate appearances) really bat 90% as often as their
    slot share says; the factor is recovered and their chances come down, while regulars stay put."""
    rng = np.random.default_rng(5)

    def season(n=6000):
        rows = []
        for i in range(n):
            prior = int(rng.choice([50, 150, 300, 600]))
            spot = int(1 + i % 9)
            keep = pc.STARTER_SHARE[spot - 1] * (0.90 if prior < 100 else 1.0)
            pa_sim = rng.poisson(4.2, 1000)
            hits = rng.binomial(pa_sim, 0.24)
            h_hist = np.bincount(np.minimum(hits, 5), minlength=6).tolist()
            zero = [1000, 0, 0, 0]
            real_pa = int(rng.binomial(5, 0.84 * keep))                   # mean 4.2 x keep, almost never zero
            rows.append({'date': '2025-%03d' % (i // 40), 'spot': spot, 'prior_pa': prior, 'mean_pa': float(pa_sim.mean()), 'actual': {'pa': real_pa, 'h': int(rng.binomial(real_pa, 0.24)), 'hr': 0, 'bb': 0, 'k': 0},
                         'h_hist': h_hist, 'hr_hist': zero, 'bb_hist': zero + [0], 'k_hist': zero + [0]})
        return rows
    res = pc.part_time_study({'2025': season(), '2026': season()}, draws=200)
    for s in ('2025', '2026'):
        f = res['factors'][s]
        assert 0.85 < f[0] < 0.96 and abs(f[3] - 1.0) < 0.04
        gaps = res['by_season'][s]['kinds']['hit']['gap_by_bin']
        assert gaps[0][1] < gaps[0][0] - 1.0 and abs(gaps[3][1] - gaps[3][0]) < 0.5


def test_two_or_more_total_bases_from_hits_by_singles():
    """PLAYER-06: the page's chance from the replay's hits-by-singles table, and the real outcome from hits and extra-base
    hits (a double or triple counts the same for two or more bases, so the history's 2B_3B class is enough)."""
    hs = [[0] * 6 for _ in range(6)]
    hs[0][0] = 250; hs[1][1] = 250; hs[1][0] = 250; hs[2][1] = 250          # none; a single; an extra-base hit; a single and an extra-base hit
    keep = pc.STARTER_SHARE[0]
    assert abs(pc.own_tb2(hs, 1) - (0 + 0 + keep + keep) / 4) < 1e-12
    g = {'game_pk': 1, 'date': '2026-05-01', 'hitters': [
        {'batter': 1, 'spot': 1, 'h_hist': [1000, 0, 0, 0, 0, 0], 'hr_hist': [1000, 0, 0, 0], 'bb_hist': [1000, 0, 0, 0, 0], 'k_hist': [1000, 0, 0, 0, 0],
         'hs_hist': hs, 'prior_pa_365': 500, 'actual': {'pa': 4, 'h': 1, 'hr': 0, 'xbh': 1, 'k': 0, 'bb': 0}},
        {'batter': 2, 'spot': 2, 'h_hist': [1000, 0, 0, 0, 0, 0], 'hr_hist': [1000, 0, 0, 0], 'bb_hist': [1000, 0, 0, 0, 0], 'k_hist': [1000, 0, 0, 0, 0],
         'hs_hist': hs, 'prior_pa_365': 500, 'actual': {'pa': 4, 'h': 1, 'hr': 0, 'xbh': 0, 'k': 0, 'bb': 0}}]}
    import tempfile, os
    d = tempfile.mkdtemp(); f = os.path.join(d, 'x.jsonl.gz')
    with gzip.open(f, 'wt') as fh:
        fh.write(json.dumps(g) + '\n')
    rows = pc.load_rows([f])
    assert [r['y_tb2'] for r in rows] == [1, 0]                                 # a double counts; one single does not


def test_paired_study_finds_a_level_fix_and_nothing_when_identical():
    """LEVEL-02's hitter check: two replays of the same hitter-games; one whose walk chances run 4 points high, the other
    on level. The level fix lowers the walk Brier; identical replays change nothing."""
    rng = np.random.default_rng(9)
    ref, new = [], []
    for i in range(30000):
        q = {'hit': rng.uniform(0.45, 0.7), 'hit2': rng.uniform(0.1, 0.3), 'hr': rng.uniform(0.05, 0.2), 'walk': rng.uniform(0.15, 0.45), 'k': rng.uniform(0.4, 0.8)}
        y = {k: int(rng.random() < v) for k, v in q.items()}
        base = {'date': '2025-%03d' % (i // 170), 'game_pk': i // 18, 'batter': i % 18, 'spot': 1 + i % 9, 'prior_pa': 300}
        ref.append({**base, **{'p_' + k: (v + 0.04 if k == 'walk' else v) for k, v in q.items()}, **{'y_' + k: v for k, v in y.items()}})
        new.append({**base, **{'p_' + k: v for k, v in q.items()}, **{'y_' + k: v for k, v in y.items()}})
    r = pc.paired_study(ref, new, draws=300)
    w = r['kinds']['walk']
    assert abs(w['ref_gap_points'] - 4.0) < 1.0 and abs(w['new_gap_points']) < 1.0
    assert w['brier_change_ci'][1] < 0 and r['summed']['brier_change_ci'][1] < 0
    assert r['kinds']['hit']['brier_change'] == 0.0
    same = pc.paired_study(ref, ref, draws=100)
    assert same['summed']['brier_change'] == 0.0 and same['pairs'] == 30000
