"""NRFI-01 and FIRST5-01's check (tools/brl_innings_cal.py) on made-up games: the real first-inning and five-inning
results are read from the scores by inning; when they come from the simulator's own chances the gaps are within noise,
and a planted excess of scoreless first innings is found."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_innings_cal', Path(__file__).parents[1] / 'tools/brl_innings_cal.py')
ic = importlib.util.module_from_spec(spec); spec.loader.exec_module(ic)


def _games(seed, n=2400, extra_scoreless=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        la, lh = rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65)                 # runs per inning
        w = 1000
        inn_a, inn_h = rng.poisson(la, (w, 9)), rng.poisson(lh, (w, 9))
        a1, h1 = inn_a[:, 0], inn_h[:, 0]
        a5, h5 = inn_a[:, :5].sum(1), inn_h[:, :5].sum(1)
        first = [int(((a1 + h1) == 0).sum()), int((a1 > 0).sum()), int((h1 > 0).sum())]
        m = np.clip(h5 - a5, -15, 15); f5m = np.bincount(m + 15, minlength=31); f5t = np.bincount(np.clip(a5 + h5, 0, 30), minlength=31)
        # the real game: one more world from the same chances, or a scoreless first more often than they say
        ra, rh = rng.poisson(la, 9), rng.poisson(lh, 9)
        if extra_scoreless and rng.random() < extra_scoreless:
            ra[0] = rh[0] = 0
        sa, sh = np.r_[0, np.cumsum(ra)], np.r_[0, np.cumsum(rh)]
        firsts = {'away': [[k + 1, int(sa[k])] for k in range(9)], 'home': [[k + 1, int(sh[k])] for k in range(9)]}
        out.append({'game_pk': i, 'date': '2025-%03d' % (i // 15), 'n': w, 'first_inning': first, 'f5_margin_hist': f5m.tolist(), 'f5_total_hist': f5t.tolist(), 'home_runs': int(sh[-1]),
                    'innings': {'actual_first_scores': firsts}})
    return out


def test_real_innings_are_read_from_the_scores_by_inning():
    g = {'home_runs': 5, 'innings': {'actual_first_scores': {'away': [[1, 0], [2, 2], [3, 2], [4, 2], [5, 3], [6, 3]], 'home': [[1, 0], [2, 0], [3, 1], [6, 4]]}}}
    assert ic.actual_innings(g) == (2, 0, 3, 4)
    # called in the top of the 6th: five full innings, the home team's five-inning score is its final
    g['innings']['actual_first_scores']['home'] = [[1, 0], [2, 0], [3, 1]]
    assert ic.actual_innings(g) == (2, 0, 3, 5)
    # shortened before the road team's 6th: the first inning still counts, five innings do not
    g['innings']['actual_first_scores']['away'] = [[1, 0], [2, 2], [3, 2]]
    assert ic.actual_innings(g) == (2, 0, None, None)


def test_own_chances_check_out_and_a_planted_scoreless_first_is_found():
    r = ic.season_check(_games(1), draws=400)
    for name in ('no_run_1st', 'road_scores_1st', 'home_scores_1st', 'home_leads_5', 'tied_5', 'over_4_5_thru_5'):
        assert abs(r[name]['gap_points']) < 2.0, (name, r[name])                  # 2,400 games: about 0.8 points of noise
    bad = ic.season_check(_games(2, extra_scoreless=0.12), draws=400)
    assert bad['no_run_1st']['gap_ci_points'][1] < 0                          # the simulator says scoreless too seldom
