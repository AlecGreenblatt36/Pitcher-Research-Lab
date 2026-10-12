import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('brl_keys_check', Path(__file__).resolve().parents[1] / 'tools' / 'brl_keys_check.py')
kc = importlib.util.module_from_spec(spec); spec.loader.exec_module(kc)


def test_first_inning_scored_reads_the_inning_starts_and_the_final():
    assert kc.first_inning_scored([[1, 0], [2, 0], [3, 2]], 4) == 2
    assert kc.first_inning_scored([[1, 0], [2, 0], [9, 0]], 1) == 9          # scored in his last turn: only the final says so
    assert kc.first_inning_scored([[1, 0], [2, 0]], 0) is None


def _game(away_runs, home_runs, starts, keys_away, keys_home, outs=(18, 12), hitters=()):
    return {'game_pk': 1, 'n': 10, 'away_runs': away_runs, 'home_runs': home_runs,
            'innings': {'actual_first_scores': starts},
            'starters': {'away': {'keys': keys_away, 'actual': {'outs': outs[0]}}, 'home': {'keys': keys_home, 'actual': {'outs': outs[1]}}},
            'hitters': list(hitters)}


def test_season_check_scores_shares_and_conditional_win_chances():
    # away scores in the 2nd, home in the 1st: home scored first; away won 3-2
    g = _game(3, 2, {'away': [[1, 0], [2, 0], [3, 1]], 'home': [[1, 0], [2, 1]]},
              keys_away=[4, 3, 5, 6, 4], keys_home=[5, 3, 5, 2, 1],
              hitters=[{'side': 'away', 'actual': {'hr': 1, 'h': 1}, 'hr_hist': [8, 2, 0, 0], 'h_hist': [3, 4, 3, 0, 0, 0], 'kw': [2, 2]}])
    r = kc.season_check([g])['all']
    f = r['first']
    assert f['n'] == 2 and f['share_sim'] == 45.0 and f['share_real'] == 50.0
    # the home team scored first (it lost; chance given first 3/5), the away team did not (it won; chance given not (5-3)/6)
    assert f['if_n'] == 1 and f['if_sim'] == 60.0 and f['if_real'] == 0.0
    assert f['else_n'] == 1 and abs(f['else_sim'] - 100 * 2 / 6) < 0.01 and f['else_real'] == 100.0
    s = r['sp6']   # away starter got 18 outs (chance given 4/6), home did not (chance given not (5-1)/8)
    assert s['if_n'] == 1 and abs(s['if_sim'] - 100 * 4 / 6) < 0.01 and s['if_real'] == 100.0
    assert s['else_n'] == 1 and s['else_sim'] == 50.0 and s['else_real'] == 0.0
    h = r['hr']    # he homered in 2 of 10 worlds, the team won both; it did homer and win
    assert h['share_sim'] == 20.0 and h['if_sim'] == 100.0 and h['if_real'] == 100.0
    assert r['h2']['share_sim'] == 30.0 and r['h2']['else_n'] == 1


def test_season_check_leaves_out_ties_and_unknown_firsts():
    g = _game(2, 2, {'away': [[1, 0]], 'home': [[1, 0]]}, [5, 2, 5, 5, 2], [5, 3, 5, 5, 3])
    out = kc.season_check([g])
    assert out['games_skipped_tied'] == 1 and out['all']['first']['n'] == 0
    g2 = _game(1, 0, {}, [5, 2, 5, 5, 2], [5, 3, 5, 5, 3])
    assert kc.season_check([g2])['all']['first']['n'] == 0 and kc.season_check([g2])['all']['sp6']['n'] == 2
