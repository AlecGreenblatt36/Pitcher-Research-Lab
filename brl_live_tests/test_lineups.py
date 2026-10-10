"""The lineup before one is posted (brl_live/lineups.py) and the report lane's study of it (LINEUP-01, LINEUP-02)."""
import importlib.util
from pathlib import Path

import numpy as np

from brl_live import lineups as LU

ROOT = Path(__file__).resolve().parents[1]
VS_R = [1, 2, 3, 4, 5, 6, 7, 8, 9]
VS_L = [1, 2, 3, 4, 5, 6, 17, 18, 19]          # three platoon bats against lefties


def _g(day, pk, order, hand, extra=()):
    return (day, pk, list(order), hand, set(order) | set(extra))


def test_last_rule_is_the_latest_complete_game():
    prior = [_g(10, 1, VS_R, 'R'), _g(11, 2, VS_L, 'L'), _g(12, 3, VS_R[:8], 'R')]
    assert LU.project(prior, 13, 'R', 'last') == (VS_L, 'last game')
    assert LU.project(prior, 11, 'R', 'last') == (VS_R, 'last game')          # nothing from the day itself
    assert LU.project([], 13, 'R', 'last') == (None, None)


def test_hand_rule_takes_the_last_lineup_against_that_hand():
    prior = [_g(10, 1, VS_L, 'L'), _g(11, 2, VS_R, 'R'), _g(12, 3, VS_R, 'R')]
    assert LU.project(prior, 13, 'L', 'hand') == (VS_L, 'last game against a lefty')
    assert LU.project(prior, 13, 'R', 'hand') == (VS_R, 'last game against a righty')
    # too long ago: back to the last game
    assert LU.project(prior, 10 + LU.HAND_DAYS + 1, 'L', 'hand') == (VS_R, 'last game')
    assert LU.project(prior, 13, None, 'hand') == (VS_R, 'last game')


def test_freq_rule_is_the_usual_lineup_and_skips_a_rest_day():
    rested = [1, 2, 3, 4, 5, 6, 7, 8, 29]                  # the number 9 hitter sat once
    prior = [_g(d, d, VS_R, 'R') for d in range(1, 6)] + [_g(6, 6, rested, 'R', extra=[9])]
    order, src = LU.project(prior, 7, 'R', 'freq')
    assert order == VS_R and src == 'usual lineup against a righty'
    assert LU.project(prior, 7, 'R', 'hand')[0] == rested
    # someone who has not batted in the team's last games is left out (a hitter who went on the injured list)
    gone = [_g(d, d, VS_R, 'R') for d in range(1, 6)] + [_g(d, d, [1, 2, 3, 4, 5, 6, 7, 8, 39], 'R') for d in range(6, 12)]
    order, _ = LU.project(gone, 12, 'R', 'freq')
    assert 9 not in order and 39 in order


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_lineups_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_study_counts_starters_named_by_each_rule(monkeypatch):
    B = _report()
    rows, sides, pk = [], {}, 1000
    rng = np.random.default_rng(0)
    for season, base in ((2025, 739000), (2026, 739040)):
        for k in range(40):
            pk += 1
            hand = 'L' if rng.random() < 0.3 else 'R'
            sides[pk] = ({'away': 1, 'home': 2}, 'R')
            for half, order in ((0, VS_L if hand == 'L' else VS_R), (1, [101 + j for j in range(9)])):
                for ab, bat in enumerate(order * 2, start=1):
                    for pn in range(2):
                        rows.append((pk, base + k, half, bat, 900, ab, pn, 1 if hand == 'R' or half == 1 else 0, season))
    a = np.array(rows)
    T = {'game': a[:, 0], 'day': a[:, 1], 'half': a[:, 2], 'batter': a[:, 3], 'pitcher': a[:, 4], 'ab': a[:, 5], 'pitch_no': a[:, 6],
         'throw_r': a[:, 7], 'season': a[:, 8]}
    monkeypatch.setattr(B, 'team_sides', lambda lo, hi: sides)
    out = B.lineup_study(T, lambda *_: None, {'evaluate': [2026]})
    s = out['seasons']['2026']['all']
    assert s['team_games'] == 80
    assert s['hand']['overlap'] == 9.0 and s['freq']['overlap'] == 9.0      # the platoon is caught every time
    assert s['last']['overlap'] < 9.0
    assert s['hand_minus_last']['overlap'] > 0
    changed = out['seasons']['2026']['opposing_hand_changed']
    assert changed['team_games'] > 0 and changed['last']['overlap'] == 6.0
