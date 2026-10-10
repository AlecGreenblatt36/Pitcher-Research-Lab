"""The game plan's lineup before one is posted: the team's last lineup (as the simulator projects it), then the bench."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_lineup_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table():
    rows = []

    def game(pk, day, half, order, pitchers):
        ab = 1
        for _ in range(2):
            for b in order:
                for pn in range(3):
                    rows.append((pk, day, half, b, pitchers[min(ab // 6, len(pitchers) - 1)], ab, pn))
                ab += 1
    game(10, 5, 0, [101, 102, 103, 104, 105, 106, 107, 108, 109], [900])
    game(10, 5, 1, [201, 202, 203, 204, 205, 206, 207, 208, 209], [800, 801])
    game(11, 6, 1, [102, 101, 103, 104, 110, 106, 107, 108, 109], [901])
    game(11, 6, 0, [301, 302, 303, 304, 305, 306, 307, 308, 309], [700, 701])
    a = np.array(rows)
    return {'game': a[:, 0], 'day': a[:, 1], 'half': a[:, 2], 'batter': a[:, 3], 'pitcher': a[:, 4], 'ab': a[:, 5], 'pitch_no': a[:, 6]}


def test_last_lineup_is_the_latest_game_in_batting_order():
    B = _report()
    hitters, relievers, last = B.recent_players(_table(), {1: [10, 11]}, {(10, 1): 'away', (11, 1): 'home'})
    assert last[1] == [102, 101, 103, 104, 110, 106, 107, 108, 109]
    assert set(hitters[1]) == {101, 102, 103, 104, 105, 106, 107, 108, 109, 110}
    assert relievers[1] == [701]


def test_no_last_lineup_without_nine_batters():
    B = _report()
    T = _table()
    keep = ~((T['game'] == 11) & np.isin(T['batter'], (108, 109)))
    T = {k: v[keep] for k, v in T.items()}
    _, _, last = B.recent_players(T, {1: [11]}, {(11, 1): 'home'})
    assert 1 not in last
