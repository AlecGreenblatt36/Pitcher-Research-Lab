"""Outs each pitcher recorded, read from plate appearances (brl_replay/harness.pitcher_outs, for PLAYER-07): within a
half-inning from the next plate appearance's outs, 3 at a half-inning's end, a caught stealing between plate appearances
to the pitcher on the mound, a walk-off's last half-inning ending short of three, and a game-ending double play in the top
of the last inning counting both outs."""
import pandas as pd

from brl_replay.harness import pitcher_outs


def _pa(rows):
    return pd.DataFrame(rows, columns=['game_pk', 'at_bat_number', 'inning', 'inning_topbot', 'outs_when_up', 'pitcher', 'outcome'])


ROWS = [
    # game 1. top 1st, starter 10: single, strikeout, (runner caught stealing: 2 outs before the next), groundout
    (1, 1, 1, 'Top', 0, 10, '1B'), (1, 2, 1, 'Top', 0, 10, 'K'), (1, 3, 1, 'Top', 2, 10, 'BIP_OUT'),
    # bottom 1st, pitcher 20: three outs, the third by reliever 21
    (1, 4, 1, 'Bot', 0, 20, 'BIP_OUT'), (1, 5, 1, 'Bot', 1, 20, 'K'), (1, 6, 1, 'Bot', 2, 21, 'BIP_OUT'),
    # top 2nd, starter 10 again: a double play leaves two outs at once, then a strikeout
    (1, 7, 2, 'Top', 0, 10, '1B'), (1, 8, 2, 'Top', 0, 10, 'BIP_OUT'), (1, 9, 2, 'Top', 2, 10, 'K'),
    # bottom 2nd (the last), one out, then a walk-off single
    (1, 10, 2, 'Bot', 0, 22, 'K'), (1, 11, 2, 'Bot', 1, 22, '1B'),
    # game 2, won by the home team after the top of the 2nd: the road team's last half-inning ends on a double play
    (2, 1, 1, 'Top', 0, 30, 'K'), (2, 2, 1, 'Top', 1, 30, 'K'), (2, 3, 1, 'Top', 2, 30, 'K'),
    (2, 4, 1, 'Bot', 0, 40, 'HR'), (2, 5, 1, 'Bot', 0, 40, 'K'), (2, 6, 1, 'Bot', 1, 40, 'K'), (2, 7, 1, 'Bot', 2, 40, 'K'),
    (2, 8, 2, 'Top', 0, 31, '1B'), (2, 9, 2, 'Top', 0, 31, 'K'), (2, 10, 2, 'Top', 1, 31, 'BIP_OUT'),
]


def test_outs_by_pitcher_with_a_change_a_caught_stealing_a_walk_off_and_a_last_double_play():
    got = pitcher_outs(_pa(ROWS), home_won={1, 2})
    assert got == {(1, 10): 6, (1, 20): 2, (1, 21): 1, (1, 22): 1, (2, 30): 3, (2, 40): 3, (2, 31): 3}


def test_a_home_loss_in_the_last_bottom_half_ends_with_three_outs():
    rows = [(3, 1, 1, 'Top', 0, 50, 'HR'), (3, 2, 1, 'Top', 0, 50, 'K'), (3, 3, 1, 'Top', 1, 50, 'K'), (3, 4, 1, 'Top', 2, 50, 'K'),
            # bottom 1st, the last: a single, then a game-ending double play
            (3, 5, 1, 'Bot', 0, 60, '1B'), (3, 6, 1, 'Bot', 0, 60, 'K'), (3, 7, 1, 'Bot', 1, 60, 'BIP_OUT')]
    assert pitcher_outs(_pa(rows), home_won=set())[(3, 60)] == 3
    assert pitcher_outs(_pa(rows))[(3, 60)] == 2            # without the final scores the last bottom half is read as a walk-off
