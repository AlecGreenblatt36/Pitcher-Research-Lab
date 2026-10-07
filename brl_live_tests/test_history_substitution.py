"""A plate appearance finished by a different batter or pitcher than it started with is attributed to the official players."""
import io
from datetime import datetime, timezone
import pandas as pd
import pytest
from brl_live.history_refresh import collapse_statcast, Source, SOURCE_FIELDS
from app.safety import Blocked


def statcast_csv(rows):
    cols = ['game_date', 'game_year', 'game_pk', 'at_bat_number', 'pitch_number', 'events', 'batter', 'pitcher', 'stand', 'p_throws', 'home_team', 'away_team',
            'inning', 'inning_topbot', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'home_score', 'away_score', 'bat_score', 'fld_score', 'bat_score_diff',
            'n_thruorder_pitcher', 'batter_days_since_prev_game', 'pitcher_days_since_prev_game', 'age_bat', 'age_pit', 'fielder_2', 'game_type']
    frame = pd.DataFrame(rows, columns=cols)
    return frame.to_csv(index=False).encode()


def pitch(ab, n, batter, pitcher, events=None):
    return ['2026-10-06', 2026, 7, ab, n, events, batter, pitcher, 'R', 'L', 'HOM', 'AWY', 1, 'Top', 0, None, None, None, 0, 0, 0, 0, 0, 1, 2, 5, 28, 30, 999, 'D']


def test_mid_pa_substitution_is_attributed_to_the_finishing_players():
    rows = [pitch(1, 1, 100, 500), pitch(1, 2, 100, 500, 'strikeout'),            # ordinary PA
            pitch(2, 1, 101, 500), pitch(2, 2, 101, 501), pitch(2, 3, 101, 501, 'walk')]   # pitcher 501 relieved 500 mid-count
    body = statcast_csv(rows)
    source = Source('statcast', 'https://baseballsavant.mlb.com/statcast_search/csv?x', '2026-10-07T06:00:00+00:00', '2026-10-07T06:00:05+00:00', body)
    expected = {7: {1: {'outcome': 'K', 'batter': 100, 'pitcher': 500}, 2: {'outcome': 'BB_HBP', 'batter': 101, 'pitcher': 501}}}
    out, diagnostics = collapse_statcast(source, '2026-10-06', expected, datetime(2026, 10, 7, 7, tzinfo=timezone.utc))
    assert len(out) == 2 and diagnostics['mid_pa_substitutions'] == 1
    sub = [r for r in out if r['at_bat_number'] == 2][0]
    assert sub['pitcher'] == 501 and sub['batter'] == 101 and sub['outcome'] == 'BB_HBP' and sub['terminal_event'] == 'walk'
    # the official feed disagreeing with the finishing players still blocks
    wrong = {7: {1: expected[7][1], 2: {'outcome': 'BB_HBP', 'batter': 101, 'pitcher': 500}}}
    with pytest.raises(Blocked):
        collapse_statcast(source, '2026-10-06', wrong, datetime(2026, 10, 7, 7, tzinfo=timezone.utc))
