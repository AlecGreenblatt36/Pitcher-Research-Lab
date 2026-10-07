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


def test_unlisted_statcast_event_on_a_cut_short_plate_appearance_is_excluded():
    # at-bat 2 ended on a runner event the official feed does not count as a plate appearance
    rows = [pitch(1, 1, 100, 500), pitch(1, 2, 100, 500, 'strikeout'),
            pitch(2, 1, 101, 500), pitch(2, 2, 101, 500, 'other_out'),
            pitch(3, 1, 102, 500, 'single')]
    body = statcast_csv(rows)
    source = Source('statcast', 'https://baseballsavant.mlb.com/statcast_search/csv?x', '2026-10-07T06:00:00+00:00', '2026-10-07T06:00:05+00:00', body)
    expected = {7: {1: {'outcome': 'K', 'batter': 100, 'pitcher': 500}, 3: {'outcome': '1B', 'batter': 102, 'pitcher': 500}}}
    out, diagnostics = collapse_statcast(source, '2026-10-06', expected, datetime(2026, 10, 7, 7, tzinfo=timezone.utc))
    assert [r['at_bat_number'] for r in out] == [1, 3]
    assert diagnostics['unlisted_events'] == {'other_out': 1} and diagnostics['excluded_non_PA_events'] == 1
    # the same event on a plate appearance the official feed counts is a disagreement, named in the block
    counted = {7: {**expected[7], 2: {'outcome': 'BIP_OUT', 'batter': 101, 'pitcher': 500}}}
    with pytest.raises(Blocked, match='other_out'):
        collapse_statcast(source, '2026-10-06', counted, datetime(2026, 10, 7, 7, tzinfo=timezone.utc))


def test_capture_notes_reach_the_history_index(monkeypatch):
    from brl_live import history_refresh as hr

    class Store:
        def __init__(self):
            self.files = {}
        def read(self, path):
            return (self.files[path], 'sha') if path in self.files else None
        def put(self, path, raw, immutable=False):
            self.files[path] = raw

    key = bytes(range(32))
    cache = hr.HistoryCache(Store(), key)
    day_before = (datetime(2026, 10, 7, 7, tzinfo=timezone.utc))
    monkeypatch.setattr(cache, 'clock', lambda: day_before)
    base = {'schema': hr.SCHEMA, 'days': {}, 'coverage_through': None}
    monkeypatch.setattr(cache, 'index', lambda: base)

    def fake_capture(day, fetcher, clock):
        return {'schema': hr.SCHEMA, 'day': day, 'captured_at': '2026-10-07T06:00:05+00:00', 'status': 'complete_day',
                'n_games': 1, 'n_PA': 2, 'rows': [], 'results': [], 'exclusions': [], 'sources': [],
                'diagnostics': {'raw_pitch_rows': 5, 'mapped_PAs': 2, 'excluded_non_PA_events': 1,
                                'unlisted_events': {'other_out': 1}, 'mid_pa_substitutions': 1, 'terminal_pitch_counts': []}}
    monkeypatch.setattr(hr, 'capture_day', fake_capture)
    monkeypatch.setattr(hr, 'capture_seed_baseline', lambda fetcher, clock: dict(fake_capture(hr.BASE_END, fetcher, clock), diagnostics={'purpose': 'seed'}))
    monkeypatch.setattr(hr.HistoryCache, 'require_complete', staticmethod(lambda index, target: None))
    index = cache.refresh(object(), recheck_days=1)
    assert index['coverage_through'] == '2026-10-06'
    assert index['days']['2026-10-06']['notes'] == {'unlisted_events': {'other_out': 1}, 'mid_pa_substitutions': 1, 'excluded_non_PA_events': 1}
    assert index['days'][hr.BASE_END]['notes'] == {}
