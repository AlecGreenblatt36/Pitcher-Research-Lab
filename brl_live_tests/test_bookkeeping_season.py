"""The sealed season backfill feeds the pitch bridge, behind the day cache and never past the cutoff."""
import gzip, json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from brl_live.bookkeeping_season import SCHEMA, load_season, season_rows, season_path, season_purpose
from cloud.security import seal
from app.safety import Blocked


class Store:
    def __init__(self): self.files = {}
    def read(self, path): return (self.files[path], 'sha') if path in self.files else None
    def put(self, path, raw, immutable=False): self.files[path] = raw


KEY = bytes(range(32))


def sealed(doc, year=2026):
    return seal(gzip.compress(json.dumps(doc).encode()), KEY, season_purpose(year))


def row(i, o='BIP_OUT', e='field_out', p=10, b=20, s='R', pt=None, c='ground'):
    return {'i': i, 'o': o, 'e': e, 'p': p, 'b': b, 's': s, 'n': 3, 'pt': pt or [{'b': 0, 's': 0, 'r': 'X', 't': 'FF', 'v': 94.0}], 'c': {'t': 'G', 'loc': 6} if c == 'ground' else c}


def test_load_and_rows_respect_cutoff_and_overlap():
    doc = {'schema': SCHEMA, 'year': 2026, 'games': {'1': {'date': '2026-09-01', 'fetched_at': '2026-10-01T00:00:00+00:00', 'rows': [row(0), row(1, 'HR', 'home_run', c={'t': 'F', 'loc': 7, 'dist': 400})]},
                                                       '2': {'date': '2026-10-06', 'fetched_at': '2026-10-07T00:00:00+00:00', 'rows': [row(0)]},
                                                       '3': {'date': '2026-10-07', 'fetched_at': '2026-10-08T00:00:00+00:00', 'rows': [row(0)]}}}
    store = Store(); store.files[season_path(2026)] = sealed(doc)
    assert load_season(store, KEY, 2025) is None
    loaded = load_season(store, KEY, 2026)
    assert loaded['games'].keys() == doc['games'].keys()
    seen = {(1, 1)}                                   # the day cache already holds game 1's second plate appearance
    rows = season_rows(loaded, '2026-10-07', seen)
    assert [(r['date_key'], r['terminal_event']) for r in rows] == [('2026-09-01', 'field_out'), ('2026-10-06', 'field_out')]
    assert rows[0]['pitches'][0]['r'] == 'X' and rows[0]['contact'] == {'t': 'G', 'loc': 6} and rows[0]['batter'] == 20 and rows[0]['pitch_number'] == 3
    assert (2, 0) in seen and (3, 0) not in seen


def test_identity_mismatch_is_blocked():
    store = Store(); store.files[season_path(2026)] = sealed({'schema': 'other', 'year': 2026, 'games': {}})
    with pytest.raises(Blocked):
        load_season(store, KEY, 2026)
    store.files[season_path(2026)] = sealed({'schema': SCHEMA, 'year': 2025, 'games': {}})
    with pytest.raises(Blocked):
        load_season(store, KEY, 2026)


def test_history_from_cache_merges_the_season_behind_the_day_cache():
    from brl_live.boxscore import bookkeeping_history_from_cache

    class Cache:
        def __init__(self, store): self.store, self.key = store, KEY
        def load_day(self, ident): raise AssertionError('no day blobs in this test')
    store = Store()
    doc = {'schema': SCHEMA, 'year': 2026, 'games': {'5': {'date': '2026-08-15', 'fetched_at': '2026-10-01T00:00:00+00:00', 'rows': [row(0), row(1, 'K', 'strikeout', pt=[{'b': 0, 's': 0, 'r': 'C', 't': 'SL', 'v': 85.0}] * 3, c=None)]}}}
    store.files[season_path(2026)] = sealed(doc)
    frame = bookkeeping_history_from_cache(Cache(store), {'days': {}}, datetime(2026, 10, 7, 12, tzinfo=timezone.utc))
    assert len(frame) == 2 and frame.attrs['season_games'] == 1 and set(frame.columns) >= {'pitches', 'contact', 'batter', 'pitcher', 'stand', 'outcome'}
    assert frame.contact.iloc[1] is None and frame.pitches.iloc[1][0]['t'] == 'SL'
    with pytest.raises(Blocked):
        bookkeeping_history_from_cache(Cache(Store()), {'days': {}}, datetime(2026, 10, 7, 12, tzinfo=timezone.utc))


def test_game_context_from_schedule_and_feed():
    from cloud.runner import schedule_context
    from brl_live.box_runner import game_context
    g = {'gamePk': 1, 'venue': {'name': 'Progressive Field'}, 'seriesDescription': 'Division Series', 'seriesGameNumber': 3, 'gamesInSeries': 5, 'doubleHeader': 'N',
         'gameNumber': 1, 'dayNight': 'day', 'description': '', 'teams': {'away': {'leagueRecord': {'wins': 2, 'losses': 0, 'pct': '1.000'}}, 'home': {'leagueRecord': {'wins': 0, 'losses': 2}}}}
    c = schedule_context(g)
    assert c == {'venue': 'Progressive Field', 'series_description': 'Division Series', 'series_game_number': 3, 'games_in_series': 5, 'double_header': 'N',
                 'game_number': 1, 'day_night': 'day', 'records': {'away': {'wins': 2, 'losses': 0}, 'home': {'wins': 0, 'losses': 2}}}
    assert schedule_context({'teams': {}}) == {'records': {'away': None, 'home': None}}
    gd = {'venue': {'name': 'Progressive Field'}, 'weather': {'condition': 'Clear', 'temp': '71', 'wind': '8 mph, Out To CF', 'other': 'x'}, 'datetime': {'officialDate': '2026-10-07'}}
    out = game_context({'context': c}, gd, '2026-10-07T12:00:00+00:00')
    assert out['weather'] == {'condition': 'Clear', 'temp': '71', 'wind': '8 mph, Out To CF'} and out['date'] == '2026-10-07' and out['series_game_number'] == 3
    assert game_context(None, {}, 'now') == {'updated_at': 'now'}


def test_physics_table_round_trip():
    from brl_live.bookkeeping_season import load_physics_table, physics_path, physics_purpose
    import pandas as pd
    store = Store()
    assert load_physics_table(store, KEY, 2026) is None
    frame = pd.DataFrame({'date_key': ['2026-04-01'], 'game_pk': [1], 'at_bat_number': [1], 'pitcher': [2], 'batter': [3], 'n': [4.0], 'ev': [100.5], 'run_value': [0.88]})
    store.files[physics_path(2026)] = seal(gzip.compress(frame.to_csv(index=False).encode()), KEY, physics_purpose(2026))
    back = load_physics_table(store, KEY, 2026)
    assert back.date_key.iloc[0] == '2026-04-01' and back.ev.iloc[0] == 100.5 and back.run_value.iloc[0] == 0.88
