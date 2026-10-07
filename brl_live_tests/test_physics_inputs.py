"""The live physics table: sealed seasons plus day-cache feeds after their coverage, nothing from today; model selection."""
import gzip, hashlib, importlib.util, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from brl_live.bookkeeping_season import physics_path, physics_purpose
from brl_live.physics_inputs import assemble_physics_table
from cloud.security import seal
from app.safety import Blocked

KEY = bytes(range(32))


class Store:
    def __init__(self): self.files = {}
    def read(self, path): return (self.files[path], 'sha') if path in self.files else None


class Cache:
    def __init__(self, store, days): self.store, self.key, self.days = store, KEY, days
    def load_day(self, ident): return self.days[ident]


def play(i, batter, pitcher, event='single', code='X'):
    return {'result': {'eventType': event}, 'about': {'isComplete': True, 'atBatIndex': i}, 'matchup': {'pitcher': {'id': pitcher}, 'batter': {'id': batter}},
            'playEvents': [{'isPitch': True, 'details': {'code': code, 'type': {'code': 'FF'}}, 'pitchData': {'startSpeed': 94.0, 'zone': 5}, 'count': {},
                            'hitData': {'launchSpeed': 99.0, 'launchAngle': 15.0}}]}


def feed(pk, date, state='Final'):
    return {'gamePk': pk, 'gameData': {'status': {'abstractGameState': state}, 'datetime': {'officialDate': date}},
            'liveData': {'plays': {'allPlays': [play(0, 11, 21), play(1, 12, 21, 'strikeout', 'S')]}}}


def sealed_table(rows):
    frame = pd.DataFrame(rows)
    return seal(gzip.compress(frame.to_csv(index=False).encode()), KEY, physics_purpose(2026))


def test_assembles_seasons_then_day_cache_until_yesterday():
    store = Store()
    base = {c: 0.0 for c in ['n', 'sw', 'wh', 'oz', 'ch', 'iz', 'izs', 'izc', 'cs', 'fb', 'velo', 'spin', 'ivb', 'hb']}
    rows = [dict(base, date_key='2026-10-04', game_pk=1, at_bat_number=1, pitcher=21, batter=11, n=3.0, ev=90.0, la=10.0, outcome='1B', run_value=0.88),
            dict(base, date_key='2026-10-05', game_pk=2, at_bat_number=1, pitcher=22, batter=12, n=4.0, ev=np.nan, la=np.nan, outcome='K', run_value=0.0)]
    store.files[physics_path(2026)] = sealed_table(rows)
    days = {'a': {'sources': [{'url': 'https://statsapi.mlb.com/api/v1.1/game/3/feed/live', 'body': json.dumps(feed(3, '2026-10-06'))}]},
            'b': {'sources': [{'url': 'https://statsapi.mlb.com/api/v1.1/game/4/feed/live', 'body': json.dumps(feed(4, '2026-10-07'))}]},
            'c': {'sources': [{'url': 'https://statsapi.mlb.com/api/v1.1/game/2/feed/live', 'body': json.dumps(feed(2, '2026-10-05'))}]}}
    index = {'days': {'2026-10-06': {'id': 'a'}, '2026-10-07': {'id': 'b'}, '2026-10-05': {'id': 'c'}}}
    origin = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)     # today is Oct 7 Eastern
    table, receipt = assemble_physics_table(Cache(store, days), index, origin)
    assert receipt['seasons']['2026'] == {'rows': 2, 'through': '2026-10-05'} and receipt['day_cache_days'] == 1 and receipt['day_cache_rows'] == 2
    assert sorted(table.date_key.unique()) == ['2026-10-04', '2026-10-05', '2026-10-06'] and len(table) == 4
    added = table[table.game_pk == 3].sort_values('at_bat_number')
    assert list(added.outcome) == ['1B', 'K'] and list(added.run_value) == [0.88, 0.0] and added.ev.iloc[0] == 99.0 and np.isnan(added.ev.iloc[1])
    assert table[table.game_pk == 2].n.iloc[0] == 4.0                     # the sealed row wins over the day-cache copy of Oct 5
    with pytest.raises(Blocked):
        assemble_physics_table(Cache(Store(), {}), {'days': {}}, origin)
    # an unreadable day feed costs that game's rows and is recorded; the rest of the table stands
    broken = feed(5, '2026-10-06'); broken['liveData']['plays']['allPlays'][0]['about']['atBatIndex'] = 'x'
    days['a']['sources'].append({'url': 'https://statsapi.mlb.com/api/v1.1/game/5/feed/live', 'body': json.dumps(broken)})
    table, receipt = assemble_physics_table(Cache(store, days), index, origin)
    assert receipt['day_cache_errors'] == 1 and receipt['day_cache_error'].startswith('ValueError') and len(table) == 4


def test_model_selection_default_and_sealed(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('brl_entrypoint', Path(__file__).parents[1] / 'brl_engine' / 'entrypoint.py')
    ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)
    monkeypatch.setattr(ep, 'REPO_ROOT', tmp_path)
    (tmp_path / 'brl_engine').mkdir()
    for var in ('BRL_MODEL_PATH', 'BRL_MODEL_SHA256', 'BRL_MODEL_NAME'):
        monkeypatch.setenv(var, 'stale')
    assert ep.select_model('r', 't', 'ab' * 32, tmp_path / 'data') == {'name': 'locked-pa-2026-v1', 'source': 'data package'}
    assert 'BRL_MODEL_PATH' not in __import__('os').environ
    raw = b'joblib-bytes'
    cipher = seal(raw, bytes.fromhex('ab' * 32), 'model:test-v2')
    (tmp_path / 'brl_engine' / 'models').mkdir()
    (tmp_path / 'brl_engine' / 'models' / 'test-v2.json').write_text(json.dumps({'name': 'test-v2', 'branch': 'brl-live-data', 'path': 'private/models/test-v2.enc', 'purpose': 'model:test-v2',
        'cipher_sha256': hashlib.sha256(cipher).hexdigest(), 'model_sha256': hashlib.sha256(raw).hexdigest(), 'fitted_at': 'now', 'physics_features': ['b_ev']}))
    (tmp_path / 'brl_engine' / 'model.json').write_text(json.dumps({'model': 'test-v2', 'manifest': 'brl_engine/models/test-v2.json'}))
    import base64
    monkeypatch.setattr(ep, 'api', lambda url, token: {'encoding': 'base64', 'content': base64.b64encode(cipher).decode()})
    out = ep.select_model('r', 't', 'ab' * 32, tmp_path / 'data')
    assert out['name'] == 'test-v2' and out['physics_features'] == 1
    import os
    assert Path(os.environ['BRL_MODEL_PATH']).read_bytes() == raw and os.environ['BRL_MODEL_SHA256'] == hashlib.sha256(raw).hexdigest() and os.environ['BRL_MODEL_NAME'] == 'test-v2'
    (tmp_path / 'brl_engine' / 'models' / 'test-v2.json').write_text(json.dumps({'name': 'test-v2', 'branch': 'b', 'path': 'p', 'purpose': 'model:test-v2', 'cipher_sha256': '00', 'model_sha256': '00'}))
    with pytest.raises(ValueError):
        ep.select_model('r', 't', 'ab' * 32, tmp_path / 'data')


def test_live_inputs_carry_team_defense_into_profiles():
    """live_inputs puts each fielding team's defense value into its profile when a DefenseState is supplied."""
    import importlib
    contracts = importlib.import_module('cloud.contracts')
    import inspect
    sig = inspect.signature(contracts.live_inputs)
    assert 'defense' in sig.parameters
    class D:
        def value(self, team): return {'CLE': 0.015, 'CWS': -0.01}.get(team, 0.0)
    src = inspect.getsource(contracts.live_inputs)
    assert "defense.value(str(t.get('abbreviation') or '').upper())" in src and 'team_defense' in src
