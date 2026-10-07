"""The shared physics features: builder and live state agree; feed and study extraction agree."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research_lab.pa_model import physics as phys

spec = importlib.util.spec_from_file_location('brl_research_tool', Path(__file__).parents[1] / 'tools' / 'brl_research.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
spec2 = importlib.util.spec_from_file_location('brl_test_research', Path(__file__).parent / 'test_research_tool.py')
helpers = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(helpers)
spec3 = importlib.util.spec_from_file_location('brl_backfill_tool', Path(__file__).parents[1] / 'tools' / 'brl_bookkeeping_backfill.py')
backfill = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(backfill)


def study_table(study):
    return phys.table_from_study(study)


def test_builder_matches_the_research_tool_for_base_features():
    pa, study = helpers.synthetic(n_games=30)
    table = study_table(study)
    shared, audit = phys.build_features(pa, table, {})
    old, audit_old = tool.build_extras(pa, tool.per_pa_physics(study), {})
    assert list(shared.columns) == phys.BASE_FEATURES == tool.EXTRA
    assert np.allclose(shared.to_numpy(), old.to_numpy(), equal_nan=True)
    assert audit['ids_match'] == audit_old['ids_match'] == len(pa)


def test_state_at_a_date_matches_the_chronological_builder():
    pa, study = helpers.synthetic(n_games=40)
    table = study_table(study)
    pa_sorted = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    outcomes = pa_sorted.set_index(['game_pk', 'at_bat_number'])['outcome']
    table = table.join(outcomes, on=['game_pk', 'at_bat_number'])
    params = {'xvalue': True, 'recent_days': 3, 'k_rate': 80.0, 'pitch_types': True}
    built, _ = phys.build_features(pa, table, params)
    for day in ('2026-05-01', '2026-05-04', '2026-05-10'):
        state = phys.PhysicsState.build(table, day, params)
        rows = np.where(pa_sorted.date_key.to_numpy() == day)[0]
        assert len(rows)
        for r in rows[:30]:
            expected = built.iloc[r].to_numpy(float)
            got = state.features(int(pa_sorted.batter[r]), int(pa_sorted.pitcher[r]))
            assert np.allclose(got, expected, atol=1e-5), (day, r, got - expected)
    assert state.names == phys.feature_names(params) == phys.BASE_FEATURES + phys.XVALUE_FEATURES + phys.RECENT_FEATURES + phys.TYPE_FEATURES
    old = table.drop(columns=['br_n'])
    with pytest.raises(ValueError):
        phys.PhysicsState.build(old, '2026-05-04', params)


def test_feed_and_study_extraction_agree():
    play = {'result': {'eventType': 'single', 'description': 'x'}, 'about': {'isComplete': True, 'atBatIndex': 3},
            'matchup': {'pitcher': {'id': 1}, 'batter': {'id': 2}, 'batSide': {'code': 'L'}, 'pitchHand': {'code': 'R'}},
            'playEvents': [{'isPitch': True, 'details': {'code': 'B', 'type': {'code': 'FF'}}, 'count': {'balls': 1, 'strikes': 0},
                            'pitchData': {'startSpeed': 95.1, 'zone': 12, 'coordinates': {'pfxX': -6.0, 'pfxZ': 10.0}, 'breaks': {'spinRate': 2300}}},
                           {'isPitch': False},
                           {'isPitch': True, 'details': {'code': 'S', 'type': {'code': 'SL'}}, 'count': {'balls': 1, 'strikes': 1}, 'pitchData': {'startSpeed': 85.0, 'zone': 14}},
                           {'isPitch': True, 'details': {'code': 'X', 'type': {'code': 'SI'}}, 'count': {'balls': 1, 'strikes': 1},
                            'pitchData': {'startSpeed': 93.0, 'zone': 5, 'coordinates': {'pfxX': 8.0, 'pfxZ': 4.0}, 'breaks': {'spinRate': 2100}},
                            'hitData': {'launchSpeed': 101.3, 'launchAngle': 12.0, 'totalDistance': 250}}]}
    feed = {'gamePk': 77, 'gameData': {'datetime': {'officialDate': '2026-06-01'}}, 'liveData': {'plays': {'allPlays': [play, dict(play, about={'isComplete': False, 'atBatIndex': 4})]}}}
    from_feed = phys.table_from_feed(feed, lambda e: {'single': '1B'}.get(e))
    research = []
    backfill.extract({'allPlays': [play]}, research)
    study = {'schema': 'x', 'year': 2026, 'games': {'77': {'date': '2026-06-01', 'rows': research}}}
    from_study = phys.table_from_study(study)
    assert len(from_feed) == 1 and len(from_study) == 1
    a, b = from_feed.iloc[0], from_study.iloc[0]
    for c in phys.TABLE_COLUMNS:
        assert a[c] == b[c] or (isinstance(a[c], float) and np.isclose(a[c], b[c])), (c, a[c], b[c])
    assert (a['n'], a['sw'], a['wh'], a['oz'], a['ch'], a['iz'], a['izs'], a['izc'], a['fb']) == (3, 2, 1, 2, 1, 1, 1, 1, 2)
    assert (a['fb_sw'], a['fb_wh'], a['br_n'], a['br_sw'], a['br_wh'], a['os_n']) == (1, 0, 1, 1, 1, 0)
    assert np.isclose(a['velo'], 188.1) and np.isclose(a['hb'], 14.0) and np.isclose(a['ivb'], 14.0) and a['spin'] == 4400 and a['ev'] == 101.3
