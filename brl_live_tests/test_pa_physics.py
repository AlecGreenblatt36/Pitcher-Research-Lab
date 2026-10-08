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
    params = {'xvalue': True, 'recent_days': 3, 'k_rate': 80.0, 'pitch_types': True, 'environment': True, 'env_days': 4, 'k_env': 20.0, 'matchup': True, 'workload': True}
    built, _ = phys.build_features(pa, table, params)
    assert built['mx_whiff'].abs().max() > 0 and built['mx_velo'].notna().all()
    assert built['p_last_n'].max() > 0 and set(np.unique(built['p_b2b'])) <= {0.0, 1.0}
    for day in ('2026-05-01', '2026-05-04', '2026-05-10'):
        state = phys.PhysicsState.build(table, day, params)
        rows = np.where(pa_sorted.date_key.to_numpy() == day)[0]
        assert len(rows)
        for r in rows[:30]:
            expected = built.iloc[r].to_numpy(float)
            got = state.features(int(pa_sorted.batter[r]), int(pa_sorted.pitcher[r]))
            assert np.allclose(got, expected, atol=1e-5), (day, r, got - expected)
    assert state.names == phys.feature_names(params) == phys.BASE_FEATURES + phys.XVALUE_FEATURES + phys.RECENT_FEATURES + phys.TYPE_FEATURES + phys.MATCHUP_FEATURES + phys.WORKLOAD_FEATURES + phys.ENV_FEATURES
    assert built['season_day'].min() >= 47 and (built.loc[pa_sorted.date_key == '2026-05-01', ['env_hr', 'env_k', 'env_bb', 'env_out']] == 0).all().all()
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


def test_defense_feature_builder_matches_state_and_is_time_valid():
    pa, study = helpers.synthetic(n_games=40)
    # make the home team's defense better: convert some of its fielded balls in play to outs
    rng = np.random.default_rng(3)
    top = pa.inning_topbot == 'Top'            # the home team fields in the top half
    hits = top & pa.outcome.isin(['1B', '2B_3B', 'OTHER_REACH'])
    flip = hits & (rng.random(len(pa)) < 0.5)
    pa.loc[flip, 'outcome'] = 'BIP_OUT'
    table = study_table(study)
    params = {'defense': True, 'k_def': 50.0, 'defense_days': 5}
    built, _ = phys.build_features(pa, table, params)
    assert list(built.columns) == phys.BASE_FEATURES + phys.DEFENSE_FEATURES
    pa_sorted = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    first = pa_sorted.date_key == '2026-05-01'
    assert (built.loc[first, 'f_def'] == 0).all()                      # nothing known on the first date
    later = pa_sorted.date_key == '2026-05-08'
    home_fielding = later & (pa_sorted.inning_topbot == 'Top')
    assert built.loc[home_fielding, 'f_def'].mean() > 0.01 > built.loc[later & ~(pa_sorted.inning_topbot == 'Top'), 'f_def'].mean()
    D = phys.DefenseState.build(pa, '2026-05-08', params)
    assert np.isclose(D.value('HOM'), built.loc[home_fielding, 'f_def'].iloc[0], atol=1e-6)
    assert np.isclose(D.value('AWY'), built.loc[later & (pa_sorted.inning_topbot == 'Bot'), 'f_def'].iloc[0], atol=1e-6)
    assert D.value('NOPE') == 0.0
    state = phys.PhysicsState.build(table, '2026-05-08', params)
    v = state.features(10, 500, team_defense=D.value('HOM'))
    assert v[-1] == np.float64(D.value('HOM')) and len(v) == len(phys.BASE_FEATURES) + 1


def test_aging_and_recency_features_match_between_builder_and_state():
    pa, study = helpers.synthetic(n_games=40)
    table = study_table(study)
    pa = pa.copy()
    pa['age_bat'] = 24 + (pa['batter'] % 12)          # batters 24 to 35, pitchers 27 and 28
    pa['age_pit'] = 27 + (pa['pitcher'] - 500)
    pa_sorted = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    table = table.join(pa_sorted.set_index(['game_pk', 'at_bat_number'])['outcome'], on=['game_pk', 'at_bat_number'])
    params = {'xvalue': True, 'recent_days': 3, 'aging': True, 'decay_days': 3, 'k_dec': 10.0}
    built, _ = phys.build_features(pa, table, params)
    assert list(built.columns[-16:]) == phys.AGING_FEATURES + phys.DECAY_FEATURES
    first = pa_sorted.date_key == '2026-05-01'
    # Nothing is known on the first date: no history gap and no recency deviation.
    assert (built.loc[first, ['b_gap', 'p_gap'] + phys.DECAY_FEATURES] == 0).all().all()
    later = pa_sorted.date_key >= '2026-05-05'
    assert (built.loc[later, 'b_gap'] > 0).all() and built.loc[later, 'b_dec_HR'].abs().max() > 0
    # Older batters (above 27) get a negative aging term once they have history; younger ones positive.
    old = later & (pa_sorted.age_bat > 27); young = later & (pa_sorted.age_bat < 27)
    assert (built.loc[old, 'b_aging'] < 0).all() and (built.loc[young, 'b_aging'] > 0).all()
    for day in ('2026-05-01', '2026-05-04', '2026-05-10'):
        state = phys.PhysicsState.build(table, day, params).with_history(pa)
        rows = np.where(pa_sorted.date_key.to_numpy() == day)[0]
        for r in rows[:30]:
            expected = built.iloc[r].to_numpy(float)
            got = state.features(int(pa_sorted.batter[r]), int(pa_sorted.pitcher[r]), age_bat=pa_sorted.age_bat[r], age_pit=pa_sorted.age_pit[r])
            assert np.allclose(got, expected, atol=1e-5, equal_nan=True), (day, r, got - expected)
    with pytest.raises(ValueError):
        phys.PhysicsState.build(table, '2026-05-04', params).features(10, 500)


def test_windowed_state_matches_the_chronological_builder():
    pa, study = helpers.synthetic(n_games=40)
    table = study_table(study)
    pa_sorted = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    outcomes = pa_sorted.set_index(['game_pk', 'at_bat_number'])['outcome']
    table = table.join(outcomes, on=['game_pk', 'at_bat_number'])
    params = {'xvalue': True, 'recent_days': 3, 'window_days': 4}
    built, _ = phys.build_features(pa, table, params)
    full, _ = phys.build_features(pa, table, {'xvalue': True, 'recent_days': 3})
    late = (pa_sorted.date_key >= '2026-05-08').to_numpy()
    assert (built.loc[late, 'p_pitch_n'] < full.loc[late, 'p_pitch_n']).any()          # older pitches leave the window
    assert (built.loc[late, 'b_bip_n'] <= full.loc[late, 'b_bip_n']).all()
    for day in ('2026-05-04', '2026-05-10'):
        state = phys.PhysicsState.build(table, day, params)
        rows = np.where(pa_sorted.date_key.to_numpy() == day)[0]
        assert len(rows)
        for r in rows[:30]:
            got = state.features(int(pa_sorted.batter[r]), int(pa_sorted.pitcher[r]))
            assert np.allclose(got, built.iloc[r].to_numpy(float), atol=1e-5), (day, r)
