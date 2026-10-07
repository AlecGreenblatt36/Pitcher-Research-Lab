"""A model bundle that declares physics features makes the provider build and append them; without a table it fails closed."""
import importlib.util
from dataclasses import asdict
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from research_lab.pa_model import physics as phys
from research_lab.pa_model.config import PAConfig
from research_lab.pa_model.features import build_time_valid_features
from research_lab.pa_model.model import fit_frozen_model
from research_lab.game_sim.locked_pa_provider import LockedPAModelProvider, LockedModelError, sha256_file, MODEL_LABELS, expected_feature_columns
from research_lab.game_sim.models import PAContext, PlayerProfile, PitcherProfile

spec = importlib.util.spec_from_file_location('brl_test_research', Path(__file__).parent / 'test_research_tool.py')
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
EB = {'batter_weight': 0.5, 'pitcher_weight': 0.5, 'park_weight': 0.1, 'batter_recent_weight': 0.1, 'pitcher_recent_weight': 0.1, 'temperature': 1.0}


def make_bundle(tmp_path, params):
    pa, study = helpers.synthetic(n_games=60)
    frames = []
    for year in (2023, 2024, 2025, 2026):
        f = pa.copy(); f['season'] = year; f['date_key'] = f['date_key'].str.replace('2026', str(year)); f['game_pk'] = f['game_pk'] + (year - 2023) * 10000
        frames.append(f)
    pa_all = pd.concat(frames, ignore_index=True)
    tables = []
    for year in (2023, 2024, 2025, 2026):
        t = phys.table_from_study(study); t['game_pk'] = t['game_pk'] + (year - 2023) * 10000; t['date_key'] = t['date_key'].str.replace('2026', str(year))
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    outcome = pa_all.set_index(['game_pk', 'at_bat_number'])['outcome']
    table = table.join(outcome, on=['game_pk', 'at_bat_number'])
    table['run_value'] = table['outcome'].map(phys.RUN_VALUE).fillna(0.0)
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final', regularization_grid=(0.01,), max_iter=30)
    features, cols = build_time_valid_features(pa_all, config)
    extras, _ = phys.build_features(pa_all, table, params)
    for c in extras.columns:
        features[c] = extras[c].to_numpy()
    fitted, _ = fit_frozen_model(features, list(cols) + list(extras.columns), config)
    bundle = {'talent_plus_context': fitted, 'talent_only': fitted, 'empirical_bayes_parameters': EB, 'model_weight': 0.9, 'empirical_bayes_weight': 0.1,
              'labels': MODEL_LABELS, 'feature_columns': {'talent_plus_context': fitted.feature_columns}, 'config': asdict(config),
              'physics_params': params, 'physics_features': list(extras.columns), 'name': 'test-physics-v2'}
    model_path = tmp_path / 'pa_model.joblib'; joblib.dump(bundle, model_path)
    history_path = tmp_path / 'history.csv.gz'; pa_all.to_csv(history_path, index=False)
    return model_path, history_path, table, pa_all, features, extras


def test_provider_appends_physics_features_and_fails_closed_without_them(tmp_path):
    params = {'xvalue': True, 'recent_days': 3}
    model_path, history_path, table, pa_all, features, extras = make_bundle(tmp_path, params)
    date = '2026-05-08'
    with pytest.raises(LockedModelError):
        LockedPAModelProvider(model_path, history_path, date, date, 'NYY', expected_sha256=sha256_file(model_path))
    provider = LockedPAModelProvider(model_path, history_path, date, date, 'NYY', expected_sha256=sha256_file(model_path), physics_table=table)
    assert provider.name == 'test-physics-v2' and list(provider.physics_features) == phys.feature_names(params)
    assert provider.provenance()['physics_rows_used'] == int((table.date_key < date).sum())
    # The appended block equals the chronological builder's row for a PA on that date.
    rows = np.where(pa_all.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True).date_key.to_numpy() == date)[0]
    ordered = pa_all.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    r = rows[0]; b, p = int(ordered.batter[r]), int(ordered.pitcher[r])
    row, _ = provider.feature_row(b, p, 'R', 'L', is_home_batter=0, inning=1, outs=0, runners=(0, 0, 0), bat_score_diff=0, times_through=1)
    n_base = len(expected_feature_columns())
    assert len(row) == n_base + len(provider.physics_features)
    assert np.allclose(row[n_base:], extras.iloc[r].to_numpy(float).astype(np.float32), atol=1e-5)
    ctx = PAContext(batter=PlayerProfile(str(b), 'b', 'R'), pitcher=PitcherProfile(str(p), 'p', 'L', 'starter'), batting_side='away', inning=1, half='top',
                    outs=0, bases=(None, None, None), batting_score=0, fielding_score=0, lineup_position=1, times_through_order=1, pitcher_batters_faced=0,
                    pitcher_runs_allowed=0, pitcher_fatigue=0.0, park_factor=1.0, weather_run_factor=1.0, batting_team_baserunning=0.0, fielding_team_defense=0.0)
    probs = provider.probabilities(ctx)
    assert abs(sum(probs.values()) - 1.0) < 1e-9 and len(probs) == 7
    # A bundle without physics features keeps the original 122-column contract.
    bundle = joblib.load(model_path); bundle.pop('physics_features'); bundle.pop('physics_params')
    plain = tmp_path / 'plain.joblib'; joblib.dump(bundle, plain)
    with pytest.raises(LockedModelError):
        LockedPAModelProvider(plain, history_path, date, date, 'NYY', expected_sha256=sha256_file(plain))


def test_provider_reads_defense_from_the_context(tmp_path):
    params = {'defense': True, 'k_def': 50.0, 'defense_days': 5}
    model_path, history_path, table, pa_all, features, extras = make_bundle(tmp_path, params)
    date = '2026-05-08'
    provider = LockedPAModelProvider(model_path, history_path, date, date, 'NYY', expected_sha256=sha256_file(model_path), physics_table=table)
    assert 'f_def' in provider.physics_features
    row_a, _ = provider.feature_row(10, 500, 'R', 'L', is_home_batter=0, inning=1, outs=0, runners=(0, 0, 0), bat_score_diff=0, times_through=1, team_defense=0.02)
    row_b, _ = provider.feature_row(10, 500, 'R', 'L', is_home_batter=0, inning=1, outs=0, runners=(0, 0, 0), bat_score_diff=0, times_through=1, team_defense=-0.02)
    assert row_a[-1] == np.float32(0.02) and row_b[-1] == np.float32(-0.02) and not np.array_equal(row_a, row_b)
    ctx = PAContext(batter=PlayerProfile('10', 'b', 'R'), pitcher=PitcherProfile('500', 'p', 'L', 'starter'), batting_side='away', inning=1, half='top',
                    outs=0, bases=(None, None, None), batting_score=0, fielding_score=0, lineup_position=1, times_through_order=1, pitcher_batters_faced=0,
                    pitcher_runs_allowed=0, pitcher_fatigue=0.0, park_factor=1.0, weather_run_factor=1.0, batting_team_baserunning=0.0, fielding_team_defense=0.02)
    p1 = provider.probabilities(ctx)
    ctx2 = PAContext(**{**ctx.__dict__, 'fielding_team_defense': -0.02})
    p2 = provider.probabilities(ctx2)
    assert p1 != p2
