"""The fit job's benchmark call accepts physics features and records them in the bundle."""
import importlib.util
from pathlib import Path
import joblib
import pandas as pd
from research_lab.pa_model import physics as phys
from research_lab.pa_model.config import PAConfig
from research_lab.pa_model.pipeline import run_benchmark

spec = importlib.util.spec_from_file_location('brl_test_research', Path(__file__).parent / 'test_research_tool.py')
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


def test_benchmark_with_extra_features_saves_a_physics_bundle(tmp_path):
    pa, study = helpers.synthetic(n_games=60)
    frames, tables = [], []
    for year in (2023, 2024, 2025, 2026):
        f = pa.copy(); f['season'] = year; f['date_key'] = f['date_key'].str.replace('2026', str(year)); f['game_pk'] = f['game_pk'] + (year - 2023) * 10000
        frames.append(f)
        t = phys.table_from_study(study); t['game_pk'] = t['game_pk'] + (year - 2023) * 10000; t['date_key'] = t['date_key'].str.replace('2026', str(year)); tables.append(t)
    pa_all, table = pd.concat(frames, ignore_index=True), pd.concat(tables, ignore_index=True)
    table = table.join(pa_all.set_index(['game_pk', 'at_bat_number'])['outcome'], on=['game_pk', 'at_bat_number'])
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final', regularization_grid=(0.01,), max_iter=30, bootstrap_replicates=20)
    extras, _ = phys.build_features(pa_all, table, {})
    report = run_benchmark(pa_all, tmp_path, config, extra_features=extras, bundle_extra={'physics_params': phys.DEFAULT_PARAMS, 'physics_features': list(extras.columns), 'name': 'test-v2'})
    bundle = joblib.load(tmp_path / 'pa_model.joblib')
    assert bundle['name'] == 'test-v2' and bundle['physics_features'] == phys.BASE_FEATURES
    assert bundle['talent_plus_context'].feature_columns[-len(phys.BASE_FEATURES):] == phys.BASE_FEATURES
    assert set(bundle['feature_columns']['talent_only']) >= set(phys.BASE_FEATURES)
    assert 'models' in report and 'blended_candidate' in report['models']


def test_history_seasons_load_regular_season_rows_in_the_package_columns(monkeypatch):
    """LONG-01: the sealed HIST-01 seasons join the data package's history with its columns, regular season only."""
    import gzip, os
    spec2 = importlib.util.spec_from_file_location('brl_fit_tool', Path(__file__).parents[1] / 'tools' / 'brl_fit_model.py')
    tool = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(tool)
    from cloud.security import seal
    key = os.urandom(32)
    frame = pd.DataFrame({'date_key': ['2019-04-01', '2019-10-05'], 'season': [2019, 2019], 'game_pk': [1, 2], 'game_type': ['R', 'D'],
                          'outcome': ['K', 'HR'], 'extra': [0, 0]})
    blob = seal(gzip.compress(frame.to_csv(index=False).encode()), key, tool.history_purpose(2019))
    monkeypatch.setattr(tool, 'read_blob', lambda repo, token, path, branch: blob if path == 'private/history/pa-2019.enc' else None)
    out = tool.load_history('r', 't', key, 'b', [2019], ['date_key', 'season', 'game_pk', 'outcome'])
    assert list(out.columns) == ['date_key', 'season', 'game_pk', 'outcome'] and out['game_pk'].tolist() == [1]
    import pytest
    with pytest.raises(ValueError):
        tool.load_history('r', 't', key, 'b', [2018], ['date_key'])


def test_game_losses_sum_the_log_loss_of_each_test_game(tmp_path):
    import numpy as np
    spec2 = importlib.util.spec_from_file_location('brl_fit_tool2', Path(__file__).parents[1] / 'tools' / 'brl_fit_model.py')
    tool = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(tool)
    labels = ['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH']
    rows = []
    for g, outcome in ((7, 'K'), (7, 'HR'), (9, 'BIP_OUT')):
        row = {'game_pk': g, 'outcome': outcome}
        for l in labels:
            row[f'p_ensemble_{l}'] = 0.5 if l == outcome else 0.5 / 6
            row[f'p_model_{l}'] = 0.25 if l == outcome else 0.75 / 6
        rows.append(row)
    path = tmp_path / 'p.csv.gz'; pd.DataFrame(rows).to_csv(path, index=False)
    out = tool.game_losses(path)
    assert out['7'][0] == 2 and abs(out['7'][1] - 2 * np.log(2)) < 1e-8 and abs(out['9'][2] - np.log(4)) < 1e-8


def test_settings_overrides_keep_the_model_name():
    """An arm's settings set config fields without touching anything else (a loop once reused the model's name)."""
    spec2 = importlib.util.spec_from_file_location('brl_fit_tool3', Path(__file__).parents[1] / 'tools' / 'brl_fit_model.py')
    tool = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(tool)
    settings = {'name': 'stab02-s1', 'history_half_life_days': 365, 'regularization_grid': [1e-3, 1e-2],
                'batter_class_prior_pa': [160, 113, 232, 299, 1390, 317, 1756], 'pitcher_class_prior_pa': [401, 223, 489, 695, 2152, 1040, 5877]}
    out = tool.config_overrides(settings)
    assert out['history_half_life_days'] == 365.0 and out['regularization_grid'] == (1e-3, 1e-2) and out['pitcher_class_prior_pa'][-1] == 5877.0
    assert settings['name'] == 'stab02-s1' and 'name' not in out
    config = PAConfig(train_years=(2016, 2024), validation_years=(2025,), test_years=(2026,), **out)
    assert config.batter_class_prior_pa[1] == 113.0
    src = (Path(__file__).parents[1] / 'tools' / 'brl_fit_model.py').read_text()
    assert 'for name in' not in src


def test_history_loader_takes_a_column_subset_without_dates(monkeypatch):
    import gzip, os
    spec2 = importlib.util.spec_from_file_location('brl_fit_tool4', Path(__file__).parents[1] / 'tools' / 'brl_fit_model.py')
    tool = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(tool)
    from cloud.security import seal
    key = os.urandom(32)
    frame = pd.DataFrame({'date_key': ['2021-04-01', '2021-04-01'], 'game_pk': [1, 1], 'game_type': ['R', 'R'], 'park': ['NYY', 'NYY'], 'outcome': ['K', 'HR']})
    blob = seal(gzip.compress(frame.to_csv(index=False).encode()), key, tool.history_purpose(2021))
    monkeypatch.setattr(tool, 'read_blob', lambda repo, token, path, branch: blob)
    out = tool.load_history('r', 't', key, 'b', [2021], ['game_pk', 'park'])
    assert list(out.columns) == ['game_pk', 'park'] and len(out) == 2
