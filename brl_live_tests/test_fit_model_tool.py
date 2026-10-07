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
