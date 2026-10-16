"""BOOST-01: the boosted matchup model stacked into the at-bat model. Out-of-fold on the training seasons, the model
in the bundle, and the live provider appending the same seven columns the benchmark wrote."""
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from research_lab.pa_model import boost as bt
from research_lab.pa_model import physics as phys
from research_lab.pa_model.config import PAConfig
from research_lab.pa_model.features import build_time_valid_features
from research_lab.pa_model.pipeline import run_benchmark

spec = importlib.util.spec_from_file_location('brl_test_research', Path(__file__).parent / 'test_research_tool.py')
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)

SMALL = {'max_iter': 30, 'round_grid': (10, 20, 30), 'folds': 3, 'min_samples_leaf': 20, 'max_leaf_nodes': 7}


def _seasons(n_games=60):
    pa, study = helpers.synthetic(n_games=n_games)
    frames, tables = [], []
    for year in (2023, 2024, 2025, 2026):
        f = pa.copy(); f['season'] = year; f['date_key'] = f['date_key'].str.replace('2026', str(year)); f['game_pk'] = f['game_pk'] + (year - 2023) * 10000
        frames.append(f)
        t = phys.table_from_study(study); t['game_pk'] = t['game_pk'] + (year - 2023) * 10000; t['date_key'] = t['date_key'].str.replace('2026', str(year)); tables.append(t)
    pa_all, table = pd.concat(frames, ignore_index=True), pd.concat(tables, ignore_index=True)
    table = table.join(pa_all.set_index(['game_pk', 'at_bat_number'])['outcome'], on=['game_pk', 'at_bat_number'])
    return pa_all, table


def test_boost_inputs_exclude_the_in_game_state_and_fit_out_of_fold():
    pa_all, _ = _seasons(40)
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final', regularization_grid=(0.01,), max_iter=30)
    features, cols = build_time_valid_features(pa_all, config)
    inputs = bt.boost_input_columns(cols)
    assert not set(inputs) & set(bt.GAME_CONTEXT) and 'platoon' in inputs and 'age_bat' in inputs and 'p_batter_K' in inputs
    from research_lab.pa_model.model import validation_partitions
    partitions, _ = validation_partitions(features, config)
    model, stacked, audit = bt.fit_boost(features, inputs, partitions['tune'], config, SMALL)
    assert stacked.shape == (len(features), 7) and np.allclose(np.exp(stacked).sum(1), 1.0, atol=1e-6)
    assert audit['rounds'] in (10, 20, 30) and audit['folds'] == 3 and audit['train_rows'] == int(features['season'].isin((2023, 2024)).sum())
    assert set(audit['tune_log_loss_by_rounds']) == {'10', '20', '30'}
    # the training rows are out of fold: a row's own model never saw it, so its stacked value differs from the final model's in-sample value
    train = features['season'].isin((2023, 2024)).to_numpy()
    x = features.loc[train, inputs].to_numpy(dtype=np.float32)
    in_sample = bt._log_probs(model, x, 7, config.min_probability)
    assert not np.allclose(in_sample, stacked[train], atol=1e-6)
    # rows after the training seasons come from the final model
    later = ~train
    assert np.allclose(bt._log_probs(model, features.loc[later, inputs].to_numpy(dtype=np.float32), 7, config.min_probability), stacked[later])


def test_benchmark_with_boost_saves_the_model_and_the_provider_reproduces_the_columns(tmp_path):
    from dataclasses import asdict
    from research_lab.game_sim.locked_pa_provider import LockedPAModelProvider, sha256_file, MODEL_LABELS, expected_feature_columns
    pa_all, table = _seasons(60)
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final', regularization_grid=(0.01,), max_iter=30, bootstrap_replicates=20)
    extras, _ = phys.build_features(pa_all, table, {})
    report = run_benchmark(pa_all, tmp_path, config, extra_features=extras,
                           bundle_extra={'physics_params': phys.DEFAULT_PARAMS, 'physics_features': list(extras.columns), 'name': 'test-boost'}, boost=SMALL)
    assert report['boost']['rounds'] in (10, 20, 30)
    bundle = joblib.load(tmp_path / 'pa_model.joblib')
    names = [f'gb_{l}' for l in MODEL_LABELS]
    assert bundle['boost_features'] == names and bundle['boost']['rounds'] == report['boost']['rounds']
    fitted = bundle['talent_plus_context']
    assert fitted.feature_columns[-7:] == names and set(bundle['feature_columns']['talent_only']) >= set(names)
    assert fitted.feature_columns == expected_feature_columns(phys.BASE_FEATURES, (), names)
    # the provider: the same bundle, the history and the physics table; its rows end with the boosted columns the
    # benchmark's final model gives for the same inputs
    history_path = tmp_path / 'history.csv.gz'; pa_all.to_csv(history_path, index=False)
    model_path = tmp_path / 'pa_model.joblib'
    date = sorted(pa_all.loc[pa_all['season'] == 2026, 'date_key'].unique())[-3]
    park = str(pa_all.loc[pa_all['date_key'] == date, 'park'].iloc[0])
    provider = LockedPAModelProvider(model_path, history_path, date, date, park, expected_sha256=sha256_file(model_path), physics_table=table)
    assert list(provider.boost_features) == names and provider._boost is not None
    rows = pa_all[(pa_all['date_key'] == date) & (pa_all['park'] == park)]
    assert len(rows) > 3
    n_in = len(expected_feature_columns(phys.BASE_FEATURES))
    for r in rows.head(6).itertuples():
        row, _ = provider.feature_row(int(r.batter), int(r.pitcher), str(r.stand), str(r.p_throws), is_home_batter=0, inning=1, outs=0, runners=(0, 0, 0),
                                      bat_score_diff=0, times_through=1)
        assert len(row) == n_in + 7
        inputs = row[:n_in][provider._boost.idx][None, :].astype(np.float32)
        expect = bt._log_probs(bundle['boost']['model'], inputs, 7, config.min_probability)[0]
        assert np.allclose(row[n_in:], expect.astype(np.float32), atol=1e-6)
        probs = provider.predict_row(row)['model']
        assert probs.shape == (7,) and abs(probs.sum() - 1) < 1e-9
    # the row cache answers the same matchup from memory
    before = len(provider._boost._cache)
    provider.feature_row(int(rows.iloc[0].batter), int(rows.iloc[0].pitcher), str(rows.iloc[0].stand), str(rows.iloc[0].p_throws), is_home_batter=1, inning=7, outs=2,
                         runners=(1, 0, 0), bat_score_diff=-2, times_through=3)
    assert len(provider._boost._cache) == before
