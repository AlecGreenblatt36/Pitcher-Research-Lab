"""LONG-01: the half-life on the cumulative counters. The chronological builder (lazy decay) and the live provider's
history state (each row weighted directly) must give the same features; a half-life of 0 keeps the frozen counts."""
import importlib.util
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_lab.pa_model.config import PAConfig
from research_lab.pa_model.features import _OnlineCounter, build_time_valid_features
from research_lab.pa_model.model import fit_frozen_model
from research_lab.game_sim.locked_pa_provider import LockedPAModelProvider, MODEL_LABELS, sha256_file

spec = importlib.util.spec_from_file_location('brl_test_research', Path(__file__).parent / 'test_research_tool.py')
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
EB = {'batter_weight': 0.5, 'pitcher_weight': 0.5, 'park_weight': 0.1, 'batter_recent_weight': 0.1, 'pitcher_recent_weight': 0.1, 'temperature': 1.0}


def seasons():
    frames = []
    for i, year in enumerate((2023, 2024, 2025, 2026)):
        pa, _ = helpers.synthetic(seed=i, n_games=40)
        pa['season'] = year; pa['date_key'] = pa['date_key'].str.replace('2026', str(year)); pa['game_pk'] = pa['game_pk'] + i * 10000
        frames.append(pa)
    return pd.concat(frames, ignore_index=True)


def test_lazy_decay_equals_direct_weights():
    c = _OnlineCounter(3, 100.0)
    days = [10, 10, 25, 60, 200]
    for d, k in zip(days, [0, 1, 2, 0, 1]):
        c.update('x', k, d)
    got = c.get('x', 260)
    want = np.zeros(3)
    for d, k in zip(days, [0, 1, 2, 0, 1]):
        want[k] += 0.5 ** ((260 - d) / 100.0)
    assert np.allclose(got.counts, want, rtol=1e-12) and abs(got.total - want.sum()) < 1e-12
    # without a half-life nothing decays
    plain = _OnlineCounter(3)
    for d, k in zip(days, [0, 1, 2, 0, 1]):
        plain.update('x', k, d)
    assert plain.get('x', 260).total == 5 and list(plain.get('x', 260).counts) == [2, 2, 1]


def test_half_life_zero_keeps_the_frozen_features():
    pa = seasons()
    base, cols = build_time_valid_features(pa, PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,)))
    off, _ = build_time_valid_features(pa, PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), history_half_life_days=0.0))
    on, _ = build_time_valid_features(pa, PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), history_half_life_days=200.0))
    assert np.array_equal(base[cols].to_numpy(), off[cols].to_numpy())
    later = base['season'] == 2026
    # a half-life discounts the earlier seasons, so the history counts shrink and the player rates move
    assert (on.loc[later, 'batter_history_pa'] < base.loc[later, 'batter_history_pa'] - 1).all()
    assert not np.allclose(on.loc[later, 'p_batter_HR'], base.loc[later, 'p_batter_HR'])


def test_builder_and_live_provider_agree_with_a_half_life(tmp_path):
    pa = seasons()
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final',
                      regularization_grid=(0.01,), max_iter=30, history_half_life_days=150.0)
    features, cols = build_time_valid_features(pa, config)
    fitted, _ = fit_frozen_model(features, list(cols), config)
    bundle = {'talent_plus_context': fitted, 'talent_only': fitted, 'empirical_bayes_parameters': EB, 'model_weight': 0.9, 'empirical_bayes_weight': 0.1,
              'labels': MODEL_LABELS, 'feature_columns': {'talent_plus_context': fitted.feature_columns}, 'config': asdict(config), 'name': 'test-long'}
    model_path = tmp_path / 'm.joblib'; joblib.dump(bundle, model_path)
    history_path = tmp_path / 'h.csv.gz'; pa.to_csv(history_path, index=False)
    date = '2026-05-07'
    provider = LockedPAModelProvider(model_path, history_path, date, date, 'NYY', expected_sha256=sha256_file(model_path))
    ordered = features
    rows = np.where(ordered['date_key'].to_numpy() == date)[0]
    checked = 0
    for r in rows[:12]:
        b, p = int(ordered['batter'][r]), int(ordered['pitcher'][r])
        block, meta = provider.talent_block(b, p, 'R', 'L')
        want = ordered.loc[r, cols[:len(block)]].to_numpy(float)
        assert np.allclose(block.astype(np.float32), want, atol=2e-6), (b, p)
        assert abs(meta['batter_history_pa'] - float(ordered.loc[r, 'batter_history_pa'])) < 1e-3
        assert abs(meta['pitcher_split_history_pa'] - float(ordered.loc[r, 'pitcher_split_history_pa'])) < 1e-3
        checked += 1
    assert checked == 12
