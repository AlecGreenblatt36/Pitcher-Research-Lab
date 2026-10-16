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


def test_class_priors_shrink_each_class_and_the_provider_agrees(tmp_path):
    """STAB-02: per-class prior sizes for batters and pitchers, the same in the builder and the live provider."""
    pa = seasons()
    big = (180.0, 230.0, 470.0, 600.0, 600.0, 1500.0, 600.0)
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final',
                      regularization_grid=(0.01,), max_iter=30, pitcher_class_prior_pa=big, batter_class_prior_pa=(180.0,) * 7)
    features, cols = build_time_valid_features(pa, config)
    plain, _ = build_time_valid_features(pa, PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,)))
    # batter classes at 180 everywhere equal the frozen scalar setting; the pitchers' home-run rates move toward the league
    assert np.allclose(features['p_batter_HR'], plain['p_batter_HR'], atol=1e-7)
    late = features['season'] == 2026
    dev = lambda f: (f.loc[late, 'p_pitcher_HR'] - f.loc[late, 'p_league_HR']).abs().mean()
    assert dev(features) < dev(plain)
    fitted, _ = fit_frozen_model(features, list(cols), config)
    bundle = {'talent_plus_context': fitted, 'talent_only': fitted, 'empirical_bayes_parameters': EB, 'model_weight': 0.9, 'empirical_bayes_weight': 0.1,
              'labels': MODEL_LABELS, 'feature_columns': {'talent_plus_context': fitted.feature_columns}, 'config': asdict(config), 'name': 'test-class'}
    model_path = tmp_path / 'm.joblib'; joblib.dump(bundle, model_path)
    history_path = tmp_path / 'h.csv.gz'; pa.to_csv(history_path, index=False)
    date = '2026-05-07'
    provider = LockedPAModelProvider(model_path, history_path, date, date, 'NYY', expected_sha256=sha256_file(model_path))
    for r in np.where(features['date_key'].to_numpy() == date)[0][:8]:
        block, _ = provider.talent_block(int(features['batter'][r]), int(features['pitcher'][r]), 'R', 'L')
        assert np.allclose(block.astype(np.float32), features.loc[r, cols[:len(block)]].to_numpy(float), atol=2e-6)


def test_physics_override_moves_only_the_named_feature():
    from research_lab.pa_model import physics as phys
    pa, study = helpers.synthetic(n_games=40)
    table = phys.table_from_study(study)
    base, _ = phys.build_features(pa, table, {})
    over, _ = phys.build_features(pa, table, {'k_overrides': {'p_velo': 400.0}})
    late = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)['date_key'] >= '2026-05-05'
    for name in ('b_ev', 'b_whiff', 'p_spin', 'p_chase', 'p_ev', 'p_whiff'):
        assert np.array_equal(base[name].to_numpy(), over[name].to_numpy(), equal_nan=True)
    # pitcher 501 throws 3 mph harder than 500: a larger prior pulls both toward the league, so the gap narrows
    assert over.loc[late, 'p_velo'].std() < base.loc[late, 'p_velo'].std() * 0.9
