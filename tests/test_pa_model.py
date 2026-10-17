from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_lab.pa_model.config import PAConfig
from research_lab.pa_model.data import build_plate_appearances
from research_lab.pa_model.evaluation import probability_metrics
from research_lab.pa_model.features import (
    build_time_valid_features,
    empirical_bayes_matchup_probabilities,
)
from research_lab.pa_model.model import (
    apply_logit_calibration,
    fit_logit_calibration,
    validation_partitions,
)
from research_lab.pa_model.outcomes import map_event
from research_lab.pa_model.pipeline import run_benchmark
from research_lab.pa_model.synthetic import make_synthetic_pa


def test_outcome_mapping_is_explicit():
    assert map_event("home_run") == "HR"
    assert map_event("grounded_into_double_play") == "BIP_OUT"
    assert map_event("field_error") == "OTHER_REACH"
    assert map_event("wild_pitch") is None


def test_config_requires_disjoint_chronological_periods():
    with pytest.raises(ValueError, match="disjoint"):
        PAConfig(train_years=(2023,), validation_years=(2023,), test_years=(2025,))
    with pytest.raises(ValueError, match="chronological"):
        PAConfig(train_years=(2024,), validation_years=(2023,), test_years=(2025,))
    with pytest.raises(ValueError, match="evaluation_mode"):
        PAConfig(evaluation_mode="unknown")


def test_builder_uses_first_pitch_context_and_terminal_outcome(tmp_path: Path):
    rows = []
    for pitch_number, event in [(1, np.nan), (4, "single")]:
        rows.append(
            {
                "game_date": "2025-04-01",
                "game_pk": 1,
                "at_bat_number": 1,
                "pitch_number": pitch_number,
                "batter": 10,
                "pitcher": 20,
                "events": event,
                "stand": "R",
                "p_throws": "R",
                "home_team": "PIT",
                "away_team": "CHC",
                "inning": 1,
                "inning_topbot": "Top",
                "outs_when_up": 0,
                "on_1b": np.nan,
                "on_2b": np.nan,
                "on_3b": np.nan,
                "bat_score_diff": 0,
                "n_thruorder_pitcher": 1,
                "game_type": "R",
            }
        )
    path = tmp_path / "raw.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    pa, report = build_plate_appearances([path])
    assert (
        len(pa) == 1
        and pa.iloc[0]["outcome"] == "1B"
        and report["eligible_pa_rows"] == 1
    )


def test_features_are_date_blocked():
    features, _ = build_time_valid_features(
        make_synthetic_pa(rows_per_year=500, years=(2023,), seed=1)
    )
    first = features[features["date_key"] == features["date_key"].min()]
    assert first[
        [column for column in features if column.startswith("p_league_")]
    ].drop_duplicates().shape[0] == 1


def test_same_date_results_cannot_change_same_date_features():
    pa = make_synthetic_pa(rows_per_year=700, years=(2023,), seed=11)
    target_date = pa["date_key"].sort_values().iloc[300]
    changed = pa.copy()
    changed.loc[changed["date_key"] == target_date, "outcome"] = "HR"
    original_features, columns = build_time_valid_features(pa)
    changed_features, _ = build_time_valid_features(changed)
    mask = original_features["date_key"] == target_date
    assert np.allclose(
        original_features.loc[mask, columns].to_numpy(float),
        changed_features.loc[mask, columns].to_numpy(float),
        equal_nan=True,
    )


def test_empirical_bayes_probabilities_are_valid():
    features, _ = build_time_valid_features(
        make_synthetic_pa(rows_per_year=600, years=(2023,), seed=2)
    )
    probs = empirical_bayes_matchup_probabilities(
        features, PAConfig().outcome_labels
    )
    assert probs.shape == (len(features), 7)
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.all(probs > 0)


def test_metrics_reward_better_probabilities():
    y = np.array([0, 1, 2, 0])
    good = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1]]
    )
    flat = np.full((4, 3), 1 / 3)
    assert probability_metrics(y, good).log_loss < probability_metrics(y, flat).log_loss


def test_validation_partitions_are_disjoint_and_chronological():
    pa = make_synthetic_pa(rows_per_year=1200, seed=19)
    features, _ = build_time_valid_features(pa)
    masks, audit = validation_partitions(features)
    assert not np.any(masks["tune"] & masks["calibration"])
    assert not np.any(masks["tune"] & masks["blend"])
    assert not np.any(masks["calibration"] & masks["blend"])
    assert pd.Timestamp(audit["tune_date_max"]) < pd.Timestamp(
        audit["calibration_date_min"]
    )
    assert pd.Timestamp(audit["calibration_date_max"]) < pd.Timestamp(
        audit["blend_date_min"]
    )


def test_affine_calibration_returns_valid_probabilities_and_reduces_bias():
    rng = np.random.default_rng(7)
    y = rng.choice(3, size=4000, p=[0.55, 0.30, 0.15])
    raw = np.tile([0.45, 0.35, 0.20], (len(y), 1))
    temperature, biases, audit = fit_logit_calibration(y, raw)
    calibrated = apply_logit_calibration(raw, temperature, biases)
    assert np.allclose(calibrated.sum(axis=1), 1)
    assert np.all(calibrated > 0)
    assert audit["post_calibration_log_loss"] < audit["pre_calibration_log_loss"]


def test_full_synthetic_benchmark_writes_reproducible_artifacts(tmp_path: Path):
    config = PAConfig(
        regularization_grid=(1e-5, 1e-4),
        blend_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        bootstrap_replicates=50,
        max_iter=100,
    )
    result = run_benchmark(
        make_synthetic_pa(rows_per_year=1200, seed=3),
        tmp_path,
        config,
    )
    assert result["data"]["rows_test"] == 1200
    assert result["claim_status"] == "chronological_development_holdout_executed"
    assert (tmp_path / "PA_BENCHMARK_RESULT.json").exists()
    assert (tmp_path / "TUNING_AUDIT.json").exists()
    assert (tmp_path / "test_predictions.csv.gz").exists()
    assert (tmp_path / "pa_model.joblib").exists()
    saved = json.loads((tmp_path / "PA_BENCHMARK_RESULT.json").read_text())
    assert saved["schema"] == "baseball_research_lab.pa_benchmark.v3"
    assert "blended_candidate" in saved["models"]
    assert saved["promotion"]["production_holdout_required"] is True
    assert saved["promotion"]["production_pa_layer_promoted"] is False
    assert len(saved["candidate_fingerprint_sha256"]) == 64
    assert all(len(value) == 64 for value in saved["artifacts"].values())


def test_locked_final_mode_has_conservative_promotion_semantics(tmp_path: Path):
    config = PAConfig(
        evaluation_mode="locked_final",
        regularization_grid=(1e-4,),
        blend_grid=(0.0, 0.5, 1.0),
        bootstrap_replicates=25,
        max_iter=80,
    )
    result = run_benchmark(
        make_synthetic_pa(rows_per_year=800, seed=23),
        tmp_path,
        config,
    )
    promotion = result["promotion"]
    assert result["claim_status"] == "chronological_locked_final_holdout_executed"
    assert promotion["production_holdout_required"] is False
    assert promotion["production_pa_layer_promoted"] == promotion[
        "locked_final_holdout_passed"
    ]
    assert promotion["game_model_promoted"] is False
    assert promotion["production_promoted"] is False
    assert promotion["promoted"] is False
