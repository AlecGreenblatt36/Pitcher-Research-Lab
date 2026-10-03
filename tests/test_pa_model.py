from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_lab.pa_model.config import PAConfig
from research_lab.pa_model.data import build_plate_appearances
from research_lab.pa_model.evaluation import probability_metrics
from research_lab.pa_model.features import build_time_valid_features, empirical_bayes_matchup_probabilities
from research_lab.pa_model.outcomes import map_event
from research_lab.pa_model.pipeline import run_benchmark
from research_lab.pa_model.synthetic import make_synthetic_pa


def test_outcome_mapping_is_explicit():
    assert map_event("home_run") == "HR"
    assert map_event("grounded_into_double_play") == "BIP_OUT"
    assert map_event("field_error") == "OTHER_REACH"
    assert map_event("wild_pitch") is None


def test_builder_uses_first_pitch_context_and_terminal_outcome(tmp_path: Path):
    rows = []
    for pitch_number, event in [(1, np.nan), (4, "single")]:
        rows.append({"game_date": "2025-04-01", "game_pk": 1, "at_bat_number": 1, "pitch_number": pitch_number, "batter": 10, "pitcher": 20, "events": event, "stand": "R", "p_throws": "R", "home_team": "PIT", "away_team": "CHC", "inning": 1, "inning_topbot": "Top", "outs_when_up": 0, "on_1b": np.nan, "on_2b": np.nan, "on_3b": np.nan, "bat_score_diff": 0, "n_thruorder_pitcher": 1, "game_type": "R"})
    path = tmp_path / "raw.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    pa, report = build_plate_appearances([path])
    assert len(pa) == 1 and pa.iloc[0]["outcome"] == "1B" and report["eligible_pa_rows"] == 1


def test_features_are_date_blocked():
    features, _ = build_time_valid_features(make_synthetic_pa(rows_per_year=500, years=(2023,), seed=1))
    first = features[features["date_key"] == features["date_key"].min()]
    assert first[[c for c in features if c.startswith("p_league_")]].drop_duplicates().shape[0] == 1


def test_empirical_bayes_probabilities_are_valid():
    features, _ = build_time_valid_features(make_synthetic_pa(rows_per_year=600, years=(2023,), seed=2))
    probs = empirical_bayes_matchup_probabilities(features, PAConfig().outcome_labels)
    assert probs.shape == (len(features), 7) and np.allclose(probs.sum(axis=1), 1) and np.all(probs > 0)


def test_metrics_reward_better_probabilities():
    y = np.array([0, 1, 2, 0])
    good = np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.7,.2,.1]])
    flat = np.full((4,3), 1/3)
    assert probability_metrics(y, good).log_loss < probability_metrics(y, flat).log_loss


def test_full_synthetic_benchmark_writes_reproducible_artifacts(tmp_path: Path):
    config = PAConfig(regularization_grid=(1e-5, 1e-4), temperature_grid=(.9, 1, 1.1), bootstrap_replicates=50, max_iter=100)
    result = run_benchmark(make_synthetic_pa(rows_per_year=1200, seed=3), tmp_path, config)
    assert result["data"]["rows_test"] == 1200
    assert (tmp_path / "PA_BENCHMARK_RESULT.json").exists() and (tmp_path / "test_predictions.csv.gz").exists() and (tmp_path / "pa_model.joblib").exists()
    assert json.loads((tmp_path / "PA_BENCHMARK_RESULT.json").read_text())["schema"] == "baseball_research_lab.pa_benchmark.v1"
