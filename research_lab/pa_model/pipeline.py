from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PAConfig
from .data import dataframe_sha256
from .evaluation import calibration_table, clustered_log_loss_difference_ci, probability_metrics
from .features import build_time_valid_features, empirical_bayes_matchup_probabilities, fit_empirical_bayes_baseline
from .model import fit_frozen_model


def _feature_families(columns: list[str]) -> dict[str, list[str]]:
    probability = [c for c in columns if c.startswith("p_")]
    talent = [c for c in columns if any(token in c for token in ("batter_", "pitcher_", "history_pa", "platoon"))]
    context_names = {"is_home_batter", "inning", "outs_when_up", "runner_1b", "runner_2b", "runner_3b", "bat_score_diff", "n_thruorder_pitcher", "batter_days_since_prev_game", "pitcher_days_since_prev_game", "age_bat", "age_pit"}
    full = list(columns)
    talent_only = sorted(set(probability + talent) - context_names)
    return {"talent_only": talent_only, "talent_plus_context": full}


def _evaluate_one(y_index: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict:
    return {**probability_metrics(y_index, probabilities).to_dict(), "calibration": calibration_table(y_index, probabilities, labels)}


def run_benchmark(pa: pd.DataFrame, output_dir: str | Path, config: PAConfig | None = None) -> dict:
    config = config or PAConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = list(config.outcome_labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    features, all_feature_columns = build_time_valid_features(pa, config)
    test_mask = features["season"].isin(config.test_years).to_numpy()
    if not test_mask.any():
        raise ValueError("test period contains no rows")
    test = features.loc[test_mask].reset_index(drop=True)
    y_test = test["outcome"].map(label_to_index).to_numpy(int)
    league_prob = test[[f"p_league_{label}" for label in labels]].to_numpy(float)
    validation = features.loc[features["season"].isin(config.validation_years)].reset_index(drop=True)
    y_validation = validation["outcome"].map(label_to_index).to_numpy(int)
    eb_parameters, eb_tuning = fit_empirical_bayes_baseline(validation, y_validation, labels, config.min_probability)
    eb_prob = empirical_bayes_matchup_probabilities(test, labels, config.min_probability, eb_parameters)
    model_runs, fitted_models = {}, {}
    for family_name, family_columns in _feature_families(all_feature_columns).items():
        fitted, tuning = fit_frozen_model(features, family_columns, config)
        probabilities = fitted.predict_proba(test)
        fitted_models[family_name] = fitted
        model_runs[family_name] = {"metrics": _evaluate_one(y_test, probabilities, labels), "tuning": tuning, "feature_count": len(family_columns), "probabilities": probabilities}
    full_prob = model_runs["talent_plus_context"].pop("probabilities")
    talent_prob = model_runs["talent_only"].pop("probabilities")
    league_metrics, eb_metrics = _evaluate_one(y_test, league_prob, labels), _evaluate_one(y_test, eb_prob, labels)
    full_metrics = model_runs["talent_plus_context"]["metrics"]
    ci_vs_eb = clustered_log_loss_difference_ci(y_test, full_prob, eb_prob, test["game_pk"].to_numpy(), config.bootstrap_replicates, config.random_seed)
    ci_context = clustered_log_loss_difference_ci(y_test, full_prob, talent_prob, test["game_pk"].to_numpy(), config.bootstrap_replicates, config.random_seed + 1)
    max_abs_gap = max(abs(row["calibration_gap"]) for row in full_metrics["calibration"])
    promotion = {"beat_empirical_bayes_with_clustered_ci": ci_vs_eb["ci_95_high"] < 0, "context_adds_signal_with_clustered_ci": ci_context["ci_95_high"] < 0, "max_absolute_class_calibration_gap_below_0_015": max_abs_gap < 0.015}
    promotion["promoted"] = all(promotion.values())
    predictions = test[["date_key", "season", "game_pk", "at_bat_number", "batter", "pitcher", "outcome"]].copy()
    for i, label in enumerate(labels):
        predictions[f"p_league_{label}"] = league_prob[:, i]
        predictions[f"p_eb_{label}"] = eb_prob[:, i]
        predictions[f"p_model_{label}"] = full_prob[:, i]
    predictions.to_csv(output / "test_predictions.csv.gz", index=False, compression="gzip")
    result = {
        "schema": "baseball_research_lab.pa_benchmark.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": "chronological_untouched_test_executed", "config": config.to_dict(),
        "data": {"rows_total": int(len(features)), "rows_train": int(features["season"].isin(config.train_years).sum()), "rows_validation": int(features["season"].isin(config.validation_years).sum()), "rows_test": int(len(test)), "games_test": int(test["game_pk"].nunique()), "date_min": str(features["date_key"].min()), "date_max": str(features["date_key"].max()), "feature_receipt_sha256": dataframe_sha256(features, ["date_key", "game_pk", "at_bat_number", "batter", "pitcher", "outcome"])},
        "baselines": {"league": league_metrics, "empirical_bayes_matchup": {**eb_metrics, "tuning": eb_tuning}}, "models": model_runs,
        "incremental_value": {"relative_log_loss_gain_vs_league": float((league_metrics["log_loss"] - full_metrics["log_loss"]) / league_metrics["log_loss"]), "relative_log_loss_gain_vs_empirical_bayes": float((eb_metrics["log_loss"] - full_metrics["log_loss"]) / eb_metrics["log_loss"]), "full_vs_empirical_bayes_clustered_ci": ci_vs_eb, "context_vs_talent_only_clustered_ci": ci_context},
        "promotion": promotion,
        "interpretation_boundary": "This benchmark validates pre-PA probability estimates. It does not by itself validate runner transitions, bullpen logic, team-run distributions, winner probabilities, or exact-score forecasts.",
    }
    tuning_audit = {name: details["tuning"] for name, details in model_runs.items()}
    for details in result["models"].values():
        details["tuning"] = {key: value for key, value in details["tuning"].items() if key != "trials"}
    (output / "PA_BENCHMARK_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output / "TUNING_AUDIT.json").write_text(json.dumps(tuning_audit, indent=2, sort_keys=True), encoding="utf-8")
    joblib.dump(fitted_models["talent_plus_context"], output / "pa_model.joblib", compress=3)
    return result
