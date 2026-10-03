from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PAConfig
from .data import dataframe_sha256
from .evaluation import (
    calibration_table,
    clustered_log_loss_difference_ci,
    probability_metrics,
)
from .features import (
    build_time_valid_features,
    empirical_bayes_matchup_probabilities,
    fit_empirical_bayes_baseline,
)
from .model import fit_frozen_model, validation_partitions


def _feature_families(columns: list[str]) -> dict[str, list[str]]:
    # The ablation removes only in-game state. Handedness, age, rest, park,
    # player history, and recent-form estimates remain available in both models.
    game_context = {
        "is_home_batter",
        "inning",
        "outs_when_up",
        "runner_1b",
        "runner_2b",
        "runner_3b",
        "bat_score_diff",
        "n_thruorder_pitcher",
    }
    return {
        "talent_only": [column for column in columns if column not in game_context],
        "talent_plus_context": list(columns),
    }


def _evaluate_one(
    y_index: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict:
    return {
        **probability_metrics(y_index, probabilities).to_dict(),
        "calibration": calibration_table(y_index, probabilities, labels),
    }


def _select_blend(
    y_index: np.ndarray,
    model_probabilities: np.ndarray,
    empirical_bayes_probabilities: np.ndarray,
    grid: tuple[float, ...],
) -> tuple[float, dict]:
    """Select a convex shrinkage ensemble on dates unused for calibration."""

    trials: list[dict] = []
    best: tuple[float, float] | None = None
    for model_weight in grid:
        weight = float(model_weight)
        probabilities = (
            weight * model_probabilities
            + (1.0 - weight) * empirical_bayes_probabilities
        )
        score = probability_metrics(y_index, probabilities).log_loss
        trials.append(
            {
                "model_weight": weight,
                "empirical_bayes_weight": 1.0 - weight,
                "selection_log_loss": float(score),
            }
        )
        candidate = (float(score), weight)
        if best is None or candidate[0] < best[0] - 1e-12:
            best = candidate
        elif abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] < best[1]:
            # Tie-break toward the lower-variance empirical-Bayes baseline.
            best = candidate

    assert best is not None
    return best[1], {
        "model_weight": float(best[1]),
        "empirical_bayes_weight": float(1.0 - best[1]),
        "best_selection_log_loss": float(best[0]),
        "trials": sorted(trials, key=lambda item: item["selection_log_loss"]),
    }


def run_benchmark(
    pa: pd.DataFrame,
    output_dir: str | Path,
    config: PAConfig | None = None,
) -> dict:
    config = config or PAConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = list(config.outcome_labels)
    label_to_index = {label: i for i, label in enumerate(labels)}

    features, all_feature_columns = build_time_valid_features(pa, config)
    partitions, partition_audit = validation_partitions(features, config)
    test_mask = features["season"].isin(config.test_years).to_numpy()
    if not test_mask.any():
        raise ValueError("test period contains no rows")
    test = features.loc[test_mask].reset_index(drop=True)
    y_test = test["outcome"].map(label_to_index).to_numpy(int)

    # The structural baseline is fit on the tuning and calibration portions of
    # the validation season. The final validation block is reserved solely for
    # selecting the convex ensemble weight.
    eb_fit_mask = partitions["tune"] | partitions["calibration"]
    eb_fit = features.loc[eb_fit_mask].reset_index(drop=True)
    y_eb_fit = eb_fit["outcome"].map(label_to_index).to_numpy(int)
    blend_frame = features.loc[partitions["blend"]].reset_index(drop=True)
    y_blend = blend_frame["outcome"].map(label_to_index).to_numpy(int)

    # Pandas 3 can expose a read-only array here. Request an owned copy because
    # normalization is intentionally in-place for memory efficiency.
    league_prob = test[
        [f"p_league_{label}" for label in labels]
    ].to_numpy(dtype=float, copy=True)
    league_prob /= league_prob.sum(axis=1, keepdims=True)
    eb_parameters, eb_tuning = fit_empirical_bayes_baseline(
        eb_fit,
        y_eb_fit,
        labels,
        config.min_probability,
    )
    eb_prob = empirical_bayes_matchup_probabilities(
        test,
        labels,
        config.min_probability,
        eb_parameters,
    )
    eb_blend_prob = empirical_bayes_matchup_probabilities(
        blend_frame,
        labels,
        config.min_probability,
        eb_parameters,
    )

    model_runs: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    blend_probabilities: dict[str, np.ndarray] = {}
    for family_name, family_columns in _feature_families(
        all_feature_columns
    ).items():
        fitted, tuning = fit_frozen_model(features, family_columns, config)
        probabilities = fitted.predict_proba(test)
        fitted_models[family_name] = fitted
        test_probabilities[family_name] = probabilities
        blend_probabilities[family_name] = fitted.predict_proba(blend_frame)
        model_runs[family_name] = {
            "metrics": _evaluate_one(y_test, probabilities, labels),
            "tuning": tuning,
            "feature_count": len(family_columns),
        }

    model_weight, blend_tuning = _select_blend(
        y_blend,
        blend_probabilities["talent_plus_context"],
        eb_blend_prob,
        config.blend_grid,
    )
    full_prob = test_probabilities["talent_plus_context"]
    talent_prob = test_probabilities["talent_only"]
    candidate_prob = model_weight * full_prob + (1.0 - model_weight) * eb_prob

    league_metrics = _evaluate_one(y_test, league_prob, labels)
    eb_metrics = _evaluate_one(y_test, eb_prob, labels)
    candidate_metrics = _evaluate_one(y_test, candidate_prob, labels)
    model_runs["blended_candidate"] = {
        "metrics": candidate_metrics,
        "tuning": blend_tuning,
        "feature_count": model_runs["talent_plus_context"]["feature_count"],
    }

    ci_vs_eb = clustered_log_loss_difference_ci(
        y_test,
        candidate_prob,
        eb_prob,
        test["game_pk"].to_numpy(),
        config.bootstrap_replicates,
        config.random_seed,
    )
    ci_context = clustered_log_loss_difference_ci(
        y_test,
        full_prob,
        talent_prob,
        test["game_pk"].to_numpy(),
        config.bootstrap_replicates,
        config.random_seed + 1,
    )
    max_abs_gap = max(
        abs(row["calibration_gap"])
        for row in candidate_metrics["calibration"]
    )
    development_gates = {
        "beat_empirical_bayes_with_clustered_ci": ci_vs_eb["ci_95_high"] < 0,
        "context_adds_signal_with_clustered_ci": ci_context["ci_95_high"] < 0,
        "max_absolute_class_calibration_gap_below_0_015": max_abs_gap < 0.015,
    }
    development_gate_passed = all(development_gates.values())
    promotion = {
        **development_gates,
        "max_absolute_class_calibration_gap": float(max_abs_gap),
        "development_gate_passed": development_gate_passed,
        "promoted_to_locked_final_evaluation": development_gate_passed,
        "production_holdout_required": True,
        "production_promoted": False,
        # Kept for backward compatibility. A development result cannot by
        # itself promote a production game model.
        "promoted": False,
    }

    predictions = test[
        [
            "date_key",
            "season",
            "game_pk",
            "at_bat_number",
            "batter",
            "pitcher",
            "outcome",
        ]
    ].copy()
    for i, label in enumerate(labels):
        predictions[f"p_league_{label}"] = league_prob[:, i]
        predictions[f"p_eb_{label}"] = eb_prob[:, i]
        predictions[f"p_model_{label}"] = full_prob[:, i]
        predictions[f"p_ensemble_{label}"] = candidate_prob[:, i]
    predictions.to_csv(
        output / "test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    result = {
        "schema": "baseball_research_lab.pa_benchmark.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": "chronological_development_holdout_executed",
        "config": config.to_dict(),
        "validation_partitions": partition_audit,
        "data": {
            "rows_total": int(len(features)),
            "rows_train": int(
                features["season"].isin(config.train_years).sum()
            ),
            "rows_validation": int(
                features["season"].isin(config.validation_years).sum()
            ),
            "rows_test": int(len(test)),
            "games_test": int(test["game_pk"].nunique()),
            "date_min": str(features["date_key"].min()),
            "date_max": str(features["date_key"].max()),
            "feature_receipt_sha256": dataframe_sha256(
                features,
                [
                    "date_key",
                    "game_pk",
                    "at_bat_number",
                    "batter",
                    "pitcher",
                    "outcome",
                ],
            ),
        },
        "baselines": {
            "league": league_metrics,
            "empirical_bayes_matchup": {
                **eb_metrics,
                "tuning": eb_tuning,
            },
        },
        "models": model_runs,
        "incremental_value": {
            "relative_log_loss_gain_vs_league": float(
                (league_metrics["log_loss"] - candidate_metrics["log_loss"])
                / league_metrics["log_loss"]
            ),
            "relative_log_loss_gain_vs_empirical_bayes": float(
                (eb_metrics["log_loss"] - candidate_metrics["log_loss"])
                / eb_metrics["log_loss"]
            ),
            "full_vs_empirical_bayes_clustered_ci": ci_vs_eb,
            "context_vs_talent_only_clustered_ci": ci_context,
        },
        "promotion": promotion,
        "interpretation_boundary": (
            "The 2025 season is now a development holdout because its v1 "
            "results informed subsequent architecture work. This result can "
            "advance the model to a locked 2026 evaluation, but it cannot "
            "validate runner transitions, bullpen logic, team-run "
            "distributions, winner probabilities, exact scores, or a "
            "production deployment."
        ),
    }

    tuning_audit = {
        "validation_partitions": partition_audit,
        "empirical_bayes": eb_tuning,
        "talent_only": model_runs["talent_only"]["tuning"],
        "talent_plus_context": model_runs["talent_plus_context"]["tuning"],
        "blended_candidate": blend_tuning,
    }
    for details in result["models"].values():
        details["tuning"] = {
            key: value
            for key, value in details["tuning"].items()
            if key != "trials"
        }
    (output / "PA_BENCHMARK_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "TUNING_AUDIT.json").write_text(
        json.dumps(tuning_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "talent_plus_context": fitted_models["talent_plus_context"],
            "talent_only": fitted_models["talent_only"],
            "empirical_bayes_parameters": eb_parameters,
            "model_weight": float(model_weight),
            "empirical_bayes_weight": float(1.0 - model_weight),
            "labels": labels,
            "config": config.to_dict(),
        },
        output / "pa_model.joblib",
        compress=3,
    )
    return result
