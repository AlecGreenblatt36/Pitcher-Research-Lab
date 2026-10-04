"""Verify the locked PA provider under the exact replay environment.

This is a compatibility/reproduction gate, not a new model evaluation.  It
rebuilds the provider's 122-feature rows from the locked PA history for three
historical 2026 dates and compares:

* provider fast NumPy model component vs the saved p_model columns;
* provider empirical-Bayes component vs the saved p_eb columns;
* provider 90/10 ensemble vs the saved p_ensemble columns;
* fast NumPy model component vs the serialized sklearn pipeline.

The receipt records package versions, artifact hashes, selected row counts,
text-column dtypes, and maximum absolute differences.  The process fails closed
if any required artifact or row is missing or if the tolerance is exceeded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_lab.game_sim.locked_pa_provider import (  # noqa: E402
    LockedPAModelProvider,
    MODEL_LABELS,
)

DEFAULT_DATES = ("2026-04-20", "2026-07-04", "2026-09-27")
EXPECTED_ROWS = {
    "2026-04-20": 801,
    "2026-07-04": 1134,
    "2026-09-27": 1025,
}
HISTORY_COLUMNS = [
    "date_key", "game_pk", "at_bat_number", "batter", "pitcher",
    "stand", "p_throws", "park", "outcome", "is_home_batter",
    "inning", "outs_when_up", "runner_1b", "runner_2b", "runner_3b",
    "bat_score_diff", "n_thruorder_pitcher",
    "batter_days_since_prev_game", "pitcher_days_since_prev_game",
    "age_bat", "age_pit",
]
JOIN_KEYS = [
    "date_key", "game_pk", "at_bat_number", "batter", "pitcher", "outcome"
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    names = [
        "pandas", "numpy", "joblib", "scikit-learn", "scipy",
    ]
    return {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in names},
    }


def saved_vector(row: pd.Series, prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"p_{prefix}_{label}"]) for label in MODEL_LABELS])


def finite_or_nan(value) -> float:
    return float(value) if not pd.isna(value) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir",
        default="pa_model_reference/model_runs/pa_locked_2026",
    )
    parser.add_argument(
        "--output",
        default=(
            "benchmark_results/locked_pa_pinned_reproduction/"
            "LOCKED_PA_PINNED_REPRODUCTION.json"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--dates", nargs="*", default=list(DEFAULT_DATES))
    args = parser.parse_args()

    reference = ROOT / args.reference_dir
    model_path = reference / "artifacts/pa_model.joblib"
    predictions_path = reference / "artifacts/test_predictions.csv.gz"
    history_path = reference / "plate_appearances.csv.gz"
    required = [model_path, predictions_path, history_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing locked PA artifacts: {missing}")

    dates = tuple(str(value)[:10] for value in args.dates)
    if not dates:
        raise RuntimeError("at least one verification date is required")

    probability_columns = [
        f"p_{prefix}_{label}"
        for prefix in ("eb", "model", "ensemble")
        for label in MODEL_LABELS
    ]
    predictions = pd.read_csv(
        predictions_path,
        usecols=JOIN_KEYS + probability_columns,
        low_memory=False,
    )
    predictions["date_key"] = predictions["date_key"].astype(str).str[:10]
    predictions = predictions[predictions["date_key"].isin(dates)].copy()

    history = pd.read_csv(history_path, usecols=HISTORY_COLUMNS, low_memory=False)
    history["date_key"] = history["date_key"].astype(str).str[:10]
    selected_history = history[history["date_key"].isin(dates)].copy()

    joined = predictions.merge(
        selected_history,
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    missing_context = joined.loc[joined["_merge"] != "both", JOIN_KEYS]
    if len(missing_context):
        raise RuntimeError(
            f"{len(missing_context)} saved prediction rows lack exact history context"
        )
    joined = joined.drop(columns="_merge")

    date_results: list[dict] = []
    global_diffs = {
        "saved_model": 0.0,
        "saved_eb": 0.0,
        "saved_ensemble": 0.0,
        "fast_vs_sklearn_model": 0.0,
        "fast_vs_sklearn_ensemble": 0.0,
    }

    for date in dates:
        rows = joined[joined["date_key"] == date].copy()
        expected = EXPECTED_ROWS.get(date)
        if expected is not None and len(rows) != expected:
            raise RuntimeError(
                f"{date}: expected {expected} saved rows, found {len(rows)}"
            )
        if rows.empty:
            raise RuntimeError(f"{date}: no verification rows")

        provider = LockedPAModelProvider(
            artifact_path=model_path,
            history_path=history_path,
            cutoff_date=date,
            game_date=date,
            park=str(rows.iloc[0]["park"]),
            history_frame=history,
            sequential_history=False,
        )

        date_diffs = {key: 0.0 for key in global_diffs}
        absolute_ensemble_errors: list[float] = []

        for row in rows.itertuples(index=False):
            provider.set_park(str(row.park))
            feature_row, _ = provider.feature_row(
                int(row.batter),
                int(row.pitcher),
                str(row.stand),
                str(row.p_throws),
                is_home_batter=int(row.is_home_batter),
                inning=int(row.inning),
                outs=int(row.outs_when_up),
                runners=(
                    int(row.runner_1b),
                    int(row.runner_2b),
                    int(row.runner_3b),
                ),
                bat_score_diff=int(row.bat_score_diff),
                times_through=int(row.n_thruorder_pitcher),
                batter_days=finite_or_nan(row.batter_days_since_prev_game),
                pitcher_days=finite_or_nan(row.pitcher_days_since_prev_game),
                age_bat=finite_or_nan(row.age_bat),
                age_pit=finite_or_nan(row.age_pit),
            )
            fast = provider.predict_row(feature_row)
            sklearn_frame = pd.DataFrame(
                [feature_row], columns=provider._fitted.feature_columns
            )
            sklearn_model = provider._fitted.predict_proba(sklearn_frame)[0]
            sklearn_ensemble = (
                provider._model_weight * sklearn_model
                + (1.0 - provider._model_weight) * fast["eb"]
            )

            saved_model = np.asarray(
                [getattr(row, f"p_model_{label}") for label in MODEL_LABELS],
                dtype=float,
            )
            saved_eb = np.asarray(
                [getattr(row, f"p_eb_{label}") for label in MODEL_LABELS],
                dtype=float,
            )
            saved_ensemble = np.asarray(
                [getattr(row, f"p_ensemble_{label}") for label in MODEL_LABELS],
                dtype=float,
            )

            current = {
                "saved_model": float(np.max(np.abs(fast["model"] - saved_model))),
                "saved_eb": float(np.max(np.abs(fast["eb"] - saved_eb))),
                "saved_ensemble": float(
                    np.max(np.abs(fast["ensemble"] - saved_ensemble))
                ),
                "fast_vs_sklearn_model": float(
                    np.max(np.abs(fast["model"] - sklearn_model))
                ),
                "fast_vs_sklearn_ensemble": float(
                    np.max(np.abs(fast["ensemble"] - sklearn_ensemble))
                ),
            }
            for key, value in current.items():
                date_diffs[key] = max(date_diffs[key], value)
                global_diffs[key] = max(global_diffs[key], value)
            absolute_ensemble_errors.extend(
                np.abs(fast["ensemble"] - saved_ensemble).tolist()
            )

        date_results.append(
            {
                "date": date,
                "rows": int(len(rows)),
                "last_history_date_used": provider.state.last_history_date,
                "history_rows_used": int(provider.state.n_rows),
                "max_abs_differences": date_diffs,
                "median_abs_saved_ensemble_difference": float(
                    np.median(absolute_ensemble_errors)
                ),
            }
        )

    pass_keys = (
        "saved_model",
        "saved_eb",
        "saved_ensemble",
        "fast_vs_sklearn_model",
        "fast_vs_sklearn_ensemble",
    )
    passed = all(global_diffs[key] <= args.tolerance for key in pass_keys)
    receipt = {
        "schema": "baseball_research_lab.locked_pa_pinned_reproduction.v1",
        "status": "PASS" if passed else "FAIL",
        "purpose": (
            "Compatibility reproduction under the exact full-season replay "
            "environment; not a new model evaluation."
        ),
        "tolerance": float(args.tolerance),
        "environment": versions(),
        "artifacts": {
            "model": {"path": str(model_path), "sha256": sha256(model_path)},
            "history": {"path": str(history_path), "sha256": sha256(history_path)},
            "saved_predictions": {
                "path": str(predictions_path),
                "sha256": sha256(predictions_path),
            },
        },
        "selected_dates": list(dates),
        "selected_rows": int(len(joined)),
        "history_text_dtypes_after_read": {
            column: str(history[column].dtype)
            for column in ("date_key", "stand", "p_throws", "park", "outcome")
        },
        "global_max_abs_differences": global_diffs,
        "dates": date_results,
        "pandas_3_compatibility_boundary": (
            "Provider explicitly normalizes date_key, stand, p_throws, park, "
            "and outcome before grouping/feature assembly."
        ),
    }
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(
            f"locked PA reproduction failed tolerance {args.tolerance}: {global_diffs}"
        )


if __name__ == "__main__":
    main()
