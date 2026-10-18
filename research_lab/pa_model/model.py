from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PAConfig
from .evaluation import probability_metrics


@dataclass
class FittedPAModel:
    estimator: Pipeline
    feature_columns: list[str]
    labels: list[str]
    regularization_c: float
    temperature: float
    class_biases: np.ndarray

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(frame[self.feature_columns])
        return apply_logit_calibration(raw, self.temperature, self.class_biases)


def apply_logit_calibration(
    probabilities: np.ndarray,
    temperature: float,
    class_biases: np.ndarray | None = None,
) -> np.ndarray:
    """Apply multiclass temperature scaling plus zero-sum class intercepts."""

    values = np.asarray(probabilities, dtype=float)
    biases = (
        np.zeros(values.shape[1], dtype=float)
        if class_biases is None
        else np.asarray(class_biases, dtype=float)
    )
    logits = np.log(np.clip(values, 1e-12, 1.0)) / max(float(temperature), 1e-6)
    logits += biases
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def validation_partitions(
    features: pd.DataFrame,
    config: PAConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Split validation dates into disjoint tune/calibrate/blend periods."""

    config = config or PAConfig()
    validation = features["season"].isin(config.validation_years).to_numpy()
    dates = np.array(
        sorted(pd.Series(features.loc[validation, "date_key"]).drop_duplicates().tolist())
    )
    if len(dates) < 4:
        raise ValueError("validation period needs at least four unique dates")
    if config.validation_tuning_fraction <= 0:
        raise ValueError("validation_tuning_fraction must be positive")
    total_reserved = (
        config.validation_tuning_fraction + config.validation_calibration_fraction
    )
    if total_reserved >= 1:
        raise ValueError("validation split fractions must leave dates for blending")

    tune_end = max(1, int(np.floor(len(dates) * config.validation_tuning_fraction)))
    calibration_end = max(
        tune_end + 1,
        int(np.floor(len(dates) * total_reserved)),
    )
    calibration_end = min(calibration_end, len(dates) - 1)
    tune_dates = set(dates[:tune_end])
    calibration_dates = set(dates[tune_end:calibration_end])
    blend_dates = set(dates[calibration_end:])
    date_values = features["date_key"]
    masks = {
        "tune": validation & date_values.isin(tune_dates).to_numpy(),
        "calibration": validation & date_values.isin(calibration_dates).to_numpy(),
        "blend": validation & date_values.isin(blend_dates).to_numpy(),
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("validation partitions must all contain rows")

    audit = {
        "validation_unique_dates": int(len(dates)),
        "tune_date_min": str(min(tune_dates)),
        "tune_date_max": str(max(tune_dates)),
        "calibration_date_min": str(min(calibration_dates)),
        "calibration_date_max": str(max(calibration_dates)),
        "blend_date_min": str(min(blend_dates)),
        "blend_date_max": str(max(blend_dates)),
        "rows": {name: int(mask.sum()) for name, mask in masks.items()},
    }
    return masks, audit


def _estimator(c: float, config: PAConfig) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                slice(0, None),
            )
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=c,
        solver="newton-cholesky",
        max_iter=config.max_iter,
        tol=1e-5,
        random_state=config.random_seed,
    )
    return Pipeline([("preprocess", preprocessing), ("model", classifier)])


def fit_logit_calibration(
    y_index: np.ndarray,
    probabilities: np.ndarray,
    l2: float = 1e-4,
) -> tuple[float, np.ndarray, dict]:
    """Fit a regularized affine calibration map on held-out dates."""

    raw = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_index, dtype=int)
    n_classes = raw.shape[1]
    row = np.arange(len(y))

    def unpack(theta: np.ndarray) -> tuple[float, np.ndarray]:
        temperature = float(np.exp(theta[0]))
        free = np.asarray(theta[1:], dtype=float)
        biases = np.concatenate([free, [-float(free.sum())]])
        return temperature, biases

    def objective(theta: np.ndarray) -> float:
        temperature, biases = unpack(theta)
        calibrated = apply_logit_calibration(raw, temperature, biases)
        nll = float(
            -np.log(np.clip(calibrated[row, y], 1e-12, 1.0)).mean()
        )
        penalty = float(l2) * (
            float(theta[0] ** 2) + float(np.mean(biases**2))
        )
        return nll + penalty

    initial = np.zeros(n_classes, dtype=float)
    bounds = [
        (float(np.log(0.5)), float(np.log(2.5))),
        *[(-0.75, 0.75)] * (n_classes - 1),
    ]
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 250, "ftol": 1e-12},
    )
    selected = result.x if result.success else initial
    temperature, biases = unpack(selected)
    calibrated = apply_logit_calibration(raw, temperature, biases)
    audit = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "temperature": float(temperature),
        "class_biases": [float(value) for value in biases],
        "pre_calibration_log_loss": probability_metrics(y, raw).log_loss,
        "post_calibration_log_loss": probability_metrics(y, calibrated).log_loss,
    }
    return temperature, biases, audit


def fit_frozen_model(
    features: pd.DataFrame,
    feature_columns: list[str],
    config: PAConfig | None = None,
) -> tuple[FittedPAModel, dict]:
    config = config or PAConfig()
    labels = list(config.outcome_labels)
    y = features["outcome"].map(
        {label: i for i, label in enumerate(labels)}
    ).to_numpy(int)
    train_mask = features["season"].isin(config.train_years).to_numpy()
    partitions, partition_audit = validation_partitions(features, config)
    tune_mask = partitions["tune"]
    calibration_mask = partitions["calibration"]
    if not train_mask.any():
        raise ValueError("training period contains no rows")

    trials: list[dict] = []
    best: tuple[float, float] | None = None
    for c in config.regularization_grid:
        estimator = _estimator(c, config)
        estimator.fit(features.loc[train_mask, feature_columns], y[train_mask])
        raw = estimator.predict_proba(features.loc[tune_mask, feature_columns])
        score = probability_metrics(y[tune_mask], raw).log_loss
        trials.append(
            {"regularization_c": float(c), "tune_log_loss": float(score)}
        )
        if best is None or score < best[0]:
            best = (score, c)

    assert best is not None
    _, best_c = best
    refit_mask = train_mask | tune_mask
    final_estimator = _estimator(best_c, config)
    final_estimator.fit(
        features.loc[refit_mask, feature_columns],
        y[refit_mask],
    )
    raw_calibration = final_estimator.predict_proba(
        features.loc[calibration_mask, feature_columns]
    )
    temperature, class_biases, calibration_audit = fit_logit_calibration(
        y[calibration_mask],
        raw_calibration,
        config.calibration_l2,
    )
    fitted = FittedPAModel(
        final_estimator,
        feature_columns,
        labels,
        float(best_c),
        float(temperature),
        class_biases,
    )
    return fitted, {
        "best_regularization_c": float(best_c),
        "best_tune_log_loss": float(best[0]),
        "calibration": calibration_audit,
        "validation_partitions": partition_audit,
        "trials": sorted(trials, key=lambda item: item["tune_log_loss"]),
    }
