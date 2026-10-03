from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
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

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return apply_temperature(self.estimator.predict_proba(frame[self.feature_columns]), self.temperature)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def _estimator(c: float, config: PAConfig) -> Pipeline:
    preprocessing = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]), slice(0, None))], remainder="drop")
    classifier = LogisticRegression(C=c, solver="newton-cholesky", max_iter=config.max_iter, tol=1e-5, random_state=config.random_seed)
    return Pipeline([("preprocess", preprocessing), ("model", classifier)])


def fit_frozen_model(features: pd.DataFrame, feature_columns: list[str], config: PAConfig | None = None) -> tuple[FittedPAModel, dict]:
    config = config or PAConfig()
    labels = list(config.outcome_labels)
    y = features["outcome"].map({label: i for i, label in enumerate(labels)}).to_numpy(int)
    train_mask = features["season"].isin(config.train_years).to_numpy()
    validation_mask = features["season"].isin(config.validation_years).to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("training and validation periods must both contain rows")
    x_train, y_train = features.loc[train_mask, feature_columns], y[train_mask]
    x_validation, y_validation = features.loc[validation_mask, feature_columns], y[validation_mask]
    trials = []
    best = None
    for c in config.regularization_grid:
        estimator = _estimator(c, config)
        estimator.fit(x_train, y_train)
        raw = estimator.predict_proba(x_validation)
        for temperature in config.temperature_grid:
            score = probability_metrics(y_validation, apply_temperature(raw, temperature)).log_loss
            trials.append({"regularization_c": float(c), "temperature": float(temperature), "validation_log_loss": float(score)})
            if best is None or score < best[0]:
                best = (score, c, temperature)
    _, best_c, best_temperature = best
    refit_mask = train_mask | validation_mask
    final_estimator = _estimator(best_c, config)
    final_estimator.fit(features.loc[refit_mask, feature_columns], y[refit_mask])
    fitted = FittedPAModel(final_estimator, feature_columns, labels, float(best_c), float(best_temperature))
    return fitted, {"best_regularization_c": float(best_c), "best_temperature": float(best_temperature), "best_validation_log_loss": float(best[0]), "trials": sorted(trials, key=lambda x: x["validation_log_loss"])}
