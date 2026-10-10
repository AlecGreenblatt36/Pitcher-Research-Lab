from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


@dataclass(frozen=True)
class ProbabilityMetrics:
    log_loss: float
    multiclass_brier: float
    top_accuracy: float
    top_label_ece: float
    classwise_ece: float

    def to_dict(self) -> dict[str, float]:
        return {"log_loss": self.log_loss, "multiclass_brier": self.multiclass_brier, "top_accuracy": self.top_accuracy, "top_label_ece": self.top_label_ece, "classwise_ece": self.classwise_ece}


def _ece_binary(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p <= high if high == 1 else p < high)
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def probability_metrics(y_index: np.ndarray, probabilities: np.ndarray) -> ProbabilityMetrics:
    n_classes = probabilities.shape[1]
    one_hot = np.eye(n_classes)[y_index]
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    top = probabilities.argmax(axis=1)
    return ProbabilityMetrics(
        log_loss=float(log_loss(y_index, probabilities, labels=list(range(n_classes)))),
        multiclass_brier=float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        top_accuracy=float(np.mean(top == y_index)),
        top_label_ece=_ece_binary((top == y_index).astype(float), probabilities.max(axis=1)),
        classwise_ece=float(np.mean([_ece_binary((y_index == i).astype(float), probabilities[:, i]) for i in range(n_classes)])),
    )


def calibration_table(y_index: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> list[dict]:
    rows = []
    for i, label in enumerate(labels):
        observed = (y_index == i).astype(float)
        predicted = probabilities[:, i]
        rows.append({"outcome": label, "n": int(len(y_index)), "observed_rate": float(observed.mean()), "predicted_rate": float(predicted.mean()), "calibration_gap": float(predicted.mean() - observed.mean()), "binary_brier": float(np.mean((predicted - observed) ** 2))})
    return rows


def clustered_log_loss_difference_ci(y_index: np.ndarray, candidate: np.ndarray, reference: np.ndarray, game_ids: np.ndarray, replicates: int = 1000, seed: int = 36) -> dict[str, float]:
    row = np.arange(len(y_index))
    differences = -np.log(np.clip(candidate[row, y_index], 1e-12, 1)) + np.log(np.clip(reference[row, y_index], 1e-12, 1))
    grouped = pd.DataFrame({"game": game_ids, "difference": differences}).groupby("game", sort=False)["difference"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sums, counts, game_count = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float), len(grouped)
    values = np.empty(replicates)
    for i in range(replicates):
        sampled = rng.integers(0, game_count, size=game_count)
        values[i] = sums[sampled].sum() / counts[sampled].sum()
    return {"difference_nats_per_pa": float(differences.mean()), "ci_95_low": float(np.quantile(values, 0.025)), "ci_95_high": float(np.quantile(values, 0.975)), "clusters": int(game_count), "replicates": int(replicates)}
