from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class PITSummary:
    n: int
    mean: float
    variance: float
    ks_statistic: float
    ks_p_value: float
    histogram: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationFit:
    slope: float
    intercept: float
    success: bool


def _validated_pmf(pmf: Sequence[float]) -> np.ndarray:
    values = np.asarray(pmf, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("pmf must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("pmf values must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("pmf must have positive mass")
    return values / total


def randomized_pit(pmf: Sequence[float], observed: int, uniform_draw: float) -> float:
    probabilities = _validated_pmf(pmf)
    if not 0 <= observed < len(probabilities):
        raise ValueError("observed value is outside PMF support")
    if not 0.0 <= uniform_draw <= 1.0:
        raise ValueError("uniform_draw must be in [0, 1]")
    below = float(probabilities[:observed].sum())
    return below + float(uniform_draw) * float(probabilities[observed])


def randomized_pit_batch(
    pmfs: Sequence[Sequence[float]],
    observed: Sequence[int],
    uniform_draws: Sequence[float],
) -> np.ndarray:
    if not (len(pmfs) == len(observed) == len(uniform_draws)):
        raise ValueError("pmfs, observed, and uniform_draws must have equal length")
    return np.asarray(
        [
            randomized_pit(pmf, int(value), float(draw))
            for pmf, value, draw in zip(pmfs, observed, uniform_draws)
        ],
        dtype=float,
    )


def pit_summary(values: Sequence[float], *, bins: int = 10) -> PITSummary:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("PIT values must be a non-empty vector")
    if np.any(array < 0) or np.any(array > 1) or not np.isfinite(array).all():
        raise ValueError("PIT values must be finite and in [0, 1]")
    ks = stats.kstest(array, "uniform")
    histogram, _ = np.histogram(array, bins=bins, range=(0.0, 1.0))
    return PITSummary(
        n=len(array),
        mean=float(array.mean()),
        variance=float(array.var(ddof=1)) if len(array) > 1 else 0.0,
        ks_statistic=float(ks.statistic),
        ks_p_value=float(ks.pvalue),
        histogram=tuple(int(value) for value in histogram),
    )


def central_fractional_weights(pmf: Sequence[float], level: float) -> np.ndarray:
    """Return fractional inclusion weights with exact forecast mass ``level``."""

    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    probabilities = _validated_pmf(pmf)
    weights = np.ones_like(probabilities)
    tail = (1.0 - level) / 2.0

    remaining = tail
    for index in range(len(probabilities)):
        removable = probabilities[index] * weights[index]
        if remaining >= removable - 1e-15:
            weights[index] = 0.0
            remaining -= removable
        else:
            weights[index] -= remaining / probabilities[index]
            break

    remaining = tail
    for index in range(len(probabilities) - 1, -1, -1):
        removable = probabilities[index] * weights[index]
        if remaining >= removable - 1e-15:
            weights[index] = 0.0
            remaining -= removable
        else:
            weights[index] -= remaining / probabilities[index]
            break

    return np.clip(weights, 0.0, 1.0)


def fractional_interval_credit(pmf: Sequence[float], observed: int, level: float) -> float:
    weights = central_fractional_weights(pmf, level)
    if not 0 <= observed < len(weights):
        raise ValueError("observed value is outside PMF support")
    return float(weights[observed])


def fractional_coverage(
    pmfs: Sequence[Sequence[float]],
    observed: Sequence[int],
    level: float,
) -> float:
    if len(pmfs) != len(observed):
        raise ValueError("pmfs and observed must have equal length")
    credits = [
        fractional_interval_credit(pmf, int(value), level)
        for pmf, value in zip(pmfs, observed)
    ]
    return float(np.mean(credits))


def finite_path_binary_scores(
    outcomes: Sequence[int],
    probabilities: Sequence[float],
    simulations: int,
    *,
    clip: float = 1e-8,
) -> dict[str, float]:
    if simulations <= 1:
        raise ValueError("simulations must exceed one")
    y = np.asarray(outcomes, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), clip, 1.0 - clip)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("outcomes and probabilities must be equal-length vectors")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("outcomes must be binary")
    raw_brier = (p - y) ** 2
    brier_bias = p * (1.0 - p) / (simulations - 1)
    raw_log_loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    log_bias = y * (1.0 - p) / (2.0 * simulations * p) + (
        (1.0 - y) * p / (2.0 * simulations * (1.0 - p))
    )
    return {
        "raw_brier": float(raw_brier.mean()),
        "corrected_brier": float((raw_brier - brier_bias).mean()),
        "raw_log_loss": float(raw_log_loss.mean()),
        "corrected_log_loss": float((raw_log_loss - log_bias).mean()),
    }


def fit_logistic_calibration(
    outcomes: Sequence[int],
    probabilities: Sequence[float],
    *,
    clip: float = 1e-6,
) -> CalibrationFit:
    y = np.asarray(outcomes, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), clip, 1.0 - clip)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("outcomes and probabilities must be equal-length vectors")
    x = np.log(p / (1.0 - p))

    def objective(theta: np.ndarray) -> float:
        z = np.clip(theta[0] + theta[1] * x, -40.0, 40.0)
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    result = optimize.minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    return CalibrationFit(float(result.x[1]), float(result.x[0]), bool(result.success))


def forward_month_splits(dates: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Create expanding-window month splits; never randomize time order."""

    months = np.asarray([str(value)[:7] for value in dates])
    unique = sorted(set(months.tolist()))
    output: list[tuple[np.ndarray, np.ndarray, str]] = []
    for index in range(1, len(unique)):
        train_months = set(unique[:index])
        test_month = unique[index]
        train = np.asarray([month in train_months for month in months], dtype=bool)
        test = months == test_month
        if train.any() and test.any():
            output.append((train, test, test_month))
    return output
