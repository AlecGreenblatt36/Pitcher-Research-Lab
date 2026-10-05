from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class BetaPrior:
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("beta prior parameters must be positive")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def effective_sample_size(self) -> float:
        return self.alpha + self.beta


@dataclass(frozen=True)
class BinomialPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total * total * (total + 1.0))

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.variance))


@dataclass(frozen=True)
class NormalPrior:
    mean: float
    variance: float

    def __post_init__(self) -> None:
        if self.variance <= 0:
            raise ValueError("normal prior variance must be positive")


@dataclass(frozen=True)
class NormalPosterior:
    mean: float
    variance: float

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.variance))


def beta_binomial_posterior(successes: int, trials: int, prior: BetaPrior) -> BinomialPosterior:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    return BinomialPosterior(prior.alpha + successes, prior.beta + trials - successes)


def dirichlet_posterior_mean(counts: Sequence[float], prior: Sequence[float]) -> np.ndarray:
    counts_array = np.asarray(counts, dtype=float)
    prior_array = np.asarray(prior, dtype=float)
    if counts_array.shape != prior_array.shape or counts_array.ndim != 1:
        raise ValueError("counts and prior must be one-dimensional with equal shape")
    if np.any(counts_array < 0) or np.any(prior_array <= 0):
        raise ValueError("counts must be non-negative and prior must be positive")
    posterior = counts_array + prior_array
    return posterior / posterior.sum()


def normal_normal_posterior(
    observed_mean: float,
    observed_variance: float,
    prior: NormalPrior,
) -> NormalPosterior:
    if observed_variance <= 0:
        raise ValueError("observed variance must be positive")
    prior_precision = 1.0 / prior.variance
    observed_precision = 1.0 / observed_variance
    variance = 1.0 / (prior_precision + observed_precision)
    mean = variance * (
        prior_precision * prior.mean + observed_precision * float(observed_mean)
    )
    return NormalPosterior(float(mean), float(variance))


def fit_beta_prior_moments(
    successes: Sequence[int],
    trials: Sequence[int],
    *,
    min_concentration: float = 2.0,
    max_concentration: float = 10000.0,
) -> BetaPrior:
    """Empirical-Bayes moment estimate with sampling-variance correction.

    This is a fast conjugate baseline, not a substitute for a full hierarchical
    posterior when richer grouping and time drift are required.
    """

    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if s.shape != n.shape or s.ndim != 1 or len(s) < 2:
        raise ValueError("successes and trials must be equal-length vectors with at least two rows")
    if np.any(n <= 0) or np.any(s < 0) or np.any(s > n):
        raise ValueError("invalid binomial observations")
    rates = s / n
    pooled = float(s.sum() / n.sum())
    observed_variance = float(np.var(rates, ddof=1))
    sampling_variance = float(np.mean(np.clip(rates * (1.0 - rates) / n, 0.0, None)))
    between_variance = max(observed_variance - sampling_variance, 1e-8)
    concentration = pooled * (1.0 - pooled) / between_variance - 1.0
    concentration = float(np.clip(concentration, min_concentration, max_concentration))
    pooled = float(np.clip(pooled, 1e-6, 1.0 - 1e-6))
    return BetaPrior(pooled * concentration, (1.0 - pooled) * concentration)


def grouped_beta_posteriors(
    successes: Sequence[int],
    trials: Sequence[int],
    groups: Sequence[Hashable],
) -> dict[Hashable, list[BinomialPosterior]]:
    if not (len(successes) == len(trials) == len(groups)):
        raise ValueError("successes, trials, and groups must have equal length")
    index_by_group: dict[Hashable, list[int]] = {}
    for index, group in enumerate(groups):
        index_by_group.setdefault(group, []).append(index)
    output: dict[Hashable, list[BinomialPosterior]] = {}
    for group, indices in index_by_group.items():
        group_successes = [successes[index] for index in indices]
        group_trials = [trials[index] for index in indices]
        prior = fit_beta_prior_moments(group_successes, group_trials)
        output[group] = [
            beta_binomial_posterior(successes[index], trials[index], prior)
            for index in indices
        ]
    return output
