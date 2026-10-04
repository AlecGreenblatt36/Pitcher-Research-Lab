"""Second-stage ablation diagnostics for the Baseball Research Lab.

This module implements the post-100-path protocol corrections requested after
reviewing the first 2025 full-season PA signal-survival ablation:

* a prior-data team-strength baseline with an explicit home-field logit offset;
* a global negative-binomial run distribution fitted on 2023-2024 only;
* finite-Monte-Carlo bias correction for log loss;
* a measurement-error de-noised calibration slope/intercept diagnostic;
* exact negative-binomial run CRPS and win probabilities.

The 1000-path rerun is still a 2025 development diagnostic. It is not a fresh
validation of the PA model, because 2025 participated in earlier PA-model
calibration/development. No 2026 result is used by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import lgamma
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit

from .replay import HistoricalGame, calibration_parameters


@dataclass(frozen=True)
class NegativeBinomialFit:
    alpha: float
    away_mean_2023_2024: float
    home_mean_2023_2024: float
    games: int
    observations: int
    method: str = "NB2 maximum likelihood with venue-specific means"

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "away_mean_2023_2024": float(self.away_mean_2023_2024),
            "home_mean_2023_2024": float(self.home_mean_2023_2024),
            "games": int(self.games),
            "observations": int(self.observations),
            "method": self.method,
        }


def fit_nb2_dispersion(games: Iterable[HistoricalGame]) -> NegativeBinomialFit:
    games = list(games)
    if not games:
        raise ValueError("at least one game is required")
    away_mean = float(np.mean([g.actual_away_runs for g in games]))
    home_mean = float(np.mean([g.actual_home_runs for g in games]))
    y = np.asarray(
        [value for g in games for value in (g.actual_away_runs, g.actual_home_runs)],
        dtype=float,
    )
    mu = np.asarray(
        [value for _ in games for value in (away_mean, home_mean)],
        dtype=float,
    )

    def objective(log_alpha: float) -> float:
        alpha = float(np.exp(log_alpha))
        size = 1.0 / alpha
        prob = size / (size + mu)
        log_pmf = (
            np.vectorize(lgamma)(y + size)
            - lgamma(size)
            - np.vectorize(lgamma)(y + 1.0)
            + size * np.log(prob)
            + y * np.log1p(-prob)
        )
        return float(-np.sum(log_pmf))

    fit = minimize_scalar(objective, bounds=(-9.0, 3.0), method="bounded")
    if not fit.success:
        raise RuntimeError(f"negative-binomial fit failed: {fit.message}")
    return NegativeBinomialFit(
        alpha=float(np.exp(fit.x)),
        away_mean_2023_2024=away_mean,
        home_mean_2023_2024=home_mean,
        games=len(games),
        observations=2 * len(games),
    )


def nb2_pmf(mean: float, alpha: float, cap: int | None = None) -> np.ndarray:
    mean = float(max(mean, 1e-9))
    alpha = float(max(alpha, 1e-9))
    if cap is None:
        variance = mean + alpha * mean * mean
        cap = int(max(40, np.ceil(mean + 12.0 * np.sqrt(variance))))
    size = 1.0 / alpha
    probability = size / (size + mean)
    k = np.arange(cap + 1, dtype=float)
    log_pmf = (
        np.vectorize(lgamma)(k + size)
        - lgamma(size)
        - np.vectorize(lgamma)(k + 1.0)
        + size * np.log(probability)
        + k * np.log1p(-probability)
    )
    pmf = np.exp(log_pmf)
    pmf[-1] += max(0.0, 1.0 - float(pmf.sum()))
    pmf = np.clip(pmf, 0.0, None)
    return pmf / pmf.sum()


def nb2_home_win_probability(away_mean: float, home_mean: float, alpha: float) -> float:
    variance_max = max(
        away_mean + alpha * away_mean * away_mean,
        home_mean + alpha * home_mean * home_mean,
    )
    cap = int(max(40, np.ceil(max(away_mean, home_mean) + 12.0 * np.sqrt(variance_max))))
    away = nb2_pmf(away_mean, alpha, cap)
    home = nb2_pmf(home_mean, alpha, cap)
    home_win = sum(home[index] * away[:index].sum() for index in range(len(home)))
    tie = float(np.dot(home, away))
    return float(home_win + 0.5 * tie)


def nb2_crps(mean: float, alpha: float, actual: int) -> float:
    variance = mean + alpha * mean * mean
    cap = int(max(50, actual + 30, np.ceil(mean + 14.0 * np.sqrt(variance))))
    cdf = np.cumsum(nb2_pmf(mean, alpha, cap))
    cdf[-1] = 1.0
    observation_cdf = (np.arange(cap + 1) >= int(actual)).astype(float)
    return float(np.sum((cdf - observation_cdf) ** 2))


def add_home_logit_offset(neutral_probability: float, prior_home_win_rate: float) -> float:
    p = float(np.clip(neutral_probability, 1e-8, 1.0 - 1e-8))
    hfa = float(np.clip(prior_home_win_rate, 1e-8, 1.0 - 1e-8))
    return float(expit(logit(p) + logit(hfa)))


def build_nb_team_baseline_map(
    games: Iterable[HistoricalGame],
    prior_games: Iterable[HistoricalGame],
    nb_fit: NegativeBinomialFit,
    shrink_games: float = 20.0,
) -> dict[int, dict]:
    """Build a strictly prior-date team baseline for the 2025 development season.

    Team offense/defense means are neutral with respect to venue. The separate
    home-field logit offset is the expanding home-win rate from all games that
    were complete strictly before the target date, initialized with 2023-2024.
    This avoids the old baseline's accidental road scoring edge from 2023-2024
    while retaining a time-valid home advantage.
    """

    games = sorted(games, key=lambda game: (game.game_date, game.game_pk))
    prior_games = list(prior_games)
    by_date: dict[str, list[HistoricalGame]] = defaultdict(list)
    for game in games:
        by_date[game.game_date].append(game)

    team_games: Counter[str] = Counter()
    team_runs_for: Counter[str] = Counter()
    team_runs_against: Counter[str] = Counter()
    total_runs = 0.0
    game_count = 0
    home_wins = 0.0

    def reveal(game: HistoricalGame) -> None:
        nonlocal total_runs, game_count, home_wins
        game_count += 1
        total_runs += game.actual_away_runs + game.actual_home_runs
        home_wins += float(game.actual_home_win)
        for team, runs_for, runs_against in (
            (game.away_team, game.actual_away_runs, game.actual_home_runs),
            (game.home_team, game.actual_home_runs, game.actual_away_runs),
        ):
            team_games[team] += 1
            team_runs_for[team] += runs_for
            team_runs_against[team] += runs_against

    for game in prior_games:
        reveal(game)

    output: dict[int, dict] = {}
    for date in sorted(by_date):
        if game_count == 0:
            league_runs = 4.4
            prior_home_win = 0.5
        else:
            league_runs = total_runs / (2.0 * game_count)
            prior_home_win = home_wins / game_count

        def rates(team: str) -> tuple[float, float]:
            count = team_games[team]
            offense = (
                team_runs_for[team] + shrink_games * league_runs
            ) / (count + shrink_games)
            defense = (
                team_runs_against[team] + shrink_games * league_runs
            ) / (count + shrink_games)
            return float(offense), float(defense)

        for game in by_date[date]:
            away_offense, away_defense = rates(game.away_team)
            home_offense, home_defense = rates(game.home_team)
            away_mean = league_runs * np.sqrt(
                (away_offense / max(league_runs, 1e-9))
                * (home_defense / max(league_runs, 1e-9))
            )
            home_mean = league_runs * np.sqrt(
                (home_offense / max(league_runs, 1e-9))
                * (away_defense / max(league_runs, 1e-9))
            )
            neutral_probability = nb2_home_win_probability(
                float(away_mean), float(home_mean), nb_fit.alpha
            )
            home_probability = add_home_logit_offset(
                neutral_probability, prior_home_win
            )
            output[int(game.game_pk)] = {
                "home_win_probability": float(home_probability),
                "neutral_home_win_probability": float(neutral_probability),
                "prior_home_win_rate": float(prior_home_win),
                "home_logit_offset": float(logit(np.clip(prior_home_win, 1e-8, 1 - 1e-8))),
                "away_mean_runs": float(away_mean),
                "home_mean_runs": float(home_mean),
                "away_crps": nb2_crps(float(away_mean), nb_fit.alpha, game.actual_away_runs),
                "home_crps": nb2_crps(float(home_mean), nb_fit.alpha, game.actual_home_runs),
                "nb_alpha": float(nb_fit.alpha),
            }

        # Reveal the entire date only after all games on the date are forecast.
        for game in by_date[date]:
            reveal(game)
    return output


def finite_path_log_loss_correction(
    y: np.ndarray, probabilities: np.ndarray, simulations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second-order delta-method correction for binomial Monte Carlo log loss.

    The raw probability is the fraction of simulation paths won by the home
    team. For observed y in {0,1}, the plug-in log loss has positive finite-N
    bias. This returns raw rows, estimated bias rows, and corrected rows.
    """

    if simulations <= 1:
        raise ValueError("simulations must exceed one")
    y = np.asarray(y, dtype=float)
    # Half-count clipping prevents a zero/one path estimate from producing an
    # infinite delta correction while preserving the operational raw forecast.
    floor = 0.5 / simulations
    p = np.clip(np.asarray(probabilities, dtype=float), floor, 1.0 - floor)
    raw = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    variance = p * (1.0 - p) / simulations
    curvature = y / (p * p) + (1.0 - y) / ((1.0 - p) ** 2)
    bias = 0.5 * variance * curvature
    corrected = raw - bias
    return raw, bias, corrected


def denoised_calibration(
    y: np.ndarray, probabilities: np.ndarray, simulations: int
) -> dict:
    """Approximate errors-in-variables correction for calibration.

    Raw calibration is fit against logit(p_hat). Monte Carlo path sampling adds
    approximately 1/[N p(1-p)] variance on that logit scale, attenuating the
    slope. We estimate a reliability ratio, divide the slope by it, and choose
    a corrected intercept that reproduces the observed event rate.
    """

    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.5 / simulations, 1 - 0.5 / simulations)
    raw = calibration_parameters(y, p)
    x = logit(p)
    raw_variance = float(np.var(x, ddof=1))
    estimated_noise = float(np.mean(1.0 / ((simulations - 1) * p * (1.0 - p))))
    reliability = float(np.clip((raw_variance - estimated_noise) / max(raw_variance, 1e-12), 1e-6, 1.0))
    raw_slope = float(raw["slope"] if raw["slope"] is not None else 0.0)
    corrected_slope = raw_slope / reliability
    target = float(y.mean())

    def mean_error(intercept: float) -> float:
        return float(expit(intercept + corrected_slope * x).mean() - target)

    try:
        corrected_intercept = float(brentq(mean_error, -20.0, 20.0))
        intercept_status = "solved to preserve observed event rate"
    except ValueError:
        corrected_intercept = float(raw["intercept"] or 0.0)
        intercept_status = "fallback to raw intercept"

    return {
        "raw": raw,
        "de_noised": {
            "slope": float(corrected_slope),
            "intercept": float(corrected_intercept),
            "reliability_ratio": reliability,
            "raw_logit_variance": raw_variance,
            "estimated_mc_logit_error_variance": estimated_noise,
            "intercept_status": intercept_status,
            "method": "first-order errors-in-variables reliability correction",
        },
    }
