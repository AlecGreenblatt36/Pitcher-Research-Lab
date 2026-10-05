"""Forecast-valid deterministic baselines for frozen full-season replays.

The functions in this module use only games and plate appearances completed on
strictly earlier dates than the target game.  Games on the same date are
forecast from the same pre-date state and are revealed together afterward.

These are comparison models, not production forecasts:

* league NB + home-field advantage;
* prior-date team offense/defense NB + home-field advantage;
* the team baseline with a conservative prior-starting-pitcher adjustment.

The starter adjustment uses runs scored while the starter was in the game,
not official earned runs.  It is intentionally simple and heavily shrunk; its
purpose is to provide a stronger baseline, not to make a pitcher-quality claim.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import exp, log
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logit

from .ablation_v2 import (
    NegativeBinomialFit,
    add_home_logit_offset,
    build_nb_team_baseline_map,
    nb2_crps,
    nb2_home_win_probability,
)
from .replay import HistoricalGame
from .starter_hazard import SHRINK_STARTS, build_decisions


def build_league_nb_baseline_map(
    games: Iterable[HistoricalGame],
    prior_games: Iterable[HistoricalGame],
    nb_fit: NegativeBinomialFit,
) -> dict[int, dict]:
    """Expanding prior-date league scoring baseline with home advantage."""
    games = sorted(games, key=lambda game: (game.game_date, game.game_pk))
    prior_games = list(prior_games)
    by_date: dict[str, list[HistoricalGame]] = defaultdict(list)
    for game in games:
        by_date[game.game_date].append(game)

    away_runs = float(sum(game.actual_away_runs for game in prior_games))
    home_runs = float(sum(game.actual_home_runs for game in prior_games))
    home_wins = float(sum(game.actual_home_win for game in prior_games))
    game_count = len(prior_games)
    output: dict[int, dict] = {}

    for date in sorted(by_date):
        if game_count:
            away_mean = away_runs / game_count
            home_mean = home_runs / game_count
            prior_home_win = home_wins / game_count
        else:
            away_mean = float(nb_fit.away_mean_2023_2024)
            home_mean = float(nb_fit.home_mean_2023_2024)
            prior_home_win = 0.5

        neutral = nb2_home_win_probability(away_mean, home_mean, nb_fit.alpha)
        # Venue-specific means already encode scoring HFA.  The logit offset is
        # retained only as a transparent diagnostic, not applied a second time.
        for game in by_date[date]:
            output[int(game.game_pk)] = {
                "home_win_probability": float(neutral),
                "neutral_home_win_probability": float(neutral),
                "prior_home_win_rate": float(prior_home_win),
                "home_logit_offset": float(logit(np.clip(prior_home_win, 1e-8, 1 - 1e-8))),
                "away_mean_runs": float(away_mean),
                "home_mean_runs": float(home_mean),
                "away_crps": nb2_crps(float(away_mean), nb_fit.alpha, game.actual_away_runs),
                "home_crps": nb2_crps(float(home_mean), nb_fit.alpha, game.actual_home_runs),
                "nb_alpha": float(nb_fit.alpha),
            }

        for game in by_date[date]:
            game_count += 1
            away_runs += game.actual_away_runs
            home_runs += game.actual_home_runs
            home_wins += float(game.actual_home_win)

    return output


def starter_run_lines(history: pd.DataFrame) -> pd.DataFrame:
    """One conservative starter line per game/team from the PA history.

    ``build_decisions`` supplies cumulative runs scored by the opponent after
    each starter PA and strictly identifies the game's first pitcher.  The
    maximum cumulative value is used as runs allowed while the starter was in
    the game.  This is not official pitcher runs/earned runs and is named
    accordingly throughout the receipts.
    """
    decisions, starts = build_decisions(history)
    cumulative = (
        decisions.groupby(["game_pk", "fielding_team", "starter"], sort=False)
        .agg(
            date_key=("date_key", "first"),
            season=("season", "first"),
            runs_while_in_game=("runs", "max"),
        )
        .reset_index()
    )
    lines = starts.merge(
        cumulative,
        on=["game_pk", "fielding_team", "starter", "date_key", "season"],
        how="left",
        validate="one_to_one",
    )
    lines["runs_while_in_game"] = lines["runs_while_in_game"].fillna(0.0).clip(lower=0.0)
    lines["bf_total"] = lines["bf_total"].clip(lower=1.0)
    return lines.sort_values(["date_key", "game_pk", "fielding_team"], kind="mergesort")


def build_starter_adjusted_nb_baseline_map(
    games: Iterable[HistoricalGame],
    prior_games: Iterable[HistoricalGame],
    history: pd.DataFrame,
    nb_fit: NegativeBinomialFit,
    *,
    shrink_bf: float = 200.0,
    expected_team_bf: float = 38.0,
) -> dict[int, dict]:
    """Team NB baseline with a prior-date starting-pitcher run adjustment.

    A starter's run rate is empirical runs scored while he was in the game per
    batter faced, shrunk by ``shrink_bf`` league-average batters faced.  The
    rate ratio affects only the expected share of opponent batters faced by the
    starter.  All target-date starts are revealed together after forecasts for
    the date are built.
    """
    games = sorted(games, key=lambda game: (game.game_date, game.game_pk))
    prior_games = list(prior_games)
    team_base = build_nb_team_baseline_map(games, prior_games, nb_fit)
    regular_history = history[history["game_type"].astype(str).eq("R")] if "game_type" in history.columns else history
    lines = starter_run_lines(regular_history)
    by_date: dict[str, list[HistoricalGame]] = defaultdict(list)
    for game in games:
        by_date[game.game_date].append(game)

    first_target_date = games[0].game_date if games else "9999-12-31"
    # Everything before the first target date initializes the baseline.  Every
    # game on or after that date, including games excluded from final scoring,
    # is revealed only after its date has been forecast.  This prevents an
    # excluded future target-season game from leaking into earlier forecasts.
    prior_lines = lines[lines["date_key"] < first_target_date].copy()
    evolving_lines = lines[lines["date_key"] >= first_target_date].copy()
    target_lines_by_date = {
        date: frame for date, frame in evolving_lines.groupby("date_key", sort=False)
    }

    pitcher_bf: Counter[int] = Counter()
    pitcher_runs: Counter[int] = Counter()
    pitcher_start_bf: Counter[int] = Counter()
    pitcher_starts: Counter[int] = Counter()
    league_bf = 0.0
    league_runs = 0.0
    league_start_bf = 0.0
    league_starts = 0

    def reveal(frame: pd.DataFrame) -> None:
        nonlocal league_bf, league_runs, league_start_bf, league_starts
        for row in frame.itertuples():
            pitcher = int(row.starter)
            bf = float(row.bf_total)
            runs = float(row.runs_while_in_game)
            pitcher_bf[pitcher] += bf
            pitcher_runs[pitcher] += runs
            pitcher_start_bf[pitcher] += bf
            pitcher_starts[pitcher] += 1
            league_bf += bf
            league_runs += runs
            league_start_bf += bf
            league_starts += 1

    reveal(prior_lines)
    output: dict[int, dict] = {}

    def starter_context(pitcher: int) -> tuple[float, float, float, float]:
        league_rate = league_runs / max(league_bf, 1.0)
        if league_rate <= 0:
            league_rate = 4.4 / expected_team_bf
        rate = (
            pitcher_runs[pitcher] + shrink_bf * league_rate
        ) / (pitcher_bf[pitcher] + shrink_bf)
        rate_ratio = float(np.clip(rate / league_rate, 0.55, 1.65))
        league_mean_bf = league_start_bf / max(league_starts, 1)
        expected_bf = (
            pitcher_start_bf[pitcher] + SHRINK_STARTS * league_mean_bf
        ) / (pitcher_starts[pitcher] + SHRINK_STARTS)
        starter_share = float(np.clip(expected_bf / expected_team_bf, 0.30, 0.72))
        adjustment = float(exp(starter_share * log(rate_ratio)))
        return adjustment, rate_ratio, float(expected_bf), starter_share

    for date in sorted(by_date):
        for game in by_date[date]:
            base = team_base[int(game.game_pk)]
            home_starter = int(game.home_starter[0])
            away_starter = int(game.away_starter[0])
            away_adjustment, home_sp_ratio, home_sp_bf, home_sp_share = starter_context(home_starter)
            home_adjustment, away_sp_ratio, away_sp_bf, away_sp_share = starter_context(away_starter)
            away_mean = float(base["away_mean_runs"] * away_adjustment)
            home_mean = float(base["home_mean_runs"] * home_adjustment)
            neutral = nb2_home_win_probability(away_mean, home_mean, nb_fit.alpha)
            home_probability = add_home_logit_offset(neutral, float(base["prior_home_win_rate"]))
            output[int(game.game_pk)] = {
                "home_win_probability": float(home_probability),
                "neutral_home_win_probability": float(neutral),
                "prior_home_win_rate": float(base["prior_home_win_rate"]),
                "home_logit_offset": float(base["home_logit_offset"]),
                "away_mean_runs": away_mean,
                "home_mean_runs": home_mean,
                "away_crps": nb2_crps(away_mean, nb_fit.alpha, game.actual_away_runs),
                "home_crps": nb2_crps(home_mean, nb_fit.alpha, game.actual_home_runs),
                "nb_alpha": float(nb_fit.alpha),
                "home_starter_run_rate_ratio": home_sp_ratio,
                "away_starter_run_rate_ratio": away_sp_ratio,
                "home_starter_expected_bf": home_sp_bf,
                "away_starter_expected_bf": away_sp_bf,
                "home_starter_expected_bf_share": home_sp_share,
                "away_starter_expected_bf_share": away_sp_share,
                "away_run_adjustment": away_adjustment,
                "home_run_adjustment": home_adjustment,
                "run_definition": "opponent runs scored while starter remained in game; not official earned runs",
            }

        reveal(target_lines_by_date.get(date, evolving_lines.iloc[0:0]))

    return output
