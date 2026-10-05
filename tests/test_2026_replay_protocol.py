from __future__ import annotations

import pandas as pd

from research_lab.game_sim.ablation_v2 import fit_nb2_dispersion
from research_lab.game_sim.replay import HistoricalGame
from research_lab.game_sim import replay_baselines


def _game(pk: int, date: str, away: str, home: str, away_runs: int, home_runs: int,
          away_starter: int = 20, home_starter: int = 10) -> HistoricalGame:
    lineup = tuple((1000 + index, "R") for index in range(9))
    return HistoricalGame(
        game_pk=pk,
        game_date=date,
        away_team=away,
        home_team=home,
        park="TEST",
        away_lineup=lineup,
        home_lineup=lineup,
        away_starter=(away_starter, "R"),
        home_starter=(home_starter, "R"),
        actual_away_runs=away_runs,
        actual_home_runs=home_runs,
        score_source="unit_test",
    )


def test_starter_adjustment_is_same_date_safe_and_future_safe(monkeypatch):
    prior_games = [
        _game(1, "2025-09-01", "A", "B", 3, 4),
        _game(2, "2025-09-02", "C", "D", 5, 2),
        _game(3, "2025-09-03", "A", "C", 1, 6),
    ]
    target_games = [
        _game(101, "2026-04-01", "A", "D", 2, 3, home_starter=10),
        _game(102, "2026-04-01", "B", "C", 4, 1, home_starter=10),
        _game(103, "2026-04-02", "A", "B", 5, 2, home_starter=10),
    ]
    lines = pd.DataFrame(
        [
            # State available before the target season.
            {"game_pk": 1, "fielding_team": "B", "starter": 10,
             "date_key": "2025-09-01", "season": 2025, "bf_total": 20.0,
             "runs_while_in_game": 1.0},
            {"game_pk": 2, "fielding_team": "D", "starter": 20,
             "date_key": "2025-09-02", "season": 2025, "bf_total": 20.0,
             "runs_while_in_game": 3.0},
            # Bad target-date performance must not affect another game on the
            # same date, but may affect the following date.
            {"game_pk": 101, "fielding_team": "D", "starter": 10,
             "date_key": "2026-04-01", "season": 2026, "bf_total": 20.0,
             "runs_while_in_game": 12.0},
            # A later excluded/unscored game is still revealed only on its own
            # date; it must never initialize the target-season state.
            {"game_pk": 999, "fielding_team": "D", "starter": 10,
             "date_key": "2026-09-20", "season": 2026, "bf_total": 20.0,
             "runs_while_in_game": 20.0},
        ]
    )
    monkeypatch.setattr(replay_baselines, "starter_run_lines", lambda history: lines)
    nb_fit = fit_nb2_dispersion(prior_games)
    mapping = replay_baselines.build_starter_adjusted_nb_baseline_map(
        target_games,
        prior_games,
        pd.DataFrame({"game_type": []}),
        nb_fit,
    )

    assert mapping[101]["home_starter_run_rate_ratio"] == mapping[102]["home_starter_run_rate_ratio"]
    assert mapping[103]["home_starter_run_rate_ratio"] > mapping[101]["home_starter_run_rate_ratio"]
    assert mapping[101]["home_starter_run_rate_ratio"] < 1.0


def test_league_baseline_reveals_whole_date_together():
    prior_games = [
        _game(1, "2025-09-01", "A", "B", 3, 4),
        _game(2, "2025-09-02", "C", "D", 5, 2),
    ]
    target_games = [
        _game(101, "2026-04-01", "A", "D", 15, 0),
        _game(102, "2026-04-01", "B", "C", 0, 15),
        _game(103, "2026-04-02", "A", "B", 2, 3),
    ]
    mapping = replay_baselines.build_league_nb_baseline_map(
        target_games, prior_games, fit_nb2_dispersion(prior_games)
    )
    assert mapping[101]["away_mean_runs"] == mapping[102]["away_mean_runs"]
    assert mapping[101]["home_mean_runs"] == mapping[102]["home_mean_runs"]
    assert mapping[103]["away_mean_runs"] != mapping[101]["away_mean_runs"]
