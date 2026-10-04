from __future__ import annotations

import pandas as pd

from research_lab.game_sim.placebo import week_ahead_placebo_orders
from research_lab.game_sim.replay import HistoricalGame


def _game(game_pk: int, date: str, away: str, home: str) -> HistoricalGame:
    lineup = tuple((1000 + i, "R") for i in range(9))
    return HistoricalGame(
        game_pk=game_pk,
        game_date=date,
        away_team=away,
        home_team=home,
        park="PARK",
        away_lineup=lineup,
        home_lineup=lineup,
        away_starter=(1, "R"),
        home_starter=(2, "R"),
        actual_away_runs=3,
        actual_home_runs=4,
        score_source="test",
    )


def _history() -> pd.DataFrame:
    rows = []

    def add_game(game_pk: int, date: str, away: str, home: str, away_pitchers, home_pitchers):
        ab = 1
        for pitcher in home_pitchers:  # top: home fields
            rows.append({
                "season": 2025, "game_type": "R", "game_pk": game_pk,
                "date_key": date, "at_bat_number": ab, "inning_topbot": "Top",
                "pitcher": pitcher, "p_throws": "R",
            })
            ab += 1
        for pitcher in away_pitchers:  # bottom: away fields
            rows.append({
                "season": 2025, "game_type": "R", "game_pk": game_pk,
                "date_key": date, "at_bat_number": ab, "inning_topbot": "Bot",
                "pitcher": pitcher, "p_throws": "L",
            })
            ab += 1

    add_game(1, "2025-04-01", "AAA", "BBB", [10, 11], [20, 21])
    add_game(2, "2025-04-02", "AAA", "CCC", [12, 13], [30, 31])  # next game: too near
    add_game(3, "2025-04-08", "DDD", "AAA", [40, 41], [14, 15, 16])  # +7 days
    add_game(4, "2025-04-08", "EEE", "FFF", [50, 51], [60, 61])
    return pd.DataFrame(rows)


def test_uses_game_about_seven_days_later_not_next_game() -> None:
    games = [
        _game(1, "2025-04-01", "AAA", "BBB"),
        _game(2, "2025-04-02", "AAA", "CCC"),
        _game(3, "2025-04-08", "DDD", "AAA"),
        _game(4, "2025-04-08", "EEE", "FFF"),
    ]
    orders, meta = week_ahead_placebo_orders(_history(), games)

    # AAA is away in target game 1 and home in source game 3.
    # Source starter 14 is removed; actual reliever order is 15, 16.
    assert orders[1]["away"] == ((15, "R"), (16, "R"))
    assert meta[1]["away"].source_game_pk == 3
    assert meta[1]["away"].days_ahead == 7


def test_missing_week_ahead_source_falls_back_to_pregame_engine() -> None:
    games = [_game(1, "2025-04-01", "AAA", "BBB")]
    orders, meta = week_ahead_placebo_orders(_history(), games)

    assert orders[1]["away"] == ()
    assert orders[1]["home"] == ()
    assert meta[1]["away"].status == "fallback_to_pregame_engine"
