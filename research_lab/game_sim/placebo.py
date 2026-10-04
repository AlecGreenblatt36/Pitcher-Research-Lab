"""Leakage diagnostic using reliever orders from a game about one week later.

This is not a forecast and must never be used in production. It is a negative
control for the same-game actual-reliever-order oracle. A future game preserves
some team bullpen identity/role structure while breaking the direct link to the
target game's realized leverage and final score.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .oracle import actual_reliever_orders
from .replay import HistoricalGame


@dataclass(frozen=True)
class PlaceboSource:
    target_game_pk: int
    target_side: str
    team: str
    source_game_pk: int | None
    source_side: str | None
    days_ahead: int | None
    reliever_count: int
    status: str

    def to_dict(self) -> dict:
        return {
            "target_game_pk": self.target_game_pk,
            "target_side": self.target_side,
            "team": self.team,
            "source_game_pk": self.source_game_pk,
            "source_side": self.source_side,
            "days_ahead": self.days_ahead,
            "reliever_count": self.reliever_count,
            "status": self.status,
        }


def week_ahead_placebo_orders(
    history: pd.DataFrame,
    games: Iterable[HistoricalGame],
    *,
    season: int = 2025,
    target_days: int = 7,
    min_days: int = 5,
    max_days: int = 9,
) -> tuple[
    dict[int, dict[str, tuple[tuple[int, str], ...]]],
    dict[int, dict[str, PlaceboSource]],
]:
    """Map each team to a non-doubleheader source game roughly seven days later.

    Selection rules:
    - source date must be strictly later than the target date;
    - calendar gap must be inside ``[min_days, max_days]``;
    - source date must contain exactly one game for that team;
    - source game must have at least one recorded reliever;
    - choose the candidate closest to ``target_days``; ties resolve by date and
      game_pk for deterministic replay;
    - if no source qualifies, return an empty order so the normal pregame
      bullpen ordering is used and mark the side as fallback.

    Future roster changes are intentionally not filtered here. This is a
    postgame negative-control diagnostic, not a forecast-valid input. Their
    frequency must be reported separately before interpreting results.
    """

    if not 0 < min_days <= target_days <= max_days:
        raise ValueError("require 0 < min_days <= target_days <= max_days")

    games = sorted(games, key=lambda g: (g.game_date, g.game_pk))
    actual_orders = actual_reliever_orders(history, season)

    schedule: dict[str, list[tuple[pd.Timestamp, int, str]]] = defaultdict(list)
    games_per_team_date: Counter[tuple[str, str]] = Counter()
    for game in games:
        date = pd.Timestamp(game.game_date)
        for team, side in ((game.away_team, "away"), (game.home_team, "home")):
            schedule[team].append((date, int(game.game_pk), side))
            games_per_team_date[(team, str(game.game_date)[:10])] += 1

    output: dict[int, dict[str, tuple[tuple[int, str], ...]]] = {}
    metadata: dict[int, dict[str, PlaceboSource]] = {}

    for game in games:
        target_date = pd.Timestamp(game.game_date)
        side_orders: dict[str, tuple[tuple[int, str], ...]] = {}
        side_meta: dict[str, PlaceboSource] = {}

        for side, team in (("away", game.away_team), ("home", game.home_team)):
            candidates: list[
                tuple[int, pd.Timestamp, int, str, tuple[tuple[int, str], ...]]
            ] = []
            for source_date, source_game_pk, source_side in schedule.get(team, []):
                days = int((source_date - target_date).days)
                if days < min_days or days > max_days:
                    continue
                date_key = source_date.strftime("%Y-%m-%d")
                if games_per_team_date[(team, date_key)] != 1:
                    continue
                order = tuple(
                    actual_orders.get(source_game_pk, {}).get(source_side, ())
                )
                if not order:
                    continue
                candidates.append(
                    (abs(days - target_days), source_date, source_game_pk, source_side, order)
                )

            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            if candidates:
                _, source_date, source_game_pk, source_side, order = candidates[0]
                days = int((source_date - target_date).days)
                side_orders[side] = order
                side_meta[side] = PlaceboSource(
                    target_game_pk=int(game.game_pk),
                    target_side=side,
                    team=team,
                    source_game_pk=int(source_game_pk),
                    source_side=source_side,
                    days_ahead=days,
                    reliever_count=len(order),
                    status="week_ahead_source",
                )
            else:
                side_orders[side] = ()
                side_meta[side] = PlaceboSource(
                    target_game_pk=int(game.game_pk),
                    target_side=side,
                    team=team,
                    source_game_pk=None,
                    source_side=None,
                    days_ahead=None,
                    reliever_count=0,
                    status="fallback_to_pregame_engine",
                )

        output[int(game.game_pk)] = side_orders
        metadata[int(game.game_pk)] = side_meta

    return output, metadata
