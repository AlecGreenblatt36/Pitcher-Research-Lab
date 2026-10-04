"""Postgame-information bullpen oracle used only as a diagnostic.

This is never a forecast. It forces the simulator to call the relievers each
team actually used, in the order they actually appeared, while leaving the
existing starter/reliever exit rules unchanged. It estimates the maximum value
available from improving reliever identity/order under the current downstream
engine. The actual order comes from the completed game and is therefore
strictly forbidden in pregame use.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import pandas as pd

from .models import GameMatchup, PitcherProfile, TeamProfile
from .starter_hazard import FittedStarterPolicy


def actual_reliever_orders(history: pd.DataFrame, season: int) -> dict[int, dict[str, tuple[tuple[int, str], ...]]]:
    frame = history[(history["season"] == season) & (history["game_type"] == "R")].copy()
    frame = frame.sort_values(["game_pk", "at_bat_number"], kind="mergesort")
    top = frame["inning_topbot"].astype(str).str.lower().str.startswith("top")
    frame["fielding_side"] = top.map({True: "home", False: "away"})
    output: dict[int, dict[str, tuple[tuple[int, str], ...]]] = {}
    for game_pk, game in frame.groupby("game_pk", sort=False):
        sides: dict[str, tuple[tuple[int, str], ...]] = {}
        for side in ("away", "home"):
            rows = game[game["fielding_side"] == side]
            seen: set[int] = set()
            order: list[tuple[int, str]] = []
            for row in rows.itertuples(index=False):
                pitcher = int(row.pitcher)
                if pitcher in seen:
                    continue
                seen.add(pitcher)
                order.append((pitcher, str(row.p_throws)))
            # First pitcher is the actual starter; remaining pitchers are the
            # postgame-information relief order.
            sides[side] = tuple(order[1:])
        output[int(game_pk)] = sides
    return output


def _ordered_bullpen(
    existing: tuple[PitcherProfile, ...],
    actual_order: tuple[tuple[int, str], ...],
    names: Mapping[int, str] | None = None,
) -> tuple[PitcherProfile, ...]:
    names = names or {}
    by_id = {int(p.player_id): p for p in existing}
    ordered: list[PitcherProfile] = []
    used: set[int] = set()
    for pitcher_id, throws in actual_order:
        pitcher_id = int(pitcher_id)
        if pitcher_id in used:
            continue
        used.add(pitcher_id)
        profile = by_id.get(pitcher_id)
        if profile is None:
            profile = PitcherProfile(
                player_id=str(pitcher_id),
                name=names.get(pitcher_id, f"MLBAM {pitcher_id}"),
                throws=str(throws),
                role="reliever",
                stamina=0.40,
                leverage=0.50,
                rest=1.0,
                expected_batters=4,
                max_batters=8,
                available=True,
            )
        ordered.append(profile)
    # If a simulation needs more arms than the real game did, fall back only
    # after the complete actual sequence has been exhausted.
    ordered.extend(p for p in existing if int(p.player_id) not in used)
    return tuple(ordered)


def apply_actual_reliever_order(
    matchup: GameMatchup,
    orders: Mapping[str, tuple[tuple[int, str], ...]],
    names: Mapping[int, str] | None = None,
) -> GameMatchup:
    away = replace(
        matchup.away,
        bullpen=_ordered_bullpen(matchup.away.bullpen, tuple(orders.get("away", ())), names),
    )
    home = replace(
        matchup.home,
        bullpen=_ordered_bullpen(matchup.home.bullpen, tuple(orders.get("home", ())), names),
    )
    return replace(matchup, away=away, home=home)


class OracleRelieverOrderPolicy(FittedStarterPolicy):
    """Fitted starter hazard + dev reliever exit + actual reliever order."""

    name = "oracle-actual-reliever-order-diagnostic"

    def select_reliever(self, bullpen, used_pitcher_ids, state, fielding_side, next_batter, rng):
        for pitcher in bullpen:
            if (
                pitcher.available
                and pitcher.player_id not in used_pitcher_ids
                and pitcher.rest > 0.05
            ):
                return pitcher
        return None
