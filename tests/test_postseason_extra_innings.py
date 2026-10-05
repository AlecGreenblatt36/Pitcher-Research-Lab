from __future__ import annotations

from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.models import (
    GameMatchup,
    PitcherProfile,
    PlayerProfile,
    SimulationConfig,
    TeamProfile,
)
from research_lab.game_sim.probability import TableProbabilityProvider


def _team(prefix: str) -> TeamProfile:
    lineup = tuple(
        PlayerProfile(str(index), f"{prefix} Hitter {index}")
        for index in range(1, 10)
    )
    starter = PitcherProfile(
        f"{prefix}-sp",
        f"{prefix} Starter",
        role="starter",
        expected_batters=40,
        max_batters=60,
    )
    return TeamProfile(prefix, prefix, lineup, starter)


def _simulate(game_type: str):
    provider = TableProbabilityProvider(
        default={
            "bip_out": 0.0,
            "strikeout": 1.0,
            "bb_hbp": 0.0,
            "single": 0.0,
            "double_triple": 0.0,
            "home_run": 0.0,
            "other_reach": 0.0,
        }
    )
    simulator = GameSimulator(
        provider,
        SimulationConfig(
            regulation_innings=1,
            max_innings=2,
            automatic_runner_in_extras=True,
            record_events=True,
        ),
    )
    matchup = GameMatchup(
        away=_team("Away"),
        home=_team("Home"),
        game_type=game_type,
    )
    return simulator.simulate(matchup, seed=11)


def _first_top_second_event(result):
    return next(
        event
        for event in result.events
        if event["inning"] == 2 and event["half"] == "top"
    )


def test_regular_season_places_automatic_runner() -> None:
    event = _first_top_second_event(_simulate("R"))
    assert event["bases_before"][1] is not None


def test_postseason_keeps_bases_empty() -> None:
    event = _first_top_second_event(_simulate("P"))
    assert event["bases_before"] == (None, None, None)
