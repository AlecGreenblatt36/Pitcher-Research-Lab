from __future__ import annotations

import numpy as np

from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.models import (
    BaseRunner,
    GameMatchup,
    PitcherProfile,
    PlayerProfile,
    SimulationConfig,
    TeamProfile,
)
from research_lab.game_sim.monte_carlo import simulate_many
from research_lab.game_sim.payload import example_payload, matchup_from_payload
from research_lab.game_sim.probability import RatingsProbabilityProvider, TableProbabilityProvider
from research_lab.game_sim.transitions import apply_outcome


def make_team(prefix: str, starter_max: int = 30) -> TeamProfile:
    lineup = tuple(
        PlayerProfile(
            player_id=f"{prefix}-h{index}",
            name=f"{prefix} Hitter {index}",
            bats="L" if index % 3 == 0 else "R",
            contact=(index - 5) * 0.05,
            power=(5 - index) * 0.04,
            discipline=0.0,
            speed=0.35 + (index % 5) * 0.12,
        )
        for index in range(1, 10)
    )
    starter = PitcherProfile(
        player_id=f"{prefix}-sp",
        name=f"{prefix} Starter",
        role="starter",
        stuff=0.2,
        command=0.2,
        contact_management=0.2,
        stamina=0.8,
        expected_batters=max(1, min(22, starter_max)),
        max_batters=starter_max,
    )
    bullpen = tuple(
        PitcherProfile(
            player_id=f"{prefix}-rp{index}",
            name=f"{prefix} Reliever {index}",
            role=("long", "middle", "setup", "closer")[min(index - 1, 3)],
            stuff=0.1 * index,
            command=0.05 * index,
            contact_management=0.05 * index,
            stamina=0.35,
            leverage=0.2 * index,
            expected_batters=4,
            max_batters=7,
        )
        for index in range(1, 5)
    )
    return TeamProfile(
        team_id=prefix,
        name=prefix,
        lineup=lineup,
        starter=starter,
        bullpen=bullpen,
        defense=0.1,
        baserunning=0.1,
    )


def make_matchup(starter_max: int = 30) -> GameMatchup:
    return GameMatchup(
        away=make_team("Away", starter_max=starter_max),
        home=make_team("Home", starter_max=starter_max),
        venue="Test Park",
    )


def test_forced_walk_advances_only_forced_runners() -> None:
    rng = np.random.default_rng(1)
    batter = PlayerProfile("b", "Batter")
    first = BaseRunner("r1", "Runner 1", 0.5, "p")
    third = BaseRunner("r3", "Runner 3", 0.5, "p")
    result = apply_outcome(
        "bb_hbp",
        [first, None, third],
        batter,
        "p",
        0,
        0.0,
        0.0,
        rng,
    )
    assert result.runs_scored == 0
    assert result.bases[0].player_id == "b"
    assert result.bases[1].player_id == "r1"
    assert result.bases[2].player_id == "r3"


def test_home_run_scores_and_clears_bases() -> None:
    rng = np.random.default_rng(2)
    batter = PlayerProfile("b", "Batter")
    bases = [
        BaseRunner("r1", "Runner 1", 0.5, "p"),
        BaseRunner("r2", "Runner 2", 0.5, "p"),
        BaseRunner("r3", "Runner 3", 0.5, "p"),
    ]
    result = apply_outcome(
        "home_run", bases, batter, "p", 1, 0.0, 0.0, rng
    )
    assert result.runs_scored == 4
    assert result.bases == [None, None, None]


def test_transition_kernel_never_duplicates_runners() -> None:
    rng = np.random.default_rng(3)
    batter = PlayerProfile("b", "Batter", speed=0.8)
    outcomes = (
        "bip_out",
        "strikeout",
        "bb_hbp",
        "single",
        "double_triple",
        "home_run",
        "other_reach",
    )
    for _ in range(1000):
        source = [
            BaseRunner(f"r{base}", f"Runner {base}", rng.random(), "p")
            if rng.random() < 0.5
            else None
            for base in range(3)
        ]
        result = apply_outcome(
            str(rng.choice(outcomes)),
            source,
            batter,
            "p",
            int(rng.integers(0, 3)),
            0.2,
            0.1,
            rng,
        )
        occupied = [runner.player_id for runner in result.bases if runner is not None]
        assert len(occupied) == len(set(occupied))
        assert result.outs_added in {0, 1, 2}
        assert result.runs_scored >= 0


def test_simulation_is_seed_deterministic() -> None:
    simulator = GameSimulator(
        RatingsProbabilityProvider(),
        SimulationConfig(record_events=True),
    )
    first = simulator.simulate(make_matchup(), seed=42)
    second = simulator.simulate(make_matchup(), seed=42)
    assert first.to_dict() == second.to_dict()


def test_game_completes_with_legal_winner() -> None:
    simulator = GameSimulator(
        RatingsProbabilityProvider(),
        SimulationConfig(record_events=False),
    )
    for seed in range(30):
        result = simulator.simulate(make_matchup(), seed=seed)
        assert 18 <= result.plate_appearances <= 220
        assert result.away_score >= 0
        assert result.home_score >= 0
        if result.winner == "away":
            assert result.away_score > result.home_score
        elif result.winner == "home":
            assert result.home_score > result.away_score
        else:
            assert result.away_score == result.home_score
        assert not result.ended_by_plate_appearance_cap


def test_three_batter_minimum_holds_for_short_starter() -> None:
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
        SimulationConfig(record_events=False, three_batter_minimum=True),
    )
    matchup = make_matchup(starter_max=1)
    result = simulator.simulate(matchup, seed=7)
    assert result.pitcher_lines[matchup.away.starter.player_id]["batters_faced"] >= 3
    assert result.pitcher_lines[matchup.home.starter.player_id]["batters_faced"] >= 3


def test_monte_carlo_summary_is_complete() -> None:
    simulator = GameSimulator(
        RatingsProbabilityProvider(),
        SimulationConfig(record_events=False),
    )
    summary = simulate_many(simulator, make_matchup(), simulations=100, seed=19)
    win = summary["win_probabilities"]
    assert abs(win["away"] + win["home"] + win["tie"] - 1.0) < 1e-12
    assert summary["simulation_count"] == 100
    assert len(summary["top_scorelines"]) > 0
    assert summary["representative_game"]["events"]
    assert summary["model_status"]["game_engine_validation"] == "not-yet-game-level-validated"


def test_example_payload_parses() -> None:
    payload = example_payload()
    matchup = matchup_from_payload(payload)
    assert matchup.away.name.startswith("San Diego")
    assert matchup.home.name.startswith("Milwaukee")
    assert len(matchup.away.lineup) == 9
    assert len(matchup.home.bullpen) == 6
