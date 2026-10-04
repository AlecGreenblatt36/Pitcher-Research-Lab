from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import BaseRunner, PlayerProfile


@dataclass
class TransitionResult:
    bases: list[BaseRunner | None]
    outs_added: int
    scored_runners: list[BaseRunner]
    retired_runner_ids: list[str]
    description: str

    @property
    def runs_scored(self) -> int:
        return len(self.scored_runners)


def _bounded_probability(value: float) -> float:
    return float(min(0.995, max(0.005, value)))


def _batter_runner(
    batter: PlayerProfile,
    responsible_pitcher_id: str | None,
) -> BaseRunner:
    return BaseRunner.from_batter(batter, responsible_pitcher_id)


def apply_outcome(
    outcome: str,
    bases: list[BaseRunner | None],
    batter: PlayerProfile,
    responsible_pitcher_id: str | None,
    outs_before: int,
    batting_team_baserunning: float,
    fielding_team_defense: float,
    rng: np.random.Generator,
) -> TransitionResult:
    """Resolve one seven-class PA outcome into a legal base/out transition.

    The transition kernel is deliberately isolated from PA probabilities. It can
    be replaced by an empirical kernel trained on historical play-by-play while
    preserving the game engine and evaluation interface.
    """

    if len(bases) != 3:
        raise ValueError("bases must contain first, second, and third")
    first, second, third = list(bases)
    scored: list[BaseRunner] = []
    retired: list[str] = []
    batter_runner = _batter_runner(batter, responsible_pitcher_id)
    baserun = float(min(2.5, max(-2.5, batting_team_baserunning)))
    defense = float(min(2.5, max(-2.5, fielding_team_defense)))

    if outcome == "strikeout":
        return TransitionResult(
            [first, second, third],
            1,
            scored,
            retired,
            f"{batter.name} struck out.",
        )

    if outcome == "bb_hbp":
        new_first, new_second, new_third = first, second, third
        if first is not None:
            if second is not None:
                if third is not None:
                    scored.append(third)
                new_third = second
            new_second = first
        new_first = batter_runner
        return TransitionResult(
            [new_first, new_second, new_third],
            0,
            scored,
            retired,
            f"{batter.name} reached on a walk or hit by pitch.",
        )

    if outcome == "home_run":
        scored.extend(runner for runner in (third, second, first) if runner is not None)
        scored.append(batter_runner)
        return TransitionResult(
            [None, None, None],
            0,
            scored,
            retired,
            f"{batter.name} homered.",
        )

    if outcome == "single":
        new_bases: list[BaseRunner | None] = [None, None, None]
        if third is not None:
            score_probability = _bounded_probability(
                0.965 + 0.025 * third.speed + 0.010 * (outs_before == 2)
            )
            if rng.random() < score_probability:
                scored.append(third)
            else:
                new_bases[2] = third

        if second is not None:
            score_probability = _bounded_probability(
                0.54
                + 0.28 * second.speed
                + 0.10 * (outs_before == 2)
                + 0.035 * baserun
                - 0.035 * defense
            )
            if rng.random() < score_probability:
                scored.append(second)
            elif new_bases[2] is None:
                new_bases[2] = second
            else:
                new_bases[1] = second

        if first is not None:
            take_third_probability = _bounded_probability(
                0.20
                + 0.26 * first.speed
                + 0.09 * (outs_before == 2)
                + 0.030 * baserun
                - 0.035 * defense
            )
            if new_bases[2] is None and rng.random() < take_third_probability:
                new_bases[2] = first
            elif new_bases[1] is None:
                new_bases[1] = first
            else:
                scored.append(new_bases[2])
                new_bases[2] = new_bases[1]
                new_bases[1] = first

        new_bases[0] = batter_runner
        return TransitionResult(
            new_bases,
            0,
            scored,
            retired,
            f"{batter.name} singled.",
        )

    if outcome == "double_triple":
        triple_probability = _bounded_probability(
            0.035 + 0.085 * batter.speed + 0.010 * baserun - 0.010 * defense
        )
        if rng.random() < triple_probability:
            scored.extend(runner for runner in (third, second, first) if runner is not None)
            return TransitionResult(
                [None, None, batter_runner],
                0,
                scored,
                retired,
                f"{batter.name} tripled.",
            )

        if third is not None:
            scored.append(third)
        if second is not None:
            scored.append(second)
        new_third: BaseRunner | None = None
        if first is not None:
            score_probability = _bounded_probability(
                0.42
                + 0.31 * first.speed
                + 0.08 * (outs_before == 2)
                + 0.035 * baserun
                - 0.040 * defense
            )
            if rng.random() < score_probability:
                scored.append(first)
            else:
                new_third = first
        return TransitionResult(
            [None, batter_runner, new_third],
            0,
            scored,
            retired,
            f"{batter.name} doubled.",
        )

    if outcome == "other_reach":
        fielder_choice_probability = 0.23 if first is not None and outs_before < 2 else 0.0
        if rng.random() < fielder_choice_probability:
            assert first is not None
            retired.append(first.player_id)
            return TransitionResult(
                [batter_runner, second, third],
                1,
                scored,
                retired,
                f"{batter.name} reached on a fielder's choice.",
            )
        if third is not None:
            scored.append(third)
        return TransitionResult(
            [batter_runner, first, second],
            0,
            scored,
            retired,
            f"{batter.name} reached on an error or other play.",
        )

    if outcome == "bip_out":
        double_play_probability = 0.0
        if first is not None and outs_before < 2:
            double_play_probability = _bounded_probability(
                0.145
                + 0.045 * defense
                - 0.055 * first.speed
                - 0.035 * batter.speed
            )
        if double_play_probability and rng.random() < double_play_probability:
            retired.extend([first.player_id, batter.player_id])
            return TransitionResult(
                [None, second, third],
                2,
                scored,
                retired,
                f"{batter.name} grounded into a double play.",
            )

        new_bases = [first, second, third]
        if third is not None and outs_before < 2:
            sacrifice_probability = _bounded_probability(
                0.135 + 0.070 * batter.power - 0.025 * defense
            )
            if rng.random() < sacrifice_probability:
                scored.append(third)
                new_bases[2] = None
                return TransitionResult(
                    new_bases,
                    1,
                    scored,
                    retired,
                    f"{batter.name} drove in a run on a sacrifice fly.",
                )

        if second is not None and new_bases[2] is None:
            advance_probability = _bounded_probability(
                0.13 + 0.12 * second.speed + 0.020 * baserun - 0.025 * defense
            )
            if rng.random() < advance_probability:
                new_bases[2] = second
                new_bases[1] = None
        if first is not None and new_bases[1] is None:
            advance_probability = _bounded_probability(
                0.05 + 0.07 * first.speed + 0.015 * baserun - 0.020 * defense
            )
            if rng.random() < advance_probability:
                new_bases[1] = first
                new_bases[0] = None

        return TransitionResult(
            new_bases,
            1,
            scored,
            retired,
            f"{batter.name} put the ball in play for an out.",
        )

    raise ValueError(f"unsupported outcome: {outcome}")
