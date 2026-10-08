from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np

from .models import GameState, PitcherLine, PitcherProfile, PlayerProfile, TeamSide


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def game_leverage(state: GameState, fielding_side: TeamSide) -> float:
    """Small, transparent leverage approximation for bullpen decisions."""

    score_diff = state.score_for(fielding_side) - state.opponent_score_for(fielding_side)
    inning_weight = min(1.0, max(0.05, (state.inning - 3) / 6.0))
    closeness = max(0.0, 1.0 - min(abs(score_diff), 6) / 6.0)
    runners = sum(runner is not None for runner in state.bases)
    traffic = runners / 3.0
    out_pressure = (2 - min(state.outs, 2)) / 2.0
    return float(min(1.0, 0.15 + 0.48 * inning_weight * closeness + 0.22 * traffic + 0.15 * out_pressure))


@dataclass
class ManagerPolicy:
    three_batter_minimum: bool = True
    # Optional fitted reliever choice (reliever_choice.RelieverChoice, BULLPEN-01). None keeps the hand-set scoring
    # below and draws exactly the same random numbers as before.
    reliever_choice: object = None

    def fatigue(
        self,
        pitcher: PitcherProfile,
        line: PitcherLine,
    ) -> float:
        workload = line.batters_faced / max(1.0, float(pitcher.max_batters))
        stamina_relief = 0.40 * pitcher.stamina
        rest_penalty = 0.30 * (1.0 - pitcher.rest)
        return float(min(1.5, max(0.0, workload - stamina_relief + rest_penalty)))

    def removal_is_legal(
        self,
        line: PitcherLine,
        inning_ended: bool,
    ) -> bool:
        if not self.three_batter_minimum:
            return True
        return inning_ended or line.batters_since_entry >= 3

    def should_remove(
        self,
        pitcher: PitcherProfile,
        line: PitcherLine,
        state: GameState,
        fielding_side: TeamSide,
        inning_ended: bool,
        rng: np.random.Generator,
    ) -> bool:
        if not self.removal_is_legal(line, inning_ended):
            return False
        if line.batters_faced >= pitcher.max_batters:
            return True

        fatigue = self.fatigue(pitcher, line)
        leverage = game_leverage(state, fielding_side)
        runs = line.runs_allowed
        expected = max(1.0, float(pitcher.expected_batters))
        workload_delta = (line.batters_faced - expected) / max(2.5, expected * 0.22)

        if line.is_starter:
            if state.inning <= 3 and line.batters_faced < expected * 0.72 and runs < 5:
                return False
            pressure = (
                -1.35
                + 1.05 * workload_delta
                + 0.34 * runs
                + 0.72 * fatigue
                + 0.22 * max(0, state.inning - 5)
                + 0.25 * leverage
            )
            if inning_ended:
                pressure += 0.42
        else:
            if line.batters_faced < min(3, pitcher.expected_batters) and not inning_ended:
                return False
            pressure = (
                -0.45
                + 1.10 * workload_delta
                + 0.42 * runs
                + 0.92 * fatigue
                + 0.38 * leverage
            )
            if inning_ended:
                pressure += 0.75
            if pitcher.role in {"closer", "setup"} and state.inning < 7:
                pressure += 0.25

        return bool(rng.random() < _sigmoid(pressure))

    def select_reliever(
        self,
        bullpen: Iterable[PitcherProfile],
        used_pitcher_ids: set[str],
        state: GameState,
        fielding_side: TeamSide,
        next_batter: PlayerProfile,
        rng: np.random.Generator,
        upcoming=None,
    ) -> PitcherProfile | None:
        candidates = [
            pitcher
            for pitcher in bullpen
            if pitcher.available
            and pitcher.player_id not in used_pitcher_ids
            and pitcher.rest > 0.05
        ]
        if not candidates:
            return None
        if self.reliever_choice is not None:
            return self.reliever_choice.select(candidates, state, fielding_side, next_batter, rng, upcoming)

        leverage = game_leverage(state, fielding_side)
        fielding_lead = state.score_for(fielding_side) - state.opponent_score_for(fielding_side)
        save_window = state.inning >= 9 and 1 <= fielding_lead <= 3
        late_tie = state.inning >= 9 and fielding_lead == 0

        scored: list[tuple[float, PitcherProfile]] = []
        for pitcher in candidates:
            score = 0.70 * pitcher.rest + 0.95 * pitcher.leverage * leverage
            role = pitcher.role
            if role == "closer":
                score += 1.60 if save_window else 1.05 if late_tie else -0.45
                if state.inning < 8:
                    score -= 1.20
            elif role == "setup":
                score += 0.95 if 7 <= state.inning <= 9 and leverage >= 0.45 else 0.10
            elif role in {"long", "long_relief"}:
                score += 1.00 if state.inning <= 5 else -0.25
            elif role in {"middle", "reliever"}:
                score += 0.40 if 5 <= state.inning <= 8 else 0.05

            if next_batter.bats != "S":
                same_side = next_batter.bats == pitcher.throws
                score += 0.18 if same_side else -0.05
            score += 0.12 * pitcher.stuff + 0.08 * pitcher.command
            score += float(rng.normal(0.0, 0.08))
            scored.append((score, pitcher))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
