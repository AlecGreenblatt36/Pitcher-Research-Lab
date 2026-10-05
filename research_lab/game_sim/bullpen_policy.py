from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RelieverCandidate:
    player_id: int
    throws: str
    role: str
    rest_days: int
    pitches_prev_1d: int
    pitches_prev_2d: int
    pitches_prev_3d: int
    consecutive_days_used: int
    leverage_skill: float
    quality: float
    available_probability: float


@dataclass(frozen=True)
class BullpenDecisionContext:
    inning: int
    outs: int
    runners_on: int
    fielding_lead: int
    leverage_index: float
    next_batter_side: str


@dataclass(frozen=True)
class ConditionalLogitArtifact:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.coefficients):
            raise ValueError("feature_names and coefficients must have equal length")


class FittedBullpenSelectionPolicy:
    """Candidate-level softmax policy loaded from a fitted artifact.

    This class supplies no hand-tuned defaults. The caller must provide the
    artifact and a pregame-valid candidate list.
    """

    def __init__(self, artifact: ConditionalLogitArtifact) -> None:
        self.artifact = artifact

    @staticmethod
    def feature_map(candidate: RelieverCandidate, context: BullpenDecisionContext) -> dict[str, float]:
        same_side = float(
            context.next_batter_side.upper() != "S"
            and context.next_batter_side.upper() == candidate.throws.upper()
        )
        return {
            "inning": float(context.inning),
            "outs": float(context.outs),
            "runners_on": float(context.runners_on),
            "fielding_lead": float(context.fielding_lead),
            "leverage_index": float(context.leverage_index),
            "same_side": same_side,
            "rest_days": float(candidate.rest_days),
            "pitches_prev_1d": float(candidate.pitches_prev_1d),
            "pitches_prev_2d": float(candidate.pitches_prev_2d),
            "pitches_prev_3d": float(candidate.pitches_prev_3d),
            "consecutive_days_used": float(candidate.consecutive_days_used),
            "leverage_skill": float(candidate.leverage_skill),
            "quality": float(candidate.quality),
            "available_probability": float(candidate.available_probability),
            "is_closer": float(candidate.role.lower() == "closer"),
            "is_setup": float(candidate.role.lower() == "setup"),
            "is_long": float(candidate.role.lower() in {"long", "long_relief"}),
        }

    def probabilities(
        self,
        candidates: Sequence[RelieverCandidate],
        context: BullpenDecisionContext,
    ) -> dict[int, float]:
        if not candidates:
            raise ValueError("candidate list cannot be empty")
        utilities = []
        for candidate in candidates:
            features = self.feature_map(candidate, context)
            missing = [name for name in self.artifact.feature_names if name not in features]
            if missing:
                raise ValueError(f"selection artifact requires unavailable features: {missing}")
            utility = self.artifact.intercept + sum(
                coefficient * features[name]
                for name, coefficient in zip(
                    self.artifact.feature_names,
                    self.artifact.coefficients,
                )
            )
            utilities.append(float(utility))
        values = np.asarray(utilities, dtype=float)
        values -= values.max()
        weights = np.exp(values)
        weights /= weights.sum()
        return {
            int(candidate.player_id): float(probability)
            for candidate, probability in zip(candidates, weights)
        }
