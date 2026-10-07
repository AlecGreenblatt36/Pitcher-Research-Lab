from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .models import OUTCOME_LABELS, PAContext


class PAProbabilityProvider(Protocol):
    name: str
    validation_status: str

    def probabilities(self, context: PAContext) -> Mapping[str, float]:
        ...


def normalize_probabilities(values: Mapping[str, float] | Sequence[float] | np.ndarray) -> dict[str, float]:
    if isinstance(values, Mapping):
        vector = np.asarray([float(values.get(label, 0.0)) for label in OUTCOME_LABELS])
    else:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (len(OUTCOME_LABELS),):
            raise ValueError(f"probability vector must have {len(OUTCOME_LABELS)} values")
    if not np.isfinite(vector).all():
        raise ValueError("probabilities must be finite")
    if (vector < 0).any():
        raise ValueError("probabilities cannot be negative")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    vector = np.clip(vector / total, 1e-9, 1.0)
    vector /= vector.sum()
    return {label: float(vector[index]) for index, label in enumerate(OUTCOME_LABELS)}


@dataclass
class RatingsProbabilityProvider:
    """Transparent fallback PA model for engine development.

    This provider is intentionally not described as the validated PA model. It
    starts from observed league outcome rates and applies modest log-space
    adjustments from player ratings and pre-PA context. The simulation engine
    accepts any provider implementing ``probabilities``; the canonical fitted
    PA model can therefore replace this provider without changing game logic.
    """

    name: str = "ratings-fallback-v1"
    validation_status: str = "engine-development-only"

    def __post_init__(self) -> None:
        self._league = normalize_probabilities(
            {
                "bip_out": 0.45316,
                "strikeout": 0.22138,
                "bb_hbp": 0.10038,
                "single": 0.14202,
                "double_triple": 0.04441,
                "home_run": 0.03032,
                "other_reach": 0.00833,
            }
        )

    def probabilities(self, context: PAContext) -> Mapping[str, float]:
        labels = list(OUTCOME_LABELS)
        base = np.asarray([self._league[label] for label in labels], dtype=float)
        score = np.log(base)
        index = {label: i for i, label in enumerate(labels)}

        batter = context.batter
        pitcher = context.pitcher
        fatigue = float(min(1.5, max(0.0, context.pitcher_fatigue)))
        tto_penalty = max(0, context.times_through_order - 1)
        platoon = 1.0 if context.platoon_advantage else 0.0

        score[index["strikeout"]] += 0.55 * pitcher.stuff - 0.42 * batter.contact + 0.14 * fatigue - 0.08 * pitcher.command
        score[index["bb_hbp"]] += 0.38 * batter.discipline - 0.46 * pitcher.command + 0.18 * fatigue
        score[index["single"]] += 0.31 * batter.contact - 0.22 * pitcher.contact_management + 0.05 * platoon
        score[index["double_triple"]] += 0.22 * batter.contact + 0.34 * batter.power - 0.28 * pitcher.contact_management + 0.07 * platoon
        score[index["home_run"]] += 0.58 * batter.power - 0.43 * pitcher.contact_management + 0.08 * platoon
        score[index["bip_out"]] += 0.20 * pitcher.contact_management - 0.19 * batter.contact - 0.08 * batter.power
        score[index["other_reach"]] += 0.08 * batter.speed - 0.06 * context.fielding_team_defense

        offense_lift = 0.045 * tto_penalty + 0.11 * fatigue
        for label in ("single", "double_triple", "home_run", "other_reach"):
            score[index[label]] += offense_lift
        score[index["bip_out"]] -= 0.055 * tto_penalty + 0.08 * fatigue

        run_environment = max(0.55, min(1.65, context.park_factor * context.weather_run_factor))
        log_environment = float(np.log(run_environment))
        score[index["single"]] += 0.20 * log_environment
        score[index["double_triple"]] += 0.55 * log_environment
        score[index["home_run"]] += 0.90 * log_environment

        if context.outs == 2:
            score[index["single"]] += 0.015
            score[index["bip_out"]] -= 0.010
        if context.score_diff <= -3 and context.inning >= 7:
            score[index["bb_hbp"]] += 0.015

        score -= score.max()
        probabilities = np.exp(score)
        probabilities /= probabilities.sum()
        return normalize_probabilities(probabilities)


@dataclass
class TableProbabilityProvider:
    default: Mapping[str, float]
    by_matchup: Mapping[tuple[str, str], Mapping[str, float]] | None = None
    name: str = "precomputed-table"
    validation_status: str = "depends-on-supplied-table"

    def probabilities(self, context: PAContext) -> Mapping[str, float]:
        table = self.by_matchup or {}
        values = table.get((context.batter.player_id, context.pitcher.player_id), self.default)
        return normalize_probabilities(values)


@dataclass
class CallableProbabilityProvider:
    predictor: Callable[[PAContext], Mapping[str, float] | Sequence[float] | np.ndarray]
    name: str = "callable-pa-provider"
    validation_status: str = "provider-supplied"

    def probabilities(self, context: PAContext) -> Mapping[str, float]:
        return normalize_probabilities(self.predictor(context))


@dataclass
class BlendedProbabilityProvider:
    primary: PAProbabilityProvider
    fallback: PAProbabilityProvider
    primary_weight: float = 0.9
    name: str = "blended-pa-provider"

    @property
    def validation_status(self) -> str:
        return f"primary={getattr(self.primary, 'validation_status', 'unknown')};fallback={getattr(self.fallback, 'validation_status', 'unknown')}"

    def probabilities(self, context: PAContext) -> Mapping[str, float]:
        weight = float(min(1.0, max(0.0, self.primary_weight)))
        first = normalize_probabilities(self.primary.probabilities(context))
        second = normalize_probabilities(self.fallback.probabilities(context))
        return normalize_probabilities({label: weight * first[label] + (1.0 - weight) * second[label] for label in OUTCOME_LABELS})
