from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class WorkloadHistory:
    pitches_prev_1d: int = 0
    pitches_prev_2d: int = 0
    pitches_prev_3d: int = 0
    appearances_prev_3d: int = 0
    consecutive_days_used: int = 0
    days_rest: int = 4

    def __post_init__(self) -> None:
        for field_name in (
            "pitches_prev_1d",
            "pitches_prev_2d",
            "pitches_prev_3d",
            "appearances_prev_3d",
            "consecutive_days_used",
            "days_rest",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True)
class PitcherDynamicState:
    pitch_count: int
    pitches_this_inning: int
    batters_faced: int
    inning: int
    times_through_order: int
    runs_allowed: int
    runners_on: int
    score_diff_fielding: int
    baseline_fastball_velocity: float | None = None
    current_fastball_velocity: float | None = None
    baseline_release_height: float | None = None
    current_release_height: float | None = None
    baseline_zone_rate: float | None = None
    recent_zone_rate: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "pitch_count",
            "pitches_this_inning",
            "batters_faced",
            "runs_allowed",
            "runners_on",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.inning < 1:
            raise ValueError("inning must be positive")
        if self.times_through_order < 1:
            raise ValueError("times_through_order must be positive")
        if self.runners_on > 3:
            raise ValueError("runners_on cannot exceed 3")


@dataclass(frozen=True)
class FatigueAdjustment:
    velocity_delta_mph: float = 0.0
    zone_logit_delta: float = 0.0
    whiff_logit_delta: float = 0.0
    hard_hit_logit_delta: float = 0.0
    command_sigma_delta: float = 0.0


class FatigueModel(Protocol):
    def predict(self, features: Mapping[str, float]) -> FatigueAdjustment:
        ...


class IdentityFatigueModel:
    """Neutral model used only when an experiment explicitly disables fatigue."""

    def predict(self, features: Mapping[str, float]) -> FatigueAdjustment:
        return FatigueAdjustment()


def _difference(current: float | None, baseline: float | None) -> float:
    if current is None or baseline is None or not np.isfinite(current) or not np.isfinite(baseline):
        return float("nan")
    return float(current - baseline)


def fatigue_features(
    state: PitcherDynamicState,
    history: WorkloadHistory,
    *,
    role: str,
    expected_pitch_count: float | None = None,
) -> dict[str, float]:
    """Build raw, audit-friendly fatigue features without hand-tuned degradation.

    The returned values are inputs to a fitted model. This function deliberately
    does not translate workload into outcome adjustments by itself.
    """

    expected = float(expected_pitch_count) if expected_pitch_count else float("nan")
    return {
        "pitch_count": float(state.pitch_count),
        "pitch_count_over_expected": (
            float(state.pitch_count - expected) if np.isfinite(expected) else float("nan")
        ),
        "pitches_this_inning": float(state.pitches_this_inning),
        "batters_faced": float(state.batters_faced),
        "inning": float(state.inning),
        "times_through_order": float(state.times_through_order),
        "runs_allowed": float(state.runs_allowed),
        "runners_on": float(state.runners_on),
        "score_diff_fielding": float(state.score_diff_fielding),
        "pitches_prev_1d": float(history.pitches_prev_1d),
        "pitches_prev_2d": float(history.pitches_prev_2d),
        "pitches_prev_3d": float(history.pitches_prev_3d),
        "appearances_prev_3d": float(history.appearances_prev_3d),
        "consecutive_days_used": float(history.consecutive_days_used),
        "days_rest": float(history.days_rest),
        "velocity_delta_observed": _difference(
            state.current_fastball_velocity,
            state.baseline_fastball_velocity,
        ),
        "release_height_delta_observed": _difference(
            state.current_release_height,
            state.baseline_release_height,
        ),
        "zone_rate_delta_observed": _difference(
            state.recent_zone_rate,
            state.baseline_zone_rate,
        ),
        "is_starter": float(role.lower() == "starter"),
        "is_reliever": float(role.lower() != "starter"),
    }


def adjustment_to_dict(adjustment: FatigueAdjustment) -> dict[str, float]:
    return {field.name: float(getattr(adjustment, field.name)) for field in fields(adjustment)}
