from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Provenance:
    source: str
    effective_at_utc: datetime
    retrieved_at_utc: datetime
    source_url: str | None = None
    checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source is required")
        object.__setattr__(self, "effective_at_utc", _utc(self.effective_at_utc, "effective_at_utc"))
        object.__setattr__(self, "retrieved_at_utc", _utc(self.retrieved_at_utc, "retrieved_at_utc"))

    def assert_available_before(self, first_pitch_utc: datetime) -> None:
        cutoff = _utc(first_pitch_utc, "first_pitch_utc")
        if self.effective_at_utc > cutoff:
            raise ValueError(
                f"source {self.source!r} is post-cutoff: {self.effective_at_utc.isoformat()} > {cutoff.isoformat()}"
            )


@dataclass(frozen=True)
class PitchMetricSnapshot:
    player_id: int
    pitch_type: str
    velocity_mean: float
    velocity_sd: float
    ivb_mean: float
    horizontal_break_mean: float
    release_extension_mean: float
    release_height_mean: float
    release_side_mean: float
    spin_rate_mean: float
    zone_rate: float
    whiff_rate: float
    usage_by_count: Mapping[str, float]
    provenance: Provenance


@dataclass(frozen=True)
class HitterMetricSnapshot:
    player_id: int
    batter_side: str
    chase_rate: float
    contact_rate: float
    whiff_rate: float
    hard_hit_rate: float
    barrel_rate: float
    ev_mean: float
    ev_95: float
    launch_angle_mean: float
    pitch_type_whiff: Mapping[str, float]
    zone_contact: Mapping[str, float]
    provenance: Provenance


@dataclass(frozen=True)
class BullpenAvailabilitySnapshot:
    player_id: int
    team_id: int
    active_as_of_first_pitch: bool
    pitches_prev_1d: int
    pitches_prev_2d: int
    pitches_prev_3d: int
    appearances_prev_3d: int
    consecutive_days_used: int
    role: str
    availability_probability: float
    provenance: Provenance


@dataclass(frozen=True)
class FielderMetricSnapshot:
    player_id: int
    position: str
    outs_above_average: float
    fielding_run_value: float
    arm_strength_mph: float | None
    provenance: Provenance


@dataclass(frozen=True)
class CatcherMetricSnapshot:
    player_id: int
    framing_runs_per_1000: float
    blocking_runs_per_1000: float
    caught_stealing_runs_per_1000: float
    provenance: Provenance


@dataclass(frozen=True)
class UmpireZoneSnapshot:
    umpire_id: int
    called_strike_logit_delta: float
    zone_height_delta_inches: float
    zone_width_delta_inches: float
    sample_called_pitches: int
    provenance: Provenance


@dataclass(frozen=True)
class EnvironmentSnapshot:
    venue_id: int
    first_pitch_utc: datetime
    temperature_f: float
    relative_humidity: float
    pressure_hpa: float
    wind_speed_mph: float
    wind_direction_degrees: float
    altitude_ft: float
    roof_status: str
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_pitch_utc", _utc(self.first_pitch_utc, "first_pitch_utc"))
        if not 0 <= self.relative_humidity <= 100:
            raise ValueError("relative_humidity must be between 0 and 100")
        if self.wind_speed_mph < 0:
            raise ValueError("wind_speed_mph cannot be negative")


@dataclass(frozen=True)
class ParkVectorSnapshot:
    venue_id: int
    single_factor: float
    double_factor: float
    triple_factor: float
    home_run_factor: float
    directional_home_run_factors: Mapping[str, float] = field(default_factory=dict)
    provenance: Provenance | None = None
