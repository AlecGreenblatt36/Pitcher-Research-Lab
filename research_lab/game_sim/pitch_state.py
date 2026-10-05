from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Protocol

import numpy as np

from .random_streams import EventRandomStreams


class PitchEvent(str, Enum):
    BALL = "ball"
    CALLED_STRIKE = "called_strike"
    SWINGING_STRIKE = "swinging_strike"
    FOUL = "foul"
    IN_PLAY = "in_play"
    HIT_BY_PITCH = "hit_by_pitch"


PITCH_EVENTS: tuple[PitchEvent, ...] = tuple(PitchEvent)


class PlateAppearanceTerminal(str, Enum):
    WALK = "walk"
    STRIKEOUT = "strikeout"
    HIT_BY_PITCH = "hit_by_pitch"
    BALL_IN_PLAY = "ball_in_play"


@dataclass(frozen=True, order=True)
class CountState:
    balls: int = 0
    strikes: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.balls <= 3:
            raise ValueError("balls must be between 0 and 3 for a live count")
        if not 0 <= self.strikes <= 2:
            raise ValueError("strikes must be between 0 and 2 for a live count")

    @property
    def label(self) -> str:
        return f"{self.balls}-{self.strikes}"


@dataclass(frozen=True)
class PitchContext:
    batter_id: int
    pitcher_id: int
    count: CountState
    pitch_number: int
    pitcher_pitch_count: int
    inning: int
    outs: int
    runners: tuple[int, int, int]
    score_diff: int
    times_through_order: int
    batter_side: str
    pitcher_hand: str
    available_pitch_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.pitch_number < 1:
            raise ValueError("pitch_number must be positive")
        if self.pitcher_pitch_count < 0:
            raise ValueError("pitcher_pitch_count cannot be negative")
        if self.inning < 1:
            raise ValueError("inning must be positive")
        if self.outs not in (0, 1, 2):
            raise ValueError("outs must be 0, 1, or 2")
        if any(value not in (0, 1) for value in self.runners):
            raise ValueError("runners must be three 0/1 occupancy flags")
        if not self.available_pitch_types:
            raise ValueError("at least one pitch type is required")

    def after_pitch(self, next_count: CountState) -> "PitchContext":
        return replace(
            self,
            count=next_count,
            pitch_number=self.pitch_number + 1,
            pitcher_pitch_count=self.pitcher_pitch_count + 1,
        )


@dataclass(frozen=True)
class PitchTransition:
    event: PitchEvent
    next_count: CountState | None
    terminal: PlateAppearanceTerminal | None


@dataclass(frozen=True)
class PitchRecord:
    pitch_number: int
    count_before: CountState
    pitch_type: str
    zone_region: str
    event: PitchEvent
    count_after: CountState | None


@dataclass(frozen=True)
class PitchSequenceResult:
    terminal: PlateAppearanceTerminal
    pitch_count: int
    records: tuple[PitchRecord, ...]


class PitchPlanProvider(Protocol):
    def probabilities(self, context: PitchContext) -> Mapping[str, float]:
        """Return probabilities for tokens formatted as ``PITCH_TYPE|ZONE``."""


class PitchEventProvider(Protocol):
    def probabilities(
        self,
        context: PitchContext,
        pitch_type: str,
        zone_region: str,
    ) -> Mapping[PitchEvent | str, float]:
        """Return a complete probability vector across ``PITCH_EVENTS``."""


def normalize_probabilities(
    probabilities: Mapping[PitchEvent | str, float],
) -> np.ndarray:
    by_name = {
        (key.value if isinstance(key, PitchEvent) else str(key)): float(value)
        for key, value in probabilities.items()
    }
    missing = [event.value for event in PITCH_EVENTS if event.value not in by_name]
    if missing:
        raise ValueError(f"pitch event probabilities missing: {missing}")
    values = np.asarray([by_name[event.value] for event in PITCH_EVENTS], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("pitch event probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("pitch event probabilities must have positive mass")
    return values / total


def normalize_token_probabilities(probabilities: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    if not probabilities:
        raise ValueError("pitch plan probabilities cannot be empty")
    tokens = list(probabilities)
    if any("|" not in token for token in tokens):
        raise ValueError("pitch plan tokens must be formatted as PITCH_TYPE|ZONE")
    values = np.asarray([float(probabilities[token]) for token in tokens], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("pitch plan probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("pitch plan probabilities must have positive mass")
    return tokens, values / total


def advance_count(count: CountState, event: PitchEvent) -> PitchTransition:
    if event is PitchEvent.BALL:
        if count.balls == 3:
            return PitchTransition(event, None, PlateAppearanceTerminal.WALK)
        return PitchTransition(event, CountState(count.balls + 1, count.strikes), None)
    if event in (PitchEvent.CALLED_STRIKE, PitchEvent.SWINGING_STRIKE):
        if count.strikes == 2:
            return PitchTransition(event, None, PlateAppearanceTerminal.STRIKEOUT)
        return PitchTransition(event, CountState(count.balls, count.strikes + 1), None)
    if event is PitchEvent.FOUL:
        return PitchTransition(event, CountState(count.balls, min(2, count.strikes + 1)), None)
    if event is PitchEvent.HIT_BY_PITCH:
        return PitchTransition(event, None, PlateAppearanceTerminal.HIT_BY_PITCH)
    if event is PitchEvent.IN_PLAY:
        return PitchTransition(event, None, PlateAppearanceTerminal.BALL_IN_PLAY)
    raise ValueError(f"unsupported pitch event: {event}")


class TablePitchPlanProvider:
    """Fail-closed count-specific pitch plan for tests and fitted-table artifacts."""

    def __init__(self, table: Mapping[str, Mapping[str, float]]) -> None:
        self._table = {str(count): dict(values) for count, values in table.items()}

    def probabilities(self, context: PitchContext) -> Mapping[str, float]:
        try:
            return self._table[context.count.label]
        except KeyError as exc:
            raise KeyError(f"no pitch plan for count {context.count.label}") from exc


class TablePitchEventProvider:
    """Fail-closed event table keyed by ``count|pitch_type|zone``."""

    def __init__(self, table: Mapping[str, Mapping[PitchEvent | str, float]]) -> None:
        self._table = {str(key): dict(values) for key, values in table.items()}

    def probabilities(
        self,
        context: PitchContext,
        pitch_type: str,
        zone_region: str,
    ) -> Mapping[PitchEvent | str, float]:
        key = f"{context.count.label}|{pitch_type}|{zone_region}"
        try:
            return self._table[key]
        except KeyError as exc:
            raise KeyError(f"no pitch-event model cell for {key}") from exc


class PitchSequenceSimulator:
    def __init__(self, pitch_plan: PitchPlanProvider, pitch_events: PitchEventProvider, *, max_pitches: int = 20) -> None:
        if max_pitches < 3:
            raise ValueError("max_pitches must be at least 3")
        self.pitch_plan = pitch_plan
        self.pitch_events = pitch_events
        self.max_pitches = max_pitches

    def simulate(
        self,
        initial: PitchContext,
        streams: EventRandomStreams,
        *,
        event_offset: int = 0,
    ) -> PitchSequenceResult:
        context = initial
        records: list[PitchRecord] = []
        for index in range(self.max_pitches):
            event_index = event_offset + index
            tokens, pitch_probabilities = normalize_token_probabilities(self.pitch_plan.probabilities(context))
            token = streams.choice(event_index, "pitch_plan", tokens, pitch_probabilities)
            pitch_type, zone_region = token.split("|", 1)
            event_probabilities = normalize_probabilities(
                self.pitch_events.probabilities(context, pitch_type, zone_region)
            )
            event = streams.choice(event_index, "pitch_event", PITCH_EVENTS, event_probabilities)
            transition = advance_count(context.count, event)
            records.append(
                PitchRecord(
                    pitch_number=context.pitch_number,
                    count_before=context.count,
                    pitch_type=pitch_type,
                    zone_region=zone_region,
                    event=event,
                    count_after=transition.next_count,
                )
            )
            if transition.terminal is not None:
                return PitchSequenceResult(transition.terminal, len(records), tuple(records))
            assert transition.next_count is not None
            context = context.after_pitch(transition.next_count)
        raise RuntimeError("pitch sequence exceeded max_pitches without a terminal event")
