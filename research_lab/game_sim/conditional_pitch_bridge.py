from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from .random_streams import EventRandomStreams

PITCH_EVENTS: tuple[str, ...] = (
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "in_play",
    "hit_by_pitch",
)
PA_OUTCOMES: tuple[str, ...] = (
    "BIP_OUT",
    "K",
    "BB_HBP",
    "1B",
    "2B_3B",
    "HR",
    "OTHER_REACH",
)
TERMINAL_GROUPS: tuple[str, ...] = ("K", "BB_HBP", "BIP")
BIP_OUTCOMES = frozenset({"BIP_OUT", "1B", "2B_3B", "HR", "OTHER_REACH"})


class TransitionProbabilityProvider(Protocol):
    def probabilities(
        self,
        *,
        batter_id: int,
        pitcher_id: int,
        balls: int,
        strikes: int,
        platoon: int,
    ) -> Mapping[str, float]:
        """Return probabilities for all six ``PITCH_EVENTS``."""


@dataclass(frozen=True)
class BridgePitch:
    pitch_number: int
    balls_before: int
    strikes_before: int
    event: str
    balls_after: int | None
    strikes_after: int | None


@dataclass(frozen=True)
class OutcomePreservingSequence:
    pa_outcome: str
    terminal_group: str
    pitches: tuple[BridgePitch, ...]

    @property
    def pitch_count(self) -> int:
        return len(self.pitches)


def terminal_group(pa_outcome: str) -> str:
    if pa_outcome == "K":
        return "K"
    if pa_outcome == "BB_HBP":
        return "BB_HBP"
    if pa_outcome in BIP_OUTCOMES:
        return "BIP"
    raise ValueError(f"unsupported PA outcome: {pa_outcome}")


def _normalized_event_vector(probabilities: Mapping[str, float]) -> np.ndarray:
    missing = [event for event in PITCH_EVENTS if event not in probabilities]
    if missing:
        raise ValueError(f"missing pitch-event probabilities: {missing}")
    values = np.asarray([float(probabilities[event]) for event in PITCH_EVENTS], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("pitch-event probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("pitch-event probabilities must have positive mass")
    return values / total


def _next_state_or_terminal(
    balls: int,
    strikes: int,
    event: str,
) -> tuple[tuple[int, int] | None, str | None]:
    if event == "ball":
        if balls == 3:
            return None, "BB_HBP"
        return (balls + 1, strikes), None
    if event in {"called_strike", "swinging_strike"}:
        if strikes == 2:
            return None, "K"
        return (balls, strikes + 1), None
    if event == "foul":
        return (balls, min(2, strikes + 1)), None
    if event == "in_play":
        return None, "BIP"
    if event == "hit_by_pitch":
        return None, "BB_HBP"
    raise ValueError(f"unsupported pitch event: {event}")


def absorbing_probabilities(
    provider: TransitionProbabilityProvider,
    *,
    batter_id: int,
    pitcher_id: int,
    platoon: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Return eventual ``[K, BB_HBP, BIP]`` probabilities for each live count."""

    solved: dict[tuple[int, int], np.ndarray] = {}
    group_index = {group: index for index, group in enumerate(TERMINAL_GROUPS)}
    for balls in range(3, -1, -1):
        for strikes in range(2, -1, -1):
            event_probabilities = _normalized_event_vector(
                provider.probabilities(
                    batter_id=batter_id,
                    pitcher_id=pitcher_id,
                    balls=balls,
                    strikes=strikes,
                    platoon=platoon,
                )
            )
            total = np.zeros(len(TERMINAL_GROUPS), dtype=float)
            self_loop = 0.0
            for event, probability in zip(PITCH_EVENTS, event_probabilities):
                next_state, terminal = _next_state_or_terminal(balls, strikes, event)
                if terminal is not None:
                    total[group_index[terminal]] += probability
                elif next_state == (balls, strikes):
                    self_loop += probability
                else:
                    assert next_state is not None
                    total += probability * solved[next_state]
            if self_loop >= 1.0 - 1e-12:
                raise ValueError(f"non-absorbing transition cell at {(balls, strikes)}")
            total /= 1.0 - self_loop
            total = np.clip(total, 0.0, None)
            total /= total.sum()
            solved[(balls, strikes)] = total
    return solved


def conditioned_event_probabilities(
    provider: TransitionProbabilityProvider,
    absorbing: Mapping[tuple[int, int], np.ndarray],
    *,
    batter_id: int,
    pitcher_id: int,
    balls: int,
    strikes: int,
    platoon: int,
    target_group: str,
) -> np.ndarray:
    """Doob h-transform of the pitch chain conditional on a terminal group."""

    if target_group not in TERMINAL_GROUPS:
        raise ValueError(f"unsupported terminal group: {target_group}")
    target_index = TERMINAL_GROUPS.index(target_group)
    current_h = float(absorbing[(balls, strikes)][target_index])
    if current_h <= 0:
        raise ValueError(
            f"terminal group {target_group} has zero probability from count {balls}-{strikes}"
        )
    raw = _normalized_event_vector(
        provider.probabilities(
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            balls=balls,
            strikes=strikes,
            platoon=platoon,
        )
    )
    weights = np.zeros_like(raw)
    for index, (event, probability) in enumerate(zip(PITCH_EVENTS, raw)):
        next_state, terminal = _next_state_or_terminal(balls, strikes, event)
        if terminal is not None:
            future_h = 1.0 if terminal == target_group else 0.0
        else:
            assert next_state is not None
            future_h = float(absorbing[next_state][target_index])
        weights[index] = probability * future_h / current_h
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        raise RuntimeError("conditional transition lost all probability mass")
    return weights / weights.sum()


def sample_conditioned_sequence(
    provider: TransitionProbabilityProvider,
    streams: EventRandomStreams,
    *,
    batter_id: int,
    pitcher_id: int,
    platoon: int,
    target_group: str,
    event_offset: int = 0,
    max_pitches: int = 30,
) -> tuple[BridgePitch, ...]:
    absorbing = absorbing_probabilities(
        provider,
        batter_id=batter_id,
        pitcher_id=pitcher_id,
        platoon=platoon,
    )
    balls = 0
    strikes = 0
    records: list[BridgePitch] = []
    for pitch_index in range(max_pitches):
        probabilities = conditioned_event_probabilities(
            provider,
            absorbing,
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            balls=balls,
            strikes=strikes,
            platoon=platoon,
            target_group=target_group,
        )
        event = streams.choice(
            event_offset + pitch_index,
            "conditional_pitch_event",
            PITCH_EVENTS,
            probabilities,
        )
        next_state, terminal = _next_state_or_terminal(balls, strikes, event)
        records.append(
            BridgePitch(
                pitch_number=pitch_index + 1,
                balls_before=balls,
                strikes_before=strikes,
                event=event,
                balls_after=None if next_state is None else next_state[0],
                strikes_after=None if next_state is None else next_state[1],
            )
        )
        if terminal is not None:
            if terminal != target_group:
                raise RuntimeError(
                    f"conditional bridge terminated at {terminal}, expected {target_group}"
                )
            return tuple(records)
        assert next_state is not None
        balls, strikes = next_state
    raise RuntimeError("conditional pitch sequence exceeded max_pitches")


def sample_outcome_preserving_sequence(
    provider: TransitionProbabilityProvider,
    streams: EventRandomStreams,
    *,
    batter_id: int,
    pitcher_id: int,
    platoon: int,
    locked_pa_probabilities: Sequence[float],
    event_offset: int = 0,
    max_pitches: int = 30,
) -> OutcomePreservingSequence:
    """Sample a locked-model PA outcome, then a compatible pitch sequence.

    The seven-outcome marginal is exactly the supplied locked probability vector;
    the Markov layer only supplies the path and pitch count conditional on that
    sampled outcome.
    """

    probabilities = np.asarray(locked_pa_probabilities, dtype=float)
    if probabilities.shape != (len(PA_OUTCOMES),):
        raise ValueError("locked_pa_probabilities must contain seven values")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("locked PA probabilities must be finite and non-negative")
    if probabilities.sum() <= 0:
        raise ValueError("locked PA probabilities must have positive mass")
    probabilities /= probabilities.sum()
    outcome = streams.choice(
        event_offset,
        "locked_pa_outcome",
        PA_OUTCOMES,
        probabilities,
    )
    group = terminal_group(outcome)
    pitches = sample_conditioned_sequence(
        provider,
        streams,
        batter_id=batter_id,
        pitcher_id=pitcher_id,
        platoon=platoon,
        target_group=group,
        event_offset=event_offset + 1,
        max_pitches=max_pitches,
    )
    return OutcomePreservingSequence(outcome, group, pitches)
