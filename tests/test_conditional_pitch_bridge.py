import numpy as np

from research_lab.game_sim.conditional_pitch_bridge import (
    PITCH_EVENTS,
    PA_OUTCOMES,
    absorbing_probabilities,
    conditioned_event_probabilities,
    sample_conditioned_sequence,
    sample_outcome_preserving_sequence,
    terminal_group,
)
from research_lab.game_sim.random_streams import EventRandomStreams


class ConstantProvider:
    def probabilities(self, **_: int) -> dict[str, float]:
        return {
            "ball": 0.33,
            "called_strike": 0.17,
            "swinging_strike": 0.13,
            "foul": 0.17,
            "in_play": 0.18,
            "hit_by_pitch": 0.02,
        }


def test_absorbing_probabilities_are_valid() -> None:
    solved = absorbing_probabilities(
        ConstantProvider(),
        batter_id=1,
        pitcher_id=2,
        platoon=1,
    )
    assert set(solved) == {(balls, strikes) for balls in range(4) for strikes in range(3)}
    for values in solved.values():
        assert values.shape == (3,)
        assert np.all(values >= 0)
        assert np.isclose(values.sum(), 1.0)


def test_conditioned_event_probabilities_sum_to_one() -> None:
    provider = ConstantProvider()
    solved = absorbing_probabilities(provider, batter_id=1, pitcher_id=2, platoon=0)
    for target in ("K", "BB_HBP", "BIP"):
        probabilities = conditioned_event_probabilities(
            provider,
            solved,
            batter_id=1,
            pitcher_id=2,
            balls=0,
            strikes=0,
            platoon=0,
            target_group=target,
        )
        assert probabilities.shape == (len(PITCH_EVENTS),)
        assert np.all(probabilities >= 0)
        assert np.isclose(probabilities.sum(), 1.0)


def test_conditioned_sequences_always_reach_requested_terminal_group() -> None:
    provider = ConstantProvider()
    for target in ("K", "BB_HBP", "BIP"):
        for path_id in range(100):
            sequence = sample_conditioned_sequence(
                provider,
                EventRandomStreams(36, 999, path_id),
                batter_id=1,
                pitcher_id=2,
                platoon=1,
                target_group=target,
            )
            assert sequence
            final = sequence[-1].event
            if target == "K":
                assert final in {"called_strike", "swinging_strike"}
                assert sequence[-1].strikes_before == 2
            elif target == "BB_HBP":
                assert final in {"ball", "hit_by_pitch"}
                if final == "ball":
                    assert sequence[-1].balls_before == 3
            else:
                assert final == "in_play"


def test_outcome_preserving_sampler_keeps_terminal_group_consistent() -> None:
    provider = ConstantProvider()
    locked = np.asarray([0.40, 0.20, 0.10, 0.15, 0.06, 0.06, 0.03])
    observed = {outcome: 0 for outcome in PA_OUTCOMES}
    for path_id in range(2000):
        result = sample_outcome_preserving_sequence(
            provider,
            EventRandomStreams(7, 1234, path_id),
            batter_id=10,
            pitcher_id=20,
            platoon=0,
            locked_pa_probabilities=locked,
        )
        observed[result.pa_outcome] += 1
        assert result.terminal_group == terminal_group(result.pa_outcome)
        assert result.pitch_count >= 1
    frequencies = np.asarray([observed[outcome] / 2000 for outcome in PA_OUTCOMES])
    assert np.max(np.abs(frequencies - locked)) < 0.035


def test_same_stream_key_is_reproducible() -> None:
    provider = ConstantProvider()
    locked = [0.45, 0.22, 0.09, 0.14, 0.05, 0.04, 0.01]
    left = sample_outcome_preserving_sequence(
        provider,
        EventRandomStreams(1, 2, 3),
        batter_id=4,
        pitcher_id=5,
        platoon=1,
        locked_pa_probabilities=locked,
    )
    right = sample_outcome_preserving_sequence(
        provider,
        EventRandomStreams(1, 2, 3),
        batter_id=4,
        pitcher_id=5,
        platoon=1,
        locked_pa_probabilities=locked,
    )
    assert left == right
