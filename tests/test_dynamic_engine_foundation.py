from datetime import datetime, timezone

import numpy as np
import pytest

from research_lab.game_sim.batted_ball import BattedBallCell, BattedBallVector, EmpiricalBattedBallKernel
from research_lab.game_sim.bullpen_policy import (
    BullpenDecisionContext,
    ConditionalLogitArtifact,
    FittedBullpenSelectionPolicy,
    RelieverCandidate,
)
from research_lab.game_sim.data_contracts import Provenance
from research_lab.game_sim.forecast_diagnostics import (
    central_fractional_weights,
    finite_path_binary_scores,
    forward_month_splits,
    randomized_pit,
)
from research_lab.game_sim.pitch_state import (
    CountState,
    PitchContext,
    PitchEvent,
    PitchSequenceSimulator,
    PlateAppearanceTerminal,
    TablePitchEventProvider,
    TablePitchPlanProvider,
    advance_count,
)
from research_lab.game_sim.random_streams import EventRandomStreams
from research_lab.game_sim.shrinkage import (
    BetaPrior,
    NormalPrior,
    beta_binomial_posterior,
    dirichlet_posterior_mean,
    normal_normal_posterior,
)


def test_event_draws_are_order_independent() -> None:
    streams = EventRandomStreams(17, 12345, 9)
    first = streams.random(4, "runner_transition")
    _ = streams.random(1, "pitch_event")
    assert first == streams.random(4, "runner_transition")
    assert streams.seed_for(4, "pitch_event") != streams.seed_for(4, "runner_transition")


def test_event_choice_is_reproducible() -> None:
    streams = EventRandomStreams(3, 7, 11)
    a = streams.choice(2, "pitch_type", ["FF", "SL"], [0.6, 0.4])
    b = streams.choice(2, "pitch_type", ["FF", "SL"], [0.6, 0.4])
    assert a == b


def test_two_strike_foul_keeps_count() -> None:
    transition = advance_count(CountState(1, 2), PitchEvent.FOUL)
    assert transition.next_count == CountState(1, 2)
    assert transition.terminal is None


def test_fourth_ball_walks() -> None:
    transition = advance_count(CountState(3, 1), PitchEvent.BALL)
    assert transition.terminal is PlateAppearanceTerminal.WALK


def test_sequence_uses_count_specific_tables() -> None:
    plans = {
        f"{balls}-{strikes}": {"FF|heart": 1.0}
        for balls in range(4)
        for strikes in range(3)
    }
    events = {}
    for balls in range(4):
        for strikes in range(3):
            events[f"{balls}-{strikes}|FF|heart"] = {
                "ball": 1.0 if balls < 3 else 0.0,
                "called_strike": 0.0,
                "swinging_strike": 0.0,
                "foul": 0.0,
                "in_play": 1.0 if balls == 3 else 0.0,
                "hit_by_pitch": 0.0,
            }
    simulator = PitchSequenceSimulator(
        TablePitchPlanProvider(plans),
        TablePitchEventProvider(events),
    )
    context = PitchContext(
        batter_id=1,
        pitcher_id=2,
        count=CountState(),
        pitch_number=1,
        pitcher_pitch_count=0,
        inning=1,
        outs=0,
        runners=(0, 0, 0),
        score_diff=0,
        times_through_order=1,
        batter_side="R",
        pitcher_hand="R",
        available_pitch_types=("FF",),
    )
    result = simulator.simulate(context, EventRandomStreams(1, 10, 0))
    assert result.terminal is PlateAppearanceTerminal.BALL_IN_PLAY
    assert result.pitch_count == 4


def test_beta_binomial_shrinks_small_sample() -> None:
    prior = BetaPrior(20, 80)
    posterior = beta_binomial_posterior(4, 5, prior)
    assert prior.mean < posterior.mean < 0.8


def test_dirichlet_probabilities_sum_to_one() -> None:
    assert np.isclose(dirichlet_posterior_mean([3, 1, 0], [1, 1, 1]).sum(), 1.0)


def test_normal_posterior_lies_between_prior_and_observation() -> None:
    posterior = normal_normal_posterior(100.0, 4.0, NormalPrior(90.0, 9.0))
    assert 90.0 < posterior.mean < 100.0


def test_randomized_pit_and_fractional_interval_math() -> None:
    assert np.isclose(randomized_pit([0.2, 0.5, 0.3], 1, 0.4), 0.4)
    pmf = np.asarray([0.1, 0.2, 0.4, 0.2, 0.1])
    weights = central_fractional_weights(pmf, 0.8)
    assert np.isclose(float((weights * pmf).sum()), 0.8)


def test_finite_path_correction_reduces_scores() -> None:
    scores = finite_path_binary_scores([1, 0], [0.6, 0.4], 100)
    assert scores["corrected_brier"] < scores["raw_brier"]
    assert scores["corrected_log_loss"] < scores["raw_log_loss"]


def test_forward_month_splits_never_use_future_months() -> None:
    dates = ["2025-03-30", "2025-04-01", "2025-04-20", "2025-05-01"]
    splits = forward_month_splits(dates)
    train, test, month = splits[-1]
    assert month == "2025-05"
    assert train.tolist() == [True, True, True, False]
    assert test.tolist() == [False, False, False, True]


def test_provenance_rejects_post_cutoff_data() -> None:
    source = Provenance(
        source="test",
        effective_at_utc=datetime(2025, 4, 1, 20, 0, tzinfo=timezone.utc),
        retrieved_at_utc=datetime(2025, 4, 1, 20, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError):
        source.assert_available_before(datetime(2025, 4, 1, 19, 0, tzinfo=timezone.utc))


def test_batted_ball_kernel_reports_backoff_level() -> None:
    cell = BattedBallCell.from_vector(BattedBallVector(103.0, 28.0, 2.0))
    assert cell.key == "100_104|fly_ball|center"
    kernel = EmpiricalBattedBallKernel(
        exact={},
        launch_spray_backoff={"fly_ball|center": [0.4, 0.1, 0.1, 0.0, 0.4, 0.0]},
        launch_backoff={},
        league=[0.7, 0.15, 0.08, 0.01, 0.05, 0.01],
    )
    probabilities, level = kernel.probabilities(BattedBallVector(103.0, 28.0, 2.0))
    assert level == "launch_spray"
    assert np.isclose(sum(probabilities.values()), 1.0)


def test_bullpen_softmax_uses_fitted_coefficients() -> None:
    def candidate(player_id: int, rest: int) -> RelieverCandidate:
        return RelieverCandidate(
            player_id=player_id,
            throws="R",
            role="middle",
            rest_days=rest,
            pitches_prev_1d=0,
            pitches_prev_2d=0,
            pitches_prev_3d=0,
            consecutive_days_used=0,
            leverage_skill=0.5,
            quality=0.5,
            available_probability=1.0,
        )

    policy = FittedBullpenSelectionPolicy(
        ConditionalLogitArtifact(("rest_days",), (1.0,))
    )
    context = BullpenDecisionContext(7, 0, 0, 0, 1.0, "R")
    probabilities = policy.probabilities([candidate(1, 1), candidate(2, 3)], context)
    assert probabilities[2] > probabilities[1]
    assert np.isclose(sum(probabilities.values()), 1.0)
