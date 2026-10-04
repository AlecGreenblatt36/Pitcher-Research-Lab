from __future__ import annotations

import pandas as pd

from research_lab.game_sim.locked_pa_provider import MODEL_LABELS, SequentialHistoryState
from research_lab.game_sim.sequential_history_compat import install_writeability_guard


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date_key": "2024-09-01",
                "game_pk": 1,
                "at_bat_number": 1,
                "batter": 10,
                "pitcher": 20,
                "stand": "R",
                "p_throws": "R",
                "park": "A",
                "outcome": "BIP_OUT",
                "age_bat": 27.0,
                "age_pit": 28.0,
            },
            {
                "date_key": "2025-04-01",
                "game_pk": 2,
                "at_bat_number": 1,
                "batter": 11,
                "pitcher": 21,
                "stand": "L",
                "p_throws": "R",
                "park": "B",
                "outcome": "K",
                "age_bat": 25.0,
                "age_pit": 29.0,
            },
            {
                "date_key": "2025-04-02",
                "game_pk": 3,
                "at_bat_number": 1,
                "batter": 11,
                "pitcher": 22,
                "stand": "L",
                "p_throws": "L",
                "park": "C",
                "outcome": "HR",
                "age_bat": 25.0,
                "age_pit": 31.0,
            },
        ]
    )


def test_later_cutoff_reveals_actual_history_even_if_prior_simulation_failed() -> None:
    """Replay state depends on the immutable history table, not prior sim success."""

    state = SequentialHistoryState(
        _history(),
        cutoff_date="2025-04-02",
        config={"batter_recent_window": 50, "pitcher_recent_window": 50},
    )

    # At the start of April 2, that day's actual PA is correctly unavailable.
    assert state.n_rows == 2
    hr_index = MODEL_LABELS.index("HR")
    assert state.batter[11][hr_index] == 0

    # Simulate the earlier game failing by doing no game-engine update at all.
    # Advancing to April 3 must still reveal April 2 from the historical table.
    state.advance_to("2025-04-03")
    assert state.n_rows == 3
    assert state.batter[11][hr_index] == 1


def test_pandas_read_only_counts_are_copied_on_first_write() -> None:
    install_writeability_guard()
    state = SequentialHistoryState(
        _history(),
        cutoff_date="2025-04-02",
        config={"batter_recent_window": 50, "pitcher_recent_window": 50},
    )

    original = state.batter[11]
    original.setflags(write=False)
    assert not original.flags.writeable

    state.advance_to("2025-04-03")

    updated = state.batter[11]
    assert updated.flags.writeable
    assert updated is not original
    assert updated[MODEL_LABELS.index("K")] == 1
    assert updated[MODEL_LABELS.index("HR")] == 1
