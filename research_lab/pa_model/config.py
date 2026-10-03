from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class PAConfig:
    """Configuration for the chronological PA benchmark.

    Feature histories are date blocked: every PA on date D is scored before any
    result from date D is revealed. The validation season is also split by date
    into disjoint tuning, calibration, and ensemble-selection periods.
    """

    train_years: tuple[int, ...] = (2023,)
    validation_years: tuple[int, ...] = (2024,)
    test_years: tuple[int, ...] = (2025,)
    outcome_labels: tuple[str, ...] = (
        "BIP_OUT",
        "K",
        "BB_HBP",
        "1B",
        "2B_3B",
        "HR",
        "OTHER_REACH",
    )
    player_prior_pa: float = 180.0
    split_prior_pa: float = 260.0
    park_prior_pa: float = 1000.0
    recent_prior_pa: float = 120.0
    batter_recent_window: int = 100
    pitcher_recent_window: int = 150
    regularization_grid: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
    validation_tuning_fraction: float = 0.50
    validation_calibration_fraction: float = 0.25
    calibration_l2: float = 1e-4
    blend_grid: tuple[float, ...] = tuple(x / 100 for x in range(101))
    bootstrap_replicates: int = 1000
    random_seed: int = 36
    min_probability: float = 1e-7
    max_iter: int = 150
    chunk_days: int = 3
    request_workers: int = 6
    request_timeout_seconds: int = 180
    request_retries: int = 5
    user_agent: str = "BaseballResearchLab/1.0 (+public research; contact repository owner)"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def with_years(
        cls,
        train_years: Sequence[int],
        validation_years: Sequence[int],
        test_years: Sequence[int],
        **kwargs,
    ) -> "PAConfig":
        return cls(
            train_years=tuple(train_years),
            validation_years=tuple(validation_years),
            test_years=tuple(test_years),
            **kwargs,
        )
