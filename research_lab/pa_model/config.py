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
    evaluation_mode: str = "development"
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

    def __post_init__(self) -> None:
        allowed_modes = {"development", "locked_final"}
        if self.evaluation_mode not in allowed_modes:
            raise ValueError(
                f"evaluation_mode must be one of {sorted(allowed_modes)}"
            )
        if not self.train_years or not self.validation_years or not self.test_years:
            raise ValueError("train, validation, and test years must be non-empty")
        train = set(self.train_years)
        validation = set(self.validation_years)
        test = set(self.test_years)
        if train & validation or train & test or validation & test:
            raise ValueError("train, validation, and test years must be disjoint")
        if max(train) >= min(validation) or max(validation) >= min(test):
            raise ValueError(
                "train, validation, and test years must be strictly chronological"
            )
        if self.validation_tuning_fraction <= 0:
            raise ValueError("validation_tuning_fraction must be positive")
        if self.validation_calibration_fraction <= 0:
            raise ValueError("validation_calibration_fraction must be positive")
        if (
            self.validation_tuning_fraction
            + self.validation_calibration_fraction
            >= 1
        ):
            raise ValueError(
                "validation fractions must leave a final ensemble-selection block"
            )
        if not self.regularization_grid or any(
            value <= 0 for value in self.regularization_grid
        ):
            raise ValueError("regularization_grid values must be positive")
        if not self.blend_grid or any(
            value < 0 or value > 1 for value in self.blend_grid
        ):
            raise ValueError("blend_grid values must be between zero and one")

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
