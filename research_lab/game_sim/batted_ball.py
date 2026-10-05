from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


BATTED_BALL_OUTCOMES: tuple[str, ...] = (
    "out",
    "single",
    "double",
    "triple",
    "home_run",
    "error_reach",
)


@dataclass(frozen=True)
class BattedBallVector:
    exit_velocity_mph: float
    launch_angle_deg: float
    spray_angle_deg: float
    hang_time_s: float | None = None
    projected_distance_ft: float | None = None


@dataclass(frozen=True)
class BattedBallCell:
    ev_bin: str
    launch_bin: str
    spray_sector: str

    @classmethod
    def from_vector(cls, vector: BattedBallVector) -> "BattedBallCell":
        ev = vector.exit_velocity_mph
        la = vector.launch_angle_deg
        spray = vector.spray_angle_deg
        if ev < 80:
            ev_bin = "lt80"
        elif ev < 90:
            ev_bin = "80_89"
        elif ev < 95:
            ev_bin = "90_94"
        elif ev < 100:
            ev_bin = "95_99"
        elif ev < 105:
            ev_bin = "100_104"
        elif ev < 110:
            ev_bin = "105_109"
        else:
            ev_bin = "110_plus"

        if la < -10:
            launch_bin = "under"
        elif la < 10:
            launch_bin = "ground_low"
        elif la < 25:
            launch_bin = "line_drive"
        elif la < 40:
            launch_bin = "fly_ball"
        elif la < 55:
            launch_bin = "high_fly"
        else:
            launch_bin = "popup"

        if spray < -15:
            spray_sector = "pull_left"
        elif spray <= 15:
            spray_sector = "center"
        else:
            spray_sector = "pull_right"
        return cls(ev_bin, launch_bin, spray_sector)

    @property
    def key(self) -> str:
        return f"{self.ev_bin}|{self.launch_bin}|{self.spray_sector}"


class EmpiricalBattedBallKernel:
    """Hierarchical table kernel with explicit, auditable backoff."""

    def __init__(
        self,
        exact: Mapping[str, Sequence[float]],
        launch_spray_backoff: Mapping[str, Sequence[float]],
        launch_backoff: Mapping[str, Sequence[float]],
        league: Sequence[float],
    ) -> None:
        self.exact = dict(exact)
        self.launch_spray_backoff = dict(launch_spray_backoff)
        self.launch_backoff = dict(launch_backoff)
        self.league = self._normalize(league)

    @staticmethod
    def _normalize(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (len(BATTED_BALL_OUTCOMES),):
            raise ValueError("batted-ball probability vector has wrong shape")
        if not np.isfinite(array).all() or np.any(array < 0) or array.sum() <= 0:
            raise ValueError("invalid batted-ball probability vector")
        return array / array.sum()

    def probabilities(self, vector: BattedBallVector) -> tuple[dict[str, float], str]:
        cell = BattedBallCell.from_vector(vector)
        candidates = (
            (cell.key, self.exact, "exact"),
            (f"{cell.launch_bin}|{cell.spray_sector}", self.launch_spray_backoff, "launch_spray"),
            (cell.launch_bin, self.launch_backoff, "launch"),
        )
        for key, table, level in candidates:
            if key in table:
                values = self._normalize(table[key])
                return dict(zip(BATTED_BALL_OUTCOMES, map(float, values))), level
        return dict(zip(BATTED_BALL_OUTCOMES, map(float, self.league))), "league"
