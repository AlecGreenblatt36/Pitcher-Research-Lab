from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class EventRandomStreams:
    """Order-independent deterministic random streams for paired simulation tests.

    A random draw is keyed by game, Monte Carlo path, event index, draw type,
    and optional sub-index. Adding a new draw in one component therefore does
    not shift unrelated draws in another component.
    """

    master_seed: int
    game_pk: int
    path_id: int
    namespace: str = "baseball-research-lab-v1"

    def seed_for(self, event_index: int, draw_type: str, sub_index: int = 0) -> int:
        if event_index < 0:
            raise ValueError("event_index must be non-negative")
        if sub_index < 0:
            raise ValueError("sub_index must be non-negative")
        if not draw_type:
            raise ValueError("draw_type is required")
        payload = "|".join(
            (
                self.namespace,
                str(int(self.master_seed)),
                str(int(self.game_pk)),
                str(int(self.path_id)),
                str(int(event_index)),
                str(draw_type),
                str(int(sub_index)),
            )
        ).encode("utf-8")
        digest = blake2b(payload, digest_size=16, person=b"BRL-RNG-v1").digest()
        return int.from_bytes(digest, "little", signed=False)

    def generator(self, event_index: int, draw_type: str, sub_index: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed_for(event_index, draw_type, sub_index))

    def random(self, event_index: int, draw_type: str, sub_index: int = 0) -> float:
        return float(self.generator(event_index, draw_type, sub_index).random())

    def normal(
        self,
        event_index: int,
        draw_type: str,
        *,
        mean: float = 0.0,
        sd: float = 1.0,
        sub_index: int = 0,
    ) -> float:
        if sd < 0:
            raise ValueError("sd cannot be negative")
        return float(self.generator(event_index, draw_type, sub_index).normal(mean, sd))

    def choice(
        self,
        event_index: int,
        draw_type: str,
        values: Sequence[T],
        probabilities: Sequence[float],
        *,
        sub_index: int = 0,
    ) -> T:
        if not values:
            raise ValueError("values cannot be empty")
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(values),):
            raise ValueError("probability length must match values")
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("probabilities must be finite and non-negative")
        total = float(p.sum())
        if total <= 0:
            raise ValueError("probabilities must have positive mass")
        p = p / total
        index = int(self.generator(event_index, draw_type, sub_index).choice(len(values), p=p))
        return values[index]
