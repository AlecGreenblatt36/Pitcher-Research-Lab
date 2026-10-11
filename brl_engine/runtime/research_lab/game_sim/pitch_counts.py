"""Pitches per plate appearance for the simulator (LEASH-03).

Each plate appearance's pitches are drawn from their real distribution for its outcome (counts of 0, 1, 2, ... pitches
by outcome class from the sealed per-PA physics tables, tools/brl_research.py pitch_counts), so a starter's line carries
a pitch count the exit hazard's pitch terms can read. The draw takes one uniform number, from its own random stream
in paired replays."""
from __future__ import annotations

import numpy as np

from .models import OUTCOME_LABELS

MODEL_LABELS = ("BIP_OUT", "K", "BB_HBP", "1B", "2B_3B", "HR", "OTHER_REACH")


class PitchCounts:
    def __init__(self, doc: dict):
        counts = doc["counts"]
        self.name = str(doc.get("name") or "pitches per plate appearance by outcome")
        self.cdf = {}
        for sim_label, model_label in zip(OUTCOME_LABELS, MODEL_LABELS):
            c = np.asarray(counts[model_label], dtype=float)
            if c.sum() <= 0:
                raise ValueError("no pitch counts for " + model_label)
            self.cdf[sim_label] = np.cumsum(c) / c.sum()

    def draw(self, outcome: str, u: float) -> int:
        """Pitches for a plate appearance with this outcome, from a uniform number u in [0, 1)."""
        c = self.cdf[outcome]
        return int(min(np.searchsorted(c, float(u), side="right"), len(c) - 1))

    def mean(self, outcome: str) -> float:
        c = self.cdf[outcome]
        p = np.diff(np.concatenate([[0.0], c]))
        return float((np.arange(len(p)) * p).sum())
