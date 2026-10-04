"""Compatibility guards for deterministic chronological replay.

Pandas 3 can expose grouped NumPy arrays as read-only. The sequential history
state mutates those count arrays as each prior date becomes available. This
module installs a narrow copy-on-first-write guard without changing any counts,
feature values, seeds, or model probabilities.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .locked_pa_provider import MODEL_LABELS, SequentialHistoryState


def install_writeability_guard() -> None:
    """Copy a grouped count array only when pandas marked it read-only."""

    def safe_increment(table: dict, key, class_index: int) -> None:
        counts = table.get(key)
        if counts is None:
            counts = np.zeros(len(MODEL_LABELS), dtype=float)
            table[key] = counts
        elif not counts.flags.writeable:
            counts = counts.copy()
            table[key] = counts
        counts[class_index] += 1.0

    def safe_increment_recent(
        cls, table: dict, queues: dict, key, class_index: int, window: int
    ) -> None:
        counts = table.get(key)
        if counts is None:
            counts = np.zeros(len(MODEL_LABELS), dtype=float)
            table[key] = counts
        elif not counts.flags.writeable:
            counts = counts.copy()
            table[key] = counts
        queue = queues.get(key)
        if queue is None:
            queue = deque()
            queues[key] = queue
        if len(queue) >= window:
            counts[int(queue.popleft())] -= 1.0
        queue.append(int(class_index))
        counts[class_index] += 1.0

    SequentialHistoryState._increment = staticmethod(safe_increment)
    SequentialHistoryState._increment_recent = classmethod(safe_increment_recent)
