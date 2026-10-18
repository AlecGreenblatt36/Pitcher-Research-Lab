"""Leakage-resistant plate-appearance probability model."""

from .config import PAConfig
from .pipeline import run_benchmark

__all__ = ["PAConfig", "run_benchmark"]
