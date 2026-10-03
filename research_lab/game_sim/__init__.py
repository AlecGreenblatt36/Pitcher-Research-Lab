"""Sequential, pluggable baseball game simulation engine."""

from .engine import GameResult, GameSimulator
from .models import (
    GameMatchup,
    OUTCOME_LABELS,
    PAContext,
    PitcherProfile,
    PlayerProfile,
    SimulationConfig,
    TeamProfile,
)
from .monte_carlo import simulate_many
from .probability import (
    BlendedProbabilityProvider,
    CallableProbabilityProvider,
    PAProbabilityProvider,
    RatingsProbabilityProvider,
    TableProbabilityProvider,
)

__all__ = [
    "BlendedProbabilityProvider",
    "CallableProbabilityProvider",
    "GameMatchup",
    "GameResult",
    "GameSimulator",
    "OUTCOME_LABELS",
    "PAContext",
    "PAProbabilityProvider",
    "PitcherProfile",
    "PlayerProfile",
    "RatingsProbabilityProvider",
    "SimulationConfig",
    "TableProbabilityProvider",
    "TeamProfile",
    "simulate_many",
]
