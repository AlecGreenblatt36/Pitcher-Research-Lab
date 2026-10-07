"""The locked provider and the fitted starter policy, built from the data files directly."""
from __future__ import annotations
from dataclasses import dataclass

import joblib

from .common import digest
from .engine_bridge import HAZARD
from research_lab.game_sim.locked_pa_provider import LockedPAModelProvider
from research_lab.game_sim.starter_hazard import FittedStarterPolicy

_HAZARD_CACHE: dict = {}


def pinned_document(kind: str) -> dict:
    if kind != 'manager':
        raise KeyError(kind)
    key = str(HAZARD)
    if key not in _HAZARD_CACHE:
        bundle = joblib.load(HAZARD)
        _HAZARD_CACHE[key] = {'kind': 'manager', 'source_sha256': digest(HAZARD), 'league_mean_bf': float(bundle['league_mean_bf']),
                              'name': 'fitted-starter-hazard-v1 (relievers: heuristic)', 'bundle': bundle}
    return _HAZARD_CACHE[key]


@dataclass
class PortableLockedProvider(LockedPAModelProvider):
    """Identical to the locked provider; the class name is kept for the public layer's imports."""


class PortableStarterPolicy(FittedStarterPolicy):
    def __init__(self, document, pitcher_exp, team_exp, team_of_pitcher, **kw):
        super().__init__(document['bundle'], pitcher_exp, team_exp, team_of_pitcher, **kw)
