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


class MatchupPitchCounts:
    """LEASH-04: pitches per plate appearance from the pitch-level chain for this matchup (research_lab.pa_model.pitch_chain
    ChainState.pitch_count_table through a provider that carries the chain), given the outcome the engine drew. A matchup
    the chain cannot price (no chain, or unknown ids) falls back to the league table by outcome."""

    wants_context = True

    def __init__(self, provider, fallback: PitchCounts, nmax: int = 20):
        self.provider, self.fallback, self.nmax = provider, fallback, int(nmax)
        self.name = 'pitches per plate appearance from the matchup chain (LEASH-04)'
        self._cache: dict = {}
        self._index = {label: i for i, label in enumerate(MODEL_LABELS)}
        self._sim_to_model = dict(zip(OUTCOME_LABELS, MODEL_LABELS))

    def _table(self, context):
        chain = getattr(self.provider, 'chain', None)
        if chain is None:
            return None
        try:
            batter_id = int(context.batter.player_id); pitcher_id = int(context.pitcher.player_id)
        except (TypeError, ValueError):
            return None
        throws = str(context.pitcher.throws); stand = self.provider._stand(context.batter.bats, throws)
        key = (batter_id, pitcher_id, stand, throws, str(self.provider.park), str(self.provider.game_date)[:10])
        cdf = self._cache.get(key)
        if cdf is None:
            from research_lab.pa_model.pitch_chain import day_ordinal
            table = chain.pitch_count_table(batter_id, pitcher_id, str(self.provider.park), stand, throws, int(day_ordinal([self.provider.game_date])[0]), self.nmax)
            cdf = np.cumsum(table, axis=1)
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = cdf
        return cdf

    def draw(self, outcome: str, u: float, context=None) -> int:
        cdf = self._table(context) if context is not None else None
        if cdf is None:
            return self.fallback.draw(outcome, u)
        row = cdf[self._index[self._sim_to_model[outcome]]]
        if row[-1] <= 0.5:                                           # an outcome the chain gives no mass (never for real matchups)
            return self.fallback.draw(outcome, u)
        return int(min(np.searchsorted(row, float(u) * row[-1], side='right'), len(row) - 1))

    def mean(self, outcome: str) -> float:
        return self.fallback.mean(outcome)
