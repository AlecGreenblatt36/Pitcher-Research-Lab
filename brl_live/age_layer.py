"""Aging and recency layer on top of the plate-appearance model (AGE-01 and AGE-02 in LEDGER.md).

The PA model rates batters and pitchers on all their plate appearances since 2023 with no time decay, and its
coefficients were fitted in 2023-2024 when histories were short. In 2026 it was too high on older hitters and too
low on young ones (and the mirror image for pitchers). This layer multiplies each plate appearance's probabilities by
exp(z W), where z holds 16 standardized features of the batter and the pitcher computed from earlier dates only
(research_lab.pa_model.physics.AgingState): years from the PA-weighted mean date of the player's history to today,
that gap times his distance from age 27, age squared, and the log of his time-decayed rate (half-life 365 days) over
his all-history rate for strikeouts, walks, home runs, singles and extra-base hits. W (16 x 7, columns summing to
zero) was fitted on the model's own predictions (research set 'stage2'): fitted on 2025 and scored on 2026, -0.00049
nats per PA; fitted through June 2026 and scored from July, -0.00107.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAYER_PATH = Path(__file__).resolve().parent / 'age_layer.json'
SIM_LABELS = ('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach')
_STATES: dict = {}


def load_layer(path: Path = LAYER_PATH) -> dict:
    doc = json.loads(Path(path).read_text())
    if tuple(doc['labels']) != ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH'):
        raise ValueError('age layer label order mismatch')
    return doc


def layer_from_receipt(receipt: dict, fit: str = 'fit_all', variant: str = 'v2') -> dict:
    """The layer dict from a stage2 research receipt (results[variant].aging_layer.coefficients[fit])."""
    c = receipt['results'][variant]['aging_layer']['coefficients']
    return {'schema': 'brl.age-layer.v1', 'columns': c['columns'], 'labels': c['labels'], 'mean': c[fit]['mean'], 'sd': c[fit]['sd'],
            'coef': c[fit]['coef'], 'params': {'aging': True, 'decay_days': 365, 'k_dec': 60.0}, 'fit': fit}


def state_for(history, date: str, params: dict):
    """AgingState at the start of `date` from the PA history; one per (history object, date, parameters)."""
    from research_lab.pa_model.physics import AgingState
    key = (id(history), len(history), str(date)[:10], tuple(sorted((k, str(v)) for k, v in params.items())))
    st = _STATES.get(key)
    if st is None:
        if len(_STATES) > 6:
            _STATES.clear()
        st = AgingState.build(history, str(date)[:10], params)
        _STATES[key] = st
    return st


class AgeAdjust:
    """Multiplies each plate appearance's probabilities by exp(z W) for its batter and pitcher.

    state: an AgingState as of the game's date; today: that date's ordinal; ages(batter_id, pitcher_id) -> (age_bat, age_pit)."""

    def __init__(self, inner, layer: dict, state, today: int, ages=None):
        self.inner, self.layer, self.state, self.today = inner, layer, state, int(today)
        self.cols = list(layer['columns'])
        self.mu, self.sd, self.W = np.asarray(layer['mean'], float), np.asarray(layer['sd'], float), np.asarray(layer['coef'], float)
        self.ages = ages
        self.cache: dict = {}
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def set_day(self, state, today: int) -> None:
        self.state, self.today, self.cache = state, int(today), {}

    def reset(self) -> None:
        self.cache = {}

    def _ages(self, b: int, p: int):
        if self.ages is not None:
            return self.ages(b, p)
        try:
            s = self.inner.state
            return self.inner.season_age(b, s.batter_age), self.inner.season_age(p, s.pitcher_age)
        except Exception:
            return None, None

    def multipliers(self, b: int, p: int) -> np.ndarray:
        key = (b, p)
        m = self.cache.get(key)
        if m is None:
            ab, ap = self._ages(b, p)
            vals = self.state.values(b, p, self.today, ab, ap)
            x = np.array([vals.get(c, np.nan) for c in self.cols], float)
            x = np.where(np.isnan(x), self.mu, x)
            m = np.exp(((x - self.mu) / self.sd) @ self.W)
            self.cache[key] = m
        return m

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        try:
            b, p = int(ctx.batter.player_id), int(ctx.pitcher.player_id)
        except (TypeError, ValueError, AttributeError):
            return base
        q = np.array([base[k] for k in SIM_LABELS], dtype=float) * self.multipliers(b, p)
        q /= q.sum()
        return dict(zip(SIM_LABELS, map(float, q)))

    def __getattr__(self, item):
        return getattr(self.inner, item)
