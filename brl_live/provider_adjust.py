"""Adjustments layered on the locked plate-appearance model at simulation time.

Both are applied inside the public layer, on top of the unchanged locked provider, and both
were measured on the 2026 out-of-sample replay before being switched on (see LEDGER.md).

1. Context offsets (ContextAdjust)
   The locked model has no interaction between batting side and inning, so it misses two
   things the data shows every season: the visiting team hits worse in the top of the first
   inning (more strikeouts, fewer extra-base hits) and the home team hits better in the
   bottom of the first; and the model runs slightly offense-heavy overall. The offsets are
   log(observed / predicted) per outcome class in six buckets (side x {1st, 2nd-8th, 9th+}),
   estimated from the model's own predictions on 2025 and on 2026 games before the forecast
   date, with 200 pseudo plate appearances of shrinkage toward zero. They are multiplied into
   the seven probabilities and renormalized. context_offsets.json carries the table and its
   provenance.

2. Per-world talent noise (TalentNoise)
   Every player's true talent is uncertain; the locked model gives one point estimate. In each
   simulated world each batter and pitcher gets a vector of log-multipliers drawn once, with
   standard deviation c * sqrt((1 - q) / (q * (n + prior))) per class (q league rate, n prior
   plate appearances, prior the model's shrinkage). Worlds then differ in who is good tonight,
   not only in dice, which widens player ranges and reduces overconfidence. c = 0 disables it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OFFSETS_PATH = Path(__file__).resolve().parent / 'context_offsets.json'
SIM_LABELS = ('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach')
MODEL_LABELS = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')


def inning_bucket(inning: int) -> str:
    return '1st' if inning == 1 else ('mid' if inning <= 8 else 'late')


def load_offsets(path: Path = OFFSETS_PATH) -> dict:
    doc = json.loads(Path(path).read_text())
    if tuple(doc['labels']) != MODEL_LABELS:
        raise ValueError('context offsets label order mismatch')
    table = {}
    for bucket, values in doc['offsets'].items():
        table[bucket] = np.exp(np.array([float(values[l]) for l in MODEL_LABELS]))
    for side in ('0', '1'):
        for inn in ('1st', 'mid', 'late'):
            if f'{side}_{inn}' not in table:
                raise ValueError('context offsets missing bucket ' + f'{side}_{inn}')
    return {'table': table, 'estimated_through': doc.get('estimated_through'), 'pseudo_pa': doc.get('pseudo_pa'),
            'source': doc.get('source')}


class ContextAdjust:
    """Multiplies the locked probabilities by exp(offset[side, inning bucket]) and renormalizes."""
    def __init__(self, inner, offsets: dict | None = None):
        self.inner = inner
        self.offsets = offsets or load_offsets()
        self.table = self.offsets['table']
        # The provider identity the engine records stays the locked model's own; the
        # adjustments are listed separately in the public box payload.
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        m = self.table[('1' if ctx.batting_side == 'home' else '0') + '_' + inning_bucket(int(ctx.inning))]
        p = np.array([base[k] for k in SIM_LABELS], dtype=float) * m
        p /= p.sum()
        return dict(zip(SIM_LABELS, map(float, p)))

    def __getattr__(self, item):
        return getattr(self.inner, item)


class EnvironmentAdjust:
    """Multiplies every plate appearance's probabilities by the game's run environment (brl_live/environment.py).

    The log-multipliers are in the model's label order; set_environment() changes them between games.
    """
    def __init__(self, inner, log_mult=None):
        self.inner = inner
        self.m = np.ones(7) if log_mult is None else np.exp(np.asarray(log_mult, float))
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def set_environment(self, log_mult):
        self.m = np.ones(7) if log_mult is None else np.exp(np.asarray(log_mult, float))

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        p = np.array([base[k] for k in SIM_LABELS], dtype=float) * self.m
        p /= p.sum()
        return dict(zip(SIM_LABELS, map(float, p)))

    def __getattr__(self, item):
        return getattr(self.inner, item)


class TalentNoise:
    """One draw of player log-multipliers per world; new_world() must be called before each world."""

    def __init__(self, inner, c: float, prior_pa: float, league: np.ndarray, counts_for):
        """counts_for(player_id, is_batter) -> prior-date plate appearances for that player."""
        self.inner, self.c, self.prior, self.league, self.counts_for = inner, float(c), float(prior_pa), np.asarray(league, float), counts_for
        self.z = {}
        self.rng = np.random.default_rng(0)
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def new_world(self, seed: int):
        self.z = {}
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, 0x54414C]))

    def _draw(self, pid, is_batter):
        key = (str(pid), is_batter)
        z = self.z.get(key)
        if z is None:
            n = float(self.counts_for(pid, is_batter))
            sigma = self.c * np.sqrt((1 - self.league) / (self.league * (n + self.prior)))
            z = self.rng.normal(0.0, 1.0, len(self.league)) * sigma - 0.5 * sigma ** 2  # E[exp(z)] = 1
            self.z[key] = z
        return z

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        if self.c == 0:
            return base
        p = np.array([base[k] for k in SIM_LABELS], dtype=float)
        q = np.exp(np.log(np.clip(p, 1e-9, 1)) + self._draw(ctx.batter.player_id, True) + self._draw(ctx.pitcher.player_id, False))
        q /= q.sum()
        return dict(zip(SIM_LABELS, map(float, q)))

    def __getattr__(self, item):
        return getattr(self.inner, item)


def history_talent_inputs(history, date: str):
    """counts_for and league rates from the assembled PA history, prior-date only.

    Independent of the private provider's internals: counts are plate appearances per
    batter and per pitcher before `date`; league rates are the outcome shares of the same rows.
    """
    import pandas as pd
    h = history.loc[history['date_key'].astype(str).str[:10] < str(date)]
    if h.empty:
        raise ValueError('No prior plate appearances for talent noise')
    label = h['outcome'].astype(str).str.upper()
    league = np.array([float((label == l).sum()) for l in MODEL_LABELS])
    league = np.clip(league / league.sum(), 1e-4, 1)
    league = league / league.sum()
    batters = h.groupby(h['batter'].astype(int)).size().to_dict()
    pitchers = h.groupby(h['pitcher'].astype(int)).size().to_dict()

    def counts_for(pid, is_batter):
        try:
            key = int(pid)
        except (TypeError, ValueError):
            return 0.0
        return float((batters if is_batter else pitchers).get(key, 0))
    return counts_for, league


def locked_counts(provider):
    """Prior-date PA counter lookup on a LockedPAModelProvider (or a wrapper exposing .state)."""
    state = provider.state

    def counts_for(pid, is_batter):
        table = state.batter if is_batter else state.pitcher
        counts = table.get(int(pid))
        return 0.0 if counts is None else float(counts.sum())
    return counts_for


def league_rates(provider) -> np.ndarray:
    counts = np.asarray(provider.state.league_counts, float)
    return counts / counts.sum()
