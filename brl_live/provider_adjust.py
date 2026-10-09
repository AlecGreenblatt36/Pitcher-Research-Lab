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


BASE_STATE_PATH = Path(__file__).resolve().parent / 'base_state_offsets.json'


def base_cell(ctx) -> str:
    """'bases|outs' for a plate appearance: bases 0 empty, 1 first only, 2 a runner in scoring position; outs 0 to 2."""
    on = [b is not None and b is not False for b in (tuple(ctx.bases) + (None, None, None))[:3]]
    cat = 0 if not any(on) else 1 if not (on[1] or on[2]) else 2
    return f'{cat}|{min(max(int(ctx.outs), 0), 2)}'


def load_base_state(path: Path = BASE_STATE_PATH) -> dict:
    doc = json.loads(Path(path).read_text())
    if tuple(doc['labels']) != MODEL_LABELS:
        raise ValueError('base-state offsets label order mismatch')
    table = {cell: np.exp(np.array([float(values[l]) for l in MODEL_LABELS])) for cell, values in doc['offsets'].items()}
    for b in range(3):
        for o in range(3):
            if f'{b}|{o}' not in table:
                raise ValueError(f'base-state offsets missing cell {b}|{o}')
    return {'table': table, 'name': doc.get('name'), 'source': doc.get('source')}


class BaseStateAdjust:
    """Multiplies each plate appearance's probabilities by the offsets of its bases (empty, first only, a runner in
    scoring position) and outs, and renormalizes (RUNS-02). The offsets are the stack's miss in each cell relative to its
    miss over the whole season, so they carry the shape by situation and leave the level to the context offsets."""
    def __init__(self, inner, offsets: dict | None = None):
        self.inner = inner
        self.offsets = offsets or load_base_state()
        self.table = self.offsets['table']
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        p = np.array([base[k] for k in SIM_LABELS], dtype=float) * self.table[base_cell(ctx)]
        p /= p.sum()
        return dict(zip(SIM_LABELS, map(float, p)))

    def __getattr__(self, item):
        return getattr(self.inner, item)


GOOD_FOR_OFFENSE = np.array([l not in ('bip_out', 'strikeout') for l in SIM_LABELS])


class DayForm:
    """A day-form shock per batting team and world (DISP-01): in each simulated world each team's chances of reaching base
    (walk, single, extra-base hit, home run, other reach) are multiplied by exp(e), e ~ N(-sigma^2/2, sigma^2) drawn once
    per team per world, and the seven probabilities renormalized. Real team runs vary more from game to game than a
    simulator with fixed rates gives (team-run SD 3.21 against 3.12 in 2026, 3.25 against 3.11 in 2025: starters' stuff,
    lineups' days, conditions). new_world(seed) must be called before each world."""
    def __init__(self, inner, sigma: float):
        self.inner, self.sigma = inner, float(sigma)
        self.e = {'away': 0.0, 'home': 0.0}
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def new_world(self, seed: int):
        r = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, 0x44415946]))
        self.e = {s: float(r.normal(0.0, self.sigma) - 0.5 * self.sigma ** 2) for s in ('away', 'home')}

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        e = self.e.get('home' if ctx.batting_side == 'home' else 'away', 0.0)
        if not e:
            return base
        p = np.array([base[k] for k in SIM_LABELS], dtype=float)
        p[GOOD_FOR_OFFENSE] *= np.exp(e)
        p /= p.sum()
        return dict(zip(SIM_LABELS, map(float, p)))

    def __getattr__(self, item):
        return getattr(self.inner, item)


class MatchupAdjust:
    """Decision-moment matchups (MATCHUP-01, MATCHUP-02): a hitter's own swing map at the moment he commits, run over
    the pitcher's arsenal as it looks at that moment, gives the pair's extra chase rate and extra in-zone swing rate
    beyond both players' averages (points). The strikeout and walk chances move by fitted log-odds per point; the other
    outcomes keep their proportions.

    table: {(batter id, pitcher id): (chase points, zone-swing points)} from earlier seasons only;
    coefs: {'K': (per chase point, per zone-swing point), 'BB': (...)}. Pairs not in the table are left alone."""
    K, BB = SIM_LABELS.index('strikeout'), SIM_LABELS.index('bb_hbp')

    def __init__(self, inner, table: dict, coefs: dict):
        self.inner = inner
        self.table = {(str(b), str(p)): (float(c), float(z)) for (b, p), (c, z) in table.items()}
        self.ck = tuple(float(x) for x in coefs['K']); self.cb = tuple(float(x) for x in coefs['BB'])
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        d = self.table.get((str(ctx.batter.player_id), str(ctx.pitcher.player_id)))
        if d is None:
            return base
        p = np.array([base[k] for k in SIM_LABELS], dtype=float)
        def shift(q, lo):
            q = min(max(q, 1e-9), 1 - 1e-9); z = np.log(q / (1 - q)) + lo
            return 1.0 / (1.0 + np.exp(-z))
        k_new = shift(p[self.K], self.ck[0] * d[0] + self.ck[1] * d[1])
        b_new = shift(p[self.BB], self.cb[0] * d[0] + self.cb[1] * d[1])
        rest = 1.0 - p[self.K] - p[self.BB]
        scale = (1.0 - k_new - b_new) / rest if rest > 0 else 1.0
        p = p * scale; p[self.K] = k_new; p[self.BB] = b_new
        p /= p.sum()
        return dict(zip(SIM_LABELS, map(float, p)))

    def __getattr__(self, item):
        return getattr(self.inner, item)


class TeamAdjust:
    """Multiplies each plate appearance's probabilities by the batting team's and the fielding team's offsets
    (brl_live/team_offsets.py). set_teams() takes {'away': log-multipliers while the away team bats, 'home': ...}."""
    def __init__(self, inner, by_side=None):
        self.inner = inner
        self.set_teams(by_side)
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def set_teams(self, by_side):
        self.m = {s: (np.ones(7) if not by_side or by_side.get(s) is None else np.exp(np.asarray(by_side[s], float))) for s in ('away', 'home')}

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        p = np.array([base[k] for k in SIM_LABELS], dtype=float) * self.m['home' if ctx.batting_side == 'home' else 'away']
        p /= p.sum()
        return dict(zip(SIM_LABELS, map(float, p)))

    def __getattr__(self, item):
        return getattr(self.inner, item)


class RoleAdjust:
    """Multiplies each plate appearance's probabilities by the pitcher's role offsets (brl_live/role_offsets.py): the
    starter by how many times he has gone through the order (1, 2, 3 or more), any other pitcher as a reliever.
    set_roles() takes {role: seven log-multipliers}."""
    def __init__(self, inner, by_role=None):
        self.inner = inner
        self.set_roles(by_role)
        self.name = getattr(inner, 'name', 'provider')
        self.validation_status = getattr(inner, 'validation_status', '')

    def set_roles(self, by_role):
        self.m = {k: np.exp(np.asarray(v, float)) for k, v in (by_role or {}).items() if v is not None}

    def probabilities(self, ctx):
        base = self.inner.probabilities(ctx)
        if not self.m:
            return base
        starter = str(getattr(ctx.pitcher, 'role', '') or '') == 'starter'
        key = ('starter_' + str(min(max(int(getattr(ctx, 'times_through_order', 1) or 1), 1), 3))) if starter else 'reliever'
        m = self.m.get(key)
        if m is None and starter:
            m = self.m.get('starter')
        if m is None:
            return base
        p = np.array([base[k] for k in SIM_LABELS], dtype=float) * m
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
