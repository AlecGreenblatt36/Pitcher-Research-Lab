"""Team-strength win chance: negative-binomial runs with shrunk, decayed offense and defense.

For a game on date D every input is a regular-season final before D. Each team gets an
offense and a defense multiplier from runs scored and allowed against the league rate,
weighted by exp(-age / tau) and pulled toward one by k league-average pseudo games; home
field is the prior-data home/away run ratio split evenly between the two sides; run totals
are negative binomial with a dispersion fitted on the last three seasons; nine-inning ties
go to the home side at the prior one-run-game home share. tau = 180 days and k = 15 were
chosen on 2025 and frozen; on the 2026 out-of-sample replay this scored 0.24576 Brier
(1.7% better than a coin flip) and is the team half of the headline blend.
"""
from __future__ import annotations

from math import lgamma, log

import numpy as np

MAXR = 30
TAU_DAYS = 180.0
K_PSEUDO_GAMES = 15.0
VERSION = 'team-nb-decay-v1 (tau 180 days, k 15, tuned on 2025)'


def nb_pmf(mu: float, alpha: float) -> np.ndarray:
    r = 1.0 / alpha
    p = r / (r + mu)
    k = np.arange(MAXR + 1)
    logpmf = np.array([lgamma(x + r) - lgamma(r) - lgamma(x + 1) for x in k]) + r * log(p) + k * log(1 - p)
    pmf = np.exp(logpmf)
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def win_prob(mu_h: float, mu_a: float, alpha: float, home_extra: float) -> tuple[float, float]:
    ph, pa = nb_pmf(mu_h, alpha), nb_pmf(mu_a, alpha)
    joint = np.outer(ph, pa)
    p_home = float(np.tril(joint, -1).sum())
    p_tie = float(np.trace(joint))
    return p_home + p_tie * home_extra, p_tie


def _days(a: str, b: str) -> int:
    from datetime import date
    return (date.fromisoformat(str(a)[:10]) - date.fromisoformat(str(b)[:10])).days


class TeamModel:
    def __init__(self, rows, tau: float = TAU_DAYS, k: float = K_PSEUDO_GAMES):
        self.tau, self.k = float(tau), float(k)
        self.rows = []
        for r in rows:
            try:
                self.rows.append({'date': str(r['date'])[:10], 'home': str(r.get('home_id', r.get('home'))), 'away': str(r.get('away_id', r.get('away'))),
                                  'home_runs': float(r['home_runs']), 'away_runs': float(r['away_runs'])})
            except (KeyError, TypeError, ValueError):
                continue
        self.rows.sort(key=lambda r: r['date'])
        self._cache = {}

    def _prior(self, date: str) -> list:
        return [r for r in self.rows if r['date'] < str(date)[:10]]

    def ratings(self, date: str):
        if date in self._cache:
            return self._cache[date]
        prior = self._prior(date)
        if len(prior) < 30:
            raise ValueError('Too few prior games for team ratings')
        age = np.array([_days(date, r['date']) for r in prior], float)
        w = np.exp(-age / self.tau)
        teams = sorted({r['home'] for r in prior} | {r['away'] for r in prior})
        idx = {t: i for i, t in enumerate(teams)}
        hi = np.array([idx[r['home']] for r in prior]); ai = np.array([idx[r['away']] for r in prior])
        hr = np.array([r['home_runs'] for r in prior]); ar = np.array([r['away_runs'] for r in prior])
        n = len(teams)
        L = float((w * (hr + ar)).sum() / (2 * w.sum()))
        H = float((w * hr).sum() / max(1e-9, (w * ar).sum()))
        hf, af = np.sqrt(H), 1 / np.sqrt(H)
        off = np.ones(n); dfn = np.ones(n); k = self.k
        for _ in range(4):
            num = np.bincount(hi, w * hr, n) + np.bincount(ai, w * ar, n) + k * L
            den = np.bincount(hi, w * L * hf * dfn[ai], n) + np.bincount(ai, w * L * af * dfn[hi], n) + k * L
            off = num / den
            num = np.bincount(hi, w * ar, n) + np.bincount(ai, w * hr, n) + k * L
            den = np.bincount(hi, w * L * af * off[ai], n) + np.bincount(ai, w * L * hf * off[hi], n) + k * L
            dfn = num / den
        recent = [r for r in prior if _days(date, r['date']) <= 3 * 365]
        runs = np.array([r['home_runs'] for r in recent] + [r['away_runs'] for r in recent])
        m, v = runs.mean(), runs.var()
        alpha = max(0.01, (v - m) / (m * m))
        one = [r for r in recent if abs(r['home_runs'] - r['away_runs']) == 1]
        home_extra = float(np.mean([r['home_runs'] > r['away_runs'] for r in one])) if one else 0.52
        out = ({t: (float(off[i]), float(dfn[i])) for t, i in idx.items()}, L, float(hf), float(af), float(alpha), home_extra, len(prior))
        self._cache[date] = out
        return out

    def probability(self, date: str, home_id, away_id) -> dict:
        rat, L, hf, af, alpha, home_extra, n = self.ratings(date)
        oh, dh = rat.get(str(home_id), (1.0, 1.0))
        oa, da = rat.get(str(away_id), (1.0, 1.0))
        mu_h, mu_a = L * hf * oh * da, L * af * oa * dh
        p, tie = win_prob(mu_h, mu_a, alpha, home_extra)
        return {'version': VERSION, 'p_home': round(float(p), 6), 'mu_home': round(float(mu_h), 4), 'mu_away': round(float(mu_a), 4),
                'alpha': round(float(alpha), 5), 'home_extra': round(home_extra, 4), 'nine_inning_tie_share': round(float(tie), 5),
                'league_runs': round(float(L), 4), 'prior_games': n, 'as_of': str(date)[:10],
                'ratings': {'home': {'offense': round(oh, 4), 'defense': round(dh, 4)}, 'away': {'offense': round(oa, 4), 'defense': round(da, 4)}}}
