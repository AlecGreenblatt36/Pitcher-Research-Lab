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
VERSION_SP = 'team-nb-decay-sp-v2 (starter adjustment: fip value, k 300, beta 1.0 x expected share; tuned on 2025)'

# Starting-pitcher adjustment (TEAM-02, LEDGER.md). A pitcher's run value allowed per plate appearance over
# the prior 365 days, from strikeouts, walks and hit batters, and home runs shrunk separately toward the
# league (k 120, 240 and 600 plate appearances) with balls in play at the league value; the opponent's
# expected runs are multiplied by (value / rotation) ** (BETA x share), where rotation is the decayed
# average over the team's prior starts of its starters' current values (the team defense already holds
# the typical starter) and share is the starter's median batters faced in his starts over the prior
# 365 days divided by 22 (0.25 to 1.25; 1.0 with fewer than three starts, 0.6 with none). Chosen on 2025
# (team-model Brier 0.24354 to 0.24304) and frozen; on 2026 the team model went from 0.24576 to 0.24497
# (-0.00079 [-0.00149, -0.00010]) and the blend with the simulator from 0.24442 to 0.24425
# (-0.00017 [-0.00051, +0.00017]).
SP_WEIGHTS = {'BB_HBP': 0.69, '1B': 0.88, '2B_3B': 1.28, 'HR': 2.0, 'OTHER_REACH': 0.80, 'K': 0.0, 'BIP_OUT': 0.0}
SP_K, SP_BETA, SP_TYPICAL_BF = 300.0, 1.0, 22.0


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

    def probability(self, date: str, home_id, away_id, starters: dict | None = None) -> dict:
        """starters: StarterAdjust.factors output; the away starter's factor scales the home team's runs."""
        rat, L, hf, af, alpha, home_extra, n = self.ratings(date)
        oh, dh = rat.get(str(home_id), (1.0, 1.0))
        oa, da = rat.get(str(away_id), (1.0, 1.0))
        mu_h, mu_a = L * hf * oh * da, L * af * oa * dh
        if starters:
            mu_h *= starters['away']['factor']; mu_a *= starters['home']['factor']
        p, tie = win_prob(mu_h, mu_a, alpha, home_extra)
        out = self._out(p, tie, mu_h, mu_a, alpha, home_extra, L, n, date, oh, dh, oa, da)
        if starters:
            out['version'] = VERSION_SP; out['starter_adjust'] = starters
        return out

    def _out(self, p, tie, mu_h, mu_a, alpha, home_extra, L, n, date, oh, dh, oa, da) -> dict:
        return {'version': VERSION, 'p_home': round(float(p), 6), 'mu_home': round(float(mu_h), 4), 'mu_away': round(float(mu_a), 4),
                'alpha': round(float(alpha), 5), 'home_extra': round(home_extra, 4), 'nine_inning_tie_share': round(float(tie), 5),
                'league_runs': round(float(L), 4), 'prior_games': n, 'as_of': str(date)[:10],
                'ratings': {'home': {'offense': round(oh, 4), 'defense': round(dh, 4)}, 'away': {'offense': round(oa, 4), 'defense': round(da, 4)}}}


class StarterAdjust:
    """Starter values, expected shares and team rotation baselines from the plate-appearance history.

    history: PA rows (date_key, game_pk, at_bat_number, pitcher, inning_topbot, home_team, away_team,
    outcome); rows: the team model's game rows (game_pk, date, home and away keys). Starters per game
    come from the history (first pitcher of each fielding side) and are matched to the rows by game_pk
    and side, so the rows may key teams by id or by abbreviation."""

    def __init__(self, history, rows, tau: float = TAU_DAYS, k: float = SP_K, beta: float = SP_BETA):
        import pandas as pd
        self.tau, self.k, self.beta = float(tau), float(k), float(beta)
        h = history[['date_key', 'game_pk', 'at_bat_number', 'pitcher', 'inning_topbot', 'outcome']].copy()
        h['date_key'] = h['date_key'].astype(str).str[:10]
        h['top'] = h['inning_topbot'].astype(str).str.lower().str.startswith('top')
        h['v'] = h['outcome'].map(SP_WEIGHTS).astype(float)
        for name, label in (('K', 'K'), ('BB', 'BB_HBP'), ('HR', 'HR')):
            h[name] = (h['outcome'] == label).astype(float)
        h['bip'] = 1.0 - h['K'] - h['BB'] - h['HR']
        h['bip_v'] = h['v'] * h['bip']
        self.daily = h.groupby(['pitcher', 'date_key']).agg(n=('v', 'size'), K=('K', 'sum'), BB=('BB', 'sum'), HR=('HR', 'sum'),
                                                            bip=('bip', 'sum'), bip_v=('bip_v', 'sum')).reset_index()
        # starters: the first plate appearance of each fielding side (top: the home team fields)
        first = h.sort_values(['game_pk', 'at_bat_number']).groupby(['game_pk', 'top']).first().reset_index()
        bf = h.groupby(['game_pk', 'pitcher']).size().rename('bf').reset_index()
        first = first.merge(bf, on=['game_pk', 'pitcher'], how='left')
        self.starts = {}
        for r in first.itertuples():
            self.starts.setdefault(int(r.game_pk), {})['home' if r.top else 'away'] = (int(r.pitcher), str(r.date_key), int(r.bf))
        self.start_rows = [(d, p, b) for g in self.starts.values() for (p, d, b) in g.values()]
        self.games = []
        for r in rows:
            try:
                self.games.append((str(r['date'])[:10], int(r['game_pk']), str(r.get('home_id', r.get('home'))), str(r.get('away_id', r.get('away')))))
            except (KeyError, TypeError, ValueError):
                continue
        self._cache = {}

    def _values(self, date: str):
        """(value by pitcher, league value, share by pitcher) as of date (inputs strictly before it)."""
        if date in self._cache:
            return self._cache[date]
        import pandas as pd
        lo = (pd.Timestamp(date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        win = self.daily[(self.daily['date_key'] < date) & (self.daily['date_key'] >= lo)]
        lg = win[['n', 'K', 'BB', 'HR', 'bip', 'bip_v']].sum()
        n_l = max(float(lg['n']), 1.0)
        rk, rb, rh = float(lg['K']) / n_l, float(lg['BB']) / n_l, float(lg['HR']) / n_l
        bip_l = float(lg['bip_v']) / max(float(lg['bip']), 1.0)
        league = rb * SP_WEIGHTS['BB_HBP'] + rh * SP_WEIGHTS['HR'] + max(0.0, 1 - rk - rb - rh) * bip_l
        per = win.groupby('pitcher')[['n', 'K', 'BB', 'HR']].sum()
        k = self.k
        n = per['n'].to_numpy(float)
        kr = (per['K'].to_numpy(float) + 0.4 * k * rk) / (n + 0.4 * k)
        bb = (per['BB'].to_numpy(float) + 0.8 * k * rb) / (n + 0.8 * k)
        hr = (per['HR'].to_numpy(float) + 2.0 * k * rh) / (n + 2.0 * k)
        value = bb * SP_WEIGHTS['BB_HBP'] + hr * SP_WEIGHTS['HR'] + np.clip(1 - kr - bb - hr, 0, None) * bip_l
        values = dict(zip((int(p) for p in per.index), (float(v) for v in value)))
        bfs: dict = {}
        for d, p, b in self.start_rows:
            if lo <= d < date:
                bfs.setdefault(p, []).append(b)
        share = {p: (float(min(1.25, max(0.25, float(np.median(b)) / SP_TYPICAL_BF))) if len(b) >= 3 else 1.0) for p, b in bfs.items()}
        out = (values, float(league), share)
        self._cache = {date: out}
        return out

    def _rotation(self, date: str, values: dict, league: float) -> dict:
        acc: dict = {}
        for d, pk, home, away in self.games:
            if d >= date:
                continue
            g = self.starts.get(pk)
            if not g:
                continue
            w = float(np.exp(-_days(date, d) / self.tau))
            for side, team in (('home', home), ('away', away)):
                if side in g:
                    a = acc.setdefault(team, [0.0, 0.0])
                    a[0] += w * values.get(g[side][0], league); a[1] += w
        return {t: a[0] / a[1] for t, a in acc.items() if a[1] > 0}

    def factors(self, date: str, home_key, away_key, home_sp, away_sp, share_scale: float = 1.0) -> dict:
        """Multipliers on the runs each side's opponent is expected to score, with their inputs."""
        values, league, share = self._values(str(date)[:10])
        rotation = self._rotation(str(date)[:10], values, league)
        out = {}
        for side, team, sp in (('home', str(home_key), home_sp), ('away', str(away_key), away_sp)):
            pid = int(sp)
            v = values.get(pid, league)
            base = rotation.get(team, league)
            s = share.get(pid, 0.6) * share_scale         # no start in the prior 365 days: an opener or a reliever
            out[side] = {'pitcher': str(pid), 'value': round(v, 5), 'rotation': round(base, 5), 'share': round(s, 3),
                         'factor': round(float((v / base) ** (self.beta * s)), 5), 'known': pid in values}
        return out
