"""Relief decision points from the plate-appearance history (RELIEF-02, RELIEF-03).

decisions(): one row per plate appearance a reliever completes in a regular-season game that goes on, with whether the
next batter of his opponents faced someone else (removed), whether the inning ended, and the facts the fitted exit
hazard reads (research_lab.game_sim.relief_exit). design() builds its feature matrix with the engine's own feature code.

HookOffsets (RELIEF-03): each team's tendency to pull relievers beyond the fitted hazard, on the log-odds scale, for the
middle of an inning and for an inning's end: the decayed sum of observed minus fitted removals over the decayed sum of
p(1 - p), shrunk by k units of that information, from decision points strictly before the date (a one-step estimate
of a team offset in the logistic model). Team-season residuals are stable (odd/even games 0.76 and 0.79, season to
season 0.3 to 0.66).
"""
from __future__ import annotations

from datetime import date as _date

import numpy as np
import pandas as pd

from research_lab.game_sim.relief_exit import COMMON, MID, features

COLS = ['game_pk', 'date_key', 'at_bat_number', 'pitcher', 'inning', 'inning_topbot', 'outs_when_up', 'runner_1b', 'runner_2b',
        'runner_3b', 'bat_score', 'fld_score', 'home_team', 'away_team', 'game_type']


def decisions(h: pd.DataFrame) -> pd.DataFrame:
    h = h[h['game_type'] == 'R'].sort_values(['game_pk', 'at_bat_number']).reset_index(drop=True)
    h['date'] = h['date_key'].astype(str).str[:10]
    h['season'] = h['date'].str[:4].astype(int)
    top = h['inning_topbot'].astype(str).str.lower().str.startswith('top')
    h['bat'] = np.where(top, 'away', 'home')
    h['team'] = np.where(top, h['home_team'], h['away_team'])          # the fielding team
    h['on'] = h[['runner_1b', 'runner_2b', 'runner_3b']].fillna(0).astype(bool).sum(axis=1)
    g = h.groupby(['game_pk', 'bat'], sort=False)
    for c in ('pitcher', 'inning', 'outs_when_up', 'on', 'bat_score', 'fld_score'):
        h['n_' + c] = g[c].shift(-1)
    h['starter'] = g['pitcher'].transform('first')
    rel = h[(h['pitcher'] != h['starter'])].copy()
    k = rel.groupby(['game_pk', 'bat', 'pitcher'])
    rel['bf'] = k.cumcount() + 1
    rel['entry_score'] = k['bat_score'].transform('first')
    rel['inherited'] = k['on'].transform('first')
    rel['entry_inning'] = k['inning'].transform('first')
    rel = rel[rel['n_pitcher'].notna()].copy()                          # the game went on
    rel['runs'] = np.maximum(0, rel['n_bat_score'] - rel['entry_score'] - rel['inherited'])
    rel['ended'] = (rel['n_inning'] != rel['inning']).astype(int)
    rel['removed'] = (rel['n_pitcher'] != rel['pitcher']).astype(int)
    rel['lead'] = rel['n_fld_score'] - rel['n_bat_score']
    # usual length (any team) and role with this team, from relief outings in the previous 365 days
    outings = rel.groupby(['game_pk', 'pitcher']).agg(date=('date', 'first'), team=('team', 'first'), bf=('bf', 'max'),
                                                       inn=('entry_inning', 'first')).reset_index()
    outings['day'] = pd.to_datetime(outings['date']).values.astype('datetime64[D]').astype(np.int64)
    exp, ninth, late = {}, {}, {}
    for pid, o in outings.sort_values('day').groupby('pitcher'):
        day = o['day'].to_numpy(); bf = o['bf'].to_numpy(float); inn = o['inn'].to_numpy(); team = o['team'].to_numpy()
        for i, gpk in enumerate(o['game_pk'].to_numpy()):
            m = (day < day[i]) & (day >= day[i] - 365)
            exp[(gpk, pid)] = int(max(3, round(float(np.median(bf[m]))))) if m.any() else 4
            mt = m & (team == team[i])
            ninth[(gpk, pid)] = float((inn[mt] >= 9).mean()) if mt.any() else 0.0
            late[(gpk, pid)] = float((inn[mt] >= 8).mean()) if mt.any() else 0.0
    key = list(zip(rel['game_pk'], rel['pitcher']))
    rel['exp'] = [exp[x] for x in key]; rel['ninth'] = [ninth[x] for x in key]; rel['late'] = [late[x] for x in key]
    rel['n_on'] = rel['n_on'].fillna(0); rel['n_outs_when_up'] = rel['n_outs_when_up'].fillna(0)
    return rel


def design(d: pd.DataFrame, names) -> np.ndarray:
    rows = [features(bf, ex, r, int(inn), ld, nn, lt, on, min(int(o), 2)) for bf, ex, r, inn, ld, nn, lt, on, o in
            zip(d['bf'], d['exp'], d['runs'], d['inning'], d['lead'], d['ninth'], d['late'], d['n_on'], d['n_outs_when_up'])]
    return np.array([[x[n] for n in names] for x in rows], float)


def fitted(d: pd.DataFrame, doc: dict) -> np.ndarray:
    """The exit hazard's probability at each legal decision point of d (mid-inning rows with fewer than three batters
    faced get nan)."""
    p = np.full(len(d), np.nan)
    for key, names, ended in (('mid', MID, 0), ('end', COMMON, 1)):
        m = ((d['ended'] == ended) & ((d['bf'] >= 3) | (ended == 1))).to_numpy()
        if m.any():
            z = float(doc[key]['intercept']) + design(d[m], names) @ np.asarray(doc[key]['beta'], float)
            p[m] = 1.0 / (1.0 + np.exp(-z))
    return p


def _ordinal(s) -> int:
    return _date.fromisoformat(str(s)[:10]).toordinal()


class HookOffsets:
    """Team offsets on the exit hazard's log-odds from decision points before a date (RELIEF-03)."""

    def __init__(self, d: pd.DataFrame, p: np.ndarray, half_life: float = 180.0, k: float = 30.0):
        ok = np.isfinite(p)
        t = pd.DataFrame({'team': d['team'].to_numpy()[ok], 'day': [_ordinal(x) for x in d['date'].to_numpy()[ok]],
                          'end': d['ended'].to_numpy()[ok], 'res': d['removed'].to_numpy()[ok] - p[ok], 'info': p[ok] * (1 - p[ok])})
        agg = t.groupby(['team', 'day', 'end'])[['res', 'info']].sum().reset_index()
        self.by = {}
        for (team, end), g in agg.groupby(['team', 'end']):
            self.by[(str(team), int(end))] = (g['day'].to_numpy(np.int64), g['res'].to_numpy(float), g['info'].to_numpy(float))
        self.half_life, self.k = float(half_life), float(k)

    def at(self, team, game_date) -> dict:
        day = _ordinal(game_date)
        out = {}
        for end, key in ((0, 'mid'), (1, 'end')):
            days, res, info = self.by.get((str(team), end), (np.zeros(0, np.int64), np.zeros(0), np.zeros(0)))
            m = days < day
            w = np.power(0.5, (day - days[m]) / self.half_life)
            out[key] = float((w * res[m]).sum() / ((w * info[m]).sum() + self.k))
        return out

    def table(self, game_date) -> dict:
        teams = sorted({t for t, _ in self.by})
        return {t: self.at(t, game_date) for t in teams}
