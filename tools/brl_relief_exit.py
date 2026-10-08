"""Fit when relievers come out (RELIEF-02): logistic hazards over every real relief decision point.

usage: python tools/brl_relief_exit.py HISTORY.csv.gz OUT.json --fit 2023,2024,2025 [--test 2026]

HISTORY is the plate-appearance history (the seed data's plate_appearances.csv.gz layout). After each plate appearance a
reliever completes in a regular-season game that goes on, the decision is whether the next batter of his team's
opponents faces someone else; decisions at an inning's end and in the middle of an inning are fitted separately, the
latter only where a change is allowed (three batters faced or more). Facts: batters faced in the outing; runs charged to
him, taken as the runs scored while he pitched minus the runners he inherited (at least zero); his median batters per
relief outing over the previous 365 days (any team; 4 without one, at least 3); his share of relief entries for this
team in the ninth or later and in the eighth or later over the previous 365 days; the inning, the fielding team's lead
after the plate appearance and, in the middle of an inning, runners and outs. Features come from
research_lab.game_sim.relief_exit.features, the code the simulator serves with. The test seasons are scored against the
engine's hand-set rule on the same facts. Only coefficients and aggregate scores are written.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
from research_lab.game_sim.relief_exit import MID, COMMON, features     # noqa: E402

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


def hand_set(d: pd.DataFrame) -> np.ndarray:
    """The engine's hand-set reliever rule (manager.ManagerPolicy.should_remove) on the same facts."""
    bf = d['bf'].to_numpy(float); exp = d['exp'].to_numpy(float); runs = d['runs'].to_numpy(float); inn = d['inning'].to_numpy(float)
    ended = d['ended'].to_numpy() == 1
    maxb = np.maximum(exp + 3, 6)
    role = np.where(d['ninth'] >= 0.6, 'closer', np.where(d['late'] >= 0.5, 'setup', 'other'))
    close = np.maximum(0.0, 1.0 - np.minimum(np.abs(d['lead'].to_numpy(float)), 6) / 6.0)
    runners = np.where(ended, 0, d['n_on'].to_numpy(float)); outs = np.where(ended, 2, d['n_outs_when_up'].to_numpy(float))
    lev = np.minimum(1.0, 0.15 + 0.48 * np.clip((inn - 3) / 6.0, 0.05, 1.0) * close + 0.22 * runners / 3.0 + 0.15 * (2 - np.minimum(outs, 2)) / 2.0)
    fat = np.clip(bf / maxb - 0.2, 0.0, 1.5)
    press = (-0.45 + 1.10 * (bf - exp) / np.maximum(2.5, exp * 0.22) + 0.42 * runs + 0.92 * fat + 0.38 * lev + 0.75 * ended
             + 0.25 * (np.isin(role, ['closer', 'setup']) & (inn < 7)))
    p = 1 / (1 + np.exp(-press))
    p = np.where(bf >= maxb, 1.0, p)
    return np.where(~ended & (bf < np.minimum(3, exp)), 0.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('history'); ap.add_argument('out'); ap.add_argument('--fit', required=True); ap.add_argument('--test', default='')
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    fit_s = {int(x) for x in a.fit.split(',')}; test_s = {int(x) for x in a.test.split(',') if x}
    d = decisions(pd.read_csv(a.history, usecols=COLS, low_memory=False))
    d = d[(d['ended'] == 1) | (d['bf'] >= 3)]                           # where a change is allowed
    doc = {'schema': 'brl.relief-exit.v1', 'name': 'reliever exits fitted on ' + ', '.join(map(str, sorted(fit_s))),
           'fitted_on': sorted(fit_s), 'decisions': {}, 'test': {}}
    for key, names, ended in (('mid', MID, 0), ('end', COMMON, 1)):
        sub = d[d['ended'] == ended]
        tr = sub[sub['season'].isin(fit_s)]
        m = LogisticRegression(C=1.0, max_iter=5000).fit(design(tr, names), tr['removed'])
        doc[key] = {'names': list(names), 'intercept': round(float(m.intercept_[0]), 6), 'beta': [round(float(b), 6) for b in m.coef_[0]]}
        doc['decisions'][key] = {'fit': int(len(tr)), 'fit_removal_rate': round(float(tr['removed'].mean()), 4)}
        te = sub[sub['season'].isin(test_s)]
        if len(te):
            p = np.clip(m.predict_proba(design(te, names))[:, 1], 1e-4, 1 - 1e-4); y = te['removed'].to_numpy()
            ph = np.clip(hand_set(te), 1e-4, 1 - 1e-4)
            doc['test'][key] = {'decisions': int(len(te)), 'actual': round(float(y.mean()), 4), 'fitted': round(float(p.mean()), 4),
                                'hand_set': round(float(ph.mean()), 4),
                                'log_lik_fitted': round(float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
                                'log_lik_hand_set': round(float(np.mean(y * np.log(ph) + (1 - y) * np.log(1 - ph))), 4)}
    doc['seconds'] = round(time.time() - t0)
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({k: doc[k] for k in ('name', 'decisions', 'test', 'seconds')}, indent=1))


if __name__ == '__main__':
    main()
