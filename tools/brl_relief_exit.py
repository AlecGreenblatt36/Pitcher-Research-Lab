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
from research_lab.game_sim.relief_exit import MID, COMMON     # noqa: E402

from brl_replay.relief_decisions import COLS, decisions, design      # noqa: E402,F401  (one implementation, shared)


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
