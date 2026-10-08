"""Fit which reliever enters (BULLPEN-01): a conditional logit over each real relief entry's choice set.

usage: python tools/brl_reliever_choice.py HISTORY.csv.gz OUT.json --fit 2023,2024,2025 [--test 2026]

HISTORY is the plate-appearance history (the seed data's plate_appearances.csv.gz layout). For every regular-season relief
entry, the choice set is the team's relievers from the replay's bullpen rule (brl_replay.harness.bullpen: relievers used
in the 14 days before the game, or in the team's last 10 games when fewer than six) minus the game's starter and the
relievers already used in that game; entries whose reliever is outside the set are counted and left out. Candidate facts
come from brl_live.live_feed.bullpen_usage and the features from research_lab.game_sim.reliever_choice.features, the code
the simulator serves with, so the fit and the simulator read a reliever the same way. Only coefficients and aggregate
scores are written.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime')); sys.path.insert(0, str(ROOT))
from brl_live.live_feed import bullpen_usage                                   # noqa: E402
from brl_replay.harness import appearances                                    # noqa: E402
from research_lab.game_sim.reliever_choice import features, situation         # noqa: E402

NAMES = ['log_share14', 'log_share30', 'pitched_d1', 'back_to_back', 'apps_d3', 'days_since', 'same_hand',
         'ninth_x_save', 'ninth_x_tie_late', 'ninth_x_before8', 'late_x_late_close', 'late_x_early', 'late_x_blowout',
         'share_x_late_close', 'share_x_blowout', 'medbf_x_early', 'medbf_x_blowout', 'medbf_x_mid', 'same_hand_x_mid',
         'quality', 'quality_x_late_close', 'quality_x_save', 'quality_x_blowout', 'ninth_x_extra',
         'same3', 'same3_x_late', 'same3_x_mid', 'ninth21_x_save', 'late21_x_late_close']


def choice_rows(h: pd.DataFrame, seasons: set) -> tuple[list, dict]:
    """(entries, counts): each entry is (season, X matrix over its candidates, index of the chosen one)."""
    h = h[h['game_type'] == 'R'].sort_values(['game_pk', 'at_bat_number']).reset_index(drop=True)
    app = appearances(h)
    app['date'] = app['date'].astype(str).str[:10]
    first_ab = h.groupby(['game_pk', 'pitcher'])['at_bat_number'].min().rename('first_ab').reset_index()
    app = app.merge(first_ab, on=['game_pk', 'pitcher'], how='left')
    top = h['inning_topbot'].astype(str).str.lower().str.startswith('top')
    h = h.assign(team=np.where(top, h['home_team'], h['away_team']),
                 runners=h['on_1b'].notna().astype(int) + h['on_2b'].notna().astype(int) + h['on_3b'].notna().astype(int))
    first = h.groupby(['game_pk', 'pitcher'], sort=False).first().reset_index()
    first_key = {(int(g), int(p)): r for g, p, r in zip(first['game_pk'], first['pitcher'], first.itertuples(index=False))}
    # How each hitter bats: switch when the less common side is at least 10% of his plate appearances.
    sides = h.groupby(['batter', 'stand']).size().unstack(fill_value=0)
    minor = sides.min(axis=1) / sides.sum(axis=1)
    bats = {int(b): ('S' if minor.loc[b] >= 0.10 else str(sides.loc[b].idxmax())) for b in sides.index}
    throws = app.groupby('pitcher')['throws'].agg(lambda s: s.mode().iloc[0]).to_dict()
    # The hitters due up when a reliever enters: for each of the next three lineup spots, the hitter who last batted
    # from that spot before the entry (a pinch hitter sent up against the new pitcher does not count).
    bat_team = np.where(top, h['away_team'], h['home_team'])
    h = h.assign(bat_team=bat_team, pa_idx=h.groupby(['game_pk', bat_team]).cumcount())
    order = {k: g['batter'].to_numpy() for k, g in h.groupby(['game_pk', 'bat_team'])}
    pa_pos = {(int(g), int(ab)): (bt, int(i)) for g, ab, bt, i in zip(h['game_pk'], h['at_bat_number'], h['bat_team'], h['pa_idx'])}

    def due_up(gpk, ab):
        bt, j = pa_pos[(int(gpk), int(ab))]
        seq = order[(int(gpk), bt)]
        out = []
        for o in range(3):
            i = j + o - 9 if j + o - 9 >= 0 else j + o
            if i < len(seq):
                out.append(bats.get(int(seq[i]), 'R'))
        return out or ['R']
    entries, counts = [], {'entries': 0, 'outside_set': 0, 'small_set': 0}
    for (gpk, team), ga in app.groupby(['game_pk', 'team']):
        date = str(ga['date'].iloc[0])
        if int(date[:4]) not in seasons:
            continue
        team_app = app[(app['team'] == team) & (app['date'] < date)]
        window = (pd.Timestamp(date) - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
        starter = ga.loc[ga['start'], 'pitcher']
        starter = int(starter.iloc[0]) if len(starter) else -1
        recent = team_app[(team_app['date'] >= window) & (~team_app['start']) & (team_app['pitcher'] != starter)]
        if recent['pitcher'].nunique() < 6:
            last = team_app.sort_values('date')['game_pk'].drop_duplicates().tail(10)
            recent = team_app[team_app['game_pk'].isin(last) & (~team_app['start']) & (team_app['pitcher'] != starter)]
        pool = [int(p) for p in recent['pitcher'].unique()]
        if not pool:
            continue
        facts = bullpen_usage(app, team, date, pool)
        used = set()
        for _, e in ga[~ga['start']].sort_values('first_ab').iterrows():
            chosen = int(e['pitcher'])
            counts['entries'] += 1
            avail = [p for p in pool if p not in used]
            if chosen not in avail:
                counts['outside_set'] += 1
            elif len(avail) < 2:
                counts['small_set'] += 1
            else:
                r = first_key[(int(gpk), chosen)]
                lead = int(r.fld_score - r.bat_score)
                hands = due_up(gpk, r.at_bat_number)
                same = [hands[0] != 'S' and throws.get(p, 'R') == hands[0] for p in avail]
                same3 = [float(sum(x != 'S' and throws.get(p, 'R') == x for x in hands)) for p in avail]
                cols = features([facts[p] for p in avail], same, situation(int(r.inning), lead, int(r.outs_when_up), int(r.runners)), same3)
                X = np.column_stack([cols[n] for n in NAMES])
                entries.append((int(date[:4]), X, avail.index(chosen)))
            used.add(chosen)
    return entries, counts


def fit(entries: list) -> np.ndarray:
    X = np.vstack([e[1] for e in entries]); g = np.concatenate([[i] * len(e[1]) for i, e in enumerate(entries)])
    y = np.zeros(len(X)); off = 0
    for e in entries:
        y[off + e[2]] = 1.0; off += len(e[1])
    starts = np.r_[0, np.cumsum([len(e[1]) for e in entries])[:-1]]

    def nll(beta):
        u = X @ beta
        m = np.maximum.reduceat(u, starts)
        ex = np.exp(u - np.repeat(m, np.diff(np.r_[starts, len(u)])))
        s = np.add.reduceat(ex, starts)
        p = ex / np.repeat(s, np.diff(np.r_[starts, len(u)]))
        val = -((u - np.repeat(m + np.log(s), np.diff(np.r_[starts, len(u)]))) * y).sum()
        return val, -(X * (y - p)[:, None]).sum(0)
    return minimize(nll, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 1000}).x


def score(entries: list, beta: np.ndarray) -> dict:
    ll, hit, unif = [], [], []
    for _, X, c in entries:
        u = X @ beta; u = u - u.max(); p = np.exp(u) / np.exp(u).sum()
        ll.append(np.log(p[c])); hit.append(float(np.argmax(u) == c)); unif.append(-np.log(len(u)))
    return {'entries': len(entries), 'log_lik_per_entry': round(float(np.mean(ll)), 4), 'uniform': round(float(np.mean(unif)), 4),
            'top1': round(float(np.mean(hit)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('history'); ap.add_argument('out'); ap.add_argument('--fit', required=True); ap.add_argument('--test', default='')
    a = ap.parse_args()
    fit_s = {int(x) for x in a.fit.split(',')}; test_s = {int(x) for x in a.test.split(',') if x}
    t0 = time.time()
    cols = ['game_pk', 'date_key', 'at_bat_number', 'inning', 'inning_topbot', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'bat_score',
            'fld_score', 'batter', 'pitcher', 'stand', 'p_throws', 'home_team', 'away_team', 'game_type', 'outcome']
    h = pd.read_csv(a.history, usecols=cols, low_memory=False)
    entries, counts = choice_rows(h, fit_s | test_s)
    train = [e for e in entries if e[0] in fit_s]; test = [e for e in entries if e[0] in test_s]
    beta = fit(train)
    doc = {'schema': 'brl.reliever-choice.v1', 'name': 'reliever choice fitted on ' + ', '.join(map(str, sorted(fit_s))),
           'fitted_on': sorted(fit_s), 'names': NAMES, 'beta': [round(float(b), 5) for b in beta], 'counts': counts,
           'train': score(train, beta), 'test': ({'seasons': sorted(test_s), **score(test, beta)} if test else None),
           'seconds': round(time.time() - t0)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({k: doc[k] for k in ('name', 'counts', 'train', 'test', 'seconds')}, indent=1))
    for n, b in zip(NAMES, beta):
        print(f'  {n:22s} {b:+.3f}')


if __name__ == '__main__':
    main()
