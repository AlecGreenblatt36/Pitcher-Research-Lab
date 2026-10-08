"""Fit the starter-leash table (LEASH-01) from a full replay's starters, simulated against actual batters faced.

usage: python tools/brl_leash.py HISTORY.csv.gz OUT.json TAG [TAG ...]

TAG names a replay on the brl-live-data branch (research/replay-TAG-*.jsonl.gz rows with starter lines). For each start
the residual is the actual batters faced minus the simulated mean; it is regressed on the facts of
research_lab.game_sim.starter_leash (computed from HISTORY's appearances before the game's date: short rest, a relief
outing before, the first and second start back from a layoff, March, September) with an intercept for a regular start.
The table keeps the coefficients and RESPONSE, the fitted hazard's mean batters faced at each expected value (measured
with league-average hitting and the team expectation at the league mean, 600 games per point, against the empirical
base-running kernel). Only coefficients and counts are written.
"""
from __future__ import annotations
import argparse, gzip, json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
from brl_replay.harness import appearances                                    # noqa: E402
from research_lab.game_sim.starter_leash import TERMS, AppearanceIndex, indicators  # noqa: E402

RESPONSE = [[8, 3.40], [10, 3.90], [12, 5.06], [14, 7.68], [16, 11.83], [18, 15.85], [19, 17.79], [20, 19.34], [21, 20.61],
            [22, 21.63], [23, 22.56], [24, 23.49], [25, 24.41], [27, 26.10], [29, 27.72]]


def replay_rows(tag: str) -> list:
    names = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', 'origin/brl-live-data', 'research'],
                           capture_output=True, text=True, check=True).stdout.split()
    out = {}
    for f in sorted(n for n in names if n.startswith(f'research/replay-{tag}-') and n.endswith('.jsonl.gz')):
        raw = subprocess.run(['git', '-C', str(ROOT), 'show', 'origin/brl-live-data:' + f], capture_output=True, check=True).stdout
        for line in gzip.decompress(raw).decode().splitlines():
            if line.strip():
                d = json.loads(line); out[d['game_pk']] = d
    return list(out.values())


def starts(rows: list, index: AppearanceIndex) -> pd.DataFrame:
    recs = []
    for d in rows:
        for side in ('home', 'away'):
            s = (d.get('starters') or {}).get(side) or {}
            a = s.get('actual') or {}
            if not s.get('bf') or not a:
                continue
            h = np.asarray(s['bf'], float)
            sim = float(np.dot(np.arange(len(h)), h) / h.sum())
            ind = indicators(index.facts(int(s['pitcher']), d['date']))
            recs.append({'date': d['date'], 'sim': sim, 'act': float(a['bf']), **ind})
    return pd.DataFrame(recs)


def fit(df: pd.DataFrame) -> tuple[dict, dict]:
    X = df[list(TERMS)].to_numpy(float); y = (df['act'] - df['sim']).to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * resid.var())
    return ({k: round(float(v), 4) for k, v in zip(TERMS, b)}, {k: round(float(v), 4) for k, v in zip(TERMS, se)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('history'); ap.add_argument('out'); ap.add_argument('tags', nargs='+')
    a = ap.parse_args()
    h = pd.read_csv(a.history, low_memory=False)
    h = h[h['game_type'] == 'R']
    app = appearances(h)
    index = AppearanceIndex.from_frame(app)
    df = pd.concat([starts(replay_rows(t), index) for t in a.tags], ignore_index=True)
    terms, se = fit(df)
    groups = {k: int(df[k].sum()) for k in TERMS if k != 'base'}
    doc = {'schema': 'brl.starter-leash.v1', 'name': 'starter leash from the ' + ', '.join(a.tags) + ' replay starts',
           'replays': a.tags, 'starts': int(len(df)), 'groups': groups, 'terms': terms, 'se': se,
           'sim_minus_actual_bf': round(float((df['sim'] - df['act']).mean()), 4), 'response': RESPONSE}
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({k: doc[k] for k in ('starts', 'groups', 'terms', 'se', 'sim_minus_actual_bf')}, indent=1))


if __name__ == '__main__':
    main()
