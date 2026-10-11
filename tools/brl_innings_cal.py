"""NRFI-01 and FIRST5-01: the first inning and the first five innings from the simulator, scored on full replays.

Reads the per-game files of replays that record, over each game's worlds, the first inning's scoring (no run, the road
team scoring, the home team scoring) and the margin and total after five innings (brl_replay/harness.py first_inning,
f5_margin_hist, f5_total_hist), with the real scores by inning (innings.actual_first_scores: each side's score before
its first plate appearance of each inning). For each chance: the mean against the actual share, by fifths, a bootstrap
over dates for the gap, and Brier against a constant. Model outputs and final scores only.

    python3 tools/brl_innings_cal.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

MAX_MARGIN = 15


def actual_innings(g):
    """(road runs in the 1st, home runs in the 1st, road runs through 5, home runs through 5) from the real scores by
    inning, or None where a side's 2nd or 6th inning is missing (a shortened game)."""
    a = ((g.get('innings') or {}).get('actual_first_scores')) or {}
    out = []
    for upto in (2, 6):
        for side in ('away', 'home'):
            d = {int(i): int(v) for i, v in (a.get(side) or [])}
            if upto not in d:
                return None
            out.append(d[upto] - d.get(1, 0))
    return out[0], out[1], out[2], out[3]


def terms(g):
    act = actual_innings(g)
    if act is None or 'first_inning' not in g:
        return None
    n = float(g['n'])
    fi = g['first_inning']
    m = np.asarray(g['f5_margin_hist'], float); k = np.arange(len(m)) - MAX_MARGIN
    t = np.asarray(g['f5_total_hist'], float); kt = np.arange(len(t))
    a1, h1, a5, h5 = act
    return {'date': g['date'],
            'p': {'no_run_1st': fi[0] / n, 'road_scores_1st': fi[1] / n, 'home_scores_1st': fi[2] / n,
                  'home_leads_5': m[k > 0].sum() / m.sum(), 'tied_5': m[k == 0].sum() / m.sum(), 'road_leads_5': m[k < 0].sum() / m.sum(),
                  'over_4_5_thru_5': t[kt >= 5].sum() / t.sum(), 'over_3_5_thru_5': t[kt >= 4].sum() / t.sum()},
            'y': {'no_run_1st': float(a1 + h1 == 0), 'road_scores_1st': float(a1 > 0), 'home_scores_1st': float(h1 > 0),
                  'home_leads_5': float(h5 > a5), 'tied_5': float(h5 == a5), 'road_leads_5': float(h5 < a5),
                  'over_4_5_thru_5': float(a5 + h5 >= 5), 'over_3_5_thru_5': float(a5 + h5 >= 4)}}


def fifths(p, y):
    q = np.quantile(p, [0, .2, .4, .6, .8, 1]); out = []
    for i in range(5):
        msk = (p >= q[i]) & ((p <= q[i + 1]) if i == 4 else (p < q[i + 1]))
        if msk.any():
            out.append([round(float(p[msk].mean()), 4), round(float(y[msk].mean()), 4), int(msk.sum())])
    return out


def season_check(games, draws=2000, seed=1):
    T = [x for x in (terms(g) for g in games) if x is not None]
    res = {'games': len(T)}
    dates = np.array([x['date'] for x in T]); keys, inv = np.unique(dates, return_inverse=True)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    sn = np.bincount(inv, minlength=len(keys)).astype(float)
    for name in T[0]['p'] if T else []:
        p = np.array([x['p'][name] for x in T]); y = np.array([x['y'][name] for x in T])
        sp, sy = np.bincount(inv, weights=p, minlength=len(keys)), np.bincount(inv, weights=y, minlength=len(keys))
        gap = (sp[idx].sum(1) - sy[idx].sum(1)) / sn[idx].sum(1)
        f = fifths(p, y)
        res[name] = {'pred': round(float(p.mean()), 4), 'actual': round(float(y.mean()), 4), 'gap_points': round(100 * float(p.mean() - y.mean()), 2),
                     'gap_ci_points': [round(100 * float(np.quantile(gap, .025)), 2), round(100 * float(np.quantile(gap, .975)), 2)],
                     'fifths': f, 'max_fifth_gap_points': round(100 * max(abs(a - b) for a, b, _ in f), 2),
                     'brier': round(float(((p - y) ** 2).mean()), 5), 'brier_const': round(float(((y.mean() - y) ** 2).mean()), 5)}
    return res


def load_games(paths):
    seen, out = set(), []
    for f in paths:
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                g = json.loads(line)
                if g['game_pk'] not in seen:
                    seen.add(g['game_pk']); out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat)); assert files, pat
        out[lab] = season_check(load_games(files))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for lab, r in out.items():
        for name, x in r.items():
            if name == 'games':
                continue
            print(lab, name, 'pred %.4f act %.4f gap %+.2f [%+.2f, %+.2f]' % (x['pred'], x['actual'], x['gap_points'], *x['gap_ci_points']),
                  'max fifth %.2f' % x['max_fifth_gap_points'], 'brier %.5f const %.5f' % (x['brier'], x['brier_const']))


if __name__ == '__main__':
    main()
