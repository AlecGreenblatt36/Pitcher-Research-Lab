"""BULLPEN-03: the bullpen's chance to pitch (the Odds tab's and the game plans' table), scored on full replays with
starter lines.

Per team-game, every bullpen candidate's share of the worlds he pitched in (starters[side]['relievers']: [pitcher,
share, pitched]) and whether he pitched; a reliever who pitched without being a candidate counts with a share of 0. The
mean chance against the actual share, by bins of the chance (under 10%, 10-25, 25-40, 40-60, 60-80, 80 and up) with a
bootstrap over dates, Brier against the season's constant share, and how many relievers who pitched were not candidates.
Model outputs and box-score facts only.

    python3 tools/brl_bullpen_cal.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

BINS = (0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.0001)
MIN_ROWS = 200


def rows(games):
    out = []
    for g in games:
        for side in ('away', 'home'):
            rel = ((g.get('starters') or {}).get(side) or {}).get('relievers')
            if not rel:
                continue
            for _, share, used in rel:
                out.append((g['date'], float(share), int(used)))
    return out


def season_check(games, draws=2000, seed=1):
    R = rows(games)
    assert R, 'no team-game carries the relievers (replay with starter_lines after BULLPEN-03)'
    dates = np.array([r[0] for r in R]); p = np.array([r[1] for r in R]); y = np.array([r[2] for r in R], float)
    keys, inv = np.unique(dates, return_inverse=True)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    def ci(mask):
        sp = np.bincount(inv[mask], weights=p[mask], minlength=len(keys)); sy = np.bincount(inv[mask], weights=y[mask], minlength=len(keys))
        sn = np.bincount(inv[mask], minlength=len(keys)).astype(float)
        num = sn[idx].sum(1); gap = (sp[idx].sum(1) - sy[idx].sum(1)) / np.maximum(num, 1)
        return [round(100 * float(np.quantile(gap, .025)), 2), round(100 * float(np.quantile(gap, .975)), 2)]
    allm = np.ones(len(p), bool)
    res = {'rows': int(len(p)), 'pitched': int(y.sum()), 'not_candidates': int(((p == 0) & (y == 1)).sum()),
           'mean_chance': round(100 * float(p.mean()), 2), 'actual_share': round(100 * float(y.mean()), 2),
           'gap_points': round(100 * float(p.mean() - y.mean()), 2), 'gap_ci_points': ci(allm),
           'brier': round(float(((p - y) ** 2).mean()), 5), 'brier_const': round(float(((y.mean() - y) ** 2).mean()), 5), 'bins': {}}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        res['bins'][f'{lo:.2f}-{min(hi, 1.0):.2f}'] = {'rows': int(m.sum()), 'chance': round(100 * float(p[m].mean()), 2), 'actual': round(100 * float(y[m].mean()), 2),
                                                       'gap_points': round(100 * float(p[m].mean() - y[m].mean()), 2), 'gap_ci_points': ci(m)}
    res['holds'] = {'(1) mean within 2': bool(abs(res['gap_points']) <= 2.0),
                    '(2) every bin within 5': bool(all(abs(v['gap_points']) <= 5.0 for v in res['bins'].values() if v['rows'] >= MIN_ROWS)),
                    '(3) Brier better than constant': bool(res['brier'] < res['brier_const'])}
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
    out['rule_holds'] = bool(all(r['holds']['(1) mean within 2'] and r['holds']['(2) every bin within 5'] for r in out.values() if isinstance(r, dict)))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
