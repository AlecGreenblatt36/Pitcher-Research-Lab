"""PLAYER-09: a starter's prop ladder for another count (hits allowed by default), scored on full replays with starter
lines.

Each start's simulated count (starters[side][key], counts of 0, 1, ...) against his real count (actual[key]). At each
half line the mean chance of more against the actual share, with a bootstrap over dates; Brier at the middle lines
against each line's constant share; the residual over the simulated variance (c); and the same with each distribution
spread to the residual variance by PLAYER-08's step (tools/brl_k_spread.py spread, c from the other season). Model
outputs and box-score facts only.

    python3 tools/brl_prop_ladder.py --key h --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('brl_k_spread', Path(__file__).resolve().parent / 'brl_k_spread.py')
ks = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ks)

LINES = tuple(x + 0.5 for x in range(2, 8))           # 2.5 to 7.5
BRIER_LINES = (3.5, 4.5, 5.5, 6.5)
BOUND = 2.0                                            # points, PLAYER-09 (1)


def starts(games, key):
    out = []
    for g in games:
        for side in ('away', 'home'):
            s = (g.get('starters') or {}).get(side) or {}; a = s.get('actual') or {}
            c = np.asarray(s.get(key) or [], float)
            if not c.size or c.sum() <= 0 or a.get(key) is None:
                continue
            p = np.zeros(ks.N); p[:min(ks.N, len(c))] = c[:ks.N]
            out.append({'date': g['date'], 'p': p / p.sum(), 'y': int(a[key])})
    return out


def score(rows, ps, draws=2000, seed=1):
    y = np.array([x['y'] for x in rows]); P = np.array(ps)
    dates = np.array([x['date'] for x in rows]); keys, inv = np.unique(dates, return_inverse=True)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    sn = np.bincount(inv, minlength=len(keys)).astype(float)
    res = {'lines': {}}
    for L in LINES:
        pr = P[:, int(np.ceil(L)):].sum(1); o = (y > L).astype(float)
        sp, so = np.bincount(inv, weights=pr, minlength=len(keys)), np.bincount(inv, weights=o, minlength=len(keys))
        gap = (sp[idx].sum(1) - so[idx].sum(1)) / sn[idx].sum(1)
        res['lines'][str(L)] = {'pred': round(float(pr.mean()), 4), 'actual': round(float(o.mean()), 4), 'gap_points': round(100 * float(pr.mean() - o.mean()), 2),
                                'gap_ci_points': [round(100 * float(np.quantile(gap, .025)), 2), round(100 * float(np.quantile(gap, .975)), 2)],
                                'brier': round(float(((pr - o) ** 2).mean()), 5), 'brier_const': round(float(((o.mean() - o) ** 2).mean()), 5)}
    bl = [res['lines'][str(L)] for L in BRIER_LINES]
    res['brier_lines'] = round(float(np.mean([b['brier'] for b in bl])), 5)
    res['brier_lines_const'] = round(float(np.mean([b['brier_const'] for b in bl])), 5)
    res['max_abs_gap'] = round(max(abs(v['gap_points']) for v in res['lines'].values()), 2)
    res['holds'] = bool(res['max_abs_gap'] <= BOUND and res['brier_lines'] < res['brier_lines_const'])
    return res


def season_check(rows, c):
    m = np.array([ks.moments(x['p'])[0] for x in rows]); y = np.array([x['y'] for x in rows])
    return {'starts': len(rows), 'sim_mean': round(float(m.mean()), 3), 'actual_mean': round(float(y.mean()), 3), 'c_used': round(c, 4),
            'raw': score(rows, [x['p'] for x in rows]), 'spread': score(rows, [ks.spread(x['p'], c) for x in rows])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='h')
    ap.add_argument('--season', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat)); assert files, pat
        rows[lab] = starts(ks.load_games(files), a.key)
    cs = {lab: ks.spread_factor(r) for lab, r in rows.items()}
    out = {'key': a.key, 'c_by_season': {k: round(v, 4) for k, v in cs.items()}}
    for lab, r in rows.items():
        other = [k for k in rows if k != lab]
        out[lab] = season_check(r, cs[other[0]] if other else cs[lab])
    out['raw_holds'] = all(out[lab]['raw']['holds'] for lab in rows)
    out['spread_holds'] = all(out[lab]['spread']['holds'] for lab in rows)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for lab in rows:
        r = out[lab]
        print(lab, 'starts', r['starts'], 'mean sim %.2f act %.2f c %.3f' % (r['sim_mean'], r['actual_mean'], r['c_used']))
        for kind in ('raw', 'spread'):
            x = r[kind]
            print('  ', kind, ' '.join('%s:%+.1f' % (L, v['gap_points']) for L, v in x['lines'].items()), 'max %.2f brier %.5f const %.5f holds %s' % (x['max_abs_gap'], x['brier_lines'], x['brier_lines_const'], x['holds']))
    print('raw holds', out['raw_holds'], 'spread holds', out['spread_holds'])


if __name__ == '__main__':
    main()
