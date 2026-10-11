"""PLAYER-07: starters' outs recorded (the pitcher-outs prop), scored on full replays.

Reads the per-game files of replays with starter lines: each starter's simulated outs over the game's worlds
(starters[side]['outs'], counts of 0 to 27 outs) and his real outs (starters[side]['actual']['outs'], read from the plate
appearances by brl_replay/harness.py pitcher_outs). For each half line from 12.5 to 20.5: the mean chance of more against
the actual share, with a bootstrap over dates; by fifths of the simulated mean, the simulated and actual mean outs; Brier
at the lines 14.5 to 17.5 together against each line's constant share; and, described only, the lines around each start's
median as a ladder would show them. Model outputs and box-score facts only.

    python3 tools/brl_outs_cal.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

LINES = tuple(x + 0.5 for x in range(12, 21))          # 12.5 to 20.5
BRIER_LINES = (14.5, 15.5, 16.5, 17.5)
LEVEL_POINTS, FIFTH_OUTS = 2.5, 0.6                    # PLAYER-07's bounds for predictions (1) and (2)
BANDS = ((0, 9), (10, 12), (13, 15), (16, 17), (18, 18), (19, 20), (21, 27))   # described; LEASH-02 gates 10 to 12


def starts(games):
    """One row per start whose starter has simulated outs and a real line: the date, his simulated outs as shares of
    0, 1, ... outs, and his real outs."""
    out = []
    for g in games:
        for side in ('away', 'home'):
            s = (g.get('starters') or {}).get(side) or {}
            a = s.get('actual') or {}
            c = np.asarray(s.get('outs') or [], float)
            if not c.size or c.sum() <= 0 or a.get('outs') is None:
                continue
            out.append({'date': g['date'], 'p': c / c.sum(), 'y': int(a['outs'])})
    return out


def more_than(p, line):
    """The chance of more than a half line (at least its next whole number of outs)."""
    return float(p[int(np.ceil(line)):].sum())


def median(p):
    """The page's quant(c, 0.5): the fewest outs whose running share reaches one half."""
    return int(np.searchsorted(np.cumsum(p), 0.5 - 1e-12))


def fifths(m, y):
    q = np.quantile(m, [0, .2, .4, .6, .8, 1]); out = []
    for i in range(5):
        msk = (m >= q[i]) & ((m <= q[i + 1]) if i == 4 else (m < q[i + 1]))
        if msk.any():
            out.append([round(float(m[msk].mean()), 3), round(float(y[msk].mean()), 3), int(msk.sum())])
    return out


def season_check(games, draws=2000, seed=1):
    S = starts(games)
    assert S, 'no start carries simulated outs with a real line (replay with starter_lines after pitcher_outs)'
    rng = np.random.default_rng(seed)
    dates = np.array([x['date'] for x in S]); keys, inv = np.unique(dates, return_inverse=True)
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    sn = np.bincount(inv, minlength=len(keys)).astype(float)
    y = np.array([x['y'] for x in S], float)
    m = np.array([float((np.arange(len(x['p'])) * x['p']).sum()) for x in S])
    res = {'starts': len(S), 'sim_mean': round(float(m.mean()), 3), 'actual_mean': round(float(y.mean()), 3),
           'slope': round(float(np.polyfit(m, y, 1)[0]), 3), 'lines': {}}
    for L in LINES:
        p = np.array([more_than(x['p'], L) for x in S]); o = (y > L).astype(float)
        sp, so = np.bincount(inv, weights=p, minlength=len(keys)), np.bincount(inv, weights=o, minlength=len(keys))
        gap = (sp[idx].sum(1) - so[idx].sum(1)) / sn[idx].sum(1)
        res['lines'][str(L)] = {'pred': round(float(p.mean()), 4), 'actual': round(float(o.mean()), 4), 'gap_points': round(100 * float(p.mean() - o.mean()), 2),
                                'gap_ci_points': [round(100 * float(np.quantile(gap, .025)), 2), round(100 * float(np.quantile(gap, .975)), 2)],
                                'brier': round(float(((p - o) ** 2).mean()), 5), 'brier_const': round(float(((o.mean() - o) ** 2).mean()), 5)}
    f = fifths(m, y)
    res['fifths'] = f
    res['max_fifth_gap_outs'] = round(max(abs(a - b) for a, b, _ in f), 3)
    bl = [res['lines'][str(L)] for L in BRIER_LINES]
    res['brier_lines'] = round(float(np.mean([b['brier'] for b in bl])), 5)
    res['brier_lines_const'] = round(float(np.mean([b['brier_const'] for b in bl])), 5)
    # described only: the ladder's lines, each start's median less 1.5 to plus 1.5
    lp, lo = [], []
    for x, yy in zip(S, y):
        md = median(x['p'])
        for L in (md - 1.5, md - 0.5, md + 0.5, md + 1.5):
            if L > 0:
                lp.append(more_than(x['p'], L)); lo.append(float(yy > L))
    lp, lo = np.array(lp), np.array(lo)
    res['ladder_lines'] = {'n': int(len(lp)), 'pred': round(float(lp.mean()), 4), 'actual': round(float(lo.mean()), 4)}
    res['bands'] = {}
    for lo_, hi_ in BANDS:
        ps = np.array([float(x['p'][lo_:hi_ + 1].sum()) for x in S]); ob = ((y >= lo_) & (y <= hi_)).astype(float)
        res['bands'][f'{lo_}-{hi_}'] = {'sim': round(100 * float(ps.mean()), 2), 'actual': round(100 * float(ob.mean()), 2), 'gap_points': round(100 * float(ps.mean() - ob.mean()), 2)}
    res['level_holds'] = all(abs(v['gap_points']) <= LEVEL_POINTS for v in res['lines'].values())
    res['fifths_hold'] = res['max_fifth_gap_outs'] <= FIFTH_OUTS
    res['brier_holds'] = res['brier_lines'] < res['brier_lines_const']
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
    out['rule_holds'] = all(r['level_holds'] and r['fifths_hold'] for r in out.values() if isinstance(r, dict))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for lab, r in out.items():
        if not isinstance(r, dict):
            continue
        print(lab, 'starts', r['starts'], 'outs sim %.2f act %.2f slope %.3f' % (r['sim_mean'], r['actual_mean'], r['slope']))
        for L, x in r['lines'].items():
            print(' ', L, 'pred %.4f act %.4f gap %+.2f [%+.2f, %+.2f]' % (x['pred'], x['actual'], x['gap_points'], *x['gap_ci_points']))
        print('  fifths', r['fifths'], 'max gap %.3f' % r['max_fifth_gap_outs'])
        print('  brier 14.5-17.5 %.5f const %.5f' % (r['brier_lines'], r['brier_lines_const']), 'ladder', r['ladder_lines'])
        print('  bands (sim, actual)', ' '.join('%s: %.1f/%.1f' % (k, v['sim'], v['actual']) for k, v in r['bands'].items()))
        print('  (1) level', r['level_holds'], '(2) fifths', r['fifths_hold'], '(3) brier', r['brier_holds'])
    print('PLAYER-07 rule ((1) and (2) in both seasons):', out['rule_holds'])


if __name__ == '__main__':
    main()
