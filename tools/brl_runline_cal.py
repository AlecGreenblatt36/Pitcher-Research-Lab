"""RUNLINE-01: the run line (each team by two runs or more) from the simulator's margins, scored on full replays.

Reads the per-game files of two replays that record each game's margin histogram (home minus away runs over the worlds,
brl_replay/harness.py margin_hist; model outputs and final scores only) and checks the piece the page would use: given
that a team wins, the simulator's chance that it wins by two or more (the run line's -1.5 side, divided by the win), for
the home team and the road team, against how often winners actually won by two or more. The page's run-line chance is
then the headline win chance times that conditional chance, so it agrees with the pick. Also reported: the simulator
alone's run-line chance (its own win chance times its conditional) and its Brier against a constant. Metrics only.

    python3 tools/brl_runline_cal.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

MAX_MARGIN = 15


def load_games(paths):
    out = []
    for f in paths:
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                g = json.loads(line)
                if 'margin_hist' not in g:
                    continue
                out.append(g)
    seen, uniq = set(), []
    for g in out:
        if g['game_pk'] not in seen:
            seen.add(g['game_pk']); uniq.append(g)
    return uniq


def game_terms(g):
    """The simulator's chances from one game's margin histogram: home win, home by 2+, road win, road by 2+, and the
    conditional chances (by 2+ given the win); the actual margin."""
    m = np.asarray(g['margin_hist'], float); n = m.sum()
    k = np.arange(len(m)) - MAX_MARGIN
    hw, h2 = m[k > 0].sum() / n, m[k >= 2].sum() / n
    aw, a2 = m[k < 0].sum() / n, m[k <= -2].sum() / n
    return {'date': g['date'], 'p_home': hw, 'p_home2': h2, 'p_away': aw, 'p_away2': a2,
            'c_home': h2 / hw if hw > 0 else np.nan, 'c_away': a2 / aw if aw > 0 else np.nan, 'margin': int(g['home_runs']) - int(g['away_runs'])}


def fifths(c, y):
    q = np.quantile(c, [0, .2, .4, .6, .8, 1]); out = []
    for i in range(5):
        msk = (c >= q[i]) & ((c <= q[i + 1]) if i == 4 else (c < q[i + 1]))
        if msk.any():
            out.append([round(float(c[msk].mean()), 4), round(float(y[msk].mean()), 4), int(msk.sum())])
    return out


def season_check(games, draws=2000, seed=1):
    T = [game_terms(g) for g in games]
    res = {'games': len(T)}
    for side, sign in (('home', 1), ('away', -1)):
        won = np.array([t['margin'] * sign > 0 for t in T])
        c = np.array([t['c_' + side] for t in T])[won]
        y = np.array([t['margin'] * sign >= 2 for t in T], float)[won]
        ok = np.isfinite(c); c, y = c[ok], y[ok]
        dates = np.array([t['date'] for t in T])[won][ok]
        keys, inv = np.unique(dates, return_inverse=True)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(keys), size=(draws, len(keys)))
        sc = np.bincount(inv, weights=c, minlength=len(keys)); sy = np.bincount(inv, weights=y, minlength=len(keys)); sn = np.bincount(inv, minlength=len(keys)).astype(float)
        gap = (sc[idx].sum(1) - sy[idx].sum(1)) / sn[idx].sum(1)
        res[side] = {'wins': int(len(c)), 'conditional_pred': round(float(c.mean()), 4), 'conditional_actual': round(float(y.mean()), 4),
                     'gap_points': round(100 * float(c.mean() - y.mean()), 2), 'gap_ci_points': [round(100 * float(np.quantile(gap, .025)), 2), round(100 * float(np.quantile(gap, .975)), 2)],
                     'fifths': fifths(c, y), 'max_fifth_gap_points': round(100 * max(abs(a - b) for a, b, _ in fifths(c, y)), 2)}
    # the simulator alone's run line (its own win chance times its conditional), against the constant base rate
    for side, sign in (('home', 1), ('away', -1)):
        p = np.array([t['p_' + side + '2'] for t in T]); y = np.array([t['margin'] * sign >= 2 for t in T], float)
        res[side]['runline_sim_brier'] = round(float(((p - y) ** 2).mean()), 5)
        res[side]['runline_const_brier'] = round(float(((y.mean() - y) ** 2).mean()), 5)
        res[side]['runline_pred'] = round(float(p.mean()), 4); res[side]['runline_actual'] = round(float(y.mean()), 4)
    return res


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
        for side in ('home', 'away'):
            x = r[side]
            print(lab, side, 'wins', x['wins'], 'cond %.4f act %.4f gap %+.2f [%+.2f, %+.2f]' % (x['conditional_pred'], x['conditional_actual'], x['gap_points'], *x['gap_ci_points']),
                  'max fifth %.2f' % x['max_fifth_gap_points'], 'runline sim %.5f const %.5f' % (x['runline_sim_brier'], x['runline_const_brier']))


if __name__ == '__main__':
    main()
