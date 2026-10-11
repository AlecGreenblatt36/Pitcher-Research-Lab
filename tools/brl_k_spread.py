"""PLAYER-08: starters' strikeout distributions with their spread made right, scored on full replays with starter lines.

Each start's simulated strikeouts (starters[side]['k'], counts of 0, 1, ...) against his real strikeouts. The simulated
distributions carry the right means but run a little narrow (the residual variance over the mean simulated variance, c,
is above one). The candidate keeps each start's simulated shape and mean and spreads it: the distribution convolved with
the three-point step {-1: w, 0: 1 - 2w, +1: w}, w = (c - 1) x his simulated variance / 2 (capped at 0.25), the mass below
zero folded onto zero; c from the other season. Scored against the raw distributions on what the page states: the chance
of more than each line from 2.5 to 8.5 (the Odds tab's ladder) against the actual share, Brier over those lines (paired
by date), and the middle range (the page's quant(c, 0.25) to quant(c, 0.75)): its own mass against how often the real
count landed there. Model outputs and box-score facts only.

    python3 tools/brl_k_spread.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

LINES = tuple(x + 0.5 for x in range(2, 9))          # 2.5 to 8.5
N = 30


def starts(games):
    out = []
    for g in games:
        for side in ('away', 'home'):
            s = (g.get('starters') or {}).get(side) or {}; a = s.get('actual') or {}
            c = np.asarray(s.get('k') or [], float)
            if not c.size or c.sum() <= 0 or a.get('k') is None:
                continue
            p = np.zeros(N); p[:min(N, len(c))] = c[:N]
            out.append({'date': g['date'], 'p': p / p.sum(), 'y': int(a['k'])})
    return out


def moments(p):
    k = np.arange(len(p)); m = float((k * p).sum()); return m, float(((k - m) ** 2 * p).sum())


def spread_factor(rows):
    r2, v = [], []
    for x in rows:
        m, var = moments(x['p']); r2.append((x['y'] - m) ** 2); v.append(var)
    return float(np.mean(r2) / np.mean(v))


def spread(p, c):
    """p convolved with {-1: w, 0: 1 - 2w, +1: w}, w = (c - 1) var / 2 capped at 0.25; mass below zero folded onto zero."""
    if c <= 1.0:
        return p
    _, v = moments(p); w = min(0.25, (c - 1.0) * v / 2.0)
    q = (1 - 2 * w) * p
    q[1:] += w * p[:-1]                    # one more
    q[:-1] += w * p[1:]                    # one fewer
    q[0] += w * p[0]                       # one fewer than zero stays at zero
    return q / q.sum()


def quant(p, q):
    return int(np.searchsorted(np.cumsum(p), q - 1e-12))


def season_check(rows, c, draws=2000, seed=1):
    y = np.array([x['y'] for x in rows])
    dates = np.array([x['date'] for x in rows]); keys, inv = np.unique(dates, return_inverse=True)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    out = {'starts': len(rows), 'c': round(c, 4)}
    briers = {}
    for lab, ps in (('reference', [x['p'] for x in rows]), ('candidate', [spread(x['p'], c) for x in rows])):
        P = np.array(ps)
        lines = {}; b = np.zeros(len(rows))
        for L in LINES:
            pr = P[:, int(np.ceil(L)):].sum(1); o = (y > L).astype(float)
            lines[str(L)] = round(100 * float(pr.mean() - o.mean()), 2); b += (pr - o) ** 2 / len(LINES)
        lo = np.array([quant(p, .25) for p in P]); hi = np.array([quant(p, .75) for p in P])
        mass = np.array([p[a:h + 1].sum() for p, a, h in zip(P, lo, hi)]); inside = ((y >= lo) & (y <= hi)).astype(float)
        out[lab] = {'line_gaps_points': lines, 'max_abs_gap': round(max(abs(v) for v in lines.values()), 2), 'brier_lines': round(float(b.mean()), 5),
                    'range_mass': round(100 * float(mass.mean()), 2), 'range_actual': round(100 * float(inside.mean()), 2),
                    'range_gap_points': round(100 * float(mass.mean() - inside.mean()), 2)}
        briers[lab] = b
    d = briers['candidate'] - briers['reference']
    s = np.bincount(inv, weights=d, minlength=len(keys)); n = np.bincount(inv, minlength=len(keys)).astype(float)
    boot = s[idx].sum(1) / n[idx].sum(1)
    out['brier_change'] = [round(float(d.mean()), 6), round(float(np.quantile(boot, .025)), 6), round(float(np.quantile(boot, .975)), 6)]
    ref, cand = out['reference'], out['candidate']
    out['holds'] = {'(1) largest line gap smaller': bool(cand['max_abs_gap'] < ref['max_abs_gap']),
                    '(2) Brier not worse': bool(out['brier_change'][2] <= 0.0001),
                    '(3) middle range within 1.0': bool(abs(cand['range_gap_points']) <= 1.0)}
    return out


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
    rows = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat)); assert files, pat
        rows[lab] = starts(load_games(files))
    cs = {lab: spread_factor(r) for lab, r in rows.items()}
    out = {'c_by_season': {k: round(v, 4) for k, v in cs.items()}}
    for lab, r in rows.items():
        other = [k for k in rows if k != lab]
        out[lab] = season_check(r, cs[other[0]] if other else cs[lab])
    out['rule_holds'] = bool(all(all(out[lab]['holds'].values()) for lab in rows))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
