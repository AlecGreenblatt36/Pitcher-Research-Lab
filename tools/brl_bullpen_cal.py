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


def fit_map(games, lo=0.002, hi=0.998, iters=50):
    """A logistic calibration of the shown chance (candidates with a share above 0): logit(p') = a + b logit(p), fitted by
    Newton's method on one season's rows; returns (a, b)."""
    R = [r for r in rows(games) if r[1] > 0]
    x = np.log(np.clip([r[1] for r in R], lo, hi) / (1 - np.clip([r[1] for r in R], lo, hi))); y = np.array([r[2] for r in R], float)
    X = np.column_stack([np.ones(len(x)), x]); w = np.array([0.0, 1.0])
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(X @ w))); g = X.T @ (y - q); H = (X * (q * (1 - q))[:, None]).T @ X
        step = np.linalg.solve(H, g); w = w + step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def apply_map(games, a, b, lo=0.002, hi=0.998):
    """The same games with every candidate's shown chance mapped (shares of 0 stay 0)."""
    out = []
    for g in games:
        g2 = dict(g); st = {}
        for side, s in ((g.get('starters') or {}).items()):
            s2 = dict(s)
            if s.get('relievers'):
                s2['relievers'] = [[pid, (1 / (1 + np.exp(-(a + b * np.log(min(max(sh, lo), hi) / (1 - min(max(sh, lo), hi))))))) if sh > 0 else 0.0, u] for pid, sh, u in s['relievers']]
            st[side] = s2
        g2['starters'] = st; out.append(g2)
    return out


LO, HI = 0.002, 0.998


def _logit(p):
    p = np.clip(np.asarray(p, float), LO, HI); return np.log(p / (1 - p))


def fit_curve(games, iters=60):
    """BULLPEN-04: logit(p') = a + b x + c x^2, x = logit(chance), by Newton's method on one season's candidates (shares
    above 0); the straight map (c = 0) if the curve is not increasing over the clipped range. Returns (a, b, c)."""
    R = [r for r in rows(games) if r[1] > 0]
    x = _logit([r[1] for r in R]); y = np.array([r[2] for r in R], float)
    def fit(cols):
        X = np.column_stack(cols); w = np.zeros(X.shape[1]); w[1] = 1.0
        for _ in range(iters):
            q = 1 / (1 + np.exp(-(X @ w))); g = X.T @ (y - q); H = (X * (q * (1 - q))[:, None]).T @ X
            step = np.linalg.solve(H, g); w = w + step
            if np.abs(step).max() < 1e-10:
                break
        return w
    a, b, c = fit([np.ones(len(x)), x, x * x])
    lo, hi = _logit(LO), _logit(HI)
    if min(b + 2 * c * lo, b + 2 * c * hi) <= 0:
        a, b = fit([np.ones(len(x)), x]); c = 0.0
    return float(a), float(b), float(c)


def curve(p, a, b, c):
    x = _logit(p); return 1 / (1 + np.exp(-(a + b * x + c * x * x)))


def candidates_check(games, abc, draws=2000, seed=1):
    """BULLPEN-04's predictions on one season's candidates with a map (a, b, c) fitted elsewhere."""
    R = [r for r in rows(games) if r[1] > 0]
    dates = np.array([r[0] for r in R]); p0 = np.array([r[1] for r in R]); y = np.array([r[2] for r in R], float)
    p1 = curve(p0, *abc)
    keys, inv = np.unique(dates, return_inverse=True)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    d = (p1 - y) ** 2 - (p0 - y) ** 2
    s = np.bincount(inv, weights=d, minlength=len(keys)); n = np.bincount(inv, minlength=len(keys)).astype(float)
    boot = s[idx].sum(1) / n[idx].sum(1)
    res = {'rows': int(len(y)), 'map': [round(v, 5) for v in abc], 'mean_raw': round(100 * float(p0.mean()), 2), 'mean_mapped': round(100 * float(p1.mean()), 2),
           'actual': round(100 * float(y.mean()), 2), 'gap_points': round(100 * float(p1.mean() - y.mean()), 2),
           'brier_raw': round(float(((p0 - y) ** 2).mean()), 5), 'brier_mapped': round(float(((p1 - y) ** 2).mean()), 5),
           'brier_change': [round(float(d.mean()), 6), round(float(np.quantile(boot, .025)), 6), round(float(np.quantile(boot, .975)), 6)], 'bins': {}}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p1 >= lo) & (p1 < hi)
        if m.any():
            res['bins'][f'{lo:.2f}-{min(hi, 1.0):.2f}'] = {'rows': int(m.sum()), 'chance': round(100 * float(p1[m].mean()), 2), 'actual': round(100 * float(y[m].mean()), 2),
                                                           'gap_points': round(100 * float(p1[m].mean() - y[m].mean()), 2)}
    res['holds'] = {'(1) mean within 1.5': bool(abs(res['gap_points']) <= 1.5),
                    '(2) every bin within 4': bool(all(abs(v['gap_points']) <= 4.0 for v in res['bins'].values() if v['rows'] >= MIN_ROWS)),
                    '(3) Brier better': bool(res['brier_change'][2] < 0)}
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
    ap.add_argument('--map', action='store_true', help='BULLPEN-04: each season with the curve fitted on the other')
    a = ap.parse_args()
    if a.map:
        G = {}
        for spec in a.season:
            lab, pat = spec.split('=', 1)
            files = sorted(glob.glob(pat)); assert files, pat
            G[lab] = load_games(files)
        out = {}
        for lab in G:
            other = [k for k in G if k != lab][0]
            out[lab] = candidates_check(G[lab], fit_curve(G[other]))
        out['pooled_map'] = [round(v, 5) for v in fit_curve([g for v in G.values() for g in v])]
        out['rule_holds'] = bool(all(all(out[lab]['holds'].values()) for lab in G))
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return
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
