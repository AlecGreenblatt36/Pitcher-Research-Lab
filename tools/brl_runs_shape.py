"""TOTALS-05: the published run distributions with their spread made right, scored on full replays.

The simulator's run histograms carry the right means but run about a tenth too narrow (TOTALS-04: a team's predictive
variance 9.2 against a residual variance of 10.0 to 10.1). The candidate keeps each team's simulated mean and replaces the
histogram by a negative binomial with that mean and the simulated variance times c (Poisson if that is not above the
mean), with c the residual over the predictive variance on the other season; the total is the two teams' convolution
(their residuals are uncorrelated, TOTALS-04). Scored against the raw histograms on what the page states: each team's
and the total's middle range (the page's quant(c, 0.25) to quant(c, 0.75)): the distribution's own mass in it against
how often the real score landed there; the chance of 10 or more total runs (mean against the actual share, and Brier);
CRPS per team and for the total (paired by date). Model outputs and box-score facts only.

    python3 tools/brl_runs_shape.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from math import lgamma, log
from pathlib import Path

import numpy as np

K = 40                                       # runs 0 to 39 per team; totals to 78


def hist_pmf(h):
    h = np.asarray(h, float)[:K]
    out = np.zeros(K); out[:len(h)] = h
    return out / out.sum()


def nb_pmf(m, v):
    """Negative binomial on 0..K-1 with mean m and variance v (Poisson when v <= m), renormalized."""
    k = np.arange(K)
    if v <= m * 1.0001:
        lp = k * log(m) - m - np.array([lgamma(x + 1) for x in k])
    else:
        r = m * m / (v - m); p = r / (r + m)
        lp = np.array([lgamma(x + r) - lgamma(r) - lgamma(x + 1) for x in k]) + r * log(p) + k * log(1 - p)
    out = np.exp(lp - lp.max())
    return out / out.sum()


def quant(pmf, q):
    """The page's quant(c, q): the fewest runs whose running share reaches q."""
    return int(np.searchsorted(np.cumsum(pmf), q - 1e-12))


def crps(pmf, y):
    c = np.cumsum(pmf); step = (np.arange(len(pmf)) >= y).astype(float)
    return float(((c - step) ** 2).sum())


def team_games(games):
    """One row per game: date, the two teams' histograms and real runs."""
    out = []
    for g in games:
        if g.get('home_runs') is None or g.get('away_runs') is None or not g.get('home_hist') or not g.get('away_hist'):
            continue
        out.append({'date': g['date'], 'h': hist_pmf(g['home_hist']), 'a': hist_pmf(g['away_hist']), 'yh': int(g['home_runs']), 'ya': int(g['away_runs'])})
    return out


def moments(p):
    k = np.arange(len(p)); m = float((k * p).sum()); return m, float(((k - m) ** 2 * p).sum())


def inflation(rows):
    """c: the mean squared residual of a team's runs about its simulated mean over the mean simulated variance."""
    r2, v = [], []
    for x in rows:
        for p, y in ((x['h'], x['yh']), (x['a'], x['ya'])):
            m, var = moments(p); r2.append((y - m) ** 2); v.append(var)
    return float(np.mean(r2) / np.mean(v))


def statements(team_pmfs, ys):
    """Per game: [total mass in the middle range, actual in it, P(total >= 10), actual >= 10, total CRPS] and per team
    [mass in its middle range, actual in it, CRPS]."""
    games, teams = [], []
    for (ph, pa), (yh, ya) in zip(team_pmfs, ys):
        t = np.convolve(ph, pa); y = yh + ya
        lo, hi = quant(t, .25), quant(t, .75)
        games.append([float(t[lo:hi + 1].sum()), float(lo <= y <= hi), float(t[10:].sum()), float(y >= 10), crps(t, y)])
        for p, yy in ((ph, yh), (pa, ya)):
            l2, h2 = quant(p, .25), quant(p, .75)
            teams.append([float(p[l2:h2 + 1].sum()), float(l2 <= yy <= h2), crps(p, yy)])
    return np.array(games), np.array(teams)


def season_check(rows, c, draws=2000, seed=1):
    ys = [(x['yh'], x['ya']) for x in rows]
    ref = [(x['h'], x['a']) for x in rows]
    new = [(nb_pmf(*(lambda m, v: (m, c * v))(*moments(x['h']))), nb_pmf(*(lambda m, v: (m, c * v))(*moments(x['a'])))) for x in rows]
    out = {'games': len(rows), 'c': round(c, 4)}
    dates = np.array([x['date'] for x in rows]); keys, inv = np.unique(dates, return_inverse=True)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    gr, tr = statements(ref, ys); gn, tn = statements(new, ys)
    tinv = np.repeat(inv, 2)
    def paired(a, b, which):
        d = b - a; s = np.bincount(which, weights=d, minlength=len(keys)); n = np.bincount(which, minlength=len(keys)).astype(float)
        boot = s[idx].sum(1) / n[idx].sum(1)
        return [round(float(d.mean()), 5), round(float(np.quantile(boot, .025)), 5), round(float(np.quantile(boot, .975)), 5)]
    for lab, g, t in (('reference', gr, tr), ('candidate', gn, tn)):
        out[lab] = {'total_range_mass': round(100 * g[:, 0].mean(), 2), 'total_range_actual': round(100 * g[:, 1].mean(), 2),
                    'total_range_gap_points': round(100 * (g[:, 0].mean() - g[:, 1].mean()), 2),
                    'ten_plus_pred': round(100 * g[:, 2].mean(), 2), 'ten_plus_actual': round(100 * g[:, 3].mean(), 2),
                    'ten_plus_gap_points': round(100 * (g[:, 2].mean() - g[:, 3].mean()), 2), 'ten_plus_brier': round(float(((g[:, 2] - g[:, 3]) ** 2).mean()), 5),
                    'total_crps': round(float(g[:, 4].mean()), 4),
                    'team_range_mass': round(100 * t[:, 0].mean(), 2), 'team_range_actual': round(100 * t[:, 1].mean(), 2),
                    'team_range_gap_points': round(100 * (t[:, 0].mean() - t[:, 1].mean()), 2), 'team_crps': round(float(t[:, 2].mean()), 4)}
    out['ten_plus_brier_change'] = paired((gr[:, 2] - gr[:, 3]) ** 2, (gn[:, 2] - gn[:, 3]) ** 2, inv)
    out['total_crps_change'] = paired(gr[:, 4], gn[:, 4], inv)
    out['team_crps_change'] = paired(tr[:, 2], tn[:, 2], tinv)
    cand = out['candidate']
    out['holds'] = {k: bool(v) for k, v in {'(1) total range within 1.5': abs(cand['total_range_gap_points']) <= 1.5,
                    '(1) team range within 1.5': abs(cand['team_range_gap_points']) <= 1.5,
                    '(2) ten plus within 1.0': abs(cand['ten_plus_gap_points']) <= 1.0,
                    '(2) ten plus Brier not worse': out['ten_plus_brier_change'][2] <= 0.0002,
                    '(3) team CRPS not worse': out['team_crps_change'][2] <= 0.002,
                    '(3) total CRPS not worse': out['total_crps_change'][2] <= 0.002}.items()}
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
        rows[lab] = team_games(load_games(files))
    cs = {lab: inflation(r) for lab, r in rows.items()}
    out = {'c_by_season': {k: round(v, 4) for k, v in cs.items()}}
    for lab, r in rows.items():
        other = [k for k in rows if k != lab]
        out[lab] = season_check(r, cs[other[0]] if other else cs[lab])
    out['rule_holds'] = bool(all(all(out[lab]['holds'].values()) for lab in rows))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
