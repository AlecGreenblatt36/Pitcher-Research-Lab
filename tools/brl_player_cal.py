"""PLAYER-03: the page's hitter chances scored on full replays, with a calibration fitted across seasons.

Reads the per-game files of two replays with hitter lines (research/replay-<tag>-<run>-s<k>of<n>.jsonl.gz on the ledger
branch: model outputs and box-score facts only) and, for each kind of chance the page shows (at least one hit, two or more
hits, a home run, a walk, a strikeout), computes the chance the way the page does (each of the slot's events kept with the
spot's share, brl_live/boxscore.STARTER_SHARE; page ownAtLeast) against what the hitter did. Then, per kind: the level
against the actual share, deciles, a logistic map logit(p') = a + b logit(p) fitted on one season and applied to the other,
the Brier change with a bootstrap over dates, and the registered decision rule. Writes a metrics-only JSON.

    python3 tools/brl_player_cal.py --season 2025=<dir or glob> --season 2026=<dir or glob> --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _starter_share():
    """brl_live/boxscore.py STARTER_SHARE, read from the source (importing the box module needs the engine's runtime)."""
    m = re.search(r'^STARTER_SHARE=\(([^)]*)\)', (ROOT / 'brl_live' / 'boxscore.py').read_text(encoding='utf-8'), re.M)
    return tuple(float(x) for x in m.group(1).split(','))


STARTER_SHARE = _starter_share()

# kind: (histogram field, at least k, actual field)
KINDS = {'hit': ('h_hist', 1, 'h'), 'hit2': ('h_hist', 2, 'h'), 'hr': ('hr_hist', 1, 'hr'), 'walk': ('bb_hist', 1, 'bb'), 'k': ('k_hist', 1, 'k')}
ALL_KINDS = tuple(KINDS) + ('tb2',)      # tb2: two or more total bases (PLAYER-06), from the hits-by-singles table
EPS = 1e-4


def own_at_least(hist, k, spot):
    """The page's ownAtLeast: P(at least k of the slot's events are his), each kept with the spot's share."""
    keep = STARTER_SHARE[min(max(int(spot or 1), 1), 9) - 1]
    n_w = sum(hist)
    if not n_w:
        return float('nan')
    below = 0.0
    for m, c in enumerate(hist):
        below += c * sum(comb(m, j) * keep ** j * (1 - keep) ** (m - j) for j in range(min(k, m + 1)))
    return 1.0 - below / n_w


def own_tb2(hs, spot):
    """His own chance of two or more total bases: each of the slot's hits kept with the spot's share; below two only when
    no hit is kept, or exactly one is and it is a single. hs[h][s]: worlds with h hits, s of them singles."""
    keep = STARTER_SHARE[min(max(int(spot or 1), 1), 9) - 1]
    n_w = sum(sum(r) for r in hs)
    if not n_w:
        return float('nan')
    below = sum(c * ((1 - keep) ** h + (s1 * keep * (1 - keep) ** (h - 1) if h else 0.0)) for h, row in enumerate(hs) for s1, c in enumerate(row))
    return 1.0 - below / n_w


def load_rows(paths):
    """Hitter-game rows: date, spot, prior PA, the page's chance of each kind and the outcome."""
    out = []
    for f in paths:
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                g = json.loads(line)
                for x in g.get('hitters') or []:
                    a = x.get('actual')
                    if not a or not a.get('pa'):
                        continue
                    r = {'date': g['date'], 'game_pk': g['game_pk'], 'spot': x['spot'], 'prior_pa': x.get('prior_pa_365') or 0}
                    for kind, (field, k, af) in KINDS.items():
                        if field in x:
                            r['p_' + kind] = own_at_least(x[field], k, x['spot'])
                            r['y_' + kind] = int((a.get(af) or 0) >= k)
                    if 'hs_hist' in x and 'xbh' in a:
                        r['p_tb2'] = own_tb2(x['hs_hist'], x['spot'])
                        r['y_tb2'] = int((a.get('h') or 0) >= 2 or (a.get('xbh') or 0) >= 1)
                    out.append(r)
    return out


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_map(p, y, iters=50):
    """Maximum-likelihood a, b of logit(p') = a + b logit(p) (Newton's method)."""
    x = logit(p); y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x]); w = np.array([0.0, 1.0])
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (y - q); H = (X * (q * (1 - q))[:, None]).T @ X
        step = np.linalg.solve(H, g); w = w + step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def apply_map(p, a, b):
    return 1 / (1 + np.exp(-(a + b * logit(p))))


def brier_change(dates, p_raw, p_cal, y, draws=2000, seed=1):
    """Calibrated minus raw Brier, mean over hitter-games, with a bootstrap over dates (paired by hitter-game)."""
    d = (p_cal - y) ** 2 - (p_raw - y) ** 2
    keys, inv = np.unique(np.asarray(dates), return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(keys)); n = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return float(d.mean()), [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def deciles(p, y):
    q = np.quantile(p, np.linspace(0, 1, 11)); out = []
    for i in range(10):
        m = (p >= q[i]) & (p <= q[i + 1]) if i == 9 else (p >= q[i]) & (p < q[i + 1])
        if m.any():
            out.append([round(float(p[m].mean()), 4), round(float(y[m].mean()), 4), int(m.sum())])
    return out


def season_arrays(rows, kind):
    sel = [r for r in rows if 'p_' + kind in r and np.isfinite(r['p_' + kind])]
    return (np.array([r['date'] for r in sel]), np.array([r['p_' + kind] for r in sel]), np.array([r['y_' + kind] for r in sel], float),
            np.array([r['spot'] for r in sel]), np.array([r['prior_pa'] for r in sel]))


def study(seasons: dict, draws=2000):
    """seasons: {label: rows}. Two labels; each is scored with the map fitted on the other."""
    labels = sorted(seasons)
    assert len(labels) == 2
    res = {'kinds': {}, 'seasons': {s: len(seasons[s]) for s in labels}}
    for kind in ALL_KINDS:
        arrs = {s: season_arrays(seasons[s], kind) for s in labels}
        if any(len(arrs[s][1]) == 0 for s in labels):
            continue
        k_res = {'by_season': {}}
        ok = True
        for s in labels:
            other = labels[1 - labels.index(s)]
            d, p, y, spot, prior = arrs[s]
            a, b = fit_map(arrs[other][1], arrs[other][2])
            pc = apply_map(p, a, b)
            dm, ci = brier_change(d, p, pc, y, draws=draws)
            row = {'n': int(len(p)), 'mean_p': round(float(p.mean()), 4), 'actual': round(float(y.mean()), 4),
                   'gap_points': round(100 * float(p.mean() - y.mean()), 2), 'brier_raw': round(float(((p - y) ** 2).mean()), 5),
                   'brier_const': round(float(((y.mean() - y) ** 2).mean()), 5),
                   'map_from_other': [round(a, 4), round(b, 4)], 'mean_cal': round(float(pc.mean()), 4),
                   'cal_gap_points': round(100 * float(pc.mean() - y.mean()), 2), 'brier_change': round(dm, 6), 'brier_change_ci': [round(ci[0], 6), round(ci[1], 6)],
                   'deciles_raw': deciles(p, y), 'deciles_cal': deciles(pc, y),
                   'top_tenth_gap_points': round(100 * (deciles(p, y)[-1][0] - deciles(p, y)[-1][1]), 2),
                   'by_spot': [[int(sp), round(float(p[spot == sp].mean()), 4), round(float(y[spot == sp].mean()), 4), int((spot == sp).sum())] for sp in range(1, 10) if (spot == sp).any()],
                   'new_hitters': ([round(float(p[prior < 200].mean()), 4), round(float(y[prior < 200].mean()), 4), int((prior < 200).sum())] if (prior < 200).any() else None)}
            row['passes'] = bool(ci[1] < 0 and abs(row['cal_gap_points']) <= 0.5)
            ok = ok and row['passes']
            k_res['by_season'][s] = row
        k_res['passes_rule'] = ok
        if ok:
            P = np.concatenate([arrs[s][1] for s in labels]); Y = np.concatenate([arrs[s][2] for s in labels])
            k_res['map_both'] = [round(v, 4) for v in fit_map(P, Y)]
        res['kinds'][kind] = k_res
    return res


PT_BINS = ((0, 100), (100, 200), (200, 400), (400, 10 ** 9))


def own_at_least_keep(hist, k, keep):
    """own_at_least with a given share kept (the spot's share times a part-time factor)."""
    n_w = sum(hist)
    if not n_w:
        return float('nan')
    below = 0.0
    for m, c in enumerate(hist):
        below += c * sum(comb(m, j) * keep ** j * (1 - keep) ** (m - j) for j in range(min(k, m + 1)))
    return 1.0 - below / n_w


def load_hist_rows(paths):
    """Hitter-game rows with the histograms kept: date, spot, prior plate appearances, simulated mean plate appearances,
    actual line and the four histograms (PLAYER-05)."""
    out = []
    for f in paths:
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                g = json.loads(line)
                for x in g.get('hitters') or []:
                    a = x.get('actual')
                    if not a or not a.get('pa') or 'hr_hist' not in x:
                        continue
                    out.append({'date': g['date'], 'spot': x['spot'], 'prior_pa': x.get('prior_pa_365') or 0, 'mean_pa': x['mean_pa'], 'actual': a,
                                'h_hist': x['h_hist'], 'hr_hist': x['hr_hist'], 'bb_hist': x['bb_hist'], 'k_hist': x['k_hist']})
    return out


def pt_bin(prior):
    for i, (lo, hi) in enumerate(PT_BINS):
        if lo <= prior < hi:
            return i
    return len(PT_BINS) - 1


def pt_factors(rows):
    """Per prior-playing-time group: actual plate appearances over simulated ones times the spot's share."""
    num, den = np.zeros(len(PT_BINS)), np.zeros(len(PT_BINS))
    for r in rows:
        b = pt_bin(r['prior_pa'])
        num[b] += r['actual']['pa']; den[b] += r['mean_pa'] * STARTER_SHARE[min(max(int(r['spot']), 1), 9) - 1]
    return [float(n / d) if d else 1.0 for n, d in zip(num, den)]


def part_time_study(seasons: dict, draws=2000):
    """PLAYER-05: each season scored with the other season's part-time factors against the spot's share alone."""
    labels = sorted(seasons)
    res = {'bins': [list(b) for b in PT_BINS], 'factors': {s: [round(v, 4) for v in pt_factors(seasons[s])] for s in labels}, 'by_season': {}}
    for s in labels:
        other = labels[1 - labels.index(s)]
        fac = pt_factors(seasons[other])
        rows = seasons[s]
        dates = np.array([r['date'] for r in rows]); bins = np.array([pt_bin(r['prior_pa']) for r in rows])
        r_s = {}
        for kind, (field, k, af) in KINDS.items():
            base = np.array([own_at_least(r[field], k, r['spot']) for r in rows])
            adj = np.array([own_at_least_keep(r[field], k, min(1.0, STARTER_SHARE[min(max(int(r['spot']), 1), 9) - 1] * fac[pt_bin(r['prior_pa'])])) for r in rows])
            y = np.array([float((r['actual'].get(af) or 0) >= k) for r in rows])
            dm, ci = brier_change(dates, base, adj, y, draws=draws)
            r_s[kind] = {'brier_change': round(dm, 7), 'brier_change_ci': [round(ci[0], 7), round(ci[1], 7)],
                         'gap_by_bin': [[round(100 * float(base[bins == b].mean() - y[bins == b].mean()), 2), round(100 * float(adj[bins == b].mean() - y[bins == b].mean()), 2), int((bins == b).sum())]
                                        for b in range(len(PT_BINS)) if (bins == b).any()]}
        res['by_season'][s] = {'factors_from_other': [round(v, 4) for v in fac], 'kinds': r_s}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', action='append', required=True, help='label=glob of per-game files')
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--part-time', action='store_true', help='PLAYER-05: score the part-time factors instead')
    a = ap.parse_args()
    if a.part_time:
        seasons = {}
        for spec in a.season:
            lab, pat = spec.split('=', 1)
            seasons[lab] = load_hist_rows(sorted(glob.glob(pat)))
        res = part_time_study(seasons, draws=a.draws)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print('factors', res['factors'])
        for s_, r in res['by_season'].items():
            for kind, x in r['kinds'].items():
                print(s_, kind, 'dBrier %.7f [%.7f, %.7f]' % (x['brier_change'], *x['brier_change_ci']), 'gaps (before, after, n) by group', x['gap_by_bin'])
        return
    seasons = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat))
        assert files, pat
        seasons[lab] = load_rows(files)
    res = study(seasons, draws=a.draws)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for kind, kr in res['kinds'].items():
        for s, r in kr['by_season'].items():
            print(kind, s, 'n', r['n'], 'p %.4f act %.4f gap %+.2f' % (r['mean_p'], r['actual'], r['gap_points']), 'top10 %+.2f' % r['top_tenth_gap_points'],
                  'map', r['map_from_other'], 'cal gap %+.2f' % r['cal_gap_points'], 'dBrier %.6f [%.6f, %.6f]' % (r['brier_change'], *r['brier_change_ci']), 'pass', r['passes'])
        print(kind, 'RULE', kr['passes_rule'], kr.get('map_both'))


if __name__ == '__main__':
    main()
