"""LEVEL-01: re-anchoring each outcome's league level in-season, scored on the real plate appearances of check-only replays.

Each date's offsets come only from the season's real plate appearances on earlier dates: per class, log((observed +
m x share) / (predicted + m x share)), with m pseudo plate appearances at the predicted shares pulling the offsets toward
zero (the context table's form, CTX-02). Applied to the date's predictions (each class's chance times exp(offset),
renormalized), the change in log-likelihood on that date's real plate appearances comes from the replay's per-game sums
(brl_replay/harness.py real_pa_check: observed counts O, predicted sums P and second moments S2, the sum of p p'): per
plate appearance it is d_y - log(sum_c p_c exp(d_c)), and the log is the cumulant series k1 + k2/2 + k3/6 of the offsets
under that plate appearance's chances. Every term's sum over plate appearances is a function of P and S2, except the
cube of each plate appearance's mean offset, which is made negligible by first shifting the offsets by the date's mean
(a shift common to every class changes no chance). Model outputs and per-game sums only.

The registered grid of m is picked on the other season (the m with the larger gain there) and scored on this one.

    python3 tools/brl_level_cal.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

LABELS = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')
GRID = (2000, 10000, 50000)                       # pseudo plate appearances (LEVEL-01's registered grid)


def by_date(games):
    """{date: [n, O (7), P (7), S2 (7x7)]} pooled over starters and relievers."""
    out = {}
    for g in games:
        rp = g.get('real_pa') or {}
        for role in ('starter', 'reliever'):
            r = rp.get(role)
            if not r or not r.get('n') or 's2' not in r:
                continue
            e = out.setdefault(g['date'], [0, np.zeros(7), np.zeros(7), np.zeros((7, 7))])
            e[0] += int(r['n']); e[1] += np.asarray(r['obs'], float); e[2] += np.asarray(r['pred'], float)
            e[3] += np.asarray(r['s2'], float).reshape(7, 7)
    return out


def ll_change(O, P, S2, d):
    """Sum over a set of plate appearances of d_y - log(sum_c p_c exp(d_c)), from their sums (see the module note)."""
    n = P.sum()
    if n <= 0 or not np.any(d):
        return 0.0
    d = d - float(P @ d) / n                      # the date's mean offset to zero: each plate appearance's mean is then small
    k1 = float(P @ d)
    m2 = float(P @ d ** 2); m1sq = float(d @ S2 @ d)
    m3 = float(P @ d ** 3); m2m1 = float((d ** 2) @ S2 @ d)
    log_z = k1 + 0.5 * (m2 - m1sq) + (m3 - 3.0 * m2m1) / 6.0
    return float(O @ d) - log_z


def offsets(O, P, m):
    n = P.sum()
    if n <= 0:
        return np.zeros(7)
    s = P / n
    return np.log((O + m * s) / (P + m * s))


def season_gains(dates, grid=GRID):
    """Per date, per m: the log-likelihood change from offsets fitted on the earlier dates; and each date's plate appearances."""
    keys = sorted(dates)
    O_c, P_c = np.zeros(7), np.zeros(7)
    gains = {m: [] for m in grid}; n = []; last = {}
    for k in keys:
        nd, Od, Pd, S2d = dates[k]
        for m in grid:
            d = offsets(O_c, P_c, m) if P_c.sum() > 0 else np.zeros(7)
            gains[m].append(ll_change(Od, Pd, S2d, d))
            last[m] = d
        n.append(nd)
        O_c += Od; P_c += Pd
    return keys, {m: np.array(v) for m, v in gains.items()}, np.array(n, float), {m: v.round(4).tolist() for m, v in last.items()}, (O_c, P_c)


def date_offsets(dates, m):
    """{date: offsets} for each date from the season's earlier dates only (what a replay or the live run applies that day)."""
    O_c, P_c, out = np.zeros(7), np.zeros(7), {}
    for k in sorted(dates):
        out[k] = [round(float(x), 5) for x in (offsets(O_c, P_c, m) if P_c.sum() > 0 else np.zeros(7))]
        O_c += dates[k][1]; P_c += dates[k][2]
    return out


def season_check(games, grid=GRID, draws=2000, seed=1):
    dates = by_date(games)
    assert dates, 'no game carries the real plate-appearance sums with second moments (a check-only replay after s2)'
    keys, gains, n, last, (O, P) = season_gains(dates, grid)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    res = {'dates': len(keys), 'pa': int(n.sum()), 'season_offsets': dict(zip(LABELS, np.log(O / P).round(4).tolist())), 'by_m': {}}
    for m in grid:
        g = gains[m]
        boot = 1000 * g[idx].sum(1) / n[idx].sum(1)
        months = {}
        for k, gi, ni in zip(keys, g, n):
            e = months.setdefault(k[:7], [0.0, 0.0]); e[0] += gi; e[1] += ni
        res['by_m'][str(m)] = {'gain_per_1000': round(1000 * float(g.sum() / n.sum()), 4),
                               'gain_ci_per_1000': [round(float(np.quantile(boot, .025)), 4), round(float(np.quantile(boot, .975)), 4)],
                               'by_month_per_1000': {mo: round(1000 * a / b, 4) for mo, (a, b) in sorted(months.items())},
                               'offsets_at_end': dict(zip(LABELS, last[m]))}
    return res


def cross_season(res):
    """Each season scored at the m with the larger gain on the other season (LEVEL-01's prediction (1))."""
    labs = [k for k in res if isinstance(res[k], dict) and 'by_m' in res[k]]
    out = {}
    for lab in labs:
        other = [x for x in labs if x != lab]
        if not other:
            continue
        bm = res[other[0]]['by_m']
        m = max(bm, key=lambda k: bm[k]['gain_per_1000'])
        r = res[lab]['by_m'][m]
        out[lab] = {'m_from_other': int(m), 'gain_per_1000': r['gain_per_1000'], 'gain_ci_per_1000': r['gain_ci_per_1000'],
                    'holds': r['gain_ci_per_1000'][0] > 0}
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
    ap.add_argument('--out', help='the scores (LEVEL-01)')
    ap.add_argument('--write-offsets', help='directory: write level_offsets_<season>.json, each date from earlier dates only (LEVEL-02)')
    ap.add_argument('--m', type=float, default=10000.0, help='pseudo plate appearances for --write-offsets (LEVEL-01 picked 10,000)')
    a = ap.parse_args()
    if a.write_offsets:
        for spec in a.season:
            lab, pat = spec.split('=', 1)
            files = sorted(glob.glob(pat)); assert files, pat
            doc = {'schema': 'brl.level-offsets.v1', 'labels': list(LABELS), 'm': a.m,
                   'source': 'LEVEL-01: log((observed + m share) / (predicted + m share)) per class from the season\'s real plate appearances on earlier dates, '
                             'against the live stack with tables from other seasons (check-only replay files ' + Path(files[0]).name.split('-s')[0] + ')',
                   'offsets': date_offsets(by_date(load_games(files)), a.m)}
            Path(a.write_offsets, f'level_offsets_{lab}.json').write_text(json.dumps(doc, indent=0))
            print('wrote', lab, len(doc['offsets']), 'dates')
        return
    assert a.out, '--out or --write-offsets'
    out = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat)); assert files, pat
        out[lab] = season_check(load_games(files))
    out['cross'] = cross_season(out)
    out['rule_holds'] = bool(out['cross']) and all(v['holds'] for v in out['cross'].values())
    Path(a.out).write_text(json.dumps(out, indent=1))
    for lab, r in out.items():
        if not isinstance(r, dict) or 'by_m' not in r:
            continue
        print(lab, 'dates', r['dates'], 'pa', r['pa'], 'season offsets', r['season_offsets'])
        for m, x in r['by_m'].items():
            print('  m', m, 'gain per 1000 PA %+.4f [%+.4f, %+.4f]' % (x['gain_per_1000'], *x['gain_ci_per_1000']), 'by month', x['by_month_per_1000'])
    print('cross-season', out['cross'], 'rule holds', out['rule_holds'])


if __name__ == '__main__':
    main()
