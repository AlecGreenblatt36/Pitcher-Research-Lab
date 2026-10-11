"""RUNLINE-02: the run line's conditional piece (given a win, the chance of winning by two or more) from the two teams'
widened run distributions instead of the simulator's own margins.

Each team's simulated runs become TOTALS-05's negative binomial (the simulated mean, the simulated variance times c, c from
the other season; tools/brl_runs_shape.py); the margin is their convolution, and given a home win the chance of two or
more is P(margin >= 2) / P(margin >= 1) (the road side the same way). Scored exactly as RUNLINE-01
(tools/brl_runline_cal.py season_check): against how often winners won by two or more, overall and by fifths of the
conditional chance. Model outputs and final scores only.

    python3 tools/brl_runline_shape.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


rl = _load('brl_runline_cal'); rs = _load('brl_runs_shape')


def margin_hist(g, c):
    """A margin histogram (index = margin + rl.MAX_MARGIN) from the two teams' widened distributions."""
    h = rs.nb_pmf(*(lambda m, v: (m, c * v))(*rs.moments(rs.hist_pmf(g['home_hist']))))
    a = rs.nb_pmf(*(lambda m, v: (m, c * v))(*rs.moments(rs.hist_pmf(g['away_hist']))))
    K = len(h); M = rl.MAX_MARGIN; out = np.zeros(2 * M + 1)
    joint = np.outer(h, a)                                   # [home runs, away runs]
    for d in range(-(K - 1), K):
        s = np.trace(joint, offset=-d)                       # home - away = d
        out[min(max(d, -M), M) + M] += s
    return (out * 1e6).tolist()


def season_check(games, c):
    shaped = [dict(g, margin_hist=margin_hist(g, c)) for g in games]
    return rl.season_check(shaped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    games = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat)); assert files, pat
        games[lab] = [g for g in rl.load_games(files) if g.get('home_hist') and g.get('away_hist')]
    cs = {lab: rs.inflation(rs.team_games(g)) for lab, g in games.items()}
    out = {'c_by_season': {k: round(v, 4) for k, v in cs.items()}}
    for lab, g in games.items():
        other = [k for k in games if k != lab]
        out[lab] = {'reference': rl.season_check(g), 'candidate': season_check(g, cs[other[0]] if other else cs[lab])}
        for kind in ('reference', 'candidate'):
            r = out[lab][kind]
            out[lab][kind + '_holds'] = bool(all(abs(r[s]['gap_points']) <= 1.5 for s in ('home', 'away')) and all(r[s]['max_fifth_gap_points'] <= 3.0 for s in ('home', 'away')))
    out['rule_holds'] = bool(all(out[lab]['candidate_holds'] for lab in games))
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    for lab in games:
        for kind in ('reference', 'candidate'):
            r = out[lab][kind]
            print(lab, kind, {s: (r[s]['conditional_pred'], r[s]['conditional_actual'], r[s]['gap_points'], r[s]['max_fifth_gap_points']) for s in ('home', 'away')}, 'holds', out[lab][kind + '_holds'])
    print('rule holds', out['rule_holds'])


if __name__ == '__main__':
    main()
