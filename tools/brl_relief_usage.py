"""RELIEF-04: relievers per team-game against the starter's outs, scored on full replays with starter lines.

Per team-game: the simulated mean starter outs (starters[side]['outs'], counts of 0 to 27 outs) and the simulated mean
pitchers used (team_pitchers_sim_mean), against the real starter's outs and the real pitchers used (actual outs,
team_actual pitchers). The slope of pitchers on the starter's outs across team-games, simulated means against real
values (with a linear relation inside each game both estimate the same slope), and the mean pitchers per team-game.
Model outputs and box-score facts only.

    python3 tools/brl_relief_usage.py --season 2025='<glob>' --season 2026='<glob>' [--out <file>]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np


def team_games(games):
    out = []
    for g in games:
        for side in ('away', 'home'):
            s = (g.get('starters') or {}).get(side) or {}
            c = np.asarray(s.get('outs') or [], float); a = s.get('actual') or {}; ta = s.get('team_actual') or {}
            if not c.size or c.sum() <= 0 or a.get('outs') is None or ta.get('pitchers') is None or s.get('team_pitchers_sim_mean') is None:
                continue
            out.append((float((np.arange(len(c)) * c).sum() / c.sum()), float(s['team_pitchers_sim_mean']), float(a['outs']), float(ta['pitchers'])))
    return np.array(out, float).reshape(-1, 4)


def season_check(games):
    r = team_games(games)
    assert len(r), 'no team-game carries simulated starter outs and pitchers with the real ones'
    so, sp, ao, ap = r.T
    bands = {}
    for lo, hi in ((0, 11), (12, 14), (15, 17), (18, 20), (21, 27)):
        m = (ao >= lo) & (ao <= hi); ms = (so >= lo) & (so < hi + 1)
        bands[f'{lo}-{hi}'] = {'real_games': int(m.sum()), 'real_pitchers': round(float(ap[m].mean()), 3) if m.any() else None,
                               'sim_games': int(ms.sum()), 'sim_pitchers': round(float(sp[ms].mean()), 3) if ms.any() else None}
    s_sim, s_real = float(np.polyfit(so, sp, 1)[0]), float(np.polyfit(ao, ap, 1)[0])
    return {'team_games': int(len(r)), 'pitchers_sim': round(float(sp.mean()), 4), 'pitchers_real': round(float(ap.mean()), 4),
            'level_gap': round(float(sp.mean() - ap.mean()), 4), 'slope_sim': round(s_sim, 4), 'slope_real': round(s_real, 4),
            'slope_gap': round(s_sim - s_real, 4), 'starter_outs_sim': round(float(so.mean()), 3), 'starter_outs_real': round(float(ao.mean()), 3),
            'by_starter_outs': bands}


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
    ap.add_argument('--out')
    a = ap.parse_args()
    out = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat)); assert files, pat
        out[lab] = season_check(load_games(files))
        r = out[lab]
        print(lab, 'team-games', r['team_games'], 'pitchers sim %.3f real %.3f' % (r['pitchers_sim'], r['pitchers_real']),
              'slope sim %+.4f real %+.4f gap %+.4f' % (r['slope_sim'], r['slope_real'], r['slope_gap']))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
