"""BULLPEN-05 (diagnostic): where the simulator's relief entries differ from real ones.

Per team-game the replay rows hold relief entries by cell (starters[side]['entries']: 'sim' rates per world and 'real'
counts; cell = entry inning bucket : the fielding team's lead bucket at the half-inning's start : whether the arm is one
of the team's top two by 14-day share; brl_replay/harness.py entry_key). Summed over team-games: entries per team-game by
inning and lead, all arms and the top two, simulated against real. Model outputs and box-score facts only.

    python3 tools/brl_bullpen_entries.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import defaultdict
from pathlib import Path

INN = {'5': '5th or earlier', '6': '6th', '7': '7th', '8': '8th', '9': '9th or later'}
LEAD = {'0': 'down 3+', '1': 'down 1-2', '2': 'tied', '3': 'up 1-3', '4': 'up 4+'}


def season_check(games):
    sim, real, n = defaultdict(float), defaultdict(float), 0
    for g in games:
        for side in ('away', 'home'):
            e = ((g.get('starters') or {}).get(side) or {}).get('entries')
            if not e:
                continue
            n += 1
            for k, v in e['sim'].items():
                sim[k] += v
            for k, v in e['real'].items():
                real[k] += v
    assert n, 'no team-game carries the entries (replay with starter_lines after BULLPEN-05)'
    out = {'team_games': n, 'cells': {}, 'by_inning': {}, 'by_lead': {}, 'top2': {}}
    keys = sorted(set(sim) | set(real))
    for k in keys:
        out['cells'][k] = [round(sim[k] / n, 4), round(real[k] / n, 4)]
    for label, pos, names in (('by_inning', 0, INN), ('by_lead', 1, LEAD)):
        for code, name in names.items():
            s = sum(sim[k] for k in keys if k.split(':')[pos] == code) / n; r = sum(real[k] for k in keys if k.split(':')[pos] == code) / n
            st = sum(sim[k] for k in keys if k.split(':')[pos] == code and k.endswith(':1')) / n; rt = sum(real[k] for k in keys if k.split(':')[pos] == code and k.endswith(':1')) / n
            out[label][name] = {'sim': round(s, 4), 'real': round(r, 4), 'sim_top2': round(st, 4), 'real_top2': round(rt, 4)}
    out['top2']['sim'] = round(sum(v for k, v in sim.items() if k.endswith(':1')) / n, 4)
    out['top2']['real'] = round(sum(v for k, v in real.items() if k.endswith(':1')) / n, 4)
    out['all'] = {'sim': round(sum(sim.values()) / n, 4), 'real': round(sum(real.values()) / n, 4)}
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
    out = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat)); assert files, pat
        out[lab] = season_check(load_games(files))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for lab, r in out.items():
        print(lab, 'team-games', r['team_games'], 'entries sim %.3f real %.3f, top two sim %.3f real %.3f' % (r['all']['sim'], r['all']['real'], r['top2']['sim'], r['top2']['real']))
        for label in ('by_inning', 'by_lead'):
            print('  ', label, ' | '.join('%s %.3f/%.3f (top2 %.3f/%.3f)' % (k, v['sim'], v['real'], v['sim_top2'], v['real_top2']) for k, v in r[label].items()))


if __name__ == '__main__':
    main()
