"""Build the simulator's running plays other than steals (brl_live/running_events.json) from the transitions lane.

usage: python tools/brl_running_events.py OUT.json research/transitions-2023-<run>.json [more seasons ...]
(paths on the ledger branch are read with git show origin/brl-live-data:<path>; local files are read directly)

For each bases mask and outs at the start of a plate appearance with runners on (or one cut short by a running play),
the chance that a running play other than a steal happens before the batted ball, and what it does: the event kind and
where each runner ends up. Stolen bases and caught stealing (including pickoffs charged as caught stealing) are the
steal step's and are left out, together with the few plays that mix a steal with another event. Patterns where runners
pass each other or the outs do not match are dropped and counted.
"""
from __future__ import annotations
import json, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

STEAL = {'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home', 'caught_stealing_2b', 'caught_stealing_3b',
         'caught_stealing_home', 'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home'}
RANK = {'1': 1, '2': 2, '3': 3, 'H': 4}


def read(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(subprocess.run(['git', 'show', 'origin/brl-live-data:' + path], capture_output=True, check=True).stdout)


def kind_of(tokens: set, pattern: str) -> str:
    if 'X' in pattern:
        if any(t.startswith('pickoff_') and 'error' not in t for t in tokens):
            return 'pickoff'
        return 'runner_out'
    for token, kind in (('wild_pitch', 'wild_pitch'), ('passed_ball', 'passed_ball'), ('balk', 'balk'), ('forced_balk', 'balk')):
        if token in tokens:
            return kind
    if any(t.startswith('pickoff_error') for t in tokens):
        return 'pickoff_error'
    if 'error' in tokens:
        return 'error'
    if 'defensive_indiff' in tokens:
        return 'defensive_indifference'
    return 'other_advance'


def legal(pattern: str, mask: int, made: int) -> bool:
    if len(pattern) != 3:
        return False
    on = []
    for slot in range(3):
        d = pattern[slot]
        if ((mask >> slot) & 1) != (d != '-'):
            return False
        if d in '123' and int(d) < slot + 1:
            return False
        if d != '-' and d != 'X':
            on.append((slot, d))
    bases = [d for _, d in on if d in '123']
    if len(bases) != len(set(bases)):
        return False
    for (s1, d1) in on:                       # runners keep their order
        for (s2, d2) in on:
            if s1 > s2 and not (d1 == 'H' and d2 == 'H') and RANK[d1] <= RANK[d2]:
                return False
    return pattern.count('X') == made


def build(receipts: list[dict]) -> dict:
    opp = Counter(); ev = defaultdict(Counter); dropped = Counter(); kinds = Counter(); games = 0; steal_rows = 0
    for r in receipts:
        games += int(r.get('games') or 0)
        for key, rows in (r.get('pre') or {}).items():
            mask, outs = (int(x) for x in key.split('|'))
            for k, n in rows.items():
                types, rest = k.split('>')
                pattern, made, _ = rest.split('|')
                made = int(made)
                opp[(mask, outs)] += n
                if types == 'none':
                    continue
                tokens = set(types.split('+'))
                if tokens & STEAL:
                    steal_rows += n
                    continue
                if not legal(pattern, mask, made):
                    dropped['illegal'] += n
                    continue
                if pattern == ''.join('-' if not (mask >> s) & 1 else str(s + 1) for s in range(3)):
                    dropped['no movement'] += n
                    continue
                kind = kind_of(tokens, pattern)
                ev[(mask, outs)][(kind, pattern, made)] += n
                kinds[kind] += n
    cells = {}
    for key in sorted(opp):
        tot = sum(ev[key].values())
        if not tot:
            continue
        rows = sorted(ev[key].items(), key=lambda kv: -kv[1])
        cells[f'{key[0]}|{key[1]}'] = {'n': opp[key], 'p': round(tot / opp[key], 6),
                                       'events': [[k, p, m, round(c / tot, 6)] for (k, p, m), c in rows]}
    per_team_game = {k: round(v / max(1, 2 * games), 4) for k, v in kinds.most_common()}
    return {'cells': cells, 'per_team_game': per_team_game, 'games': games, 'dropped': dict(dropped), 'steal_plays_left_to_the_steal_step': steal_rows}


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    receipts = [read(p) for p in paths]
    seasons = sorted(int(r['season']) for r in receipts)
    doc = {'schema': 'brl.running-events.v1', 'name': 'running plays from ' + ', '.join(map(str, seasons)) + ' play-by-play',
           'seasons': seasons, 'sources': paths, **build(receipts)}
    Path(out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({k: doc[k] for k in ('name', 'games', 'per_team_game', 'dropped', 'steal_plays_left_to_the_steal_step')}, indent=1))


if __name__ == '__main__':
    main()
