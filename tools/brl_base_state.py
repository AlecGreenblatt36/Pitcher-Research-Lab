"""Base-state offsets (RUNS-02): the probability stack's miss by bases and outs, relative to its miss over the season.

usage: python tools/brl_base_state.py OUT.json TAG [TAG ...]

TAG names a check-only replay on the brl-live-data branch run with real_pa_check (research/replay-TAG-*.jsonl.gz rows
carry, per game, the observed and predicted counts per outcome class in nine cells: bases empty, first only, a runner in
scoring position, by outs). For each replay the season's level is observed over predicted per class over all cells; a
cell's offset is log(cell ratio / season ratio), shrunk toward its bases group (all outs) by K pseudo plate appearances at
the cell's predicted shares. With several replays the offsets are averaged, weighted by plate appearances. The level is
left to the context offsets (CTX-02); this table only carries the shape by situation. Only offsets and counts are written.
"""
from __future__ import annotations
import argparse, gzip, json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')
K = 2000.0


def cells(tag: str) -> dict:
    names = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', 'origin/brl-live-data', 'research'],
                           capture_output=True, text=True, check=True).stdout.split()
    out, seen = {}, set()
    for f in sorted(n for n in names if n.startswith(f'research/replay-{tag}-') and n.endswith('.jsonl.gz')):
        raw = subprocess.run(['git', '-C', str(ROOT), 'show', 'origin/brl-live-data:' + f], capture_output=True, check=True).stdout
        for line in gzip.decompress(raw).decode().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d['game_pk'] in seen:
                continue
            seen.add(d['game_pk'])
            for b, o, n, obs, pred in (d.get('real_pa') or {}).get('bases') or []:
                e = out.setdefault((int(b), int(o)), [0, np.zeros(7), np.zeros(7)])
                e[0] += int(n); e[1] += np.asarray(obs, float); e[2] += np.asarray(pred, float)
    return out


def shape(c: dict, k: float = K) -> tuple[dict, dict]:
    """(offsets by cell, counts): log(cell ratio / season ratio), shrunk toward the bases group's value."""
    lvl = sum(v[1] for v in c.values()) / sum(v[2] for v in c.values())
    group = {}
    for b in range(3):
        o = sum(c[(b, x)][1] for x in range(3) if (b, x) in c); p = sum(c[(b, x)][2] for x in range(3) if (b, x) in c)
        group[b] = (o / p) / lvl
    off, n = {}, {}
    for (b, x), (cnt, o, p) in c.items():
        share = p / cnt
        off[(b, x)] = np.log(((o + k * share * group[b] * lvl) / (p + k * share)) / lvl)
        n[(b, x)] = cnt
    return off, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('tags', nargs='+'); ap.add_argument('--name', default='')
    a = ap.parse_args()
    parts = [shape(cells(t)) for t in a.tags]
    keys = sorted(parts[0][0])
    offsets, counts = {}, {}
    for key in keys:
        w = np.array([p[1][key] for p in parts], float)
        v = sum(p[0][key] * wi for p, wi in zip(parts, w)) / w.sum()
        offsets[f'{key[0]}|{key[1]}'] = {l: round(float(x), 5) for l, x in zip(LABELS, v)}
        counts[f'{key[0]}|{key[1]}'] = int(w.sum())
    doc = {'schema': 'brl.base-state-offsets.v1', 'labels': list(LABELS), 'pseudo_pa': K,
           'name': a.name or 'bases and outs shape from ' + ', '.join(a.tags),
           'source': ("log(cell ratio / season ratio) of observed over predicted per class, the full probability stack on "
                      "every real plate appearance (check-only replays " + ', '.join(a.tags) + "); cells are bases (0 empty, "
                      "1 first only, 2 a runner in scoring position) by outs; shrunk toward the bases group by "
                      f"{int(K)} pseudo plate appearances at the cell's predicted shares (RUNS-02)."),
           'plate_appearances': counts, 'offsets': offsets}
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({c: {l: round(float(np.exp(v)), 3) for l, v in o.items()} for c, o in offsets.items()}, indent=0)[:3000])


if __name__ == '__main__':
    main()
