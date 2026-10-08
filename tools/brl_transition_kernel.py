"""Build the engine's empirical base-running kernel (brl_live/transitions.json) from the transitions lane's counts.

usage: python tools/brl_transition_kernel.py OUT.json research/transitions-2023-<run>.json [more seasons ...]
(paths on the ledger branch are read with git show origin/brl-live-data:<path>; local files are read directly)

Cells: for each outcome, bases mask and outs at contact, the share of each joint destination pattern of the runners and
the batter, from every listed season; cells with fewer than MIN_CELL plate appearances are left out (the engine keeps
its hand-set rule there); single-occurrence patterns in large cells are dropped as parse noise; illegal patterns
(a runner moving backward, two runners on one base, more than three outs) are dropped and counted.
Tilts: for each advance decision, a logistic regression of the aggressive result on the runner's speed (engine scale,
bucket midpoints) with outs as factors; center is the mean speed of runners in that decision.
"""
from __future__ import annotations
import json, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SIM = {'BIP_OUT': 'bip_out', 'K': 'strikeout', 'BB_HBP': 'bb_hbp', '1B': 'single', '2B_3B': 'double_triple', 'HR': 'home_run', 'OTHER_REACH': 'other_reach'}
MIN_CELL = 30
# Box-score wording the counts cannot separate, measured on 2023-2025 Statcast regular-season plate appearances: the share
# of sacrifice flies among outs with fewer than two out on which the runner from third scored (not double plays; 5,954
# plays), and the share of fielder's choices among batters reaching without a hit or an out, by the bases occupied.
DESCRIPTIONS = {'sac_fly_share': 0.588, 'fielders_choice_share': {'0': 0.0, '1': 0.514, '2': 0.173, '3': 0.573, '4': 0.441, '5': 0.602, '6': 0.555, '7': 0.574}}
MID = {'0': 0.22, '1': 0.35, '2': 0.45, '3': 0.55, '4': 0.65, '5': 0.78}
AGGRESSIVE = {'single_from_2nd': 'H', 'single_from_1st': '3H', 'double_from_1st': 'H', 'out_from_3rd': 'H', 'out_from_2nd': '3H',
              'double_play_runner': ('dp',), 'double_play_batter': ('dp',)}


def read(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    raw = subprocess.run(['git', 'show', 'origin/brl-live-data:' + path], capture_output=True, check=True).stdout
    return json.loads(raw)


def legal(pattern: str, mask: int, outs: int) -> bool:
    if len(pattern) != 4 or '?' in pattern:
        return False
    occupied = []
    for slot in range(3):
        d = pattern[slot]
        if ((mask >> slot) & 1) != (d != '-'):
            return False
        if d in '123':
            if int(d) < slot + 1:
                return False
            occupied.append(d)
    if pattern[3] == '-':
        return False
    if pattern[3] in '123':
        occupied.append(pattern[3])
    if len(occupied) != len(set(occupied)):
        return False
    return outs + pattern.count('X') <= 3


def logistic(x, y, w, factors, iters=50):
    """Weighted logistic regression of y on [1, x, factor dummies]; returns the slope on x."""
    levels = sorted(set(factors))[1:]
    X = np.column_stack([np.ones(len(x)), x] + [np.asarray([f == l for f in factors], float) for l in levels])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (w * (y - p)); H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-9:
            break
    se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    return float(b[1]), se


def build(docs: list[dict]) -> dict:
    cells = defaultdict(Counter)
    dropped = Counter()
    dec = defaultdict(Counter)
    for doc in docs:
        for key, pats in doc['contact'].items():
            outcome, mask, outs = key.split('|')
            for pm, n in pats.items():
                pattern, made = pm.split('|')
                if not legal(pattern, int(mask), int(outs)):
                    dropped['illegal'] += n
                    continue
                cells[(SIM[outcome], int(mask), int(outs))][pattern] += n
        for key, res in doc['decisions'].items():
            for r, n in res.items():
                dec[key][r] += n
    out_cells, small = {}, 0
    for (outcome, mask, outs), c in sorted(cells.items()):
        total = sum(c.values())
        if total < MIN_CELL:
            small += total
            continue
        keep = {p: n for p, n in c.items() if not (n == 1 and total >= 500)}
        dropped['singletons'] += total - sum(keep.values())
        t = sum(keep.values())
        out_cells[f'{outcome}|{mask}|{outs}'] = [[p, round(n / t, 6), n] for p, n in sorted(keep.items(), key=lambda kv: -kv[1])]
    tilts = {}
    for decision, aggressive in AGGRESSIVE.items():
        x, y, w, f = [], [], [], []
        speed_sum = speed_n = 0.0
        for key, res in dec.items():
            d, outs, b = key.split('|')
            if d != decision or b == 'na':
                continue
            for r, n in res.items():
                x.append(MID[b]); y.append(1.0 if r in aggressive else 0.0); w.append(float(n)); f.append(outs)
                speed_sum += MID[b] * n; speed_n += n
        if not x:
            continue
        beta, se = logistic(np.asarray(x), np.asarray(y), np.asarray(w), f)
        tilts[decision] = {'beta': round(beta, 3), 'se': round(se, 3), 'center': round(speed_sum / speed_n, 4), 'n': int(speed_n)}
    return {'cells': out_cells, 'tilts': tilts, 'dropped': dict(dropped), 'plate_appearances_in_small_cells': small}


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    docs = [read(p) for p in paths]
    for p, d in zip(paths, docs):
        if d.get('status') != 'completed':
            raise SystemExit(f'{p} is not a completed transitions receipt')
    k = build(docs)
    doc = {'schema': 'brl.transition-kernel.v1', 'name': 'base running from ' + ', '.join(str(d['season']) for d in docs) + ' play-by-play',
           'seasons': [d['season'] for d in docs], 'sources': paths, 'min_cell': MIN_CELL,
           'plate_appearances': sum(d.get('plays', 0) for d in docs), 'descriptions': DESCRIPTIONS, **k}
    Path(out).write_text(json.dumps(doc, indent=0) + '\n')
    print(json.dumps({'cells': len(k['cells']), 'tilts': k['tilts'], 'dropped': k['dropped'], 'small': k['plate_appearances_in_small_cells']}, indent=1))


if __name__ == '__main__':
    main()
