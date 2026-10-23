"""LEG-02's forward score: the day's top same-game slips (2 and 3 legs, one per player, from the board's verified legs),
their chance counted in the worlds against the multiplied chance, graded on the final box. Reads the ledger (boxes keep
their world bits there for two days). Appends a row per slip to research/leg02/record.jsonl and prints the running record.

    python3 tools/brl_leg02_score.py <ledger.json> [--record research/leg02/record.jsonl] [--slips 5]

Gate (LEDGER LEG-02): after 60 or more games, the worlds' chances within 2 points or 1.96 standard errors of the hit
rate, and not worse than the multiplied chance by Brier.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from brl_leg_board_score import page_legs, outcome  # noqa: E402
from brl_sgp_scan import slip_chance, lead_legs  # noqa: E402


def game_slips(ledger: dict, pk: str, per_size: int):
    fx = [(k, f) for k, f in ledger['forecasts'].items() if str(f['game_pk']) == pk]
    if not fx:
        return []
    fid, f = max(fx, key=lambda x: x[1]['version'])
    box = (ledger.get('box_scores') or {}).get(fid); actual = (ledger.get('actual_boxes') or {}).get(pk); score = (ledger.get('actuals') or {}).get(pk)
    if not (box and box.get('world_legs') and actual and score):
        return []
    legs = page_legs(f, box, ledger.get('market') or {})
    lead = lead_legs(legs, 24)
    out = []
    for k in (2, 3):
        cands = []
        for combo in itertools.combinations(lead, k):
            sc = slip_chance(box, list(combo))
            if sc:
                cands.append((sc, list(combo)))
        cands.sort(key=lambda c: -c[0][0])
        for (p, raw, prod), combo in cands[:per_size]:
            ys = [outcome(l, actual, score) for l in combo]
            if any(y is None for y in ys):
                continue
            out.append({'game_pk': int(pk), 'date': f['date'], 'legs': [l['text'] for l in combo], 'k': k, 'worlds': round(p, 4), 'raw': round(raw, 4),
                        'product': round(prod, 4), 'hit': int(all(ys))})
    return out


def summarize(rows):
    out = {'slips': len(rows), 'games': len({r['game_pk'] for r in rows})}
    for k in (2, 3):
        rk = [r for r in rows if r['k'] == k]
        if not rk:
            continue
        w = np.array([r['worlds'] for r in rk]); pr = np.array([r['product'] for r in rk]); y = np.array([r['hit'] for r in rk], float)
        se = math.sqrt(max(w.mean() * (1 - w.mean()), 1e-9) / len(rk))
        out[f'{k}_legs'] = {'n': len(rk), 'worlds_said': round(float(w.mean()), 4), 'product_said': round(float(pr.mean()), 4), 'hit': round(float(y.mean()), 4),
                            'se': round(se, 4), 'brier_worlds': round(float(((w - y) ** 2).mean()), 5), 'brier_product': round(float(((pr - y) ** 2).mean()), 5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger'); ap.add_argument('--record', default=str(ROOT / 'research/leg02/record.jsonl')); ap.add_argument('--slips', type=int, default=5)
    a = ap.parse_args()
    ledger = json.loads(Path(a.ledger).read_text())
    rec = Path(a.record); rec.parent.mkdir(parents=True, exist_ok=True)
    rows, have = [], set()
    if rec.exists():
        for line in rec.read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows.append(r); have.add((r['game_pk'], tuple(r['legs'])))
    new = []
    for pk in (ledger.get('actuals') or {}):
        for r in game_slips(ledger, str(pk), a.slips):
            if (r['game_pk'], tuple(r['legs'])) not in have:
                new.append(r); have.add((r['game_pk'], tuple(r['legs'])))
    if new:
        with rec.open('a') as fh:
            for r in new:
                fh.write(json.dumps(r) + '\n')
    rows += new
    print(json.dumps({'new_slips': len(new), **summarize(rows)}, indent=1))


if __name__ == '__main__':
    main()
