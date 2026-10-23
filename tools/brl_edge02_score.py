"""EDGE-02's forward score: Kalshi's pre-game snapshot (ledger['kalshi'][game]['pre'], the last capture taken 30 minutes or
more before first pitch) against the final box, for the board's legs Kalshi listed. Buy our side where our chance and the
asking price differ by 5 points or more, one contract at the ask, Kalshi's full fee 0.07 x p x (1 - p). Also the
pre-specified secondary: the no side alone on 2+ hits and 2+ total bases. Prints the running record and appends a row per
game to research/edge02/record.jsonl (metrics only: game, legs, side, our chance, the ask, the result).

    python3 tools/brl_edge02_score.py <ledger.json> [--record research/edge02/record.jsonl]

The gate (LEDGER EDGE-02): after 3,000 or more qualifying lines over 60 or more dates, the return per dollar at the ask
after the fee above zero with a date-bootstrap 95% interval excluding zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from brl_leg_board_score import page_legs, outcome  # noqa: E402
from brl_sgp_scan import kalshi_for  # noqa: E402

GAP = 0.05


def fee(p):
    return 0.07 * p * (1 - p)


def score_game(ledger: dict, pk: str):
    fx = [(k, f) for k, f in ledger['forecasts'].items() if str(f['game_pk']) == pk]
    if not fx:
        return []
    fid, f = max(fx, key=lambda x: x[1]['version'])
    box = (ledger.get('box_scores') or {}).get(fid); actual = (ledger.get('actual_boxes') or {}).get(pk); score = (ledger.get('actuals') or {}).get(pk)
    snap = ((ledger.get('kalshi') or {}).get(pk) or {}).get('pre')
    if not (box and actual and score and snap):
        return []
    data = {'kalshi': {pk: snap}}
    rows = []
    for l in page_legs(f, box, ledger.get('market') or {}):
        q = kalshi_for(data, pk, l, f)
        if not q:
            continue
        y = outcome(l, actual, score)
        if y is None:
            continue
        rows.append({'game_pk': int(pk), 'date': f['date'], 'leg': l['text'], 'fam': l['fam'], 'pid': l.get('pid') or '', 'ours': round(l['p'], 4),
                     'ask': round(q['ask'], 4), 'vol': q['vol'], 'captured_at': snap['captured_at'], 'minutes_before': snap.get('minutes_to_first_pitch'), 'hit': int(bool(y))})
    return rows


def summarize(rows: list[dict]):
    """Return per dollar for the qualifying lines (ours 5+ points above the ask, buying at the ask), with a date bootstrap."""
    q = [r for r in rows if r['ours'] - r['ask'] >= GAP]
    out = {'lines_priced': len(rows), 'qualifying': len(q), 'dates': len({r['date'] for r in q})}
    if q:
        cost = np.array([r['ask'] for r in q]); win = np.array([r['hit'] for r in q], float); d = np.array([r['date'] for r in q])
        profit = win - cost - fee(cost)
        out['won'] = round(float(win.mean()), 4); out['price_mean'] = round(float(cost.mean()), 4); out['return_per_dollar'] = round(float(profit.sum() / cost.sum()), 4)
        keys, inv = np.unique(d, return_inverse=True)
        if len(keys) >= 2:
            sp = np.bincount(inv, weights=profit); sc = np.bincount(inv, weights=cost); rng = np.random.default_rng(2)
            idx = rng.integers(0, len(keys), size=(2000, len(keys))); b = sp[idx].sum(1) / sc[idx].sum(1)
            out['return_ci'] = [round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]
    no = [r for r in rows if r['fam'] in ('hitter:hits2+:no', 'hitter:tb2+:no') and r['ours'] - r['ask'] >= GAP]
    if no:
        cost = np.array([r['ask'] for r in no]); win = np.array([r['hit'] for r in no], float); profit = win - cost - fee(cost)
        out['no_side_2plus'] = {'lines': len(no), 'won': round(float(win.mean()), 4), 'return_per_dollar': round(float(profit.sum() / cost.sum()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger'); ap.add_argument('--record', default=str(ROOT / 'research/edge02/record.jsonl'))
    a = ap.parse_args()
    ledger = json.loads(Path(a.ledger).read_text())
    rec = Path(a.record); rec.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    rows = []
    if rec.exists():
        for line in rec.read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows.append(r); have.add((r['game_pk'], r['leg']))
    new = []
    for pk in (ledger.get('kalshi') or {}):
        if pk not in (ledger.get('actuals') or {}):
            continue
        for r in score_game(ledger, pk):
            if (r['game_pk'], r['leg']) not in have:
                new.append(r); have.add((r['game_pk'], r['leg']))
    if new:
        with rec.open('a') as fh:
            for r in new:
                fh.write(json.dumps(r) + '\n')
    rows += new
    print(json.dumps({'new_lines': len(new), **summarize(rows)}, indent=1))


if __name__ == '__main__':
    main()
