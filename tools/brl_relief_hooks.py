"""Team hook offsets for the fitted reliever exits (RELIEF-03), as served: brl_live/relief_hooks.json.

usage: python tools/brl_relief_hooks.py HISTORY.csv.gz OUT.json --doc brl_live/relief_exit.json --as-of YYYY-MM-DD
                                       [--half-life 90] [--k 10]

Every regular-season relief decision point before --as-of (brl_replay.relief_decisions), the exit hazard's probability
from --doc, and per team the decayed sum of observed minus fitted removals over the decayed sum of p(1 - p) plus k, for
the middle of an inning and for an inning's end. Only per-team offsets and counts are written.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
from brl_replay.relief_decisions import COLS, HookOffsets, decisions, fitted   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('history'); ap.add_argument('out'); ap.add_argument('--doc', required=True); ap.add_argument('--as-of', required=True)
    ap.add_argument('--half-life', type=float, default=90.0); ap.add_argument('--k', type=float, default=10.0)
    a = ap.parse_args()
    d = decisions(pd.read_csv(a.history, usecols=COLS, low_memory=False))
    d = d[d['date'] < a.as_of]
    hooks = HookOffsets(d, fitted(d, json.loads(Path(a.doc).read_text())), half_life=a.half_life, k=a.k)
    table = {t: {k: round(v, 5) for k, v in o.items()} for t, o in hooks.table(a.as_of).items()}
    doc = {'schema': 'brl.relief-hooks.v1', 'name': f'reliever hooks by team through {d["date"].max()}', 'as_of': a.as_of,
           'through': str(d['date'].max()), 'half_life': a.half_life, 'k': a.k, 'exit_doc': str(a.doc), 'decisions': int(len(d)),
           'teams': table}
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({k: doc[k] for k in ('name', 'decisions')}), {t: table[t] for t in list(table)[:5]})


if __name__ == '__main__':
    main()
