"""LEG-01 to the page: the verified cells at 70% and up from a leg book run (tools/brl_leg_book.py --as-shown), written into
brl_live/page/template.html as LEG_TABLE (one line), so the leg board lists only legs whose kind, at their chance, held up
in both seasons re-run. Aggregates only: counts, mean stated chances and hit rates per cell.

    python3 tools/brl_leg_table.py <leg book json> [--template brl_live/page/template.html] [--min 0.7]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r'^  var LEG_TABLE = .*;$', re.M)


def table(book: dict, lo: float = 0.7) -> dict:
    if book.get('variant') != 'as shown on the page':
        raise SystemExit('the page table needs the leg book run with --as-shown')
    years = sorted(book['seasons'])
    if len(years) != 2:
        raise SystemExit('two seasons needed')
    verified = set(book.get('verified') or [])
    cells = {}
    for cell in sorted(verified):
        if float(cell.rsplit('|', 1)[1]) < lo - 1e-9:
            continue
        row = []
        for y in years:
            c = book['seasons'][y]['bins'][cell]
            row += [int(c['n']), round(float(c['stated']), 3), round(float(c['hit']), 3)]
        cells[cell] = row + [1]
    return {'experiment': 'LEG-01', 'seasons': [int(y) for y in years], 'min': lo, 'cells': cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('book')
    ap.add_argument('--template', default=str(ROOT / 'brl_live/page/template.html'))
    ap.add_argument('--min', type=float, default=0.7)
    a = ap.parse_args()
    t = table(json.loads(Path(a.book).read_text()), a.min)
    path = Path(a.template); html = path.read_text(encoding='utf-8')
    if len(LINE.findall(html)) != 1:
        raise SystemExit('LEG_TABLE line not found exactly once')
    html = LINE.sub(lambda m: '  var LEG_TABLE = ' + json.dumps(t, separators=(',', ':')) + ';', html)
    path.write_text(html, encoding='utf-8')
    print(json.dumps({'cells': len(t['cells']), 'bytes': len(json.dumps(t, separators=(',', ':')))}))


if __name__ == '__main__':
    main()
