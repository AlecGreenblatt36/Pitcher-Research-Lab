"""ENG-01: the machine-readable registration record. Every experiment row of LEDGER.md (the tables whose first column is an
ID such as VALUE-17) becomes one record: id, what was tried, the registration and results column, the status column, whether
the row says it was registered before its run, the lane run ids it cites, and the frozen commits it names. Written to
research/registry.json by `python tools/registry.py`; `python tools/registry.py --check VALUE-17` exits non-zero unless that
ID has a row that says it was registered before the run, or that it is a diagnostic (descriptive, no prediction). The discovery
lane runs this before an experiment starts, so no lane run happens without a ledger row. Maintenance runs (ids starting with
DATA-, MAINT- or no id in the commit message) are not checked.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r'^\| ([A-Z][A-Z0-9]*-[0-9]+[A-Za-z0-9]*) \|')
RUN_RE = re.compile(r'\b(3[0-9]{10})\b')
COMMIT_RE = re.compile(r'\b([0-9a-f]{40}|[0-9a-f]{9,12})\b')
REGISTERED_PHRASES = ('Registered in the commit that added this row', 'registered in the commit that added this row', 'Registered before', 'registered before the run',
                      'registered before the lane runs', 'registered before the runs')


def records(text: str) -> list[dict]:
    out = []
    header = None
    for line in text.split('\n'):
        if line.startswith('| ID |'):
            header = [h.strip() for h in line.strip().strip('|').split('|')]
            continue
        m = ID_RE.match(line)
        if not m or header is None:
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        rec = {'id': m.group(1), 'columns': {header[i] if i < len(header) else f'col{i}': cells[i] for i in range(len(cells))}}
        body = ' '.join(cells[1:])
        rec['registered_before_run'] = any(p in body for p in REGISTERED_PHRASES)
        rec['diagnostic'] = ('diagnostic' in body.lower()) and not rec['registered_before_run']
        rec['runs'] = sorted(set(RUN_RE.findall(body)))
        rec['frozen_commits'] = sorted(set(c for c in COMMIT_RE.findall(body) if len(c) >= 9 and any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c)))[:6]
        rec['status'] = cells[-1] if cells else ''
        out.append(rec)
    return out


def build() -> dict:
    text = (ROOT / 'LEDGER.md').read_text()
    recs = records(text)
    by_id = {}
    for r in recs:                      # a repeated id (an edge-track row and a research row) keeps both, in order
        by_id.setdefault(r['id'], []).append(r)
    return {'schema': 'brl.registry.v1', 'source': 'LEDGER.md', 'experiments': int(len(by_id)), 'rows': int(len(recs)), 'records': by_id}


def check(exp_id: str) -> int:
    reg = build()
    rows = reg['records'].get(exp_id)
    if not rows:
        print(f'registry: no ledger row for {exp_id}; register it in LEDGER.md before the run', file=sys.stderr); return 2
    if any(r['registered_before_run'] for r in rows):
        print(f'registry: {exp_id} registered ({len(rows)} row(s))'); return 0
    if any(r['diagnostic'] for r in rows):
        print(f'registry: {exp_id} is a diagnostic (descriptive, no prediction); allowed'); return 0
    print(f'registry: {exp_id} has a ledger row but none says it was registered before the run or that it is a diagnostic', file=sys.stderr); return 3


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == '--check':
        exp_id = argv[1].strip()
        if not exp_id or exp_id.startswith(('DATA-', 'MAINT-')):
            print('registry: maintenance run, no check'); return 0
        return check(exp_id)
    reg = build()
    (ROOT / 'research').mkdir(exist_ok=True)
    (ROOT / 'research' / 'registry.json').write_text(json.dumps(reg, indent=1))
    n_reg = sum(1 for rows in reg['records'].values() for r in rows if r['registered_before_run'])
    print(f"registry: {reg['experiments']} experiments, {reg['rows']} rows, {n_reg} rows registered before their run")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
