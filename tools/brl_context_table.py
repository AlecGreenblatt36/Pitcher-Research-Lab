"""Context offsets (brl_live/provider_adjust.ContextAdjust) for a PA model, from a check-only replay of that model
(n_sims 0 with real_pa_check): the bare model's predictions and the observed outcomes of every real plate appearance,
summed in the six batting-side x inning buckets, offset = log((observed + k x share) / (predicted + k x share)) per class,
share = the bucket's predicted shares, k = 200 pseudo plate appearances (CTX-02's estimator; it reproduces the live
tables from their own check runs to four decimals).
Usage: python tools/brl_context_table.py <check replay tag[,tag2,...]> <out.json> [model name]   (several tags pool their pitches)"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[1])
LABELS = ['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH']


def _git(*args, binary=False):
    out = subprocess.run(['git', '-C', REPO, *args], capture_output=True)
    return out.stdout if binary else out.stdout.decode()


def bucket_sums(rows) -> tuple[dict, dict, str]:
    obs, pred, last = {}, {}, ''
    for r in rows:
        last = max(last, str(r.get('date', '')))
        for g in (r.get('real_pa') or {}).get('groups', []):
            key = f'{g[0]}_{g[1]}'
            obs[key] = obs.get(key, 0) + np.asarray(g[5], float)
            pred[key] = pred.get(key, 0) + np.asarray(g[6], float)
    return obs, pred, last


def offsets(obs: dict, pred: dict, k: float = 200.0) -> dict:
    out = {}
    for key in sorted(obs):
        share = pred[key] / pred[key].sum()
        out[key] = {l: round(float(v), 4) for l, v in zip(LABELS, np.log((obs[key] + k * share) / (pred[key] + k * share)))}
    return out


def main():
    tag, target = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else ''
    tags = [t for t in tag.split(',') if t]
    files = sorted(f for f in _git('ls-tree', '-r', '--name-only', 'origin/brl-live-data', 'research').split()
                   if any(f.startswith(f'research/replay-{t}-') for t in tags) and f.endswith('.jsonl.gz'))
    rows = [json.loads(l) for f in files for l in gzip.decompress(_git('show', 'origin/brl-live-data:' + f, binary=True)).decode().splitlines() if l.strip()]
    obs, pred, last = bucket_sums(rows)
    doc = {'schema': 'brl.context-offsets.v1', 'labels': LABELS, 'pseudo_pa': 200.0, 'estimated_through': last,
           'source': f"log(observed/predicted) per class from {model or 'the model'}'s own bare predictions on every regular-season plate appearance of the "
                     f"check-only replay {tag} ({len(files)} shards, {len(rows)} games); buckets are batting side (0 away, 1 home) x inning (1st, 2nd-8th as mid, "
                     "9th and later as late); 200 pseudo plate appearances per bucket at the bucket's predicted shares pull offsets toward zero (tools/brl_context_table.py)",
           'offsets': offsets(obs, pred)}
    Path(target).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({'games': len(rows), 'estimated_through': last, 'buckets': len(doc['offsets'])}))


if __name__ == '__main__':
    main()
