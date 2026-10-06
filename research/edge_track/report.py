"""Paired edge experiment report; consumes frozen, already-scored records.

This function does not fit or promote a model. Data-integrity, clock and holdout
checks remain required even when the numerical gate passes. Execute with two
JSON inputs containing lists of rows: game_pk,date,metric,entity_id,score,
baseline_score. Every requested metric needs the exact same case identities.
"""
from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def compare(current, candidate, *, seed=20261006, n_boot=5000):
    def index(rows):
        out={}
        for r in rows:
            key=(r['game_pk'],r['date'],r['metric'],r['entity_id'])
            if key in out:raise ValueError('Duplicate metric case')
            if not all(math.isfinite(r[k]) for k in ('score','baseline_score')):
                raise ValueError('Nonfinite score: report separately, never clip')
            out[key]=r
        return out
    a,b=index(current),index(candidate)
    if not a or set(a)!=set(b):raise ValueError('Current and candidate cases do not exactly match')
    if any(a[k]['baseline_score']!=b[k]['baseline_score'] for k in a):
        raise ValueError('Comparator changed between candidate and parent')
    grouped=defaultdict(lambda:defaultdict(list))
    for k in sorted(a):
        game,day,metric,_=k
        grouped[metric][(game,day)].append((a[k]['score'],b[k]['score'],a[k]['baseline_score']))
    effects={}
    for metric,games in sorted(grouped.items()):
        means={key:np.mean(rows,axis=0) for key,rows in games.items()}
        days=sorted(set(day for _,day in games))
        by_day=[np.array([means[g] for g in sorted(means) if g[1]==day]) for day in days]
        sums=np.array([v.sum(axis=0) for v in by_day]);counts=np.array([len(v) for v in by_day])
        original=sums.sum(axis=0)/counts.sum()
        rng=np.random.default_rng(seed);boots=[]
        for _ in range(n_boot):
            draw=rng.integers(len(days),size=len(days))
            values=sums[draw].sum(axis=0)/counts[draw].sum()
            boots.append(values[1]-values[0])
        lo,hi=np.quantile(boots,[.025,.975])
        effects[metric]={'current':float(original[0]),'candidate':float(original[1]),
            'simple_baseline':float(original[2]),'delta':float(original[1]-original[0]),
            'ci95':[float(lo),float(hi)],'n_games':len(games),'n_date_blocks':len(days),
            'n_cases':sum(len(v) for v in games.values()),'mcse_numerical':None,
            'mcse_note':'Not recoverable from case score summaries; retain world-level paired diagnostics separately'}
    primary=('win_brier','win_log_loss')
    passes=all(m in effects and effects[m]['delta']<0 and effects[m]['ci95'][1]<0
               and effects[m]['n_date_blocks']>=2 for m in primary)
    return {'effects':effects,'numerical_game_gate':'PASS' if passes else 'FAIL_OR_NOT_RUN',
        'adoption_authorized':False,'n_bootstrap':n_boot,'seed':seed,
        'weighting':'Case mean within game, equal games; paired date-block bootstrap',
        'notice':'Numerical gate only. No data-provenance/holdout/MC audit is inferred from supplied scores.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('current');parser.add_argument('candidate');parser.add_argument('--output',required=True)
    args=parser.parse_args();paths=[Path(args.current),Path(args.candidate)]
    result=compare(*(json.loads(p.read_text()) for p in paths))
    result['input_sha256']=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
