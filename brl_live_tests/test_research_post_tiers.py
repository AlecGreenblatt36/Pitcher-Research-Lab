"""POST-03's machinery on made-up seasons: when managers leave their best starters in longer in October, the study sees it;
when they treat everyone alike, it does not."""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _research():
    spec = importlib.util.spec_from_file_location('brl_research_post_test', ROOT / 'tools' / 'brl_research.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _season(rng, year, planted):
    games, gid = {}, 0
    kbb = {p: rng.uniform(0.02, 0.28) for p in range(1, 61)}          # sixty starters with their strikeout-minus-walk rates
    def rows_for(p, bf, other):
        out = []
        for i in range(bf):
            u = rng.random()
            o = 'K' if u < 0.2 + kbb[p] / 2 else ('BB_HBP' if u < 0.2 + kbb[p] / 2 + max(0.02, 0.1 - kbb[p] / 2) else 'BIP_OUT')
            out.append({'p': p, 'half': 'top', 'o': o})
        out += [{'p': other, 'half': 'top', 'o': 'BIP_OUT'} for _ in range(15)]
        return out
    for p in range(1, 61):
        for k in range(25):                                            # regular-season starts around 22 batters
            gid += 1
            rows = rows_for(p, int(rng.normal(22, 2)), 900)
            games[str(gid)] = {'game_type': 'R', 'rows': [dict(r, i=i) for i, r in enumerate(rows)]}
        for k in range(3):                                             # postseason starts: aces kept in when planted
            gid += 1
            ratio = (1.0 if kbb[p] >= 0.2 else 0.8) if planted else 0.87
            rows = rows_for(p, max(1, int(rng.normal(22 * ratio, 2))), 901)
            games[str(gid)] = {'game_type': 'D', 'rows': [dict(r, i=i) for i, r in enumerate(rows)]}
    return {'games': games}


def _run(planted):
    R = _research()
    rng = np.random.default_rng(7)
    docs = {y: _season(rng, y, planted) for y in (2024, 2025)}
    R.read_blob = lambda repo, token, path, branch: path
    R.study_path = lambda y: y
    R.study_purpose = lambda y: y
    R.unseal = lambda raw, key, purpose: gzip.compress(json.dumps(docs[raw]).encode())
    return R.postseason_tiers('r', 't', b'k', 'b', seasons=(2024, 2025), boot=400)


def test_tiers_see_a_planted_difference_and_not_a_null_one():
    out = _run(True)
    assert out['starts'] > 300 and out['top_minus_bottom']['diff'] > 0.12 and out['top_minus_bottom']['ci'][0] > 0.05
    null = _run(False)
    assert abs(null['top_minus_bottom']['diff']) < 0.04 and null['top_minus_bottom']['ci'][0] < 0 < null['top_minus_bottom']['ci'][1]
