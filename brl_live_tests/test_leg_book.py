"""LEG-01's leg book: the page's run and strikeout reshaping ported exactly (so 'as shown' chances are the page's), the over
and push rules, the two-sided legs, and the edge test's arithmetic."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import brl_leg_book as lb  # noqa: E402

TEMPLATE = ROOT / 'brl_live/page/template.html'


def _js():
    s = TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]

    def fn(name):
        m = re.search(r'\n  function ' + name + r'\(.*?\n  \}\n', s, re.S) or re.search(r'\n  function ' + name + r'\([^\n]*\}\n', s)
        return m.group(0)
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s).group(0) for v in ('RUN_SPREAD', 'LANCZOS', 'K_SPREAD'))
    return consts + ''.join(fn(n) for n in ('total', 'mean', 'sd', 'lgam', 'nbRuns', 'kShape'))


def test_reshaping_matches_the_page():
    runs = [3, 40, 120, 180, 190, 160, 120, 80, 50, 30, 15, 8, 4]
    ks = [0, 2, 10, 40, 90, 130, 140, 110, 70, 30, 10, 3]
    c = lambda h: json.dumps({'counts': [[i, v] for i, v in enumerate(h) if v]})
    out = subprocess.run(['node', '-e', _js() + f'console.log(JSON.stringify([nbRuns({c(runs)}), kShape({c(ks)}).counts.map(function(x){{return x[1]/10000;}})]))'],
                         capture_output=True, text=True, check=True).stdout
    a, b = json.loads(out)
    assert np.allclose(a, lb.nb_runs(runs), atol=1e-12)
    assert np.allclose(b, lb.k_shape(ks), atol=1e-12)
    assert lb.RUN_SPREAD == float(re.search(r'RUN_SPREAD = ([\d.]+)', TEMPLATE.read_text()).group(1))
    assert lb.K_SPREAD == float(re.search(r'K_SPREAD = ([\d.]+)', TEMPLATE.read_text()).group(1))


def test_over_and_push_rules():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert lb.over(p, 1.5) == 0.7 and lb.over(p, 3.5) == 0.0
    assert abs(lb.push_free_over(p, 2) - 0.4 / 0.7) < 1e-12          # a whole-number line: the push is taken out


def test_two_sided_legs_carry_both_sides():
    legs = lb.two_sided('win:home', 0.62, True, {'date': 'd', 'game_pk': 1})
    sides = {l['fam']: (l['p'], l['y']) for l in legs}
    assert len(legs) == 2 and abs(sum(v[0] for v in sides.values()) - 1) < 1e-12
    assert sorted(v[1] for v in sides.values()) == [0, 1]


def test_edge_returns_arithmetic():
    rng = np.random.default_rng(0)
    d = np.array(['a'] * 50 + ['b'] * 50)
    po = np.full(100, 0.6); pm = np.full(100, 0.5); y = np.array([1] * 60 + [0] * 40, float)
    e = lb.edge_returns(d, po, pm, y, rng)['5pts']
    assert e['bets'] == 100 and e['our_side_won'] == 0.6 and e['price_mean'] == 0.5
    assert abs(e['return_fair'] - 0.2) < 1e-9                          # win 60 at 0.50 each: 10 profit on 50 staked
    assert e['return_with_hold'] < e['return_fair']
