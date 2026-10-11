"""TOTALS-05 on the page: nbRuns keeps the simulator's mean and widens its variance by RUN_SPREAD, as
tools/brl_runs_shape.py scored it; totalShape is the two teams' convolution; the Odds tab and the after-game ranges use them."""
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'
spec = importlib.util.spec_from_file_location('brl_runs_shape', Path(__file__).parents[1] / 'tools/brl_runs_shape.py')
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)


def _script():
    return TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]


def _fn(s, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\(.*?\n  \}\n', s, re.S); assert m, name; return m.group(0)


def _line(s, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\([^\n]*\}\n', s); assert m, name; return m.group(0)


def test_page_shape_matches_the_scored_tool():
    s = _script()
    consts = re.search(r'\n  var RUN_SPREAD = .*?;\n', s).group(0) + re.search(r'\n  var LANCZOS = .*?;\n', s).group(0)
    fns = ''.join(_line(s, n) for n in ('total', 'mean', 'sd', 'probAtLeast', 'mode', 'asDist', 'runShape')) + ''.join(_fn(s, n) for n in ('quant', 'lgam', 'nbRuns', 'totalShape'))
    a = {'n': 1000, 'counts': [[0, 90], [1, 160], [2, 180], [3, 170], [4, 140], [5, 100], [6, 70], [7, 50], [8, 25], [9, 15]]}
    h = {'n': 1000, 'counts': [[0, 60], [1, 120], [2, 160], [3, 170], [4, 160], [5, 120], [6, 90], [7, 60], [8, 40], [9, 20]]}
    js = ("const a=" + json.dumps(a) + ",h=" + json.dumps(h) + ";const p=nbRuns(a),t=totalShape(a,h);"
          "process.stdout.write(JSON.stringify({p:p,q:[quant(runShape(a),.25),quant(runShape(a),.75)],t:[quant(t,.25),quant(t,.75),probAtLeast(t,10),mode(t)],g:[lgam(1),lgam(5),lgam(0.5)],c:RUN_SPREAD}));")
    out = json.loads(subprocess.run(['node', '-e', consts + fns + js], check=True, capture_output=True, text=True).stdout)
    assert abs(out['g'][0]) < 1e-9 and abs(out['g'][1] - np.log(24)) < 1e-9 and abs(out['g'][2] - 0.5 * np.log(np.pi)) < 1e-9
    ha = np.zeros(10); [ha.__setitem__(k, v) for k, v in a['counts']]; hh = np.zeros(10); [hh.__setitem__(k, v) for k, v in h['counts']]
    pa, ph = rs.hist_pmf(ha), rs.hist_pmf(hh)
    ma, va = rs.moments(pa); mh, vh = rs.moments(ph)
    want = rs.nb_pmf(ma, out['c'] * va)
    assert np.allclose(out['p'], want, atol=1e-9)
    tot = np.convolve(want, rs.nb_pmf(mh, out['c'] * vh))
    assert out['t'][:2] == [rs.quant(tot, .25), rs.quant(tot, .75)] and abs(out['t'][2] - tot[10:].sum()) < 1e-9
    assert out['q'] == [rs.quant(want, .25), rs.quant(want, .75)]


def test_odds_tab_and_after_game_ranges_use_the_shape():
    s = _script()
    odds = _fn(s, 'oddsTab')
    assert 'runShape(ra)' in odds and 'runsChart(sa, sh)' in odds and 'totalShape(ra, rh)' in odds and 'mode(ts)' in odds and 'probAtLeast(ts, 10)' in odds
    assert 'mean(tr)' in odds                                  # the expected total stays the simulator's own mean
    assert 'c = runShape(c);' in _fn(s, 'howClose')


def test_page_strikeout_spread_matches_the_scored_tool():
    """PLAYER-08: kShape is tools/brl_k_spread.py's spread with K_SPREAD, and the ladder, the ranges and the after-game
    range read it."""
    spec2 = importlib.util.spec_from_file_location('brl_k_spread', Path(__file__).parents[1] / 'tools/brl_k_spread.py')
    ks = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(ks)
    s = _script()
    consts = re.search(r'\n  var K_SPREAD = .*?;\n', s).group(0)
    fns = ''.join(_line(s, n) for n in ('total', 'mean')) + _fn(s, 'kShape')
    counts = [[2, 30], [3, 90], [4, 160], [5, 210], [6, 200], [7, 150], [8, 90], [9, 50], [10, 20]]
    js = "process.stdout.write(JSON.stringify(kShape({counts:" + json.dumps(counts) + "}).counts.map(function(x){return x[1]/10000;})));"
    got = np.array(json.loads(subprocess.run(['node', '-e', consts + fns + js], check=True, capture_output=True, text=True).stdout))
    raw = np.zeros(ks.N); [raw.__setitem__(k, v) for k, v in counts]
    c = float(re.search(r'\n  var K_SPREAD = (.*?);\n', s).group(1))
    want = ks.spread(raw / raw.sum(), c)
    assert np.allclose(got, want[:len(got)], atol=1e-12) and abs(want[len(got):].sum()) < 1e-12
    assert 'propLadder(kShape(c)' in s and 'mode(kShape(d.K))' in s and 'var ks = kShape(p.distributions.K)' in s
