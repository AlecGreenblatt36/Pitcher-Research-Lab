"""The Odds tab's prop numbers. Hitters (ownAtLeast): each of his lineup slot's events is his with his spot's share of the
slot's plate appearances (PLAYER-02), so the table's 1+ hit and HR agree with the Summary's meters, which the box computes
the same way (brl_live/boxscore.py STARTER_SHARE); 2+ hits and walks thin the same way. Starters (kLadder): the chance of
more than each half line around his median, over and under adding to one (PLAYER-04)."""
import json
import re
import subprocess
from math import comb
from pathlib import Path

import numpy as np

from brl_live.boxscore import STARTER_SHARE

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'


def _script():
    return TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]


def _fn(script, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\(.*?\n  \}\n', script, re.S)
    assert m, name
    return m.group(0)


def _line(script, name):
    """A one-line function (its closing brace on the same line)."""
    m = re.search(r'\n  function ' + re.escape(name) + r'\([^\n]*\}\n', script)
    assert m, name
    return m.group(0)


def _run(js):
    s = _script()
    consts = re.search(r'\n  var SLOT_SHARE = .*?;\n', s, re.S).group(0)
    fns = ''.join(_line(s, n) for n in ('pct', 'total', 'probAtLeast')) + ''.join(_fn(s, n) for n in ('quant', 'ownAtLeast', 'amOdds', 'kLadder'))
    out = subprocess.run(['node', '-e', consts + fns + js], check=True, capture_output=True, text=True)
    return out.stdout


def test_the_page_shares_are_the_boxs_shares():
    s = _script()
    got = json.loads(re.search(r'\n  var SLOT_SHARE = (.*?);\n', s, re.S).group(1))
    assert got == list(STARTER_SHARE)


def test_own_chances_thin_each_event_with_the_spots_share():
    rng = np.random.default_rng(3)
    for spot in (1, 5, 9):
        counts = [[k, int(c)] for k, c in enumerate(rng.integers(10, 400, 6))]
        n = sum(c for _, c in counts)
        keep = STARTER_SHARE[spot - 1]
        got = json.loads(_run('process.stdout.write(JSON.stringify([1,2,3].map(function(k){return ownAtLeast({counts:' + json.dumps(counts) + '},k,' + str(spot) + ');})));'))
        for k, g in zip((1, 2, 3), got):
            below = sum(c * sum(comb(m, j) * keep ** j * (1 - keep) ** (m - j) for j in range(min(k, m + 1))) for m, c in counts) / n
            assert abs(g - (1 - below)) < 1e-12
        # k = 1 is the box's hit_probability formula
        box = 1 - sum(c * (1 - keep) ** m for m, c in counts) / n
        assert abs(got[0] - box) < 1e-12
        # thinning only lowers a chance, and the slot's own chance bounds it
        raw = [sum(c for m, c in counts if m >= k) / n for k in (1, 2, 3)]
        assert all(g <= r + 1e-12 for g, r in zip(got, raw))


def test_strikeout_ladder_centers_on_the_median_and_over_plus_under_is_one():
    counts = [[0, 10], [1, 40], [2, 90], [3, 150], [4, 190], [5, 180], [6, 150], [7, 100], [8, 50], [9, 30], [10, 10]]
    html = _run('process.stdout.write(kLadder({counts:' + json.dumps(counts) + '}));')
    rows = re.findall(r'<tr><th scope="row">([\d.]+)</th><td>(\d+)%<small>([^<]+)</small></td><td>(\d+)%<small>([^<]+)</small></td></tr>', html)
    assert [r[0] for r in rows] == ['3.5', '4.5', '5.5', '6.5']          # the median is 5
    n = sum(c for _, c in counts)
    for line, over, o_odds, under, u_odds in rows:
        p = sum(c for k, c in counts if k > float(line)) / n
        assert int(over) == round(p * 100) and int(over) + int(under) in (99, 100, 101)
        assert (o_odds.startswith('−') or o_odds == '+100') if p >= 0.5 else o_odds.startswith('+')
    # only the lines PLAYER-04 checked (2.5 to 8.5): an opener's median of 0 gets none, an ace's median of 9 two
    low = _run('process.stdout.write(kLadder({counts:[[0,500],[1,300],[2,200]]}));')
    assert low == ''
    ace = _run('process.stdout.write(kLadder({counts:[[7,100],[8,200],[9,300],[10,250],[11,150]]}));')
    assert re.findall(r'<th scope="row">([\d.]+)</th>', ace) == ['7.5', '8.5']
