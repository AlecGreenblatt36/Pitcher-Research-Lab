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
    consts = re.search(r'\n  var SLOT_SHARE = .*?;\n', s, re.S).group(0) + re.search(r'\n  var K_SPREAD = .*?;\n', s).group(0)
    fns = ''.join(_line(s, n) for n in ('pct', 'total', 'mean', 'probAtLeast', 'kLadder', 'outsLadder')) + ''.join(_fn(s, n) for n in ('quant', 'ownAtLeast', 'amOdds', 'propLadder', 'kShape'))
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
    import importlib.util
    spec = importlib.util.spec_from_file_location('brl_k_spread', Path(__file__).parents[1] / 'tools/brl_k_spread.py')
    ks = importlib.util.module_from_spec(spec); spec.loader.exec_module(ks)
    c = float(re.search(r'\n  var K_SPREAD = (.*?);\n', _script()).group(1))
    raw = np.zeros(ks.N); [raw.__setitem__(k, v) for k, v in counts]
    spread = ks.spread(raw / raw.sum(), c)                                   # PLAYER-08: the page states the spread distribution
    for line, over, o_odds, under, u_odds in rows:
        p = float(spread[int(float(line) + 0.5):].sum())
        assert int(over) == round(p * 100) and int(over) + int(under) in (99, 100, 101)
        assert (o_odds.startswith('−') or o_odds == '+100') if p >= 0.5 else o_odds.startswith('+')
    # only the lines PLAYER-04 checked (2.5 to 8.5): an opener's median of 0 gets none, an ace's median of 9 two
    low = _run('process.stdout.write(kLadder({counts:[[0,700],[1,200],[2,100]]}));')
    assert low == ''
    ace = _run('process.stdout.write(kLadder({counts:[[7,100],[8,200],[9,300],[10,250],[11,150]]}));')
    assert re.findall(r'<th scope="row">([\d.]+)</th>', ace) == ['7.5', '8.5']


def test_bullpen_table_lists_likely_arms_and_folds_the_rest():
    s = _script()
    m = re.search(r'\n  var PEN_LIST_MIN = .*?;\n', s).group(0)
    fns = ''.join(_line(s, n) for n in ('pct', 'esc')) + ''.join(_fn(s, n) for n in ('lastName', 'penTable'))
    js = ("const assert=require('assert');"
          "const pen=[{name:'Tanner Scott',appearance_probability:.62},{name:'Edwin Diaz',appearance_probability:.49},"
          "{name:'Blake Snell',appearance_probability:.06},{name:'Shohei Ohtani',appearance_probability:.05}];"
          "const h=penTable(pen);"
          "assert.equal((h.match(/<tr>/g)||[]).length,3);"                       # the header and two arms
          "assert.ok(h.indexOf('Less likely: Snell\\u00a06%, Ohtani\\u00a05%.')>0,h);"
          "const low=penTable(pen.slice(2));assert.equal((low.match(/<tr>/g)||[]).length,3);assert.ok(low.indexOf('Less likely')<0,low);")
    subprocess.run(['node', '-e', m + fns + js], check=True)


def test_outs_ladder_uses_the_same_rule_at_the_outs_lines():
    """PLAYER-07's ladder: outs recorded around the median, within the lines it checked (12.5 to 20.5); the header names the prop."""
    counts = [[k, c] for k, c in zip(range(10, 24), [5, 10, 20, 40, 70, 110, 150, 170, 150, 110, 70, 40, 20, 10])]
    html = _run('process.stdout.write(outsLadder({counts:' + json.dumps(counts) + '}));')
    assert '<th scope="col">Outs</th>' in html
    assert re.findall(r'<th scope="row">([\d.]+)</th>', html) == ['15.5', '16.5', '17.5', '18.5']      # the median is 17
    n = sum(c for _, c in counts)
    over = re.findall(r'<th scope="row">16.5</th><td>(\d+)%', html)[0]
    assert int(over) == round(100 * sum(c for k, c in counts if k >= 17) / n)
    assert _run('process.stdout.write(outsLadder({counts:[[1,500],[2,300],[3,200]]}));') == ''          # an opener: none
    long = _run('process.stdout.write(outsLadder({counts:[[19,100],[20,200],[21,400],[22,300]]}));')   # median 21: only 19.5 and 20.5
    assert re.findall(r'<th scope="row">([\d.]+)</th>', long) == ['19.5', '20.5']


def test_outs_ladder_only_for_boxes_simulated_with_the_exit_correction():
    """LEASH-02 and PLAYER-07: the outs ladder held only with the corrected starter-exit hazard, so it shows for boxes whose
    adjustments name it (regular-season games) and never for the plain hazard's (the postseason, earlier boxes)."""
    s = _script()
    fns = re.search(r'\n  var K_SPREAD = .*?;\n', s).group(0) + ''.join(_line(s, n) for n in ('pct', 'total', 'mean', 'probAtLeast', 'kLadder', 'outsLadder', 'exitsByInning')) + ''.join(_fn(s, n) for n in ('quant', 'amOdds', 'propLadder', 'ladders', 'kShape'))
    js = ("const assert=require('assert');"
          "const d={K:{counts:[[3,100],[5,300],[7,100]]},outs:{counts:[[12,100],[15,200],[16,200],[18,300],[21,200]]}};"
          "assert.ok(exitsByInning({adjustments:['context offsets','LEASH-01 leash; starter exits by inning (LEASH-02)']}));"
          "assert.ok(!exitsByInning({adjustments:['postseason starter usage (expected batters x0.91)']}));assert.ok(!exitsByInning({}));assert.ok(!exitsByInning(null));"
          "assert.ok(ladders(d,true).indexOf('>Outs<')>0 && ladders(d,true).indexOf('>Strikeouts<')>0);"
          "assert.ok(ladders(d,false).indexOf('>Outs<')<0 && ladders(d,false).indexOf('>Strikeouts<')>0);")
    subprocess.run(['node', '-e', fns + js], check=True)
    assert re.search(r'\n  var OUTS_LADDER = true;\n', s)


def test_price_board_gives_expected_runs_for_the_total_not_an_over_price():
    """TOTALS-03: no over chance of ours beat the closing total, so the board's total row gives the runs we expect (the
    taught mix of the simulator and the line from TOTAL_TAUGHT.from, else the simulator's mean), never an over price."""
    s = _script()
    tt = re.search(r'\n  var TOTAL_TAUGHT = .*?;', s).group(0)
    fns = ''.join(_line(s, n) for n in ('esc', 'dist', 'total', 'mean', 'amText')) + ''.join(_fn(s, n) for n in ('amOdds', 'taughtTotalFor', 'lineBoard'))
    js = ("const assert=require('assert');"
          "function homeProb(f){return 0.51;} function nickname(n){return n;} function timeET(t){return '3:15 AM';}"
          "const box={total_run_distribution:{counts:[[4,200],[6,300],[8,300],[10,200]]}};"
          "const mkt={provider:'DraftKings',away_ml:-107,home_ml:-113,over_under:6,over_odds:-112,under_odds:-108,captured_at:'x'};"
          "const f={date:'2026-10-11',away:{abbr:'LAD',name:'Dodgers'},home:{abbr:'MIL',name:'Brewers'}};"
          "const h=lineBoard(f,mkt,box);"
          "const want=(0.32*7.0+0.78*6-0.543).toFixed(1)+' runs';"
          "assert.ok(h.indexOf('Total 6</th><td>o\\u2212112 / u\\u2212108</td><td>'+want+'</td></tr>')>0,h);"
          "assert.ok(h.indexOf('we do not price the over')>0,h);"
          "const old=lineBoard(Object.assign({},f,{date:'2026-10-01'}),mkt,box);assert.ok(old.indexOf('<td>7.0 runs</td>')>0,old);")
    subprocess.run(['node', '-e', tt + fns + js], check=True)


def test_total_runs_states_no_over_chance_of_ours():
    """TOTALS-03: the Total runs section gives the runs we expect and Vegas's total with Vegas's over chance, never the
    simulator's own over chance; and the Odds tab without a box stops after the win chances instead of failing."""
    s = _script()
    odds = _fn(s, 'oddsTab')
    assert 'Vegas total: ' in odds and ': simulator ' not in odds and 'p_over' in odds
    assert re.search(r'h \+= "</section>";\n    if \(!box\) return h \+ ', odds)
    assert 'function marketTotal' not in s


def test_hits_allowed_ladder_shows_for_every_box():
    """PLAYER-09: the hits-allowed ladder held on both replay sets (corrected and plain starter exits), so ladders() adds it
    whether or not the outs ladder shows, at the lines 2.5 to 7.5 around the median."""
    s = _script()
    fns = re.search(r'\n  var K_SPREAD = .*?;\n', s).group(0) + ''.join(_line(s, n) for n in ('pct', 'total', 'mean', 'probAtLeast', 'kLadder', 'outsLadder', 'hLadder')) + ''.join(_fn(s, n) for n in ('quant', 'amOdds', 'propLadder', 'ladders', 'kShape'))
    js = ("const assert=require('assert');"
          "const d={K:{counts:[[3,100],[5,300],[7,100]]},H:{counts:[[2,100],[4,300],[5,300],[7,200],[9,100]]}};"
          "const a=ladders(d,false);assert.ok(a.indexOf('>Hits allowed<')>0 && a.indexOf('>Strikeouts<')>0 && a.indexOf('>Outs<')<0,a);"
          "const rows=(a.split('>Hits allowed<')[1].match(/<th scope=\"row\">([\\d.]+)<\\/th>/g)||[]).map(function(x){return x.replace(/<[^>]+>/g,'');});"
          "assert.deepEqual(rows,['3.5','4.5','5.5','6.5']);"
          "assert.equal(ladders({}, true), '');")
    subprocess.run(['node', '-e', fns + js], check=True)


def test_hitter_props_list_a_strikeout_column():
    """PROD-16: the hitter props table adds 1+ K with the same own-chance rule as the other columns."""
    s = _script()
    assert '<th>Walk</th><th>1+ K</th>' in s and 'pf(ownAtLeast(d.K, 1, b.spot))' in s
