"""SPLITS-01's Expected split by pitcher hand on the page (splitExpected): his level against all pitchers, lightly shrunk,
times his side's league rate against that hand over its rate against both; misses add his own swings against that hand
with 1,600 pseudo-swings at that rate. The same arithmetic as tools/brl_report.py zone_estimates on two columns."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'


def _script():
    return TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]


def _fn(script, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\(.*?\n  \}\n', script, re.S)
    assert m, name
    return m.group(0)


def _cells(rng, ab, avg, swings, miss, ooz_pitches, chase):
    """13 zone cells [zone, pitches, swings, misses, ab, hits, tb] with the totals asked for, spread over the zones."""
    zones = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
    out = []
    for i, z in enumerate(zones):
        inz = z <= 9
        a = ab // 13 + (1 if i < ab % 13 else 0); h = int(round(a * avg))
        sw = (swings // 9) if inz else int(round(ooz_pitches / 4 * chase))
        p = sw * 2 if inz else ooz_pitches // 4
        out.append([z, p, sw, int(round(sw * miss)), a, h, h + (h // 3)])
    return out


def _run(js):
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('SPLIT_EXP_M', 'SPLIT_CARRY'))
    fns = ''.join(_fn(s, n) for n in ('hzTotal', 'hzCells', 'hzSplit', 'hzLeague', 'isSwitch', 'splitExpected'))
    out = subprocess.run(['node', '-e', "const assert=require('assert');" + consts + fns + js], check=True, capture_output=True, text=True)
    return out.stdout


def test_expected_split_matches_the_studys_scaled_and_shrunk_rates():
    rng = np.random.default_rng(1)
    me = {'R': {'all': _cells(rng, 400, .290, 900, .25, 800, .30)}, 'L': {'all': _cells(rng, 120, .200, 300, .35, 260, .36)}}
    lgR = {'R': {'all': _cells(rng, 90000, .250, 200000, .24, 150000, .29)}, 'L': {'all': _cells(rng, 40000, .235, 90000, .27, 70000, .31)}}
    card = {'side': 'R', 'zones': me}
    lg = {'zones': {'R': lgR, 'L': lgR}}
    got = json.loads(_run("process.stdout.write(JSON.stringify([splitExpected(" + json.dumps(card) + "," + json.dumps(lg) + ",'R'),splitExpected("
                          + json.dumps(card) + "," + json.dumps(lg) + ",'L')]));"))
    tot = lambda cells, ooz=False: np.array([c for c in cells if (c[0] >= 11) == ooz or not ooz], float).sum(0)

    def py(hand, num, den, ooz, m=None, clip=True):
        mine, both = tot(me[hand]['all'], ooz), tot(me['R']['all'], ooz) + tot(me['L']['all'], ooz)
        l1 = tot(lgR[hand]['all'], ooz); la_ = tot(lgR['R']['all'], ooz) + tot(lgR['L']['all'], ooz)
        la = la_[num] / la_[den]; h_all = (both[num] + 30 * la) / (both[den] + 30); sc = l1[num] / l1[den] * h_all / la
        if clip:
            sc = min(0.995, max(0.005, sc))
        return (mine[num] + m * sc) / (mine[den] + m) if m else sc
    for k, hand in ((0, 'R'), (1, 'L')):
        g = got[k]
        assert abs(g['avg'] - py(hand, 5, 4, False)) < 1e-9 and abs(g['slg'] - py(hand, 6, 4, False, clip=False)) < 1e-9
        assert abs(g['chase'] - py(hand, 2, 1, True)) < 1e-9 and abs(g['miss'] - py(hand, 3, 2, False, m=1600)) < 1e-9
    # the expected average against lefties sits near his level times the league's split, not at his raw .200
    assert got[1]['avg'] > 0.25 and got[1]['avg'] < got[0]['avg']


def test_no_expected_split_for_switch_hitters_both_hands_or_missing_league():
    rng = np.random.default_rng(2)
    z = {'R': {'all': _cells(rng, 300, .27, 700, .25, 600, .3)}, 'L': {'all': _cells(rng, 100, .25, 250, .25, 200, .3)}}
    lgz = {'R': {'all': _cells(rng, 9000, .25, 20000, .24, 15000, .29)}, 'L': {'all': _cells(rng, 4000, .24, 9000, .27, 7000, .31)}}
    lg = {'zones': {'R': lgz, 'L': lgz}}
    out = _run("const lg=" + json.dumps(lg) + ",z=" + json.dumps(z) + ";"
               "assert.equal(splitExpected({side:'L',side_by_hand:{R:'L',L:'R'},zones:z},lg,'R'),null);"
               "assert.equal(splitExpected({side:'R',zones:z},lg,'all'),null);"
               "assert.equal(splitExpected({side:'R',zones:z},{},'R'),null);"
               "assert.ok(splitExpected({side:'L',zones:z},lg,'L').avg>0);process.stdout.write('ok');")
    assert out == 'ok'
