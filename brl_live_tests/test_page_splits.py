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
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('SPLIT_EXP_M', 'SPLIT_CARRY', 'PSPLIT_EXP_M'))
    fns = ''.join(_fn(s, n) for n in ('hzTotal', 'hzCells', 'hzSplit', 'hzLeague', 'isSwitch', 'splitExpected', 'splitCore', 'pitcherSplitExpected'))
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


def _hz_run(js):
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('HZ_EXP_M', 'HZ_ANCHOR'))
    metrics = re.search(r'\n  var HZ_METRICS = \{.*?\n  \};\n', s, re.S).group(0)
    fns = ''.join(_fn(s, n) for n in ('hzTotal', 'hzCells', 'hzSplit', 'hzLeague', 'isSwitch', 'hzAnchor', 'hzExpected'))
    out = subprocess.run(['node', '-e', "const assert=require('assert');" + consts + metrics + fns + js], check=True, capture_output=True, text=True)
    return out.stdout


def test_hot_zones_against_a_hand_anchor_on_his_level_against_both_hands():
    # ZONES-03: the prior's level from his pitches against both hands, relative to his side's league rate on them
    rng = np.random.default_rng(3)
    me = {'R': {'all': _cells(rng, 400, .290, 900, .25, 800, .30)}, 'L': {'all': _cells(rng, 120, .330, 300, .35, 260, .36)}}
    lgz = {'R': {'all': _cells(rng, 90000, .250, 200000, .24, 150000, .29)}, 'L': {'all': _cells(rng, 40000, .235, 90000, .27, 70000, .31)}}
    card = {'side': 'R', 'zones': me}
    lg = {'zones': {'R': lgz, 'L': lgz}}
    got = json.loads(_hz_run("const card=" + json.dumps(card) + ",lg=" + json.dumps(lg) + ";"
                             "const an=hzAnchor(card,lg,'L','all');"
                             "process.stdout.write(JSON.stringify([hzExpected(HZ_METRICS.avg,'avg',card.zones.L.all,lg.zones.R.L.all,an),"
                             "hzExpected(HZ_METRICS.avg,'avg',card.zones.L.all,lg.zones.R.L.all,null),"
                             "hzExpected(HZ_METRICS.miss,'miss',card.zones.R.all,lg.zones.R.R.all,hzAnchor(card,lg,'R','all')),"
                             "hzExpected(HZ_METRICS.miss,'miss',card.zones.R.all,lg.zones.R.R.all,null),"
                             "hzAnchor({side:'R',side_by_hand:{R:'L',L:'R'},zones:card.zones},lg,'L','all'),hzAnchor(card,lg,'all','all')]));"))
    tot = lambda cells: np.array(cells, float).sum(0)
    L, mine_all = np.array(lgz['L']['all'], float), tot(me['R']['all']) + tot(me['L']['all'])
    lg_both = tot(lgz['R']['all']) + tot(lgz['L']['all'])
    la = lg_both[5] / lg_both[4]; h_all = (mine_all[5] + 30 * la) / (mine_all[4] + 30)
    for c, l in zip(me['L']['all'], L):
        lz = l[5] / l[4] if l[4] else L[:, 5].sum() / L[:, 4].sum()
        sc = min(0.995, max(0.005, lz * h_all / la))
        assert abs(got[0][str(c[0])] - (c[5] + 320 * sc) / (c[4] + 320)) < 1e-9
    # his raw .330 against lefties no longer lifts the whole chart: the anchored chart sits lower than the old one
    assert np.mean(list(got[0].values())) < np.mean(list(got[1].values()))
    assert got[2] == got[3]                                   # misses against righties keep the old anchor
    assert got[4] is None and got[5] is None                  # switch hitters and both hands: no anchor


def test_pitcher_expected_split_is_his_level_with_his_hands_league_split():
    rng = np.random.default_rng(4)
    me = {'R': {'all': _cells(rng, 500, .220, 900, .30, 800, .33)}, 'L': {'all': _cells(rng, 300, .290, 600, .22, 500, .28)}}
    lgz = {'R': {'R': {'all': _cells(rng, 90000, .240, 200000, .25, 150000, .30)}, 'L': {'all': _cells(rng, 40000, .250, 90000, .24, 70000, .29)}},
           'L': {'R': {'all': _cells(rng, 80000, .255, 180000, .22, 140000, .28)}, 'L': {'all': _cells(rng, 30000, .228, 70000, .28, 50000, .31)}}}
    card = {'throws': 'L', 'zones': me}
    got = json.loads(_run("process.stdout.write(JSON.stringify([pitcherSplitExpected(" + json.dumps(card) + "," + json.dumps({'zones': lgz}) + ",'R'),"
                          "pitcherSplitExpected(" + json.dumps(card) + "," + json.dumps({'zones': lgz}) + ",'all')]));"))
    tot = lambda cells: np.array(cells, float).sum(0)
    both = tot(me['R']['all']) + tot(me['L']['all'])
    l1, la_ = tot(lgz['R']['L']['all']), tot(lgz['R']['L']['all']) + tot(lgz['L']['L']['all'])      # batter side R and both, against lefty pitchers
    la = la_[5] / la_[4]; h_all = (both[5] + 30 * la) / (both[4] + 30)
    assert abs(got[0]['avg'] - min(0.995, max(0.005, l1[5] / l1[4] * h_all / la))) < 1e-9
    assert got[1] is None


def test_pitcher_zone_expected_uses_the_batter_side_m():
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('HZ_EXP_M', 'HZ_ANCHOR', 'PZ_EXP_M'))
    metrics = re.search(r'\n  var HZ_METRICS = \{.*?\n  \};\n', s, re.S).group(0)
    fns = ''.join(_fn(s, n) for n in ('hzTotal', 'hzExpected'))
    rng = np.random.default_rng(5)
    mine, lg = _cells(rng, 200, .300, 500, .3, 400, .3), _cells(rng, 50000, .240, 120000, .25, 90000, .3)
    js = ("const mine=" + json.dumps(mine) + ",lg=" + json.dumps(lg) + ";"
          "const a=hzExpected(HZ_METRICS.avg,'avg',mine,lg,null,PZ_EXP_M.R),m=hzExpected(HZ_METRICS.miss,'miss',mine,lg,null,PZ_EXP_M.R);"
          # AVG at the scaled rate: his level times the league's zone rate over the league's overall rate, his own zone counts all but ignored
          "const t=hzTotal(mine),lt=hzTotal(lg),la=lt[5]/lt[4],h=(t[5]+30*la)/(t[4]+30);"
          "lg.forEach(function(c){var sc=Math.min(0.995,Math.max(0.005,(c[4]?c[5]/c[4]:la)*h/la));assert.ok(Math.abs(a[c[0]]-sc)<2e-4,[c[0],a[c[0]],sc]);});"
          # misses with m 80: his own swings count
          "const c0=mine[0],l0=lg[0],lm=hzTotal(lg)[3]/hzTotal(lg)[2],hm=(t[3]+30*lm)/(t[2]+30),sc0=Math.min(0.995,Math.max(0.005,(l0[3]/l0[2])*hm/lm));"
          "assert.ok(Math.abs(m[c0[0]]-(c0[3]+80*sc0)/(c0[2]+80))<1e-9);process.stdout.write('ok');")
    out = subprocess.run(['node', '-e', "const assert=require('assert');" + consts + metrics + fns + js], check=True, capture_output=True, text=True)
    assert out.stdout == 'ok'


def test_legend_names_the_anchor_only_where_it_applies():
    s = _script()
    consts = re.search(r'\n  var HZ_ANCHOR = .*?;\n', s, re.S).group(0)
    js = ("var state={hzSheetHand:'sp'};"
          "assert.ok(hzAnchorWords('avg','R',{side:'R'}).indexOf('against this hand')>0);"
          "assert.equal(hzAnchorWords('miss','R',{side:'R'}),'');assert.ok(hzAnchorWords('miss','L',{side:'R'}).length>0);"
          "assert.equal(hzAnchorWords('avg','R',{side:'L',side_by_hand:{R:'L',L:'R'}}),'');assert.equal(hzAnchorWords('avg','all',{side:'R'}),'');"
          "assert.ok(hzSheetAnchorWords('avg').indexOf('switch hitters')>0);assert.ok(hzSheetAnchorWords('miss').indexOf('against lefties')>0);"
          "state.hzSheetHand='R';assert.equal(hzSheetAnchorWords('miss'),'');state.hzSheetHand='all';assert.equal(hzSheetAnchorWords('avg'),'');")
    subprocess.run(['node', '-e', "const assert=require('assert');" + consts + _fn(s, 'isSwitch') + _fn(s, 'hzAnchorWords') + _fn(s, 'hzSheetAnchorWords') + js], check=True)
