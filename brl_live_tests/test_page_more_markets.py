"""The Odds tab's other markets (moreMarkets), off until RUNLINE-01, FIRST5-01 and NRFI-01 hold. The run line keeps the
pick: our win chance times the simulator's chance of winning by two or more given a win; the favorite takes -1.5. The
first five innings push on a tie; the first inning reads the box's no-run count."""
import json
import re
import subprocess
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'


def _script():
    return TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]


def _fn(script, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\(.*?\n  \}\n', script, re.S)
    assert m, name
    return m.group(0)


def _line(script, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\([^\n]*\}\n', script)
    assert m, name
    return m.group(0)


def _run(js, ph, flags):
    s = _script()
    pre = ('function homeProb(f){return ' + str(ph) + ';}var MORE_MARKETS=' + json.dumps(flags) + ';')
    fns = ''.join(_line(s, n) for n in ('pct', 'esc', 'total', 'dist', 'mmShare')) + ''.join(_fn(s, n) for n in ('amOdds', 'runLineMarket', 'moreMarkets'))
    return subprocess.run(['node', '-e', pre + fns + js], check=True, capture_output=True, text=True).stdout


BOX = {'margin_distribution': {'n': 1000, 'counts': [[-3, 100], [-1, 300], [1, 250], [2, 200], [4, 150]]},
       'f5_margin_distribution': {'n': 1000, 'counts': [[-1, 300], [0, 200], [2, 500]]},
       'f5_total_distribution': {'n': 1000, 'counts': [[3, 400], [5, 350], [8, 250]]},
       'first_inning': {'n': 1000, 'no_run': 480, 'away_scores': 300, 'home_scores': 320}}
F = {'away': {'abbr': 'LAD'}, 'home': {'abbr': 'MIL'}}


def test_run_line_keeps_the_pick_and_the_favorite_lays_the_run_and_a_half():
    # home wins 600 of 1,000 worlds, 350 by two or more: given a win, 7/12; the pick says home 51%
    rl = lambda mkt: 'process.stdout.write(JSON.stringify(runLineMarket(' + json.dumps(F) + ',' + json.dumps(BOX) + ',' + json.dumps(mkt) + ')));'
    got = json.loads(_run(rl(None), 0.51, {}))
    assert got['fav'] == 'home' and got['dog'] == 'away' and abs(got['p'] - 0.51 * 350 / 600) < 1e-12
    # the road team favored by our pick: its win by two or more given a win is 100/400
    got = json.loads(_run(rl(None), 0.40, {}))
    assert got['fav'] == 'away' and abs(got['p'] - 0.60 * 100 / 400) < 1e-12
    # with a line, the book's favorite lays the run and a half even where our pick leans the other way
    got = json.loads(_run(rl({'home_ml': 120, 'away_ml': -140}), 0.51, {}))
    assert got['fav'] == 'away' and abs(got['p'] - 0.49 * 100 / 400) < 1e-12
    got = json.loads(_run(rl({'home_ml': -150, 'away_ml': 130}), 0.45, {}))
    assert got['fav'] == 'home' and abs(got['p'] - 0.45 * 350 / 600) < 1e-12


def test_markets_show_only_when_switched_on():
    off = _run('process.stdout.write(moreMarkets(' + json.dumps(F) + ',' + json.dumps(BOX) + '));', 0.51, {'runline': False, 'f5': False, 'nrfi': False})
    assert off == ''
    on = _run('process.stdout.write(moreMarkets(' + json.dumps(F) + ',' + json.dumps(BOX) + ',null));', 0.51, {'runline': True, 'f5': True, 'nrfi': True})
    assert 'MIL −1.5 <b>30%</b>' in on and 'LAD +1.5 <b>70%</b>' in on          # 0.51 x 7/12 = 0.2975
    assert 'tied after five 20%, a push' in on and 'LAD <b>37%</b>' in on and 'MIL <b>63%</b>' in on   # 0.3/0.8 and 0.5/0.8
    assert 'Over <b>60%</b>' in on and 'Under <b>40%</b>' in on                       # five or more through five: 600 of 1,000
    assert 'A run <b>52%</b>' in on and 'No run <b>48%</b>' in on


def test_day_form_worlds_take_over_the_market_numbers_when_the_box_has_them():
    """DISP-03: with market worlds in the box, the run line reads their margins; without, the plain ones."""
    box = dict(BOX, market_worlds={'n': 1000, 'margin_distribution': {'n': 1000, 'counts': [[-3, 150], [-1, 250], [1, 200], [3, 400]]},
                                   'total_run_distribution': {'n': 1000, 'counts': [[5, 500], [9, 500]]}})
    got = json.loads(_run('process.stdout.write(JSON.stringify(runLineMarket(' + json.dumps(F) + ',' + json.dumps(box) + ',null)));', 0.51, {}))
    assert got['fav'] == 'home' and abs(got['p'] - 0.51 * 400 / 600) < 1e-12        # by two or more given a win: 400 of 600
