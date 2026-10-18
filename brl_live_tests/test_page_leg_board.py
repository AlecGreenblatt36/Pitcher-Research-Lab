"""LEG-01's leg board: the page computes every leg's chance the way the leg book scored it (tools/brl_leg_book.py,
--as-shown), so a verified cell describes the number the page shows. One made-up game is priced both ways and every family
the two share must agree."""
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


def _script():
    return TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]


def _fn(s, name):
    m = re.search(r'\n  function ' + name + r'\(.*?\n  \}\n', s, re.S)
    one = re.search(r'\n  function ' + name + r'\([^\n]*\}\n', s)
    if one and (not m or len(one.group(0)) < len(m.group(0))):
        return one.group(0)
    assert m, name
    return m.group(0)


def _counts(h, offset=0):
    return {'counts': [[i + offset, int(v)] for i, v in enumerate(h) if v]}


def _game():
    rng = np.random.default_rng(4)
    def hist(lam, n=10000, size=None):
        x = rng.poisson(lam, n); return np.bincount(x, minlength=size or x.max() + 1).tolist()
    margin = np.zeros(31, int)
    for d, c in zip(range(-15, 16), rng.multinomial(10000, np.exp(-np.abs(np.arange(-15, 16) - 0.4) / 2.5) / np.exp(-np.abs(np.arange(-15, 16) - 0.4) / 2.5).sum())):
        margin[d + 15] = c
    g = {'date': '2026-10-18', 'game_pk': 1, 'n': 10000, 'ties': 0, 'home_wins': 5600, 'home_runs': 4, 'away_runs': 3,
         'home_hist': hist(4.6), 'away_hist': hist(4.1), 'margin_hist': margin.tolist(), 'first_inning': [7100],
         'f5_total_hist': hist(2.7), 'f5_margin_hist': margin.tolist(),
         'innings': {'actual_first_scores': {'away': [[i, 1] for i in range(1, 10)], 'home': [[i, 2] for i in range(1, 9)]}},
         'hitters': [{'batter': 10 + i, 'spot': i + 1, 'h_hist': hist(1.05), 'hr_hist': hist(0.15), 'k_hist': hist(0.9), 'bb_hist': hist(0.4),
                      'actual': {'pa': 4, 'h': 1, 'hr': 0, 'k': 1, 'bb': 0}} for i in range(3)],
         'starters': {'home': {'pitcher': 99, 'k': hist(5.8), 'outs': hist(16.5), 'h': hist(5.2), 'bb': hist(1.8), 'actual': {'k': 6, 'outs': 17, 'h': 5, 'bb': 2}}}}
    box = {'margin_distribution': {'counts': [[d - 15, int(c)] for d, c in enumerate(g['margin_hist']) if c]},
           'team_run_distributions': {'away': _counts(g['away_hist']), 'home': _counts(g['home_hist'])},
           'first_inning': {'n': 10000, 'no_run': 7100}, 'f5_total_distribution': _counts(g['f5_total_hist']),
           'teams': {'away': {'batting': [], 'pitching': []},
                     'home': {'batting': [{'player_id': x['batter'], 'name': 'Hitter ' + 'ABC'[i], 'spot': x['spot'],
                                           'distributions': {'H': _counts(x['h_hist']), 'HR': _counts(x['hr_hist']), 'K': _counts(x['k_hist']), 'BB': _counts(x['bb_hist'])}}
                                          for i, x in enumerate(g['hitters'])],
                              'pitching': [{'name': 'Some Starter', 'role': 'starter', 'distributions': {'K': _counts(g['starters']['home']['k']), 'outs': _counts(g['starters']['home']['outs']),
                                                                                                      'H': _counts(g['starters']['home']['h']), 'BB': _counts(g['starters']['home']['bb'])}}]}}}
    return g, box


def test_page_legs_match_the_leg_book():
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('RUN_SPREAD', 'LANCZOS', 'K_SPREAD', 'SLOT_SHARE', 'LEG_LINES'))
    names = ('dist', 'total', 'mean', 'sd', 'lgam', 'nbRuns', 'kShape', 'ownAtLeast', 'mmShare', 'lastName', 'shortNames', 'marketProb',
             'pmfOf', 'overArr', 'convArr', 'legBin', 'gameLegs')
    g, box = _game()
    f = {'game_pk': 1, 'home_win_probability': 0.56, 'away': {'abbr': 'AWY'}, 'home': {'abbr': 'HOM'}}
    js = 'var D = {market: {}};' + consts + ''.join(_fn(s, n) for n in names) + \
        f'console.log(JSON.stringify(gameLegs({json.dumps(f)}, {json.dumps(box)}).map(function (l) {{ return [l.fam, l.p]; }})))'
    page = dict(json.loads(subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout))
    book = {}
    for l in lb.game_legs(g, as_shown=True):
        book[l['fam']] = l['p']
    shared = set(page) & set(book)
    for must in ('win:home:yes', 'runline:home-1.5:yes', 'total:over8.5:no', 'team_total:home:over3.5:yes', 'first_inning:run_scored:yes',
                 'hitter:hits1+:yes', 'hitter:hr:no', 'starter:k:over5.5:yes', 'starter:outs:over15.5:no', 'starter:h:over4.5:yes', 'starter:bb:over1.5:no'):
        assert must in shared, must
    # both key hitter legs by family only, so each dict keeps the last hitter priced: the same hitter on both sides
    worst = max(abs(page[k] - book[k]) for k in shared)
    assert worst < 1e-9, worst


def test_bins_match_the_leg_book():
    s = _script()
    js = _fn(s, 'legBin') + 'console.log(JSON.stringify([0.7, 0.6999999, 0.75, 0.8, 0.89999, 0.9, 0.99, 1.0].map(legBin)))'
    out = json.loads(subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout)
    py = [f"{[lo for lo, hi in lb.BINS if lo <= p < hi][0]:.1f}" for p in (0.7, 0.6999999, 0.75, 0.8, 0.89999, 0.9, 0.99, 1.0)]
    assert out == py
