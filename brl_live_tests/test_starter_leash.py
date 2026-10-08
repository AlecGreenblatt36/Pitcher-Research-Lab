"""The starter leash (research_lab.game_sim.starter_leash, LEASH-01): facts come from earlier appearances only, the
shift moves p_exp along the hazard's response, no shift leaves p_exp alone, the fit recovers planted terms, and the
production helper reads a start's facts from the plate-appearance history."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_lab.game_sim.starter_leash import AppearanceIndex, Leash, indicators

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brl_leash', ROOT / 'tools' / 'brl_leash.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
DOC = json.loads((ROOT / 'brl_live' / 'leash.json').read_text())


def test_facts_come_from_earlier_appearances():
    rows = [(7, '2026-03-27', True), (7, '2026-04-01', True), (7, '2026-04-06', True), (7, '2026-05-10', True),
            (7, '2026-05-15', True), (7, '2026-05-20', True), (7, '2026-05-23', False), (7, '2026-05-25', True)]
    idx = AppearanceIndex(rows)
    assert idx.facts(7, '2026-03-27') == {'days': None, 'prev_start': None, 'since_layoff': 1, 'month': 3}
    assert idx.facts(7, '2026-04-01') == {'days': 5, 'prev_start': True, 'since_layoff': 2, 'month': 4}
    assert idx.facts(7, '2026-04-06')['since_layoff'] == 3
    assert idx.facts(7, '2026-05-10') == {'days': 34, 'prev_start': True, 'since_layoff': 1, 'month': 5}
    assert idx.facts(7, '2026-05-15')['since_layoff'] == 2 and idx.facts(7, '2026-05-20')['since_layoff'] == 3
    f = idx.facts(7, '2026-05-25')            # two days after a relief outing
    assert f['days'] == 2 and f['prev_start'] is False and f['since_layoff'] == 3
    ind = indicators(f)
    assert ind['short'] == 1 and ind['relief'] == 1 and ind['lay1'] == 0 and ind['sep'] == 0
    assert indicators(idx.facts(7, '2026-09-30'))['sep'] == 1
    assert idx.facts(99, '2026-06-01')['since_layoff'] == 1          # never seen: first start back


def test_shift_moves_p_exp_along_the_response():
    lz = Leash(DOC)
    regular = {'days': 5, 'prev_start': True, 'since_layoff': 3, 'month': 6}
    opener = {'days': 3, 'prev_start': False, 'since_layoff': 3, 'month': 6}
    for p in (17.0, 22.0, 26.5, 31.0):
        new = lz.adjust(p, regular)
        assert abs(lz.response(new) - (lz.response(p) + lz.shift(regular))) < 1e-9 and new > p
    short = lz.adjust(19.0, opener)
    assert short < 15 and abs(lz.response(short) - max(3.0, lz.response(19.0) + lz.shift(opener))) < 1e-9
    none = Leash({'terms': {}, 'response': DOC['response']})
    assert all(abs(none.adjust(p, opener) - p) < 1e-9 for p in (9.0, 15.0, 22.0, 33.0))
    assert lz.shift(opener) == DOC['terms']['base'] + DOC['terms']['short'] + DOC['terms']['relief']


def test_fit_recovers_planted_terms():
    rng = np.random.default_rng(3)
    n = 6000
    df = pd.DataFrame({'base': 1.0, 'short': (rng.random(n) < 0.05) * 1.0, 'relief': (rng.random(n) < 0.08) * 1.0,
                       'lay1': (rng.random(n) < 0.07) * 1.0, 'lay2': (rng.random(n) < 0.06) * 1.0,
                       'mar': (rng.random(n) < 0.03) * 1.0, 'sep': (rng.random(n) < 0.15) * 1.0})
    planted = {'base': 0.6, 'short': -7.0, 'relief': -2.0, 'lay1': -1.8, 'lay2': -0.8, 'mar': 0.4, 'sep': -0.9}
    df['sim'] = 22 + rng.normal(0, 1, n)
    df['act'] = df['sim'] + sum(planted[k] * df[k] for k in planted) + rng.normal(0, 3, n)
    terms, se = tool.fit(df)
    assert all(abs(terms[k] - planted[k]) < 4 * se[k] + 0.05 for k in planted)


def test_production_helper_reads_the_history():
    from brl_live.boxscore import leash_pexp
    rows = []
    def game(gpk, d, home_sp, away_sp, game_type='R'):
        for ab, (top, p) in enumerate(((True, home_sp), (True, home_sp), (False, away_sp), (False, away_sp)), start=1):
            rows.append({'game_pk': gpk, 'date_key': d, 'at_bat_number': ab, 'inning': 1, 'inning_topbot': 'Top' if top else 'Bot',
                         'home_team': 'AAA', 'away_team': 'BBB', 'pitcher': p, 'p_throws': 'R', 'outcome': 'K', 'game_type': game_type})
    game(1, '2026-06-01', 10, 20); game(2, '2026-06-06', 11, 20); game(3, '2026-06-08', 10, 21)
    rows.append(dict(rows[-1], at_bat_number=5, pitcher=11))       # 11 relieves in game 3 (two days before)
    h = pd.DataFrame(rows)
    lz = Leash(DOC)
    out = leash_pexp(lz, h, {10: 22.0, 11: 21.0}, ['11', '10'], '2026-06-10', 21.9)
    f11 = {'days': 2, 'prev_start': False, 'since_layoff': 2, 'month': 6}
    f10 = {'days': 2, 'prev_start': True, 'since_layoff': 3, 'month': 6}
    assert abs(out[11] - lz.adjust(21.0, f11)) < 1e-9 and abs(out[10] - lz.adjust(22.0, f10)) < 1e-9 and out[11] < 21.0
