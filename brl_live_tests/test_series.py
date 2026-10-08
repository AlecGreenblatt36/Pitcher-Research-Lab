import math

import pytest

from brl_live.series import HOME_EDGE, series_outlook, series_probability


def test_series_probability_matches_enumeration():
    p = {1: .6, 2: .55, 3: .4, 4: .45, 5: .6}
    # Best of five from 0-0, enumerated over every sequence of results.
    total = 0.0
    for bits in range(32):
        a = b = 0; pr = 1.0
        for k in range(1, 6):
            if a == 3 or b == 3:
                break
            w = (bits >> (k - 1)) & 1
            pr *= p[k] if w else 1 - p[k]
            a += w; b += 1 - w
        if a == 3 and all(((bits >> j) & 1) == 0 for j in range(a + b, 5)):
            total += pr
    assert series_probability(5, 0, 0, p)['high'] == pytest.approx(total)
    out = series_probability(7, 3, 2, {6: .5, 7: .5})
    assert out['high'] == pytest.approx(.75) and set(out['finals']) == {'4-2', '4-3', '3-4'}


def ledger(series_record, state='pregame'):
    def forecast(pk, date, away, home, p):
        return {'game_pk': pk, 'date': date, 'game_type': 'D', 'version': 1, 'saved_at': date + 'T12:00:00+00:00', 'home_win_probability': p,
                'away': {'abbr': away, 'name': away + ' Club'}, 'home': {'abbr': home, 'name': home + ' Club'}}
    led = {'forecasts': {'a': forecast(1, '2026-10-06', 'AAA', 'BBB', .45), 'b': forecast(2, '2026-10-07', 'AAA', 'BBB', .40)},
           'context': {'2': {'date': '2026-10-07', 'series_description': 'NL Division Series', 'series_game_number': 4,
                             'games_in_series': 5, 'records': series_record}},
           'actuals': {}, 'live': {}}
    if state == 'final':
        led['actuals']['2'] = {'away': 5, 'home': 2}
    if state == 'live':
        led['live']['2'] = {'state': {}}
    return led


def test_outlook_runs_the_series_out_from_the_score():
    led = ledger({'away': {'wins': 2, 'losses': 1}, 'home': {'wins': 1, 'losses': 2}})
    out = series_outlook(led, {'blend': {'a': .45, 'b': .40}})['series']
    assert len(out) == 1
    s = out[0]
    assert (s['high'], s['low']) == ('AAA', 'BBB')          # game 4 of a 2-2-1 series is at the lower seed
    assert s['wins'] == {'AAA': 2, 'BBB': 1}
    assert [g['n'] for g in s['games']] == [4, 5]
    assert s['games'][1]['home'] == 'AAA' and s['games'][1]['source'] == 'series estimate'
    strength = ((-math.log(.45 / .55) + HOME_EDGE) + (-math.log(.40 / .60) + HOME_EDGE)) / 2
    assert s['strength'] == pytest.approx(strength, abs=1e-4)
    p4, p5 = .60, 1 / (1 + math.exp(-(strength + HOME_EDGE)))
    assert s['advance']['AAA'] == pytest.approx(p4 + (1 - p4) * p5, abs=1e-3)


def test_a_final_not_yet_in_the_series_record_is_counted_and_ends_the_series():
    led = ledger({'away': {'wins': 2, 'losses': 1}, 'home': {'wins': 1, 'losses': 2}}, state='final')
    s = series_outlook(led, {'blend': {'a': .45, 'b': .40}})['series'][0]
    assert s['wins'] == {'AAA': 3, 'BBB': 1} and s['winner'] == 'AAA' and s['advance'] is None
    assert [g['n'] for g in s['games']] == [4] and s['games'][0]['winner'] == 'AAA'


def test_live_game_is_marked():
    led = ledger({'away': {'wins': 2, 'losses': 1}, 'home': {'wins': 1, 'losses': 2}}, state='live')
    s = series_outlook(led, {'blend': {}})['series'][0]
    assert s['games'][0]['state'] == 'live' and s['games'][0]['p_home'] == .40
