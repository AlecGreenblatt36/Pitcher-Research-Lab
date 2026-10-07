"""Versioned headline recipes: our independent model, the market-combined headline, and scoring by the recipe in force at first pitch."""
import json
import math

from brl_live import headline as hl
from brl_live.record import build_record

PARAMS = {'schema': 'brl.headline-params.v1', 'versions': [
    {'name': 'equal-v1', 'effective_from': '2026-01-01T00:00:00+00:00', 'kind': 'equal'},
    {'name': 'taught-v2', 'effective_from': '2026-10-08T00:00:00+00:00', 'kind': 'taught',
     'ours': {'features': ['x_sim', 'x_team', 'sp_home', 'sp_away'], 'intercept': 0.05, 'coef': [0.5, 0.6, -0.4, 0.3]},
     'headline': {'intercept': 0.0, 'market': 0.9, 'ours': 0.15}}]}
TM = {'p_home': 0.55, 'starter_adjust': {'home': {'factor': 0.9}, 'away': {'factor': 1.1}},
      'ratings': {'home': {'offense': 1.02, 'defense': 0.98}, 'away': {'offense': 1.0, 'defense': 1.01}}}


def lg(p):
    return math.log(p / (1 - p))


def test_versions_and_recipes(monkeypatch):
    monkeypatch.setattr(hl, 'PARAMS', PARAMS)
    old, new = hl.version_at('2026-10-07T20:00:00Z'), hl.version_at('2026-10-08T20:00:00Z')
    assert old['name'] == 'equal-v1' and new['name'] == 'taught-v2'
    p_old, how_old = hl.ours(0.6, TM, old)
    assert abs(p_old - 1 / (1 + math.exp(-(0.5 * lg(0.6) + 0.5 * lg(0.55))))) < 1e-12 and 'equal' in how_old
    p_new, how_new = hl.ours(0.6, TM, new)
    z = 0.05 + 0.5 * lg(0.6) + 0.6 * lg(0.55) - 0.4 * math.log(0.9) + 0.3 * math.log(1.1)
    assert abs(p_new - 1 / (1 + math.exp(-z))) < 1e-12 and how_new == 'taught-v2'
    # a box from before the starter adjustment lacks features: the equal blend stands in
    p_fallback, how_fb = hl.ours(0.6, {'p_home': 0.55}, new)
    assert abs(p_fallback - p_old) < 1e-12 and 'equal' in how_fb
    # the headline uses the market only when one was captured, and only in the taught recipe
    h, how = hl.headline(p_new, 0.58, new)
    assert abs(h - 1 / (1 + math.exp(-(0.9 * lg(0.58) + 0.15 * lg(p_new))))) < 1e-12 and 'market' in how
    assert hl.headline(p_new, None, new)[0] == p_new and hl.headline(p_old, 0.58, old)[0] == p_old


def ledger(first_pitch, captured):
    f = {'game_pk': 1, 'date': '2026-10-08', 'version': 1, 'saved_at': '2026-10-08T18:00:00Z', 'scheduled_start': '2026-10-08T23:00:00Z',
         'home_win_probability': 0.6, 'team_baseline_probability': 0.5, 'away': {'abbr': 'LAD'}, 'home': {'abbr': 'ATL'}}
    return {'forecasts': {'a': f}, 'publications': {'a': {'published_at': '2026-10-08T18:01:00Z', 'commit': 'c' * 40}},
            'actuals': {'1': {'home': 5, 'away': 2, 'first_pitch_observed_at': first_pitch}},
            'market': {'1': {'p_home': 0.58, 'captured_at': captured}}, 'box_scores': {'a': {'team_model': TM}}}


def test_record_scores_with_the_recipe_in_force_at_first_pitch(monkeypatch):
    monkeypatch.setattr(hl, 'PARAMS', PARAMS)
    rec = build_record(ledger('2026-10-08T23:05:00Z', '2026-10-08T22:50:00Z'))
    g = rec['games'][0]
    p_new = hl.ours(0.6, TM, PARAMS['versions'][1])[0]
    assert g['headline_version'] == 'taught-v2' and abs(g['p_ours'] - round(p_new, 4)) < 1e-9 and g['p_market'] == 0.58
    assert abs(rec['blend']['a'] - round(hl.headline(p_new, 0.58, PARAMS['versions'][1])[0], 6)) < 1e-9
    assert [r['key'] for r in rec['ladder']] == ['coin', 'home', 'market', 'team', 'sim', 'ours', 'blend']
    # a line captured after first pitch is not used, so the headline is our model alone
    rec = build_record(ledger('2026-10-08T23:05:00Z', '2026-10-08T23:10:00Z'))
    assert rec['games'][0]['p_market'] is None and abs(rec['blend']['a'] - rec['ours']['a']) < 1e-9
    # a game that started before the switch keeps the old recipe even when scored after it
    old = ledger('2026-10-07T23:05:00Z', '2026-10-07T22:50:00Z')
    old['forecasts']['a']['saved_at'] = '2026-10-07T18:00:00Z'; old['publications']['a']['published_at'] = '2026-10-07T18:01:00Z'
    rec = build_record(old)
    assert rec['games'][0]['headline_version'] == 'equal-v1' and abs(rec['blend']['a'] - rec['ours']['a']) < 1e-9
    assert rec['disagreements']['threshold'] == 0.05
