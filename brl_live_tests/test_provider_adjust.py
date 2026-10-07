import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from brl_live.provider_adjust import (ContextAdjust, TalentNoise, load_offsets, history_talent_inputs,
                                      inning_bucket, SIM_LABELS, MODEL_LABELS)

BASE = dict(zip(SIM_LABELS, (.4, .2, .1, .16, .07, .05, .02)))


class Flat:
    name = 'locked-test'
    validation_status = 'synthetic'

    def probabilities(self, c):
        return dict(BASE)


def ctx(side='away', inning=1, batter='100', pitcher='200'):
    return SimpleNamespace(batting_side=side, inning=inning, batter=SimpleNamespace(player_id=batter), pitcher=SimpleNamespace(player_id=pitcher))


def test_offsets_file_is_complete_and_small():
    off = load_offsets()
    assert set(off['table']) == {f'{s}_{i}' for s in '01' for i in ('1st', 'mid', 'late')}
    for m in off['table'].values():
        assert m.shape == (7,) and np.all(m > 0.8) and np.all(m < 1.25)
    assert off['estimated_through'] == '2026-09-27'


def test_context_adjust_matches_the_measured_pattern():
    off = load_offsets()
    p = ContextAdjust(Flat(), off)
    top1 = p.probabilities(ctx('away', 1)); bot1 = p.probabilities(ctx('home', 1)); mid = p.probabilities(ctx('away', 5))
    for q in (top1, bot1, mid):
        assert abs(sum(q.values()) - 1) < 1e-9 and set(q) == set(SIM_LABELS)
    assert top1['strikeout'] > BASE['strikeout'] and top1['double_triple'] < BASE['double_triple']
    assert bot1['single'] > BASE['single'] and bot1['home_run'] > BASE['home_run']
    assert abs(mid['bip_out'] - BASE['bip_out']) < 0.01
    assert p.name == 'locked-test'


def test_inning_bucket():
    assert [inning_bucket(i) for i in (1, 2, 8, 9, 14)] == ['1st', 'mid', 'mid', 'late', 'late']


def test_talent_noise_is_seeded_per_world_and_centred():
    league = np.array([.45, .22, .09, .14, .045, .035, .02])
    counts = lambda pid, is_batter: 300.0
    t = TalentNoise(Flat(), 1.0, 180.0, league, counts)
    t.new_world(5); a = t.probabilities(ctx()); a2 = t.probabilities(ctx())
    t.new_world(5); b = t.probabilities(ctx())
    t.new_world(6); c = t.probabilities(ctx())
    assert a == a2 == b and a != c
    assert abs(sum(a.values()) - 1) < 1e-9
    # many worlds average back to roughly the base probabilities (Jensen correction)
    acc = np.zeros(7)
    for w in range(2000):
        t.new_world(w); acc += np.array([t.probabilities(ctx())[k] for k in SIM_LABELS])
    acc /= 2000
    assert np.all(np.abs(acc - np.array([BASE[k] for k in SIM_LABELS])) < 0.006)


def test_talent_noise_zero_is_identity_and_unknown_player_gets_prior_only():
    league = np.array([.45, .22, .09, .14, .045, .035, .02])
    t0 = TalentNoise(Flat(), 0.0, 180.0, league, lambda *a: 0.0)
    t0.new_world(1); assert t0.probabilities(ctx()) == BASE
    seen = {}
    def counts(pid, is_batter): seen[(pid, is_batter)] = True; return 0.0
    t = TalentNoise(Flat(), 1.0, 180.0, league, counts); t.new_world(1); t.probabilities(ctx(batter='x', pitcher='y'))
    assert seen == {('x', True): True, ('y', False): True}


def test_history_talent_inputs_are_prior_date_only():
    h = pd.DataFrame({'date_key': ['2026-10-01'] * 4 + ['2026-10-06'] * 3, 'batter': [1, 1, 2, 2, 1, 1, 1], 'pitcher': [9, 9, 9, 8, 9, 9, 9],
                      'outcome': ['K', 'BIP_OUT', '1B', 'HR', 'HR', 'HR', 'HR']})
    counts, league = history_talent_inputs(h, '2026-10-06')
    assert counts(1, True) == 2 and counts(2, True) == 2 and counts(9, False) == 3 and counts(8, False) == 1 and counts(77, True) == 0
    assert abs(league.sum() - 1) < 1e-9 and league[MODEL_LABELS.index('HR')] == pytest.approx(0.25, abs=2e-3)
    with pytest.raises(ValueError):
        history_talent_inputs(h, '2026-10-01')
