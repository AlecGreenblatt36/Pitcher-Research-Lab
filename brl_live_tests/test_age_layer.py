"""The aging layer: multipliers from the AgingState, cached per batter and pitcher, neutral at zero coefficients."""
import numpy as np
import pytest

from brl_live.age_layer import AgeAdjust, layer_from_receipt
from brl_live.provider_adjust import SIM_LABELS
from research_lab.pa_model import physics as phys

BASE = np.array([0.45, 0.22, 0.09, 0.14, 0.05, 0.03, 0.02])
COLS = phys.AGING_FEATURES + phys.DECAY_FEATURES


class _P:
    def __init__(self, pid):
        self.player_id = str(pid)


class _Ctx:
    def __init__(self, b, p):
        self.batter, self.pitcher = _P(b), _P(p)


class _Inner:
    name = 'inner'

    def probabilities(self, ctx):
        return dict(zip(SIM_LABELS, BASE.tolist()))


def state():
    st = phys.AgingState({'aging': True, 'decay_days': 365, 'k_dec': 60.0})
    day = 739000
    st.absorb_date(day, [10, 10, 11], [500, 500, 500], ['HR', 'K', 'BIP_OUT'])
    st.absorb_date(day + 400, [10, 11], [500, 500], ['HR', 'K'])
    return st, day + 401


def layer(coef):
    return {'columns': COLS, 'labels': ['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH'], 'mean': [0.0] * 16, 'sd': [1.0] * 16, 'coef': coef}


def test_zero_layer_is_neutral_and_weights_move_the_right_class():
    st, today = state()
    adj = AgeAdjust(_Inner(), layer([[0.0] * 7 for _ in COLS]), st, today, ages=lambda b, p: (30.0, 28.0))
    assert adj.probabilities(_Ctx(10, 500))['home_run'] == pytest.approx(BASE[5])
    coef = [[0.0] * 7 for _ in COLS]; coef[COLS.index('b_gap')] = [0, 0, 0, 0, 0, 1.0, 0]
    adj = AgeAdjust(_Inner(), layer(coef), st, today, ages=lambda b, p: (30.0, 28.0))
    q = adj.probabilities(_Ctx(10, 500))
    assert q['home_run'] > BASE[5] and sum(q.values()) == pytest.approx(1.0)
    assert (10, 500) in adj.cache
    adj.reset(); assert not adj.cache


def test_layer_from_receipt_reads_the_stage2_coefficients():
    rec = {'results': {'v2': {'aging_layer': {'coefficients': {'columns': COLS, 'labels': ['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH'],
                                                                'fit_2025': {'mean': [0.0] * 16, 'sd': [1.0] * 16, 'coef': [[0.0] * 7] * 16}}}}}}
    lay = layer_from_receipt(rec, fit='fit_2025')
    assert lay['columns'] == COLS and lay['params']['decay_days'] == 365 and lay['fit'] == 'fit_2025'
