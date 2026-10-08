"""Role offsets: time-valid, pulled toward zero, and applied by the pitcher's role and times through the order."""
import numpy as np

from brl_live import role_offsets as ro
from brl_live.provider_adjust import RoleAdjust, SIM_LABELS

BASE = np.array([0.45, 0.22, 0.09, 0.14, 0.05, 0.03, 0.02])


def rows_for(days=30, boost=1.1, seed=1):
    """Starters strike out 10% more than predicted the first time through; relievers as predicted."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(days):
        day = f'2026-05-{1 + d:02d}'
        for role, tto, n in (('starter', 1, 300), ('starter', 2, 250), ('reliever', 0, 280)):
            true = BASE.copy()
            if role == 'starter' and tto == 1:
                true[1] *= boost
            true /= true.sum()
            obs = np.bincount(rng.choice(7, size=n, p=true), minlength=7)
            out.append({'date': day, 'role': role, 'tto': tto, 'side': '0', 'bucket': 'mid', 'n': n, 'obs': obs.tolist(),
                        'pred': (BASE * n).round(4).tolist(), 'pred_ctx': (BASE * n).round(4).tolist()})
    return out


def test_offsets_are_time_valid_and_find_the_starters_strikeouts():
    rows = rows_for()
    dates = sorted({r['date'] for r in rows})
    tb = ro.by_date(rows, dates, k=500.0)
    assert all(v == 0.0 for role in tb[dates[0]].values() for v in role)
    last = tb[dates[-1]]
    k = ro.LABELS.index('K')
    assert last['starter_1'][k] > 0.05 and abs(last['reliever'][k]) < 0.05 and abs(last['starter_2'][k]) < 0.05
    rows2 = [dict(r) for r in rows]
    for r in rows2:
        if r['date'] == dates[-1]:
            r['obs'] = [0, r['n'], 0, 0, 0, 0, 0]
    assert ro.by_date(rows2, dates[:-1], k=500.0) == ro.by_date(rows, dates[:-1], k=500.0)


def test_current_table_and_multipliers_match_the_state():
    rows = rows_for()
    table = ro.current_table(rows, k=500.0, half_life=90.0, source='test')
    m = ro.log_multipliers(table, table['estimated_through'])
    assert np.allclose(m['starter_1'], table['roles']['starter_1'], atol=1e-4)
    later = ro.log_multipliers(table, '2026-12-31')
    k = ro.LABELS.index('K')
    assert abs(later['starter_1'][k]) < abs(m['starter_1'][k])


class _Inner:
    name = 'inner'

    def probabilities(self, ctx):
        return dict(zip(SIM_LABELS, BASE.tolist()))


class _P:
    def __init__(self, role):
        self.role = role


class _Ctx:
    def __init__(self, role, tto):
        self.pitcher = _P(role); self.times_through_order = tto


def test_role_adjust_uses_role_and_times_through_the_order():
    ra = RoleAdjust(_Inner(), {'starter_1': [0, 0.1, 0, 0, 0, 0, 0], 'starter_3': [0, -0.1, 0, 0, 0, 0, 0], 'reliever': [0] * 7})
    k = SIM_LABELS[1]
    first = ra.probabilities(_Ctx('starter', 1))[k]; third = ra.probabilities(_Ctx('starter', 4))[k]
    rel = ra.probabilities(_Ctx('reliever', 1))[k]; second = ra.probabilities(_Ctx('starter', 2))[k]
    assert first > 0.22 > third and abs(rel - 0.22) < 1e-12 and abs(second - 0.22) < 1e-12
    assert abs(sum(ra.probabilities(_Ctx('starter', 1)).values()) - 1) < 1e-12
