"""Team offsets: time-valid, credit the right side, combine batting and fielding teams, and adjust probabilities."""
import numpy as np
import pytest

from brl_live import team_offsets as to
from brl_live.provider_adjust import TeamAdjust, SIM_LABELS

BASE = np.array([0.45, 0.22, 0.09, 0.14, 0.05, 0.03, 0.02])


def rows_for(n_days=40, k_boost=1.3, seed=0):
    """Teams A, B, C, D play in pairs each day; team A's pitchers and fielders get 30% more strikeouts than predicted."""
    rng = np.random.default_rng(seed)
    out, teams = [], ['A', 'B', 'C', 'D']
    for d in range(n_days):
        day = f'2026-05-{1 + d % 28:02d}' if d < 28 else f'2026-06-{1 + (d - 28):02d}'
        order = list(rng.permutation(teams))
        for home, away in ((order[0], order[1]), (order[2], order[3])):
            for bat, fld in ((away, home), (home, away)):
                n = 38
                true = BASE.copy()
                if fld == 'A':
                    true[1] *= k_boost
                true /= true.sum()
                obs = np.bincount(rng.choice(7, size=n, p=true), minlength=7)
                pred = BASE * n
                out.append({'date': day, 'team': bat, 'side': 'bat', 'n': n, 'obs': obs.tolist(), 'pred': pred.round(4).tolist()})
                out.append({'date': day, 'team': fld, 'side': 'fld', 'n': n, 'obs': obs.tolist(), 'pred': pred.round(4).tolist()})
    return out


def test_offsets_are_time_valid_and_credit_the_fielding_team():
    rows = rows_for()
    dates = sorted({r['date'] for r in rows})
    table = to.by_date(rows, dates, k=200.0)
    # Nothing is known on the first date.
    assert all(v == 0.0 for t in table[dates[0]].values() for s in t.values() for v in s)
    last = table[dates[-1]]
    k_index = to.LABELS.index('K')
    assert last['A']['fld'][k_index] > 0.12, last['A']
    assert abs(last['B']['fld'][k_index]) < 0.12 and abs(last['A']['bat'][k_index]) < 0.12
    # Changing a late row never changes an earlier date's table.
    rows2 = [dict(r) for r in rows]
    for r in rows2:
        if r['date'] == dates[-1]:
            r['obs'] = [0, 38, 0, 0, 0, 0, 0]
    assert to.by_date(rows2, dates[:-1], k=200.0) == to.by_date(rows, dates[:-1], k=200.0)


def test_opponents_pair_batting_and_fielding_rows():
    rows = rows_for(n_days=3)
    opp = to._opponents(rows)
    for r in rows:
        o = opp.get((r['date'], r['team'], r['side']))
        assert o is not None and o != r['team']


def test_current_table_and_game_multipliers():
    rows = rows_for()
    table = to.current_table(rows, k=200.0, half_life=60.0, source='test')
    assert table['schema'] == 'brl.team-offsets.v1' and table['estimated_through'] == max(r['date'] for r in rows)
    m = to.game_log_multipliers(table, away='B', home='A', day=table['estimated_through'])
    k_index = to.LABELS.index('K')
    # The away team bats against A's fielders: more strikeouts. A at the plate faces B's ordinary fielders.
    assert m['away'][k_index] > 0.1 and abs(m['home'][k_index]) < m['away'][k_index]
    later = to.game_log_multipliers(table, away='B', home='A', day='2026-12-31')
    assert abs(later['away'][k_index]) < abs(m['away'][k_index])          # decayed toward zero with time
    unknown = to.game_log_multipliers(table, away='X', home='Y')
    assert unknown == {'away': [0.0] * 7, 'home': [0.0] * 7}


class _Ctx:
    def __init__(self, side):
        self.batting_side = side


class _Inner:
    name = 'inner'

    def probabilities(self, ctx):
        return dict(zip(SIM_LABELS, BASE.tolist()))


def test_team_adjust_uses_the_batting_side():
    adj = TeamAdjust(_Inner(), {'away': [0, np.log(1.5), 0, 0, 0, 0, 0], 'home': [0.0] * 7})
    away = adj.probabilities(_Ctx('away')); home = adj.probabilities(_Ctx('home'))
    assert away['strikeout'] > home['strikeout'] and home['strikeout'] == pytest.approx(BASE[1])
    assert sum(away.values()) == pytest.approx(1.0)
    adj.set_teams(None)
    assert adj.probabilities(_Ctx('away'))['strikeout'] == pytest.approx(BASE[1])


def test_adjusted_provider_applies_team_offsets_when_switched_on(monkeypatch):
    from brl_live import boxscore
    table = to.current_table(rows_for(), k=200.0, source='test')
    monkeypatch.setattr(to, 'load_table', lambda path=to.TABLE_PATH: table)
    on, _, label = boxscore.adjusted_provider(_Inner(), None, table['estimated_through'], settings={'team_offsets': True}, teams=('B', 'A'))
    off, _, label_off = boxscore.adjusted_provider(_Inner(), None, table['estimated_through'], settings={'team_offsets': False}, teams=('B', 'A'))
    assert any('team offsets through' in x for x in label) and not label_off
    assert on.probabilities(_Ctx('away'))['strikeout'] > off.probabilities(_Ctx('away'))['strikeout']
