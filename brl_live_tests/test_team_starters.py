"""TEAM-02: the team model's starting-pitcher adjustment."""
import numpy as np
import pandas as pd
from brl_live.team_model import TeamModel, StarterAdjust


def history_and_rows():
    """Two teams over 40 days; A's starters: ace 1 (strikeouts) and 2 (homers allowed); B's: 3 and 4 (league-ish)."""
    rng = np.random.default_rng(3)
    rows, pa = [], []
    for d in range(40):
        date = (pd.Timestamp('2026-05-01') + pd.Timedelta(days=d)).strftime('%Y-%m-%d')
        pk = 1000 + d
        a_sp, b_sp = (1, 3) if d % 2 == 0 else (2, 4)
        home, away = ('A', 'B') if d % 2 == 0 else ('B', 'A')
        rows.append({'game_pk': pk, 'date': date, 'home_id': home, 'away_id': away, 'home_runs': int(rng.integers(0, 8)), 'away_runs': int(rng.integers(0, 8))})
        ab = 0
        for half, sp in (('Top', a_sp if home == 'A' else b_sp), ('Bot', b_sp if home == 'A' else a_sp)):
            for i in range(24):
                ab += 1
                if sp == 1:
                    o = 'K' if i % 2 == 0 else 'BIP_OUT'
                elif sp == 2:
                    o = 'HR' if i % 4 == 0 else ('BB_HBP' if i % 4 == 1 else 'BIP_OUT')
                else:
                    o = ['K', 'BIP_OUT', 'BIP_OUT', '1B', 'BB_HBP', 'BIP_OUT', 'HR', 'BIP_OUT'][i % 8]
                pa.append({'date_key': date, 'game_pk': pk, 'at_bat_number': ab, 'pitcher': sp, 'inning_topbot': half, 'outcome': o})
    return pd.DataFrame(pa), rows


def test_starter_factors_follow_quality_and_share():
    h, rows = history_and_rows()
    sa = StarterAdjust(h, rows)
    f = sa.factors('2026-06-10', 'A', 'B', 1, 4)
    assert f['home']['factor'] < 1.0 and f['home']['known']          # the strikeout ace allows fewer runs than A's rotation
    g = sa.factors('2026-06-10', 'A', 'B', 2, 4)
    assert g['home']['factor'] > 1.0                                   # the homer-prone one more
    assert abs(f['away']['share'] - 24 / 22) < 1e-3                    # 20 starts of 24 batters (shown to 3 places)
    u = sa.factors('2026-06-10', 'A', 'B', 99, 4)                      # unknown pitcher: league value, opener share
    assert u['home']['share'] == 0.6 and not u['home']['known']
    post = sa.factors('2026-06-10', 'A', 'B', 1, 4, share_scale=0.91)
    assert abs(post['home']['share'] - f['home']['share'] * 0.91) < 1e-3
    # nothing on or after the date enters: a start that day changes nothing
    before = sa.factors('2026-05-21', 'A', 'B', 1, 4)
    assert before['home']['value'] == StarterAdjust(h[h['date_key'] < '2026-05-21'], rows).factors('2026-05-21', 'A', 'B', 1, 4)['home']['value']


def test_probability_moves_with_the_starters():
    h, rows = history_and_rows()
    tm, sa = TeamModel(rows), StarterAdjust(h, rows)
    base = tm.probability('2026-06-10', 'A', 'B')
    ace = tm.probability('2026-06-10', 'A', 'B', starters=sa.factors('2026-06-10', 'A', 'B', 1, 4))
    weak = tm.probability('2026-06-10', 'A', 'B', starters=sa.factors('2026-06-10', 'A', 'B', 2, 4))
    assert weak['p_home'] < base['p_home'] < ace['p_home']
    assert ace['version'].startswith('team-nb-decay-sp-v2') and 'starter_adjust' in ace and base['version'].startswith('team-nb-decay-v1')
    assert ace['mu_away'] < base['mu_away']
