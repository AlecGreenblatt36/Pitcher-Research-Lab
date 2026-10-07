import numpy as np
import pytest
from brl_live.team_model import TeamModel, nb_pmf, win_prob


def rows(n_days=120, seed=0):
    rng = np.random.default_rng(seed); out = []
    teams = ['A', 'B', 'C', 'D']; strength = {'A': 1.25, 'B': 1.0, 'C': 1.0, 'D': 0.8}
    for d in range(n_days):
        date = f'2026-{4 + d // 30:02d}-{1 + d % 30:02d}'
        for home, away in (('A', 'D'), ('B', 'C')) if d % 2 else (('D', 'A'), ('C', 'B')):
            out.append({'date': date, 'home_id': home, 'away_id': away,
                        'home_runs': rng.poisson(4.6 * strength[home] / strength[away]), 'away_runs': rng.poisson(4.3 * strength[away] / strength[home])})
    return out


def test_nb_pmf_and_home_field_tie_rule():
    pmf = nb_pmf(4.5, 0.1)
    assert pmf.sum() == pytest.approx(1.0) and (pmf * np.arange(31)).sum() == pytest.approx(4.5, abs=0.02)
    p, tie = win_prob(4.5, 4.5, 0.1, 0.52)
    assert 0.5 < p < 0.53 and 0.05 < tie < 0.15


def test_better_team_is_favored_and_inputs_are_prior_date():
    tm = TeamModel(rows())
    last = max(r['date'] for r in tm.rows)
    strong_home = tm.probability('2026-09-01', 'A', 'D'); weak_home = tm.probability('2026-09-01', 'D', 'A')
    assert strong_home['p_home'] > 0.6 and weak_home['p_home'] < 0.4
    assert strong_home['ratings']['home']['offense'] > 1.05 and strong_home['prior_games'] == len(tm.rows)
    assert tm.probability('2026-05-15', 'A', 'D')['prior_games'] < len(tm.rows)
    even = tm.probability('2026-09-01', 'B', 'C')
    assert 0.5 < even['p_home'] < 0.58          # home field alone
    with pytest.raises(ValueError):
        tm.probability('2026-04-02', 'A', 'D')  # fewer than 30 prior games


def test_unknown_team_gets_league_average():
    tm = TeamModel(rows())
    p = tm.probability('2026-09-01', 'ZZ', 'YY')
    assert 0.5 < p['p_home'] < 0.58 and p['ratings']['home'] == {'offense': 1.0, 'defense': 1.0}


def test_record_prefers_box_team_model():
    from brl_live.record import build_record
    f = {'game_pk': 1, 'version': 1, 'saved_at': '2026-10-07T18:00:00Z', 'date': '2026-10-07', 'home_win_probability': 0.6, 'team_baseline_probability': 0.50,
         'away': {'abbr': 'LAD'}, 'home': {'abbr': 'ATL'}}
    ledger = {'forecasts': {'a': f}, 'publications': {'a': {'commit': 'a' * 40, 'published_at': '2026-10-07T18:05:00Z'}},
              'actuals': {'1': {'away': 2, 'home': 5, 'first_pitch_observed_at': '2026-10-07T22:10:00Z'}},
              'box_scores': {'a': {'game_pk': 1, 'team_model': {'p_home': 0.58}}}}
    rec = build_record(ledger)
    g = rec['games'][0]
    assert g['p_team'] == 0.58 and g['p_team_runtime'] == 0.5
    assert rec['blend']['a'] > 0.58 and rec['blend']['a'] < 0.6
