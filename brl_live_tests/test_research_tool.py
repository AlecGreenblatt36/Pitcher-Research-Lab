"""The research lane's physics features are time-valid and feed the locked fitting code."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('brl_research', Path(__file__).parents[1] / 'tools' / 'brl_research.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


def synthetic(seed=0, n_games=40, pa_per_side=12):
    rng = np.random.default_rng(seed)
    rows, study = [], {'schema': tool.STUDY_SCHEMA, 'year': 2026, 'games': {}}
    dates = [f'2026-05-{d:02d}' for d in range(1, 11)]
    for g in range(n_games):
        pk = 1000 + g; date = dates[g % len(dates)]
        plays = []
        for i in range(2 * pa_per_side):
            batter = 10 + (i % pa_per_side); pitcher = 500 + (i // pa_per_side)
            probs = [0.45, 0.23, 0.09, 0.14, 0.04, 0.03, 0.02] if batter != 10 else [0.25, 0.23, 0.09, 0.14, 0.09, 0.18, 0.02]   # batter 10: loud contact, more extra bases
            outcome = rng.choice(['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH'], p=probs)
            top = i < pa_per_side
            rows.append({'date_key': date, 'season': 2026, 'game_pk': pk, 'at_bat_number': i + 1, 'batter': batter, 'pitcher': pitcher, 'outcome': outcome,
                         'stand': 'R', 'p_throws': 'L', 'park': 'NYY', 'platoon': 1, 'is_home_batter': int(not top), 'inning': 1 + (i % pa_per_side) // 3, 'outs_when_up': i % 3,
                         'runner_1b': 0, 'runner_2b': 0, 'runner_3b': 0, 'bat_score_diff': 0, 'n_thruorder_pitcher': 1, 'batter_days_since_prev_game': 1,
                         'pitcher_days_since_prev_game': 5, 'age_bat': 28, 'age_pit': 29, 'inning_topbot': 'Top' if top else 'Bot',
                         'home_team': 'HOM', 'away_team': 'AWY', 'home_score': 0, 'away_score': 0, 'bat_score': 0, 'fld_score': 0})
            velo = 92.0 + (pitcher - 500) * 3.0 + rng.normal(0, 0.5)
            pitches = [['FF', 'C', 0, 0, round(velo, 1), 85.0, 2200, -5.0, 9.0, 0.1, 2.5, -1.0, 6.0, 6.3, 5],
                       ['SL', 'S' if batter % 2 else 'F', 0, 1, 84.0, 78.0, 2500, 4.0, 1.0, 0.9, 1.5, -1.0, 6.0, 6.3, 14],
                       ['FF', 'X', 0, 2, round(velo, 1), 85.0, 2200, -5.0, 9.0, 0.0, 2.6, -1.0, 6.0, 6.3, 4]]
            hit = [100.0 if batter == 10 else 85.0, 20.0, 300, 100.0, 100.0, 'line_drive', 'hard', '8'] if outcome not in ('K', 'BB_HBP') else None
            plays.append({'i': i, 'inning': 1 + i // 6, 'half': 'top', 'o': outcome, 'e': 'x', 'p': pitcher, 'b': batter, 's': 'R', 't': 'L', 'pitches': pitches, 'hit': hit})
        study['games'][str(pk)] = {'date': date, 'game_type': 'R', 'fetched_at': '2026-10-01T00:00:00+00:00', 'rows': plays}
    return pd.DataFrame(rows), study


def test_physics_features_are_time_valid_and_player_specific():
    pa, study = synthetic()
    physics = tool.per_pa_physics(study)
    assert set(physics.columns) >= {'game_pk', 'at_bat_number', 'pitcher', 'batter', 'n', 'sw', 'wh', 'fb', 'velo', 'ev', 'la'}
    assert (physics.n == 3).all() and (physics.fb == 2).all()
    extras, audit = tool.build_extras(pa, physics)
    assert audit['joined'] == len(pa) and audit['ids_match'] == len(pa)
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    first_day = ordered.date_key == '2026-05-01'
    # Nothing is known on the first date: every value is the (empty) league prior, identical across players.
    assert extras.loc[first_day, 'b_ev'].nunique() == 1 and extras.loc[first_day, 'p_velo'].nunique() == 1 and extras.loc[first_day, 'b_bip_n'].eq(0).all()
    later = ordered.date_key >= '2026-05-05'
    # Batter 10 hits the ball 100 mph, everyone else 85: his exit velocity feature is the highest; pitcher 501 throws harder than 500.
    assert extras.loc[later & (ordered.batter == 10), 'b_ev'].mean() > extras.loc[later & (ordered.batter == 11), 'b_ev'].mean() + 3
    assert extras.loc[later & (ordered.pitcher == 501), 'p_velo'].mean() > extras.loc[later & (ordered.pitcher == 500), 'p_velo'].mean() + 1
    # Odd batters whiff on every slider; even ones foul it off.
    assert extras.loc[later & (ordered.batter == 11), 'b_whiff'].mean() > extras.loc[later & (ordered.batter == 12), 'b_whiff'].mean()
    assert np.isfinite(extras.to_numpy()).all()
    # A velocity change feature only exists once the pitcher has 100 prior fastballs and a prior outing.
    assert (extras.loc[first_day, 'p_velo_delta'] == 0).all()


def test_fitting_path_accepts_extra_columns():
    from research_lab.pa_model.config import PAConfig
    from research_lab.pa_model.features import build_time_valid_features
    from research_lab.pa_model.model import fit_frozen_model
    pa, study = synthetic(n_games=60)
    # three seasons so the locked split works: copy the synthetic season with shifted dates
    frames = []
    for year in (2023, 2024, 2025, 2026):
        f = pa.copy(); f['season'] = year; f['date_key'] = f['date_key'].str.replace('2026', str(year)); f['game_pk'] = f['game_pk'] + (year - 2023) * 10000
        frames.append(f)
    pa_all = pd.concat(frames, ignore_index=True)
    physics = pd.concat([tool.per_pa_physics(study).assign(game_pk=lambda d, y=y: d.game_pk + (y - 2023) * 10000) for y in (2023, 2024, 2025, 2026)], ignore_index=True)
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final', regularization_grid=(0.01,), max_iter=30)
    features, cols = build_time_valid_features(pa_all, config)
    extras, audit = tool.build_extras(pa_all, physics)
    for c in tool.EXTRA:
        features[c] = extras[c].to_numpy()
    fitted, tuning = fit_frozen_model(features, list(cols) + tool.EXTRA, config)
    p = fitted.predict_proba(features[features.season == 2026])
    assert p.shape[1] == 7 and np.allclose(p.sum(axis=1), 1.0)


def test_variant_columns_and_time_validity_of_xvalue_and_recent():
    pa, study = synthetic(n_games=40)
    physics = tool.per_pa_physics(study)
    extras, audit = tool.build_extras(pa, physics, {'xvalue': True, 'recent_days': 3})
    assert list(extras.columns) == tool.EXTRA + tool.XVALUE_COLS + tool.RECENT_COLS
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    first = ordered.date_key == '2026-05-01'
    assert extras.loc[first, 'b_xv'].nunique() == 1 and (extras.loc[first, tool.RECENT_COLS] == 0).all().all()
    later = ordered.date_key >= '2026-05-06'
    # batter 10's 100-mph contact lands in a richer cell than the 85-mph contact of everyone else
    assert extras.loc[later & (ordered.batter == 10), 'b_xv'].mean() > extras.loc[later & (ordered.batter == 11), 'b_xv'].mean()
    assert np.isfinite(extras.to_numpy()).all()
    assert tool.columns_for({}) == tool.EXTRA and tool.columns_for({'recent_days': 30}) == tool.EXTRA + tool.RECENT_COLS
    assert set(tool.SETS['physics2']) >= {'base', 'k_low', 'k_high', 'xvalue', 'recent30', 'xvalue_recent30'}
