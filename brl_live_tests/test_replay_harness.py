"""The cloud replay harness runs end to end on synthetic data with a plain bundle and a physics bundle."""
import importlib.util
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from research_lab.game_sim.locked_pa_provider import sha256_file
from brl_replay.games import load_history, reconstruct, final_scores
from brl_replay.harness import appearances, replay_dates, bullpen, starter_profile

spec = importlib.util.spec_from_file_location('brl_test_provider', Path(__file__).parent / 'test_provider_physics.py')
prov = importlib.util.module_from_spec(spec); spec.loader.exec_module(prov)


def test_reconstruct_and_replay(tmp_path):
    params = {'xvalue': True, 'recent_days': 3}
    model_path, history_path, table, pa_all, features, extras = prov.make_bundle(tmp_path, params)
    h = load_history(history_path)
    games = reconstruct(h)
    assert len(games) == pa_all.game_pk.nunique() and games.valid_lineups.all()
    assert set(games.columns) >= {'game_pk', 'date', 'home', 'away', 'park', 'away_lineup', 'home_lineup', 'away_starter', 'home_starter', 'home_runs', 'away_runs', 'ambiguous'}
    app = appearances(h)
    assert set(app.columns) >= {'game_pk', 'pitcher', 'date', 'team', 'bf', 'start'}
    hazard = Path(__import__('os').environ.get('BRL_DATA_ROOT', '')) / 'model_runs/starter_hazard_v1/starter_hazard.joblib'
    if not hazard.exists():
        import pytest; pytest.skip('starter hazard bundle not available locally')
    games['date'] = games['date'].astype(str)
    g = games[(games.season == 2026) & (~games.ambiguous)].sort_values(['date', 'game_pk'])
    dates = sorted(g.date.unique())[-2:]
    g = g[g.date.isin(dates)]
    # plain bundle (no physics) on the same data
    bundle = joblib.load(model_path); bundle.pop('physics_features'); bundle.pop('physics_params')
    from research_lab.pa_model.config import PAConfig
    from research_lab.pa_model.features import build_time_valid_features
    from research_lab.pa_model.model import fit_frozen_model
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final', regularization_grid=(0.01,), max_iter=30)
    feats, cols = build_time_valid_features(pa_all, config)
    fitted, _ = fit_frozen_model(feats, list(cols), config)
    bundle['talent_plus_context'] = fitted
    plain = tmp_path / 'plain.joblib'; joblib.dump(bundle, plain)
    out_plain = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4, log=lambda m: None)
    out_phys = replay_dates(h, app, g, dates, model_path=model_path, model_sha256=sha256_file(model_path), history_path=history_path, hazard_path=hazard, n_sims=4,
                            physics_table=table, log=lambda m: None)
    assert len(out_plain) == len(out_phys) == len(g)
    for r in out_phys:
        assert r['n'] == 4 and r['home_wins'] + r['ties'] <= 4 and sum(r['home_hist']) == 4 and 'home_runs' in r
        assert sum(r['margin_hist']) == 4 and sum(r['margin_hist'][16:]) == r['home_wins']      # the margin's joint histogram
        assert sum(r['f5_margin_hist']) == 4 and sum(r['f5_total_hist']) == 4 and 0 <= r['first_inning'][0] <= 4
    # a run environment that makes home runs ten times likelier in every game moves the run histograms up; same seeds otherwise
    import numpy as np
    z = np.zeros(7); z[5] = np.log(10.0)
    out_env = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                           environment={int(pk): z for pk in g['game_pk']}, log=lambda m: None)
    runs = lambda rows: sum(sum(k * c for k, c in enumerate(r['home_hist'])) + sum(k * c for k, c in enumerate(r['away_hist'])) for r in rows)
    assert len(out_env) == len(out_plain) and runs(out_env) > runs(out_plain)
    same = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                        environment={}, log=lambda m: None)
    assert [r['home_hist'] for r in same] == [r['home_hist'] for r in out_plain]
    # team offsets: the home team's hitters ten times likelier to homer raises the home runs; empty offsets change nothing
    home_hr = {d: {str(t): {'bat': z.tolist(), 'fld': [0.0] * 7} for t in g['home']} for d in dates}
    out_team = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                            team_offsets=home_hr, log=lambda m: None)
    home_runs = lambda rows: sum(sum(k * c for k, c in enumerate(r['home_hist'])) for r in rows)
    assert home_runs(out_team) > home_runs(out_plain)
    none = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                        team_offsets={}, log=lambda m: None)
    assert [r['home_hist'] for r in none] == [r['home_hist'] for r in out_plain]
    # aging layer: zero coefficients change nothing; a large home-run weight on the batter's history gap raises scoring
    from research_lab.pa_model import physics as phys
    cols = phys.AGING_FEATURES + phys.DECAY_FEATURES
    zero = {'columns': cols, 'labels': ['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH'], 'mean': [0.0] * 16, 'sd': [1.0] * 16,
            'coef': [[0.0] * 7 for _ in cols], 'params': {'aging': True, 'decay_days': 365, 'k_dec': 60.0}}
    same_age = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                            age_layer=zero, log=lambda m: None)
    assert [r['home_hist'] for r in same_age] == [r['home_hist'] for r in out_plain]
    big = dict(zero, coef=[[0.0] * 7 for _ in cols]); big['coef'][cols.index('b_gap')] = [0, 0, 0, 0, 0, 40.0, 0]
    out_age = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                           age_layer=big, log=lambda m: None)
    assert runs(out_age) > runs(out_plain)
    # hitter lines (PLAYER-03): each lineup hitter's histograms count every world once and agree with his chances
    out_hl = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                          hitter_lines=True, log=lambda m: None)
    for r in out_hl:
        assert len(r['hitters']) == 18
        for x in r['hitters']:
            assert all(sum(x[k]) == 4 for k in ('h_hist', 'hr_hist', 'bb_hist', 'k_hist'))
            for p_, k in (('p_hit', 'h_hist'), ('p_hr', 'hr_hist'), ('p_bb', 'bb_hist'), ('p_k', 'k_hist')):
                assert abs(x[p_] - (1 - x[k][0] / 4)) < 1e-9

    # BULLPEN-03: each side's bullpen candidates with their share of worlds pitched in and whether they pitched
    out_sl = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                          starter_lines=True, log=lambda m: None)
    seen = 0
    for r in out_sl:
        for side in ('away', 'home'):
            st = r['starters'][side]
            if 'relievers' not in st:
                continue
            seen += 1
            rel = st['relievers']
            assert all(0.0 <= share <= 1.0 and used in (0, 1) for _, share, used in rel)
            assert abs(sum(share for _, share, _ in rel) - (st['team_pitchers_sim_mean'] - 1)) < 1e-3 * (len(rel) + 1)
            assert sum(used for _, _, used in rel) == st['team_actual']['pitchers'] - 1
            # BULLPEN-05: relief entries by inning, lead and top-two arms, simulated per world and real
            ent = st['entries']
            assert abs(sum(ent['sim'].values()) - (st['team_pitchers_sim_mean'] - 1)) < 1e-3 * (len(ent['sim']) + 1)
            assert sum(ent['real'].values()) == st['team_actual']['pitchers'] - 1
            assert all(len(k.split(':')) == 3 for k in list(ent['sim']) + list(ent['real']))
    assert seen > 0
