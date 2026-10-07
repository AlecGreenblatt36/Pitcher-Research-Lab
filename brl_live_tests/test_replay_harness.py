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
