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
            # KEYS-02: worlds his team won with his home run (at most the worlds he homered) and with his two or more hits
            assert 0 <= x['kw'][0] <= 4 - x['hr_hist'][0] and 0 <= x['kw'][1] <= 4 - x['h_hist'][0] - x['h_hist'][1]

    # BULLPEN-03: each side's bullpen candidates with their share of worlds pitched in and whether they pitched
    out_sl = replay_dates(h, app, g, dates, model_path=plain, model_sha256=sha256_file(plain), history_path=history_path, hazard_path=hazard, n_sims=4,
                          starter_lines=True, log=lambda m: None)
    seen = 0
    for r in out_sl:
        # KEYS-02: per side over the n worlds: scored first, that and won, won, starter 18+ outs, that and won
        ka, kh = r['starters']['away']['keys'], r['starters']['home']['keys']
        assert kh[2] == r['home_wins'] and ka[2] == r['n'] - r['home_wins'] - r['ties']
        assert ka[0] + kh[0] <= r['n'] and all(k[1] <= min(k[0], k[2]) and k[4] <= min(k[3], k[2]) for k in (ka, kh))
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


def test_postseason_game_applies_the_live_usage_and_reconstruct_keeps_the_game_type():
    """POST-06: a postseason game's starters and teams get the round's factor on expected batters (each starter his
    tier's), the league fallback follows, and the reliever exit offsets come back by decision kind; a regular-season
    game and a missing table change nothing; reconstructed games carry their game type."""
    from brl_replay.harness import postseason_game
    rows = []
    for i in range(240):          # 240 regular-season plate appearances before the date: pitcher 7 strikes out 30%, walks 5%
        rows.append({'date_key': '2026-06-%02d' % (1 + i % 28), 'game_pk': 100 + i // 30, 'at_bat_number': i % 30 + 1, 'pitcher': 7, 'batter': 5000 + i % 9,
                     'outcome': 'K' if i % 10 < 3 else ('BB_HBP' if i % 20 == 3 else 'BIP_OUT'), 'game_type': 'R', 'inning_topbot': 'Top',
                     'home_team': 'H', 'away_team': 'A', 'inning': 1 + (i % 30) // 4, 'p_throws': 'R', 'stand': 'R', 'park': 'H', 'season': 2026,
                     'home_score': 0, 'away_score': 0, 'runner_1b': 0, 'runner_2b': 0, 'runner_3b': 0, 'outs_when_up': 0, 'bat_score': 0, 'fld_score': 0})
    h = pd.DataFrame(rows)
    post = {'scale': {'L': 0.91}, 'tiers': {'cuts': [0.1518, 0.1876], 'scale': [0.81, 0.89, 0.94]}, 'exit_offsets': {'L': {'mid': 0.8, 'end': -0.4}, 'D': 0.4}}
    pexp = {7: 24.0, 8: 20.0}; texp = {'A': 22.0, 'H': 23.0}
    pe, te, lg, offs = postseason_game(post, 'L', pexp, texp, 21.0, h, '2026-10-10', (7, 8))
    assert abs(pe[7] - 24.0 * 0.94) < 1e-9        # 30% strikeouts minus 5% walks = 0.25, the top tier
    assert abs(pe[8] - 20.0 * 0.91) < 1e-9        # no record this season: the round's factor
    assert abs(te['A'] - 22.0 * 0.91) < 1e-9 and abs(lg - 21.0 * 0.91) < 1e-9 and offs == {'mid': 0.8, 'end': -0.4}
    pe, te, lg, offs = postseason_game(post, 'D', pexp, texp, 21.0, h, '2026-10-10', (7, 8))
    assert abs(pe[8] - 20.0) < 1e-9 and offs == {'mid': 0.4, 'end': 0.4}    # no scale for D in this table; one number means both kinds
    assert postseason_game(post, 'R', pexp, texp, 21.0, h, '2026-10-10', (7, 8)) == (pexp, texp, 21.0, None)
    assert postseason_game(None, 'L', pexp, texp, 21.0, h, '2026-10-10', (7, 8)) == (pexp, texp, 21.0, None)
    # reconstruct keeps the type
    g = h.copy(); g['inning_topbot'] = ['Top' if k % 2 == 0 else 'Bot' for k in range(len(g))]
    g.loc[g['game_pk'] == 100, 'game_type'] = 'L'
    games = reconstruct(g)
    assert set(games.columns) >= {'game_type'} and games.set_index('game_pk').loc[100, 'game_type'] == 'L' and (games[games['game_pk'] != 100]['game_type'] == 'R').all()


def test_bullpen_takes_the_day_roster_as_the_pool():
    """BULLPEN-10: with the day's roster the pool is its pitchers minus the starter, including a starter in the pen and
    an arm with no outing on record (a right-handed unknown arm); without it the 14-day rule holds."""
    rows = []
    for d in range(1, 20):
        date = '2026-06-%02d' % d
        rows.append({'game_pk': d, 'pitcher': 1, 'date': date, 'team': 'AAA', 'bf': 4, 'entry_inning': 9, 'throws': 'R', 'start': False, 'finished': True, 'k': 1})
        rows.append({'game_pk': d, 'pitcher': 2, 'date': date, 'team': 'AAA', 'bf': 4, 'entry_inning': 7, 'throws': 'L', 'start': False, 'finished': False, 'k': 1})
        rows.append({'game_pk': d, 'pitcher': 10 + d % 5, 'date': date, 'team': 'AAA', 'bf': 22, 'entry_inning': 1, 'throws': 'R', 'start': True, 'finished': False, 'k': 5})
    app = pd.DataFrame(rows)
    old = bullpen(app, 'AAA', '2026-06-20', 10)
    assert sorted(int(p.player_id) for p in old) == [1, 2]
    pen = bullpen(app, 'AAA', '2026-06-20', 10, roster=[1, 2, 10, 11, 12, 99])
    assert sorted(int(p.player_id) for p in pen) == [1, 2, 11, 12, 99]
    by = {int(p.player_id): p for p in pen}
    assert by[2].throws == 'L' and by[99].throws == 'R' and dict(by[99].usage)['new_arm'] == 1.0 and dict(by[11].usage)['starts30'] >= 1 and dict(by[1].usage)['starts30'] == 0
    assert by[1].role == 'closer' and by[99].expected_batters == 4
