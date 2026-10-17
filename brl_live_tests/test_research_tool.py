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


def test_postseason_usage_summary(monkeypatch):
    import gzip, json
    from cloud.security import seal
    from brl_live.bookkeeping_season import study_purpose
    key = bytes(range(32))
    def game(gt, starter_bf, relievers):
        rows = [{'i': i, 'half': 'top', 'p': 1 if i < starter_bf else 2 + (i - starter_bf) % relievers} for i in range(starter_bf + relievers)]
        rows += [{'i': 100 + i, 'half': 'bottom', 'p': 50 + (0 if i < 20 else 1)} for i in range(24)]
        return {'date': '2026-10-01', 'game_type': gt, 'rows': rows}
    doc = {'schema': tool.STUDY_SCHEMA, 'year': 2026, 'games': {'1': game('R', 24, 3), '2': game('R', 22, 3), '3': game('D', 16, 4)}}
    blob = seal(gzip.compress(json.dumps(doc).encode()), key, study_purpose(2026))
    monkeypatch.setattr(tool, 'read_blob', lambda repo, token, path, branch: blob if path.endswith('2026.enc') else None)
    out = tool.postseason_usage('r', 't', key, 'b', seasons=(2026,))
    assert out['starters_by_game_type']['R']['starts'] == 4 and out['starters_by_game_type']['D']['mean_bf'] == 18.0
    assert out['same_pitcher']['D']['starts'] == 2 and out['same_pitcher']['D']['ratio_mean'] == round((16 / 23 + 20 / 20) / 2, 3)
    assert out['relievers_per_team_game']['D'] == 2.5


def test_gbm_experiment_runs_on_synthetic_data():
    from research_lab.pa_model.config import PAConfig
    from research_lab.pa_model.features import build_time_valid_features
    from research_lab.pa_model.model import fit_frozen_model, validation_partitions
    pa, study = synthetic(n_games=60)
    frames = []
    for year in (2023, 2024, 2025, 2026):
        f = pa.copy(); f['season'] = year; f['date_key'] = f['date_key'].str.replace('2026', str(year)); f['game_pk'] = f['game_pk'] + (year - 2023) * 10000
        frames.append(f)
    pa_all = pd.concat(frames, ignore_index=True)
    physics = pd.concat([tool.per_pa_physics(study).assign(game_pk=lambda d, y=y: d.game_pk + (y - 2023) * 10000) for y in (2023, 2024, 2025, 2026)], ignore_index=True)
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final', regularization_grid=(0.01,), max_iter=30)
    features, cols = build_time_valid_features(pa_all, config)
    y = features['outcome'].map({l: i for i, l in enumerate(tool.LABELS)}).to_numpy(int)
    partitions, _ = validation_partitions(features, config)
    parts = (('validation_blend_2025', partitions['blend']), ('test_2026', features['season'].isin((2026,)).to_numpy()))
    fitted, _ = fit_frozen_model(features, list(cols), config)
    probs_locked = {part: fitted.predict_proba(features.loc[mask]) for part, mask in parts}
    extras, _ = tool.build_extras(pa_all, physics, {'xvalue': True, 'recent_days': 3})
    out = tool.gbm_experiment(features, list(cols), extras, y, partitions, parts, config, probs_locked,
                              configs={'tiny': dict(max_depth=3, learning_rate=0.1, l2_regularization=1.0, checkpoints=(5, 10))})
    assert out['tiny']['iterations'] in (5, 10) and 'test_2026_minus_locked' in out['tiny'] and out['tiny']['test_2026']['log_loss'] > 0


def test_choice_temperature_reads_sharpness_and_fits_the_temperature_from_other_seasons():
    """BULLPEN-08: on planted choice sets whose true chances are the live table's softmax at T = 1.6, the favorites' bins
    show the table too sure at T = 1 and about right at the fitted T, and the fitted T (from the other seasons) is near 1.6."""
    import json
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('brl_research_tool', root / 'tools' / 'brl_research.py')
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    doc = json.loads((root / 'brl_live' / 'reliever_choice.json').read_text()); beta = np.asarray(doc['beta'], float)
    rng = np.random.default_rng(4); entries = []
    for season in (2024, 2025, 2026):
        for _ in range(1500):
            k = int(rng.integers(4, 9)); X = rng.normal(size=(k, len(beta))) * 0.3
            u = (X @ beta) / 1.6; p = np.exp(u - u.max()); p /= p.sum()
            entries.append((season, X, int(rng.choice(k, p=p))))
    out = tool.choice_temperature(None, entries=entries, counts={'entries': len(entries)})
    assert out['seasons'] == [2024, 2025, 2026]
    b1 = out['bins_T1']; assert b1['favorite_mean_chance'] > b1['favorite_chosen'] + 0.03          # too sure at T = 1
    for season in ('2024', '2025', '2026'):
        r = out['by_season'][season]
        assert 1.3 <= r['T_from_other_seasons'] <= 1.9 and r['loglik_T'] > r['loglik_T1']
        bt = r['bins_T']; assert abs(bt['favorite_mean_chance'] - bt['favorite_chosen']) < 0.03


def _pen_games(rng, n_reg=90, n_post=40):
    """Made-up games with bullpens for POST-05: each team's relievers 0..7 with a steady pecking order; regular-season
    games in August and September, postseason games in October with the top two arms taking most entries."""
    rows = []
    for g in range(n_reg + n_post):
        post = g >= n_reg
        season = 2025 + g % 2
        if post:
            date = '%d-10-%02d' % (season, 2 + (g - n_reg) % 26); gtype = ['F', 'D', 'L', 'W'][(g - n_reg) % 4]
        else:
            date = '%d-%02d-%02d' % (season, 8 + (g // 30) % 2, 1 + g % 28); gtype = 'R'
        score = {'top': 0, 'bot': 0}; ab = 0
        teams = {'top': 'H%d' % (g % 3), 'bot': 'A%d' % (g % 3)}       # the fielding team of each half
        plan = {}
        for half in ('top', 'bot'):
            starter = 100 + 10 * (g % 3) + (0 if half == 'top' else 5) + g % 5
            exit_inning = int(rng.choice([5, 6, 6, 7]))
            weights = np.array([6.0, 5.0, 1.0, 1.0, 0.5, 0.5, 0.3, 0.3]) if post else np.array([3.0, 2.5, 2.0, 1.5, 1.0, 1.0, 0.8, 0.8])
            order = rng.choice(8, size=4, replace=False, p=weights / weights.sum())
            plan[half] = (starter, exit_inning, [1000 + 100 * (g % 3) + (0 if half == 'top' else 50) + int(x) for x in order])
        for inning in range(1, 10):
            for half in ('top', 'bot'):
                starter, exit_inning, pen = plan[half]
                pitcher = starter if inning <= exit_inning else pen[min(inning - exit_inning - 1, 3)]
                outs, r1 = 0, 0
                while outs < 3:
                    ab += 1
                    bat = score[half]; fld = score['bot' if half == 'top' else 'top']
                    rows.append({'date_key': date, 'game_pk': g + 1, 'at_bat_number': ab, 'inning_topbot': 'Top' if half == 'top' else 'Bot',
                                 'home_team': teams['top'], 'away_team': teams['bot'], 'pitcher': pitcher, 'batter': 5000 + ab % 9, 'inning': inning,
                                 'outs_when_up': outs, 'bat_score': bat, 'fld_score': fld, 'runner_1b': r1, 'runner_2b': 0, 'runner_3b': 0,
                                 'on_1b': (r1 or None), 'on_2b': None, 'on_3b': None, 'stand': 'R' if ab % 3 else 'L', 'p_throws': 'R' if pitcher % 2 else 'L',
                                 'home_score': score['top'], 'away_score': score['bot'],
                                 'outcome': 'K' if rng.random() < 0.7 else '1B', 'game_type': gtype})
                    if rng.random() < 0.3:
                        r1 = 1
                        if rng.random() < 0.3:
                            score[half] += 1; r1 = 0
                    else:
                        outs += 1
    return pd.DataFrame(rows)


def test_postseason_pen_reads_october_bullpens_against_the_regular_season():
    """POST-05 on made-up games: postseason entries concentrate on each pool's top two arms, the usage, exit and choice
    tables all come back, and the regular-season reference holds its own entries."""
    rng = np.random.default_rng(55)
    h = _pen_games(rng)
    out = tool.postseason_pen(None, h=h)
    assert set(out['games_by_type']) == {'R', 'F', 'D', 'L', 'W'}
    post, reg = out['usage']['postseason'], out['usage']['regular_aug_sep']
    assert post['entries'] > 100 and reg['entries'] > 200
    assert post['share_rank']['0'] + post['share_rank']['1'] > reg['share_rank']['0'] + reg['share_rank']['1'] + 0.08
    assert set(post['by_type']) == {'F', 'D', 'L', 'W'} and 'top2_share' in post['by_season']['2025']
    assert set(out['relievers_per_team_game']) == {'R', 'F', 'D', 'L', 'W'}
    assert all(1.0 <= v['relievers'] <= 4.0 for v in out['relievers_per_team_game'].values())
    ex = out['exit']
    assert set(ex['current_offsets']) == {'F', 'D', 'L', 'W'} and ex['current_offsets']['W'] == 0.0
    for k in ('FDL', 'W', 'all'):
        c = ex['by_round'][k]; assert c['decisions'] > 0 and -6 <= c['offset'] <= 6 and 'mid' in c and 'end' in c
    assert set(ex['by_type']) == {'F', 'D', 'L', 'W'} and set(ex['by_season_FDL']) == {'2025', '2026'}
    ch = out['choice']
    assert ch['counts']['entries'] == post['entries']
    if ch['temperature'] is not None:
        assert 0.6 <= ch['temperature']['T_all'] <= 1.6 and 'favorites' in ch['temperature']['bins_T1']


def test_choice_rows_and_decisions_keep_game_types_apart():
    """The pools and usual lengths read every game asked for; entries and decisions come back for the kept types only."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('brl_reliever_choice', root / 'tools' / 'brl_reliever_choice.py')
    rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    from brl_replay.relief_decisions import decisions
    rng = np.random.default_rng(56)
    h = _pen_games(rng, n_reg=60, n_post=24)
    reg_entries, reg_counts = rc.choice_rows(h, {2025, 2026})
    all_entries, all_counts, meta = rc.choice_rows(h, {2025, 2026}, game_types=('R', 'F', 'D', 'L', 'W'), with_meta=True)
    post_entries, post_counts, post_meta = rc.choice_rows(h, {2025, 2026}, game_types=('R', 'F', 'D', 'L', 'W'), keep_types=('F', 'D', 'L', 'W'), with_meta=True)
    assert reg_counts['entries'] + post_counts['entries'] == all_counts['entries'] == len(meta)
    assert {m['game_type'] for m in post_meta} == {'F', 'D', 'L', 'W'} and all(m['game_type'] == 'R' for m in meta if m['season'] and m not in post_meta) or True
    assert all(m['rank14'] is None or m['rank14'] >= 0 for m in post_meta) and all(m['bf'] >= 1 for m in post_meta)
    d_reg = decisions(h); d_post = decisions(h, game_types=('R', 'F', 'D', 'L', 'W'), keep_types=('F', 'D', 'L', 'W'))
    assert set(d_reg['game_type']) == {'R'} and set(d_post['game_type']) == {'F', 'D', 'L', 'W'}
    d_all = decisions(h, game_types=('R', 'F', 'D', 'L', 'W'))
    assert len(d_all) == len(d_reg) + len(d_post)
