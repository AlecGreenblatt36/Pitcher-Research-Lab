"""PITCH-01/02: the pitch-level chain. Sanity of the count chain, date blocking, and parity between the chronological
builder and the state built to a cutoff."""
import json
import numpy as np
import pandas as pd
import pytest

from research_lab.pa_model import pitch_chain as pc


def synthetic(seed=0, days=40, games_per_day=4, pa_per_game=30):
    """Hitters and pitchers with known tendencies: batter 1 chases and whiffs, pitcher 100 lives up in the zone."""
    rng = np.random.default_rng(seed)
    rows, pa_rows = [], []
    start = pd.Timestamp('2026-04-01')
    batters = list(range(1, 19)); pitchers = list(range(100, 108))
    for d in range(days):
        day = (start + pd.Timedelta(days=d)).strftime('%Y-%m-%d')
        for g in range(games_per_day):
            pk = 1000 + d * 10 + g
            park = ['NYY', 'BOS', 'COL', 'SEA', 'LAD', 'CHC', 'HOU', 'ATL', 'SF', 'SD', 'TEX', 'MIA'][g % 12]
            for i in range(pa_per_game):
                b = batters[(i + g) % len(batters)]; p = pitchers[(d + g + i // 15) % len(pitchers)]
                stand = 'L' if b % 2 else 'R'; throws = 'L' if p % 3 == 0 else 'R'
                balls = strikes = 0; n = 0; outcome = None
                while outcome is None:
                    n += 1
                    fam = rng.choice(['FF', 'SL', 'CH'], p=[0.6, 0.25, 0.15] if p != 100 else [0.8, 0.1, 0.1])
                    zone = int(rng.choice([2, 5, 8, 12, 14], p=[0.2, 0.3, 0.2, 0.15, 0.15] if p != 100 else [0.45, 0.25, 0.1, 0.15, 0.05]))
                    out = zone > 9
                    p_swing = (0.25 if out else 0.65) + (0.25 if (b == 1 and out) else 0.0)
                    swing = rng.random() < p_swing
                    if not swing:
                        code = 'B' if out else ('C' if rng.random() < 0.9 else 'B')
                    else:
                        p_whiff = 0.22 + (0.3 if b == 1 else 0.0) + (0.1 if fam == 'SL' else 0.0)
                        if rng.random() < p_whiff:
                            code = 'S'
                        elif rng.random() < 0.45:
                            code = 'F'
                        else:
                            code = 'X'
                    rows.append([day, pk, i + 1, n, p, b, stand, throws, balls, strikes, fam, code, 92.0, 2200, -5.0, 9.0, 0.1, 2.5, zone, 6.3])
                    if code == 'B':
                        balls += 1
                        if balls == 4: outcome = 'BB_HBP'
                    elif code in ('C', 'S'):
                        strikes += 1
                        if strikes == 3: outcome = 'K'
                    elif code == 'F':
                        strikes = min(2, strikes + 1)
                    else:
                        pr = [0.60, 0.22, 0.07, 0.10, 0.01] if park == 'COL' else [0.69, 0.21, 0.06, 0.03, 0.01]
                        outcome = str(rng.choice(['BIP_OUT', '1B', '2B_3B', 'HR', 'OTHER_REACH'], p=pr))
                    if n > 25:
                        outcome = 'BIP_OUT'
                for r in rows[-n:]:
                    r.append(outcome)
                pa_rows.append({'date_key': day, 'game_pk': pk, 'at_bat_number': i + 1, 'batter': b, 'pitcher': p, 'stand': stand, 'p_throws': throws,
                                'park': park, 'outcome': outcome, 'season': 2026})
    pitches = pd.DataFrame(rows, columns=['date_key', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter', 'stand', 'p_throws', 'balls', 'strikes',
                                          'ptype', 'code', 'speed', 'spin', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone', 'ext', 'outcome'])
    pitches['last'] = 0
    return pd.DataFrame(pa_rows), pitches


def test_chain_probabilities_sum_to_one_and_read_the_tendencies():
    pa, pitches = synthetic()
    feats, audit = pc.build_features(pa, pitches, {'half_life': 0.0})
    probs = np.exp(feats[[f'ch_{l}' for l in pc.LABELS]].to_numpy())
    assert np.allclose(probs.sum(1), 1.0, atol=2e-3)
    assert (feats['ch_pitches'] > 2.5).all() and (feats['ch_pitches'] < 8).all()
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number']).reset_index(drop=True)
    parks = {int(g): str(k) for g, k in zip(ordered['game_pk'], ordered['park'])}
    state = pc.ChainState.build(pitches, '2026-05-11', {'half_life': 0.0}, parks)
    day = int(pc.day_ordinal(['2026-05-11'])[0])
    # batter 1 chases and whiffs: against the same pitcher the chain gives him more strikeouts and fewer walks than batter 3
    f1 = np.exp(state.features(1, 101, 'NYY', 'L', 'R', day)[:7]); f3 = np.exp(state.features(3, 101, 'NYY', 'L', 'R', day)[:7])
    assert f1[1] > f3[1] + 0.02 and f1[2] < f3[2]
    # Coors gives more home runs than the Bronx for the same matchup
    hr_col = np.exp(state.features(4, 101, 'COL', 'R', 'R', day)[5]); hr_nyy = np.exp(state.features(4, 101, 'NYY', 'R', 'R', day)[5])
    assert hr_col > hr_nyy * 1.2
    # the chain beats league-only tables on its own pitches (PITCH-01's check)
    rep = audit['per_pitch_check']
    assert rep['swing']['gain_per_1000'] > 5 and rep['whiff']['gain_per_1000'] > 0 and rep['outcome']['n'] > 1000


def test_first_day_is_the_league_and_nothing_leaks_from_the_same_day():
    pa, pitches = synthetic(days=3)
    feats, _ = pc.build_features(pa, pitches)
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number']).reset_index(drop=True)
    first = ordered['date_key'] == '2026-04-01'
    f = feats[first]
    # with nothing absorbed every matchup is the prior (platoon aside): identical rows within a platoon
    same = f[(ordered.loc[first, 'stand'] == ordered.loc[first, 'p_throws']).to_numpy()]
    assert same.round(9).drop_duplicates().shape[0] == 1
    # dropping the first day's pitches does not change the first day's features
    feats2, _ = pc.build_features(pa, pitches[pitches['date_key'] != '2026-04-01'])
    assert np.allclose(feats[first].to_numpy(), feats2[first].to_numpy())


def test_state_at_cutoff_equals_the_chronological_builder():
    pa, pitches = synthetic(days=12)
    params = {'half_life': 30.0}
    feats, _ = pc.build_features(pa, pitches, params)
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number']).reset_index(drop=True)
    for date in ('2026-04-05', '2026-04-12'):
        parks = {int(g): str(k) for g, k in zip(ordered['game_pk'], ordered['park'])}
        state = pc.ChainState.build(pitches, date, params, parks)
        rows = np.where(ordered['date_key'].to_numpy() == date)[0]
        day = pc.day_ordinal([date])[0]
        for r in rows[:25]:
            got = state.features(int(ordered.batter[r]), int(ordered.pitcher[r]), str(ordered.park[r]), str(ordered.stand[r]), str(ordered.p_throws[r]), int(day))
            assert np.allclose(got, feats.iloc[r].to_numpy(float), atol=1e-5), (date, r)
    # advancing the state one day at a time gives the same as building fresh
    q = pc.prepare(pitches, parks)
    inc = pc.ChainState(params); inc.absorb_through(q, int(pc.day_ordinal(['2026-04-05'])[0])); inc.absorb_through(q, int(pc.day_ordinal(['2026-04-12'])[0]))
    fresh = pc.ChainState.build(pitches, '2026-04-12', params, parks)
    r = rows[0]; day = pc.day_ordinal(['2026-04-12'])[0]
    a = inc.features(int(ordered.batter[r]), int(ordered.pitcher[r]), str(ordered.park[r]), str(ordered.stand[r]), str(ordered.p_throws[r]), int(day))
    b = fresh.features(int(ordered.batter[r]), int(ordered.pitcher[r]), str(ordered.park[r]), str(ordered.stand[r]), str(ordered.p_throws[r]), int(day))
    assert np.allclose(a, b)


def test_region_and_result_classes():
    reg = pc.region_of([1, 5, 9, 11, 14, None, None, None], [0, 0, 0, 0, 0, 0.1, 0.1, 1.5], [3.0, 2.5, 1.7, 3.8, 1.0, 3.2, 1.8, 2.0])
    assert reg.tolist() == [0, 1, 2, 3, 4, 0, 2, 4]
    cls = pc.result_class(['B', 'C', 'S', 'F', 'X', 'H', '*B', 'W', 'T', 'D'])
    assert cls.tolist() == [0, 1, 2, 3, 4, 5, 0, 2, 3, 4]


def test_provider_appends_chain_features_and_matches_the_builder(tmp_path):
    """A bundle with chain features makes the provider build the chain at the cutoff and append the same numbers the
    chronological builder wrote; without a pitch table it fails closed."""
    import importlib.util
    from dataclasses import asdict
    from pathlib import Path
    import joblib
    from research_lab.pa_model.config import PAConfig
    from research_lab.pa_model.features import build_time_valid_features
    from research_lab.pa_model.model import fit_frozen_model
    from research_lab.game_sim.locked_pa_provider import LockedPAModelProvider, LockedModelError, sha256_file, MODEL_LABELS, expected_feature_columns
    pa, pitches = synthetic(seed=3, days=36)
    for col, val in (('platoon', 0), ('is_home_batter', 0), ('inning', 1), ('outs_when_up', 0), ('runner_1b', 0), ('runner_2b', 0), ('runner_3b', 0),
                     ('bat_score_diff', 0), ('n_thruorder_pitcher', 1), ('batter_days_since_prev_game', 1), ('pitcher_days_since_prev_game', 5), ('age_bat', 28), ('age_pit', 29)):
        pa[col] = val
    frames, pitch_frames = [], []
    for i, year in enumerate((2023, 2024, 2025, 2026)):
        f = pa.copy(); f['season'] = year; f['date_key'] = f['date_key'].str.replace('2026', str(year)); f['game_pk'] = f['game_pk'] + i * 100000; frames.append(f)
        q = pitches.copy(); q['date_key'] = q['date_key'].str.replace('2026', str(year)); q['game_pk'] = q['game_pk'] + i * 100000; pitch_frames.append(q)
    pa_all = pd.concat(frames, ignore_index=True); pitch_all = pd.concat(pitch_frames, ignore_index=True)
    config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final', regularization_grid=(0.01,), max_iter=30)
    features, cols = build_time_valid_features(pa_all, config)
    params = {'half_life': 300.0}
    chain, _ = pc.build_features(pa_all, pitch_all, params)
    for c in chain.columns:
        features[c] = chain[c].to_numpy()
    fitted, _ = fit_frozen_model(features, list(cols) + list(chain.columns), config)
    EB = {'batter_weight': 0.5, 'pitcher_weight': 0.5, 'park_weight': 0.1, 'batter_recent_weight': 0.1, 'pitcher_recent_weight': 0.1, 'temperature': 1.0}
    bundle = {'talent_plus_context': fitted, 'talent_only': fitted, 'empirical_bayes_parameters': EB, 'model_weight': 0.9, 'empirical_bayes_weight': 0.1,
              'labels': MODEL_LABELS, 'feature_columns': {'talent_plus_context': fitted.feature_columns}, 'config': asdict(config),
              'chain_params': dict(pc.DEFAULT_PARAMS, **params), 'chain_features': list(chain.columns), 'name': 'test-chain'}
    model_path = tmp_path / 'pa_model.joblib'; joblib.dump(bundle, model_path)
    history_path = tmp_path / 'history.csv.gz'; pa_all.to_csv(history_path, index=False)
    date = '2026-05-03'
    with pytest.raises(LockedModelError):
        LockedPAModelProvider(model_path, history_path, date, date, 'NYY', expected_sha256=sha256_file(model_path))
    provider = LockedPAModelProvider(model_path, history_path, date, date, 'NYY', expected_sha256=sha256_file(model_path), pitch_table=pitch_all)
    assert list(provider.chain_features) == pc.FEATURES and provider.chain is not None
    ordered = pa_all.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    rows = np.where((ordered['date_key'].to_numpy() == date) & (ordered['park'].to_numpy() == 'NYY'))[0]
    assert len(rows) > 5
    n_base = len(expected_feature_columns())
    for r in rows[:8]:
        b, p = int(ordered['batter'][r]), int(ordered['pitcher'][r])
        row, _ = provider.feature_row(b, p, str(ordered['stand'][r]), str(ordered['p_throws'][r]), is_home_batter=0, inning=1, outs=0, runners=(0, 0, 0), bat_score_diff=0, times_through=1)
        assert len(row) == n_base + len(pc.FEATURES)
        assert np.allclose(row[n_base:], chain.iloc[r].to_numpy(float).astype(np.float32), atol=1e-5)


def test_live_pitch_table_assembles_seasons_then_the_day_cache():
    """PITCH-02 at forecast time: the sealed season tables, then every final feed in the day cache after their coverage,
    nothing from today; the feed rows carry the count before each pitch."""
    import gzip
    from datetime import datetime, timezone
    from cloud.security import seal
    from brl_live.physics_inputs import assemble_pitch_table
    key = bytes(range(32))

    class Store:
        def __init__(self): self.files = {}
        def read(self, path): return (self.files[path], 'sha') if path in self.files else None

    class Cache:
        def __init__(self, store, days): self.store, self.key, self.days = store, key, days
        def load_day(self, ident): return self.days[ident]

    def ev(code, ptype='FF', balls=0, strikes=0):
        return {'isPitch': True, 'details': {'code': code, 'type': {'code': ptype}}, 'pitchData': {'startSpeed': 94.0, 'zone': 5, 'coordinates': {'pX': 0.1, 'pZ': 2.5}},
                'count': {'balls': balls, 'strikes': strikes}}

    def play(i, batter, pitcher, event, events):
        return {'result': {'eventType': event}, 'about': {'isComplete': True, 'atBatIndex': i},
                'matchup': {'pitcher': {'id': pitcher}, 'batter': {'id': batter}, 'batSide': {'code': 'L'}, 'pitchHand': {'code': 'R'}}, 'playEvents': events}

    def feed(pk, date, state='Final'):
        return {'gamePk': pk, 'gameData': {'status': {'abstractGameState': state}, 'datetime': {'officialDate': date}},
                'liveData': {'plays': {'allPlays': [play(0, 11, 21, 'single', [ev('B', balls=1), ev('X', 'SL', 1, 0)]),
                                                    play(1, 12, 21, 'strikeout', [ev('C', strikes=1), ev('S', strikes=2), ev('S', strikes=3)])]}}}
    _, pitches = synthetic(seed=5, days=3)
    pitches['date_key'] = '2026-10-10'
    store = Store()
    store.files['private/statcast/pitches-2026.enc'] = seal(gzip.compress(pitches.to_csv(index=False).encode()), key, 'pitch-table-season-2026')
    days = {'d1': {'sources': [{'url': 'https://statsapi.mlb.com/api/v1.1/game/900/feed/live', 'body': json.dumps(feed(900, '2026-10-12'))}]},
            'd2': {'sources': [{'url': 'https://statsapi.mlb.com/api/v1.1/game/901/feed/live', 'body': json.dumps(feed(901, '2026-10-13', 'Live'))}]},
            'd3': {'sources': [{'url': 'https://statsapi.mlb.com/api/v1.1/game/902/feed/live', 'body': json.dumps(feed(902, '2026-10-14'))}]}}
    index = {'days': {'2026-10-12': {'id': 'd1'}, '2026-10-13': {'id': 'd2'}, '2026-10-14': {'id': 'd3'}}}
    origin = datetime(2026, 10, 14, 16, 0, tzinfo=timezone.utc)      # noon Eastern on the 14th: the 14th is today
    table, receipt = assemble_pitch_table(Cache(store, days), index, origin, years=(2025, 2026))
    assert receipt['seasons'] == {'2026': {'rows': len(pitches), 'through': '2026-10-10'}} and receipt['day_cache_rows'] == 5
    feed_rows = table[table['game_pk'] == 900].sort_values(['at_bat_number', 'pitch_number'])
    assert feed_rows['balls'].tolist() == [0, 1, 0, 0, 0] and feed_rows['strikes'].tolist() == [0, 0, 0, 1, 2]
    assert feed_rows['code'].tolist() == ['B', 'X', 'C', 'S', 'S'] and feed_rows['outcome'].tolist() == ['1B', '1B', 'K', 'K', 'K'] and feed_rows['last'].tolist() == [0, 1, 0, 0, 1]
    assert 902 not in set(table['game_pk']) and 901 not in set(table['game_pk']) and receipt['through'] == '2026-10-12'
    # the assembled table prepares like a sealed one
    q = pc.prepare(table, {900: 'NYY'})
    assert len(q) == len(table) and set(q['park'].unique()) >= {'NYY', 'UNK'}


def test_v2_options_keep_parity_and_add_the_extra_features():
    pa, pitches = synthetic(seed=9, days=10)
    params = {'two_strike_groups': True, 'pitcher_cells': True, 'extra_features': True, 'half_life': 60.0}
    feats, audit = pc.build_features(pa, pitches, params)
    assert list(feats.columns) == pc.FEATURES + pc.EXTRA_FEATURES and audit['per_pitch_check']['swing']['n'] == len(pitches)
    probs = np.exp(feats[[f'ch_{l}' for l in pc.LABELS]].to_numpy()); assert np.allclose(probs.sum(1), 1.0, atol=2e-3)
    assert ((feats['ch_sw00'] > 0) & (feats['ch_sw00'] < 1) & (feats['ch_wh02'] >= 0) & (feats['ch_ball00'] + feats['ch_ip00'] <= 1.0001)).all()
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number']).reset_index(drop=True)
    parks = {int(g): str(k) for g, k in zip(ordered['game_pk'], ordered['park'])}
    state = pc.ChainState.build(pitches, '2026-04-08', params, parks)
    day = int(pc.day_ordinal(['2026-04-08'])[0])
    for r in np.where(ordered['date_key'].to_numpy() == '2026-04-08')[0][:20]:
        got = state.features(int(ordered.batter[r]), int(ordered.pitcher[r]), str(ordered.park[r]), str(ordered.stand[r]), str(ordered.p_throws[r]), day)
        assert np.allclose(got, feats.iloc[r].to_numpy(float), atol=1e-5)
