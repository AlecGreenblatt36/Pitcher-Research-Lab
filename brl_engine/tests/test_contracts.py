"""Own-runtime contracts: pregame inputs, forecast record, audits, store behaviour."""
import json, os
import numpy as np
import pandas as pd
import pytest
from cloud.contracts import (live_inputs, config_for, draw_seeds, summarize, check_public, score_versions,
                             assert_finished_before_start, FORECAST_FIELDS, N)
from cloud.runner import LocalStore, fingerprint_from_forecasts, Runner
from cloud.security import seal, unseal, key_bytes
from app.safety import Blocked
from app.results import parse_final


def history():
    rows = []
    for g in range(1, 16):
        date = f'2026-09-{g:02d}'
        for half, (team, opp, lineup, pitcher) in enumerate([('AWY', 'HOM', range(101, 110), 401), ('HOM', 'AWY', range(201, 210), 301)]):
            for i, b in enumerate(lineup):
                rows.append({'game_pk': g, 'date_key': date, 'at_bat_number': half * 9 + i + 1, 'batter': b, 'pitcher': pitcher,
                             'inning': 1 + i // 3, 'inning_topbot': 'Top' if half == 0 else 'Bot', 'home_team': 'HOM', 'away_team': 'AWY',
                             'p_throws': 'R', 'outcome': 'K', 'stand': 'L' if b % 2 else 'R'})
        for pid, inn, top in [(302, 8, 'Bot'), (303, 9, 'Bot'), (402, 8, 'Top'), (403, 9, 'Top')]:
            rows.append({'game_pk': g, 'date_key': date, 'at_bat_number': 30 + pid % 10, 'batter': 150, 'pitcher': pid, 'inning': inn,
                         'inning_topbot': top, 'home_team': 'HOM', 'away_team': 'AWY', 'p_throws': 'L', 'outcome': 'BIP_OUT', 'stand': 'R'})
    return pd.DataFrame(rows)


def pregame_feed(official=True, probable=True):
    players = {}
    for pid in list(range(101, 110)) + list(range(201, 210)) + [301, 302, 303, 401, 402, 403]:
        players['ID' + str(pid)] = {'id': pid, 'fullName': f'Player {pid}', 'batSide': {'code': 'R'}, 'pitchHand': {'code': 'L'}}
    return {'gamePk': 999, 'gameData': {
        'status': {'abstractGameState': 'Preview', 'detailedState': 'Scheduled'}, 'game': {'type': 'D'},
        'datetime': {'dateTime': '2026-10-07T22:08:00Z', 'officialDate': '2026-10-07'},
        'teams': {'away': {'id': 1, 'name': 'Away Nine', 'abbreviation': 'AWY'}, 'home': {'id': 2, 'name': 'Home Nine', 'abbreviation': 'HOM'}},
        'probablePitchers': {'away': {'id': 301}, 'home': {'id': 401}} if probable else {}, 'players': players, 'venue': {'name': 'Test Park'}},
        'liveData': {'boxscore': {'teams': {
            'away': {'battingOrder': list(range(109, 100, -1)) if official else [], 'bullpen': [302, 303, 304, 305], 'pitchers': []},
            'home': {'battingOrder': [], 'bullpen': [], 'pitchers': []}}}, 'plays': {'allPlays': []}}}


def test_live_inputs_official_and_projected_lineups():
    receipt = {'finished_at': '2026-10-07T18:00:00+00:00'}
    game, matchup, notes, statuses, fp = live_inputs(pregame_feed(), receipt, history())
    assert statuses == {'away': 'official', 'home': 'projected'}
    assert [p.player_id for p in matchup.away.lineup] == [str(i) for i in range(109, 100, -1)]
    assert [p.player_id for p in matchup.home.lineup] == [str(i) for i in range(201, 210)]
    assert matchup.away.starter.player_id == '301' and {p.player_id for p in matchup.away.bullpen} == {'302', '303', '304', '305'}
    assert {p.player_id for p in matchup.home.bullpen} == {'402', '403'}      # history fallback
    assert game['home']['abbr'] == 'HOM' and game['game_type'] == 'D' and game['away']['starter']['player_id'] == '301'
    assert notes['history_through_used'] == '2026-09-15' and len(fp) == 64
    game2, m2, _, _, fp2 = live_inputs(pregame_feed(official=False), receipt, history())
    assert fp2 != fp and [p.player_id for p in m2.away.lineup] == [str(i) for i in range(101, 110)]


def test_live_inputs_refuses_started_games_and_missing_starters():
    with pytest.raises(Blocked):
        live_inputs(pregame_feed(probable=False), {'finished_at': '2026-10-07T18:00:00+00:00'}, history())
    with pytest.raises(Blocked):
        live_inputs(pregame_feed(), {'finished_at': '2026-10-07T23:00:00+00:00'}, history())
    f = pregame_feed(); f['gameData']['status']['abstractGameState'] = 'Live'
    with pytest.raises(Blocked):
        live_inputs(f, {'finished_at': '2026-10-07T18:00:00+00:00'}, history())


class R:
    def __init__(self, w, a, h): self.winner, self.away_score, self.home_score, self.ended_by_plate_appearance_cap, self.provider_name = w, a, h, False, 'locked-pa-2026-v1'


def test_summary_matches_the_public_record():
    game, matchup, notes, statuses, fp = live_inputs(pregame_feed(), {'finished_at': '2026-10-07T18:00:00+00:00'}, history())
    rng = np.random.default_rng(0)
    results = [R('home' if rng.random() < 0.55 else 'away', 3, 4) for _ in range(N)]
    f = summarize(game, statuses, results, '2026-10-07T18:00:00+00:00', '2026-10-07T18:04:00+00:00', {'probability': 0.52}, notes, fp, 1, 'run-1')
    assert set(f) == FORECAST_FIELDS and f['automatic_runner'] is False and f['postseason_regular_bullpen_logic'] is True
    assert 0.5 < f['home_win_probability'] < 0.6 and f['n_simulations'] == N and f['team_baseline_probability'] == 0.52
    check_public(f)
    bad = dict(f); bad['extra'] = 1
    with pytest.raises(Blocked):
        check_public(bad)
    with pytest.raises(Blocked):
        summarize(game, statuses, results[:10], '2026-10-07T18:00:00+00:00', '2026-10-07T18:04:00+00:00', None, notes, fp, 1, 'x')
    guard = pregame_feed()
    assert_finished_before_start(game, guard, '2026-10-07T18:04:00+00:00')
    with pytest.raises(Blocked):
        assert_finished_before_start(game, guard, '2026-10-07T22:30:00+00:00')


def test_config_seeds_and_scores():
    assert config_for('R')['automatic_runner_in_extras'] and not config_for('D')['automatic_runner_in_extras']
    assert len(draw_seeds(1)) == N and (draw_seeds(1) == draw_seeds(1)).all() and not (draw_seeds(1) == draw_seeds(2)).all()
    forecasts = {'a': {'game_pk': 1, 'version': 1, 'home_win_probability': 0.6, 'team_baseline_probability': 0.5, 'market_probability': None},
                 'b': {'game_pk': 1, 'version': 2, 'home_win_probability': 0.7, 'team_baseline_probability': 0.5, 'market_probability': None}}
    pubs = {'a': {'published_at': '2026-10-07T18:00:00Z'}, 'b': {'published_at': '2026-10-07T23:00:00Z'}}
    actuals = {'1': {'away': 1, 'home': 4, 'first_pitch_observed_at': '2026-10-07T22:10:00Z'}}
    s = score_versions(forecasts, pubs, actuals)
    assert s['n_games'] == 1 and s['model_brier'] == pytest.approx(0.16) and s['team_brier'] == pytest.approx(0.25) and s['market_brier'] is None


def test_local_store_and_sealing(tmp_path):
    key = key_bytes('ab' * 32)
    store = LocalStore(tmp_path, key)
    ident = store.publish({'game_pk': 1, 'version': 1, 'x': 1})
    with pytest.raises(Blocked):
        store.publish({'game_pk': 1, 'version': 1, 'x': 1})
    store.private('inputs', 'abc', {'secret': 1})
    raw = (tmp_path / 'private/inputs/abc.enc').read_bytes()
    assert raw[:8] == b'BRLAESG1' and json.loads(unseal(raw, key, 'private:inputs:abc')) == {'secret': 1}
    with pytest.raises(Exception):
        unseal(raw, key, 'private:inputs:other')
    store.persist()
    again = LocalStore(tmp_path, key)
    assert ident in again.ledger['forecasts'] and fingerprint_from_forecasts(again, 1)[0][0] == ident


def test_parse_final_reads_runs_ids_and_first_pitch():
    feed = {'gamePk': 5, 'gameData': {'status': {'abstractGameState': 'Final'}, 'teams': {'away': {'id': 1}, 'home': {'id': 2}}, 'datetime': {'officialDate': '2026-10-06'}},
            'liveData': {'linescore': {'teams': {'away': {'runs': 3}, 'home': {'runs': 1}}},
                         'plays': {'allPlays': [{'about': {'startTime': '2026-10-06T22:09:00Z'}, 'playEvents': [{'isPitch': False}, {'isPitch': True, 'startTime': '2026-10-06T22:11:49.337Z'}]}]}}}
    r = parse_final(feed, 5, '2026-10-07T00:00:00Z')
    assert r['away'] == 3 and r['home'] == 1 and r['team_ids'] == {'away': 1, 'home': 2}
    assert r['first_pitch_observed_at'].startswith('2026-10-06T22:11:49')


def test_runner_isolates_blocked_games(tmp_path):
    class Net:
        def json(self, url):
            return {'dates': [{'games': [{'gamePk': 1, 'gameDate': 'x', 'status': {'abstractGameState': 'Preview'}, 'teams': {'away': {'team': {'name': 'A'}}, 'home': {'team': {'name': 'B'}}}},
                                         {'gamePk': 2, 'gameDate': 'x', 'status': {'abstractGameState': 'Preview'}, 'teams': {'away': {'team': {'name': 'C'}}, 'home': {'team': {'name': 'D'}}}}]}]}, {}
    seen = []
    class Mine(Runner):
        def process(self, pk, item):
            seen.append(pk)
            if pk == 1:
                raise Blocked('no starter')
    store = LocalStore(tmp_path, key_bytes('cd' * 32))
    Mine(Net(), store, None).iteration()
    assert seen == [1, 2] and store.ledger['status']['1']['state'] == 'blocked' and store.ledger['iteration']['blocked']['1']['error'] == 'Blocked: no starter'


def test_team_results_schedule_documents(tmp_path):
    import gzip
    from app.team_baseline import load_results
    doc = {'dates': [{'date': '2026-04-01', 'games': [
        {'gamePk': 1, 'gameType': 'R', 'officialDate': '2026-04-01', 'status': {'abstractGameState': 'Final'},
         'teams': {'away': {'score': 3, 'team': {'id': 119}}, 'home': {'score': 5, 'team': {'id': 144}}}},
        {'gamePk': 2, 'gameType': 'S', 'status': {'abstractGameState': 'Final'}, 'teams': {'away': {'score': 1, 'team': {'id': 1}}, 'home': {'score': 2, 'team': {'id': 2}}}},
        {'gamePk': 3, 'gameType': 'R', 'status': {'abstractGameState': 'Postponed'}, 'teams': {'away': {'team': {'id': 1}}, 'home': {'team': {'id': 2}}}}]}]}
    (tmp_path / 'team_results_2026.json.gz').write_bytes(gzip.compress(json.dumps(doc).encode()))
    rows, meta = load_results(tmp_path)
    assert rows == [{'game_pk': 1, 'date': '2026-04-01', 'away_id': 119, 'home_id': 144, 'away_runs': 3, 'home_runs': 5}]
