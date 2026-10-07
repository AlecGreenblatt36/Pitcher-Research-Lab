import pytest
from brl_live.market import implied, vig_free_home, parse_scoreboard, schedule_games, match, capture
from brl_live.record import build_record


def test_moneyline_math():
    assert implied(-150) == pytest.approx(0.6) and implied(130) == pytest.approx(100 / 230)
    assert vig_free_home(-150, 130) == pytest.approx(0.6 / (0.6 + 100 / 230))
    assert vig_free_home(-110, -110) == pytest.approx(0.5)


def board(home_ml=-150, away_ml=130, state='pre'):
    return {'events': [{'date': '2026-10-07T22:08Z', 'competitions': [{'status': {'type': {'state': state}},
        'competitors': [{'homeAway': 'home', 'team': {'displayName': 'Atlanta Braves', 'abbreviation': 'ATL'}},
                        {'homeAway': 'away', 'team': {'displayName': 'Los Angeles Dodgers', 'abbreviation': 'LAD'}}],
        'odds': [{'provider': {'name': 'ESPN BET'}, 'homeTeamOdds': {'moneyLine': home_ml}, 'awayTeamOdds': {'moneyLine': away_ml}, 'overUnder': 8.5}]}]},
        {'date': '2026-10-07T23:00Z', 'competitions': [{'competitors': [{'homeAway': 'home', 'team': {'displayName': 'X'}}, {'homeAway': 'away', 'team': {'displayName': 'Y'}}], 'odds': []}]}]}


def schedule(state='Preview'):
    return {'dates': [{'games': [{'gamePk': 1, 'gameDate': '2026-10-07T22:08:00Z', 'status': {'abstractGameState': state},
                                  'teams': {'away': {'team': {'name': 'Los Angeles Dodgers', 'abbreviation': 'LAD'}}, 'home': {'team': {'name': 'Atlanta Braves', 'abbreviation': 'ATL'}}}}]}]}


def test_parse_and_match():
    lines = parse_scoreboard(board())
    assert len(lines) == 1 and lines[0]['p_home'] == pytest.approx(vig_free_home(-150, 130), abs=1e-4) and lines[0]['provider'] == 'ESPN BET'
    games = schedule_games(schedule())
    assert match(games, lines)[1]['home_ml'] == -150


def test_capture_keeps_only_pregame_lines():
    calls = []
    def fetch(url):
        calls.append(url)
        return (schedule() if 'statsapi' in url else board()), {}
    ledger = {}
    r = capture(fetch, '2026-10-07', ledger, '2026-10-07T20:00:00+00:00')
    assert r['captured_pregame'] == 1 and ledger['market']['1']['p_home'] > 0.5 and ledger['market']['1']['captured_at'] == '2026-10-07T20:00:00+00:00'
    assert any('dates=20261007' in u for u in calls)
    def fetch_live(url):
        return (schedule('Live') if 'statsapi' in url else board(home_ml=-300, away_ml=250)), {}
    r = capture(fetch_live, '2026-10-07', ledger, '2026-10-07T23:00:00+00:00')
    assert r['captured_pregame'] == 0 and ledger['market']['1']['home_ml'] == -150


def test_record_scores_market_only_when_captured_before_first_pitch():
    f = {'game_pk': 1, 'version': 1, 'saved_at': '2026-10-07T18:00:00Z', 'date': '2026-10-07', 'home_win_probability': 0.6, 'team_baseline_probability': 0.55,
         'away': {'abbr': 'LAD'}, 'home': {'abbr': 'ATL'}}
    ledger = {'forecasts': {'a': f}, 'publications': {'a': {'commit': 'a' * 40, 'published_at': '2026-10-07T18:05:00Z'}},
              'actuals': {'1': {'away': 2, 'home': 5, 'first_pitch_observed_at': '2026-10-07T22:10:00Z'}},
              'market': {'1': {'p_home': 0.62, 'captured_at': '2026-10-07T21:50:00Z'}}}
    rec = build_record(ledger)
    assert rec['games'][0]['p_market'] == 0.62 and {r['key']: r['n'] for r in rec['ladder']}['market'] == 1
    ledger['market']['1']['captured_at'] = '2026-10-07T22:30:00Z'
    rec = build_record(ledger)
    assert rec['games'][0]['p_market'] is None and {r['key']: r['n'] for r in rec['ladder']}['market'] == 0


def test_new_espn_moneyline_shapes():
    from brl_live.market import _moneylines
    comp = {'competitors': [{'homeAway': 'home', 'team': {'abbreviation': 'ATL'}}, {'homeAway': 'away', 'team': {'abbreviation': 'LAD'}}],
            'odds': [{'provider': {'name': 'ESPN BET'}, 'moneyline': {'home': {'close': {'odds': '+130'}, 'open': {'odds': '+125'}}, 'away': {'close': {'odds': '-150'}}}, 'overUnder': 7.5}]}
    assert _moneylines(comp) == (130, -150, 'ESPN BET', 7.5)
    comp2 = {'competitors': comp['competitors'], 'odds': [{'provider': {'name': 'X'}, 'details': 'LAD -150', 'moneyline': {}}]}
    assert _moneylines(comp2) is None
    comp3 = {'competitors': comp['competitors'], 'odds': [{'provider': {'name': 'X'}, 'moneyline': {'home': 'EVEN', 'away': {'odds': '-120'}}}]}
    assert _moneylines(comp3) == (100, -120, 'X', None)


def test_totals_from_scoreboard_odds():
    from brl_live.market import _totals, _total_line
    assert _total_line('o7.5') == 7.5 and _total_line('u8') == 8.0 and _total_line(9) == 9.0 and _total_line('x') is None
    comp = {'odds': [{'overUnder': 7.5, 'overOdds': -108, 'underOdds': -112}]}
    assert _totals(comp) == (7.5, -108, -112)
    comp = {'odds': [{'total': {'over': {'close': {'line': 'o8.5', 'odds': '+100'}}, 'under': {'close': {'line': 'u8.5', 'odds': '-120'}}}}]}
    assert _totals(comp) == (8.5, 100, -120)
    assert _totals({'odds': [{'details': 'LAD -150'}]}) is None


def test_capture_keeps_the_first_line_seen():
    def fetch_at(home_ml, away_ml):
        return lambda url: ((schedule() if 'statsapi' in url else board(home_ml=home_ml, away_ml=away_ml)), {})
    ledger = {}
    capture(fetch_at(-150, 130), '2026-10-07', ledger, '2026-10-07T15:00:00+00:00')
    first = ledger['market']['1']['p_home']
    capture(fetch_at(-180, 155), '2026-10-07', ledger, '2026-10-07T20:00:00+00:00')
    m = ledger['market']['1']
    assert m['first_p_home'] == first and m['first_captured_at'] == '2026-10-07T15:00:00+00:00' and m['p_home'] > first


def test_record_counts_line_moves_toward_our_number():
    f = {'game_pk': 1, 'version': 1, 'saved_at': '2026-10-07T18:00:00Z', 'date': '2026-10-07', 'home_win_probability': 0.66, 'team_baseline_probability': 0.62,
         'away': {'abbr': 'LAD'}, 'home': {'abbr': 'ATL'}}
    ledger = {'forecasts': {'a': f}, 'publications': {'a': {'commit': 'a' * 40, 'published_at': '2026-10-07T18:05:00Z'}},
              'actuals': {'1': {'away': 2, 'home': 5, 'first_pitch_observed_at': '2026-10-07T22:10:00Z'}},
              'market': {'1': {'p_home': 0.58, 'captured_at': '2026-10-07T21:50:00Z', 'first_p_home': 0.52, 'first_captured_at': '2026-10-07T15:00:00Z'}}}
    rec = build_record(ledger)
    assert rec['line_moves'] == {'threshold_logit': 0.1, 'n': 1, 'toward_ours': 1}           # ours about 0.64 from 0.52: the move to 0.58 is toward us
    ledger['market']['1']['p_home'] = 0.48
    assert build_record(ledger)['line_moves'] == {'threshold_logit': 0.1, 'n': 1, 'toward_ours': 0}
    ledger['market']['1']['p_home'] = 0.52
    assert build_record(ledger)['line_moves']['n'] == 0                                       # the line did not move
