"""EDGE-02's capture: Kalshi's open markets for today's pending games, matched to the box's players, with the snapshot
taken 30 minutes or more before first pitch kept for scoring."""
from brl_live import kalshi_props as kp


def test_event_ticker_uses_eastern_time_and_team_codes():
    assert kp.event_ticker('KXMLBHIT', '2026-10-24T00:00:00Z', 'TB', 'MIL') == 'KXMLBHIT-26OCT232000TBMIL'
    assert kp.event_ticker('KXMLBGAME', '2026-11-01T23:05:00Z', 'TB', 'CLE') == 'KXMLBGAME-26NOV011805TBCLE'   # after daylight time ends


def test_markets_parse_to_compact_rows():
    names = {kp.norm_name('José Ramírez'): '608070', kp.norm_name('Jacob Misiorowski'): '694819'}
    prop = kp.parse_market({'ticker': 'KXMLBHIT-26OCT232000TBMIL-MILJRAMIREZ-2', 'title': 'José Ramírez: 2+ hits?', 'status': 'active',
                            'yes_bid_dollars': '0.2100', 'yes_ask_dollars': '0.2300', 'no_bid_dollars': '0.7700', 'no_ask_dollars': '0.7900',
                            'last_price_dollars': '0.2200', 'volume_fp': '812.50'}, 'KXMLBHIT', names)
    assert prop['k'] == 'prop' and prop['stat'] == 'hits' and prop['n'] == 2 and prop['pid'] == '608070'
    assert prop['yb'] == 0.21 and prop['ya'] == 0.23 and prop['na'] == 0.79 and prop['last'] == 0.22 and prop['vol'] == 812.5
    ks = kp.parse_market({'ticker': 'KXMLBKS-26OCT232000TBMIL-MILJMISIOROWSKI32-7', 'title': 'Jacob Misiorowski: 7+ strikeouts?', 'yes_bid': 52, 'yes_ask': 53}, 'KXMLBKS', names)
    assert ks['stat'] == 'k' and ks['n'] == 7 and ks['pid'] == '694819' and ks['ya'] == 0.53   # cents when no dollar field
    unknown = kp.parse_market({'ticker': 'x', 'title': 'Someone Else: 1+ hits?'}, 'KXMLBHIT', names)
    assert unknown['pid'] is None and unknown['name'] == 'Someone Else'
    total = kp.parse_market({'ticker': 'KXMLBTOTAL-26OCT232000TBMIL-7', 'title': 'Over 6.5 runs scored', 'floor_strike': 6.5, 'yes_ask_dollars': '0.52'}, 'KXMLBTOTAL', names)
    assert total['k'] == 'game' and total['line'] == 6.5 and 'team' not in total
    win = kp.parse_market({'ticker': 'KXMLBGAME-26OCT232000TBMIL-MIL', 'title': 'Milwaukee wins', 'yes_sub_title': 'Milwaukee', 'yes_ask_dollars': '0.63'}, 'KXMLBGAME', names)
    assert win['team'] == 'MIL'
    assert kp.parse_market({'ticker': 'x', 'title': 'Milwaukee wins', 'status': 'settled'}, 'KXMLBGAME', names) is None


def _ledger(start='2026-10-24T00:00:00Z'):
    return {'forecasts': {'f1': {'game_pk': 849805, 'date': '2026-10-23', 'version': 1, 'scheduled_start': start,
                                 'away': {'abbr': 'TB'}, 'home': {'abbr': 'MIL'}},
                          'f0': {'game_pk': 849815, 'date': '2026-10-19', 'version': 1, 'scheduled_start': '2026-10-20T00:08:00Z', 'away': {'abbr': 'CLE'}, 'home': {'abbr': 'TB'}}},
            'status': {'849805': {'state': 'Preview'}}, 'actuals': {}, 'box_scores': {'f1': {'teams': {'away': {'batting': [{'player_id': '650490', 'name': 'Yandy Díaz'}], 'pitching': []}, 'home': {'batting': [], 'pitching': []}},
                                                                                       'starters': {'home': {'name': 'Jacob Misiorowski', 'player_id': '694819'}}}}}


def _fetch(calls):
    def fetch_json(url):
        calls.append(url)
        if 'KXMLBHIT-' in url:
            return {'markets': [{'ticker': 'KXMLBHIT-26OCT232000TBMIL-TBYDIAZ2-1', 'title': 'Yandy Díaz: 1+ hits?', 'status': 'active', 'yes_bid_dollars': '0.60', 'yes_ask_dollars': '0.62'}]}, {}
        if 'KXMLBKS-' in url:
            raise OSError('timed out')
        return {'markets': []}, {}
    return fetch_json


def test_capture_fetches_each_series_by_event_and_keeps_the_pregame_snapshot():
    ledger = _ledger(); calls = []
    r = kp.capture(_fetch(calls), ledger, '2026-10-23', '2026-10-23T22:00:00Z')      # two hours before first pitch
    assert r['games'] == 1 and r['markets'] == 1 and len(calls) == len(kp.SERIES)
    assert all('event_ticker=' in u and '-26OCT232000TBMIL' in u for u in calls)
    row = ledger['kalshi']['849805']
    assert row['markets'][0]['pid'] == '650490' and row['errors'] == {'KXMLBKS': 'OSError'}
    assert row['pre']['captured_at'] == '2026-10-23T22:00:00Z' and row['minutes_to_first_pitch'] == 120.0
    # 45 minutes out: the pre snapshot moves with the capture; 10 minutes out: it stays
    kp.capture(_fetch([]), ledger, '2026-10-23', '2026-10-23T23:15:00Z')
    assert ledger['kalshi']['849805']['pre']['captured_at'] == '2026-10-23T23:15:00Z'
    kp.capture(_fetch([]), ledger, '2026-10-23', '2026-10-23T23:50:00Z')
    assert ledger['kalshi']['849805']['pre']['captured_at'] == '2026-10-23T23:15:00Z' and ledger['kalshi']['849805']['captured_at'] == '2026-10-23T23:50:00Z'
    # a game under way is skipped
    ledger['status']['849805']['state'] = 'In Progress'
    r = kp.capture(_fetch([]), ledger, '2026-10-23', '2026-10-24T00:30:00Z')
    assert r['games'] == 0 and r['skipped'] == 1


def test_page_carries_todays_pending_games_without_the_snapshot():
    from brl_live.box_page import trim_boxes
    public = {'date': '2026-10-23', 'box_scores': {}, 'forecasts': {}, 'actuals': {'849815': {}},
              'kalshi': {'849805': {'captured_at': '2026-10-23T22:00:00Z', 'markets': [], 'pre': {'x': 1}},
                         '849815': {'captured_at': '2026-10-19T22:00:00Z', 'markets': [], 'pre': {'x': 1}}}}
    out = trim_boxes(public)
    assert list(out['kalshi']) == ['849805'] and 'pre' not in out['kalshi']['849805']
