"""Several books' lines from the odds page's embedded JSON, and their consensus."""
import json

from brl_live.market_books import parse_page, consensus_home, consensus_total, page_url
from brl_live.market import vig_free_home


def page(rows):
    doc = {'props': {'pageProps': {'oddsTables': [{'oddsTableModel': {'gameRows': rows}}]}}}
    return '<html><script id="__NEXT_DATA__" type="application/json">' + json.dumps(doc) + '</script></html>'


def test_parse_and_consensus():
    rows = [{'gameView': {'homeTeam': {'fullName': 'New York Yankees'}, 'awayTeam': {'fullName': 'Tampa Bay Rays'}, 'startDate': '2026-10-08T00:08:00+00:00'},
             'oddsViews': [{'sportsbook': 'draftkings', 'openingLine': {'homeOdds': -150, 'awayOdds': 130}, 'currentLine': {'homeOdds': -160, 'awayOdds': 135}},
                           {'sportsbook': 'fanduel', 'openingLine': {}, 'currentLine': {'homeOdds': -155, 'awayOdds': 130}},
                           None, {'sportsbook': 'caesars', 'currentLine': {'homeOdds': None, 'awayOdds': 120}}]}]
    games = parse_page(page(rows))
    assert len(games) == 1 and set(games[0]['books']) == {'draftkings', 'fanduel'}
    p, n = consensus_home(games[0]['books'])
    assert n == 2 and abs(p - (vig_free_home(-160, 135) + vig_free_home(-155, 130)) / 2) < 1e-12
    assert consensus_home(games[0]['books'], 'open') == (vig_free_home(-150, 130), 1)
    trows = [{'gameView': rows[0]['gameView'], 'oddsViews': [{'sportsbook': 'a', 'currentLine': {'total': 7.5, 'overOdds': -110, 'underOdds': -110}},
                                                            {'sportsbook': 'b', 'currentLine': {'total': 7.5, 'overOdds': -105, 'underOdds': -115}},
                                                            {'sportsbook': 'c', 'currentLine': {'total': 8, 'overOdds': 100, 'underOdds': -120}}]}]
    tg = parse_page(page(trows), 'totals')
    total, p_over, n = consensus_total(tg[0]['books'])
    assert total == 7.5 and n == 2 and abs(p_over - (0.5 + vig_free_home(-105, -115)) / 2) < 1e-12
    assert parse_page('<html>no data</html>') == [] and 'totals/full-game' in page_url('2026-10-08', 'totals')
