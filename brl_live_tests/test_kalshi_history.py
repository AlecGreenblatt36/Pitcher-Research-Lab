"""KALSHI-01: parsing Kalshi's event tickers and market titles (no network)."""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brl_kalshi_history_t', ROOT / 'tools' / 'brl_kalshi_history.py')
kh = importlib.util.module_from_spec(spec); spec.loader.exec_module(kh)


def test_event_start_reads_eastern_time_and_teams():
    start, teams = kh.event_start('KXMLBHIT-26AUG171940ATHKC')
    assert start == datetime(2026, 8, 17, 23, 40, tzinfo=timezone.utc) and teams == 'ATHKC'
    start, _ = kh.event_start('KXMLBGAME-26NOV011805TBCLE')             # after daylight time ends: 18:05 EST is 23:05 UTC
    assert start == datetime(2026, 11, 1, 23, 5, tzinfo=timezone.utc)
    assert kh.event_start('not-a-ticker') == (None, None)


def test_doubleheader_tickers_keep_their_time_and_teams():
    start, teams = kh.event_start('KXMLBGAME-26AUG171840STLCING2')       # game 2 of a doubleheader
    assert start == datetime(2026, 8, 17, 22, 40, tzinfo=timezone.utc) and teams == 'STLCIN'
    _, teams = kh.event_start('KXMLBGAME-26SEP221305TBNYYG1')
    assert teams == 'TBNYY'


def test_describe_reads_props_and_keeps_game_markets_as_given():
    d = kh.describe({'title': 'Victor Mesa Jr.: 2+ hits?'})
    assert d == {'kind': 'prop', 'name': 'Victor Mesa Jr.', 'at_least': 2, 'stat': 'hits'}
    d = kh.describe({'title': 'José Ramírez: 1+ hits + runs + RBIs?'})
    assert d['kind'] == 'prop' and d['at_least'] == 1 and d['stat'] == 'hits + runs + rbis'
    g = kh.describe({'title': 'Tampa Bay vs Cleveland Winner?', 'yes_sub_title': 'Tampa Bay', 'floor_strike': None})
    assert g['kind'] == 'game' and g['yes_sub_title'] == 'Tampa Bay'


def test_names_match_across_accents_and_suffixes():
    assert kh.norm_name('José Ramírez') == kh.norm_name('Jose Ramirez')
    assert kh.norm_name('Victor Mesa Jr.') == kh.norm_name('Victor Mesa')


def test_prop_lines_per_series():
    spec = 'KXMLBHIT:1,2 KXMLBTB:2'
    assert kh.prop_lines(spec, 'KXMLBHIT') == [1, 2]
    assert kh.prop_lines(spec, 'KXMLBTB') == [2]
    assert kh.prop_lines(spec, 'KXMLBKS') == []                          # every strikeout line
    assert kh.prop_lines('1 2', 'KXMLBHIT') == [1, 2]                     # the old form still works


def test_game_matched_by_teams_not_only_time():
    day = [{'gamePk': 1, 'gameDate': '2026-03-29T17:35:00Z', 'teams': {'away': {'team': {'abbreviation': 'BAL'}}, 'home': {'team': {'abbreviation': 'MIN'}}}},
           {'gamePk': 2, 'gameDate': '2026-03-29T17:35:00Z', 'teams': {'away': {'team': {'abbreviation': 'KC'}}, 'home': {'team': {'abbreviation': 'ATL'}}}},
           {'gamePk': 3, 'gameDate': '2026-03-29T17:35:00Z', 'teams': {'away': {'team': {'abbreviation': 'TEX'}}, 'home': {'team': {'abbreviation': 'PHI'}}}}]
    mlb = kh.MLB(lambda url: {'dates': [{'games': day}]})
    start, teams = kh.event_start('KXMLBGAME-26MAR291335KCATL')
    assert mlb.game_for(start, teams)['gamePk'] == 2
    assert mlb.game_for(start, 'NYYSF') is None                         # no game with those teams: never a neighbour by time
