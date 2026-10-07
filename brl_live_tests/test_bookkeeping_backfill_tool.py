"""The backfill tool keeps only bookkeeping fields from the official play-by-play."""
import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('brl_bookkeeping_backfill', Path(__file__).parents[1] / 'tools' / 'brl_bookkeeping_backfill.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


def test_extract_rows():
    play = {'result': {'eventType': 'single', 'description': 'A singles on a line drive to left fielder B.'}, 'about': {'isComplete': True, 'atBatIndex': 4},
            'matchup': {'pitcher': {'id': 1}, 'batter': {'id': 2}, 'batSide': {'code': 'L'}},
            'playEvents': [{'isPitch': True, 'details': {'code': 'B', 'type': {'code': 'FF'}}, 'pitchData': {'startSpeed': 95.1}, 'count': {'balls': 1, 'strikes': 0}},
                           {'isPitch': False}, {'isPitch': True, 'details': {'code': 'X', 'type': {'code': 'SL'}}, 'pitchData': {'startSpeed': 85.0},
                            'hitData': {'trajectory': 'line_drive', 'location': '7', 'launchSpeed': 101.3, 'totalDistance': 250.0}, 'count': {'balls': 1, 'strikes': 0}}]}
    skipped = {'result': {'eventType': 'stolen_base_2b'}, 'about': {'isComplete': True, 'atBatIndex': 5}, 'matchup': {'pitcher': {'id': 1}}, 'playEvents': []}
    rows = tool.extract({'allPlays': [play, skipped, dict(play, about={'isComplete': False, 'atBatIndex': 6})]})
    assert rows == [{'i': 4, 'o': '1B', 'e': 'single', 'p': 1, 'b': 2, 's': 'L', 'n': 2,
                     'pt': [{'b': 0, 's': 0, 'r': 'B', 't': 'FF', 'v': 95.1}, {'b': 1, 's': 0, 'r': 'X', 't': 'SL', 'v': 85.0}],
                     'c': {'t': 'L', 'loc': 7, 'dist': 250, 'ev': 101}}]


def test_completed_games_filter(monkeypatch):
    doc = {'dates': [{'date': '2026-04-01', 'games': [
        {'gamePk': 1, 'gameType': 'R', 'officialDate': '2026-04-01', 'status': {'abstractGameState': 'Final', 'detailedState': 'Final'}},
        {'gamePk': 2, 'gameType': 'R', 'officialDate': '2026-04-01', 'status': {'abstractGameState': 'Final', 'detailedState': 'Postponed'}},
        {'gamePk': 3, 'gameType': 'S', 'officialDate': '2026-04-01', 'status': {'abstractGameState': 'Final', 'detailedState': 'Final'}},
        {'gamePk': 4, 'gameType': 'D', 'officialDate': '2026-04-01', 'status': {'abstractGameState': 'Live', 'detailedState': 'In Progress'}},
        {'gamePk': 5, 'gameType': 'F', 'officialDate': '2026-04-01', 'status': {'abstractGameState': 'Final', 'detailedState': 'Completed Early: Rain'}}]}]}
    monkeypatch.setattr(tool, 'get_json', lambda url, *a, **k: doc)
    assert [g['game_pk'] for g in tool.completed_games(2026, '2026-10-06')] == [1, 5]
