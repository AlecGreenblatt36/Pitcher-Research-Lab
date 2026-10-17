"""ROSTER-01: the daily roster builder without the network: team-dates from a schedule document, pitchers from a roster
document (two-way players count), the day table, the missing list and the receipt counts."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('brl_rosters', Path(__file__).resolve().parents[1] / 'tools' / 'brl_rosters.py')
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)


def fake_get(url):
    if 'schedule' in url:
        g = lambda pk, date, gt, a, h: {'gamePk': pk, 'gameType': gt, 'officialDate': date, 'status': {'abstractGameState': 'Final'},
                                       'teams': {'away': {'team': {'id': a, 'abbreviation': 'A%d' % a}}, 'home': {'team': {'id': h, 'abbreviation': 'H%d' % h}}}}
        return {'dates': [{'date': '2025-06-01', 'games': [g(1, '2025-06-01', 'R', 108, 114), g(2, '2025-06-01', 'S', 109, 115)]},
                          {'date': '2025-10-05', 'games': [g(3, '2025-10-05', 'D', 108, 114), {'gamePk': 4, 'gameType': 'R', 'officialDate': '2025-10-05', 'status': {'abstractGameState': 'Preview'}, 'teams': {}}]}]}
    if '/teams/114/roster' in url and '2025-10-05' in url:
        return None                                                               # a day the API has no roster for
    tid = int(url.split('/teams/')[1].split('/')[0])
    return {'roster': [{'person': {'id': 10 * tid + 1}, 'position': {'type': 'Pitcher'}}, {'person': {'id': 10 * tid + 2}, 'position': {'type': 'Catcher'}},
                       {'person': {'id': 10 * tid + 3}, 'position': {'type': 'Two-Way Player'}}, {'person': {'id': 10 * tid + 1}, 'position': {'type': 'Pitcher'}}]}


def test_build_reads_pitchers_per_team_date_and_counts_the_misses():
    teams, days = rs.team_dates(fake_get, 2025)
    assert teams == {108: 'A108', 114: 'H114'} and set(days) == {('2025-06-01', 108), ('2025-06-01', 114), ('2025-10-05', 108), ('2025-10-05', 114)}
    assert days[('2025-10-05', 108)] == 'D'                                       # spring games and unfinished games are left out
    doc = rs.build(fake_get, 2025, workers=2, log=lambda m: None)
    assert doc['days']['2025-06-01'] == {'A108': [1081, 1083], 'H114': [1141, 1143]}
    assert doc['days']['2025-10-05'] == {'A108': [1081, 1083]} and doc['missing'] == [['2025-10-05', 'H114']]
    rec = rs.summary(doc)
    assert rec['team_dates'] == 4 and rec['read'] == 3 and rec['missing'] == 1 and rec['pitchers_per_roster_mean'] == 2.0 and rec['public_data_only']
    assert rs.rosters_path(2025) == 'research/rosters-2025.json.gz' and rec['path'] == 'research/rosters-2025.json.gz'
