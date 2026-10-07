"""Game conditions from the public schedule: weather, wind, roof and home plate umpire."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('brl_game_conditions', Path(__file__).resolve().parents[1] / 'tools/brl_game_conditions.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def test_parse_game_reads_weather_wind_roof_and_umpire():
    g = {'gamePk': 1, 'gameDate': '2026-06-01T23:05:00Z', 'dayNight': 'night', 'doubleHeader': 'N', 'scheduledInnings': 9,
         'weather': {'condition': 'Partly Cloudy', 'temp': '78', 'wind': '12 mph, Out To CF'},
         'officials': [{'official': {'id': 9, 'fullName': 'Ump One'}, 'officialType': 'First Base'},
                       {'official': {'id': 7, 'fullName': 'Ump Plate'}, 'officialType': 'Home Plate'}],
         'venue': {'id': 17, 'name': 'Wrigley Field', 'fieldInfo': {'roofType': 'Open', 'turfType': 'Grass'}}}
    r = mod.parse_game(g, '2026-06-01')
    assert r['temp_f'] == 78 and r['wind_mph'] == 12 and r['wind_dir'] == 'Out To CF' and r['hp_umpire_id'] == 7 and r['roof'] == 'Open'
    g['weather'] = {'condition': 'Dome', 'temp': '72', 'wind': '0 mph, None'}; g['officials'] = []
    r = mod.parse_game(g, '2026-06-01')
    assert r['wind_mph'] == 0 and r['wind_dir'] == 'None' and r['hp_umpire_id'] is None
    g['weather'] = {}
    r = mod.parse_game(g, '2026-06-01')
    assert r['temp_f'] is None and r['wind_mph'] is None and r['condition'] is None
