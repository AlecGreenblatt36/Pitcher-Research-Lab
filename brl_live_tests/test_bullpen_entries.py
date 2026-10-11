"""BULLPEN-05's summary (tools/brl_bullpen_entries.py) adds the cells up by inning and lead and keeps the top two apart."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('brl_bullpen_entries', Path(__file__).parents[1] / 'tools/brl_bullpen_entries.py')
be = importlib.util.module_from_spec(spec); spec.loader.exec_module(be)


def test_cells_add_up():
    g = {'game_pk': 1, 'starters': {'home': {'entries': {'sim': {'9:3:1': 0.6, '8:3:1': 0.5, '6:1:0': 0.9}, 'real': {'9:3:1': 1, '7:2:0': 1}}},
                                    'away': {'entries': {'sim': {'5:0:0': 1.2}, 'real': {'5:0:0': 1, '6:0:0': 1}}}}}
    r = be.season_check([g])
    assert r['team_games'] == 2 and abs(r['all']['sim'] - 1.6) < 1e-9 and abs(r['all']['real'] - 2.0) < 1e-9
    assert abs(r['top2']['sim'] - 0.55) < 1e-9 and abs(r['top2']['real'] - 0.5) < 1e-9
    assert abs(r['by_inning']['9th or later']['sim_top2'] - 0.3) < 1e-9 and r['by_lead']['up 1-3']['real'] == 0.5
