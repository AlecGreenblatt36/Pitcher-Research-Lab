"""CHAIN-01: the sealed study flattens to one row per pitch with the count before it and the plate appearance's outcome."""
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location('brl_pitch_table', Path(__file__).parents[1] / 'tools' / 'brl_pitch_table.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


def test_rows_from_study():
    study = {'games': {'7': {'date': '2026-05-01', 'rows': [
        {'i': 0, 'o': 'K', 'p': 500, 'b': 10, 's': 'L', 't': 'R', 'pitches': [['FF', 'C', 0, 0, 95.1, 87.0, 2300, -6.0, 15.0, 0.2, 2.9, -1.5, 5.9, 6.4, 5],
                                                                             ['SL', 'S', 0, 1, 85.0, 78.0, 2500, 4.0, 1.0, 0.9, 1.5, -1.4, 5.8, 6.2, 14],
                                                                             ['SL', 'S', 0, 2, 84.0, 77.0, 2450, 4.2, 0.8, 1.1, 1.1, -1.4, 5.8, 6.2, None]], 'hit': None},
        {'i': 1, 'o': 'HR', 'p': 500, 'b': None, 's': 'R', 't': 'R', 'pitches': [['CH', 'X', 0, 0, 86.0, 79.0, 1700, -8.0, 4.0, -0.2, 2.2, -1.5, 5.9, 6.4, 5]], 'hit': [108.0, 28.0]}]}}}
    f = pd.DataFrame(tool.rows_from_study(study), columns=tool.PITCH_COLUMNS)
    assert len(f) == 4 and f['pitch_number'].tolist() == [1, 2, 3, 1] and f['last'].tolist() == [0, 0, 1, 1]
    assert f['at_bat_number'].tolist() == [1, 1, 1, 2] and f['batter'].tolist() == [10, 10, 10, -1] and f['outcome'].tolist() == ['K', 'K', 'K', 'HR']
    assert f.loc[2, 'zone'] is None or pd.isna(f.loc[2, 'zone'])
    assert f['strikes'].tolist() == [0, 1, 2, 0] and f.loc[0, 'speed'] == 95.1 and f.loc[3, 'p_throws'] == 'R'
