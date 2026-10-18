"""DQ-02: the matchup report's record counts a game once, on the last date it was graded (a postponed or suspended game
was graded on both dates), and a postponed game is not final."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('brl_report_rec_t', ROOT / 'tools' / 'brl_report.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def grade(chases):
    return {'pairs': 1, 'outside_pitches': 10, 'in_recommended': 4, 'usual_expected': 3.0, 'chases': chases,
            'chases_expected_league': 3.0, 'chases_expected_map': 2.5, 'bins': {}, 'by_group': {}}


def test_a_game_graded_twice_counts_once_on_its_last_date():
    rb = load()
    days = {'2025-04-05': {'games': {'778443': {'game_pk': 778443, 'grade': grade(2)}}},
            '2025-04-06': {'games': {'778443': {'game_pk': 778443, 'grade': grade(5)}, '9': {'game_pk': 9, 'grade': grade(1)}}}}
    rec = rb.record_from(days)
    assert rec['games'] == 2 and rec['chases'] == 6 and rec['duplicates_dropped'] == 1
