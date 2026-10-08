"""The engine's empirical base-running kernel (research_lab.game_sim.transitions.EmpiricalKernel) and its builder
(tools/brl_transition_kernel.py): patterns move the right runners, missing cells keep the hand-set rule, speed tilts
move the advance decisions, and whole games stay legal."""
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.models import BaseRunner, GameMatchup, PitcherProfile, PlayerProfile, SimulationConfig, TeamProfile
from research_lab.game_sim.transitions import EmpiricalKernel, apply_outcome

spec = importlib.util.spec_from_file_location('brl_transition_kernel', Path(__file__).resolve().parents[1] / 'tools' / 'brl_transition_kernel.py')
tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)


def runner(pid, speed=0.5):
    return BaseRunner(pid, pid, speed, 'P1')


def test_a_pattern_moves_each_runner_and_scores_in_order():
    k = EmpiricalKernel({'cells': {'single|7|1': [['3HH1', 1.0]], 'bip_out|5|0': [['X-HX', 1.0]]}})
    bases = [runner('r1'), runner('r2'), runner('r3')]
    t = k.apply('single', bases, PlayerProfile('b', 'Batter'), 'P1', 1, 0.0, 0.0, np.random.default_rng(0))
    assert [b.player_id if b else None for b in t.bases] == ['b', None, 'r1'] and [r.player_id for r in t.scored_runners] == ['r3', 'r2']
    assert t.outs_added == 0
    t = k.apply('bip_out', [runner('r1'), None, runner('r3')], PlayerProfile('b', 'Batter'), 'P1', 0, 0.0, 0.0, np.random.default_rng(0))
    assert t.bases == [None, None, None] and t.outs_added == 2 and t.retired_runner_ids == ['r1', 'b'] and [r.player_id for r in t.scored_runners] == ['r3']
    assert 'double play' in t.description


def test_missing_cells_and_home_runs_keep_the_hand_set_rule():
    k = EmpiricalKernel({'cells': {'home_run|0|0': [['---X', 1.0]]}})
    for outcome, bases in (('single', [runner('r1'), None, None]), ('home_run', [None, None, None])):
        a = k.apply(outcome, list(bases), PlayerProfile('b', 'Batter'), 'P1', 0, 0.0, 0.0, np.random.default_rng(3))
        b = apply_outcome(outcome, list(bases), PlayerProfile('b', 'Batter'), 'P1', 0, 0.0, 0.0, np.random.default_rng(3))
        assert ([x.player_id if x else None for x in a.bases], a.outs_added, a.runs_scored) == ([x.player_id if x else None for x in b.bases], b.outs_added, b.runs_scored)


def test_speed_tilts_the_advance_decision():
    doc = {'cells': {'single|2|1': [['-H-1', 0.5], ['-3-1', 0.5]]}, 'tilts': {'single_from_2nd': {'beta': 4.0, 'center': 0.5}}}
    k = EmpiricalKernel(doc)
    rng = np.random.default_rng(1)
    share = lambda s: np.mean([k.apply('single', [None, runner('r2', s), None], PlayerProfile('b', 'B'), 'P1', 1, 0.0, 0.0, rng).runs_scored for _ in range(4000)])
    fast, avg, slow = share(0.9), share(0.5), share(0.1)
    assert slow < avg < fast and abs(avg - 0.5) < 0.03
    assert abs(fast - 1 / (1 + np.exp(-4.0 * 0.4))) < 0.03


def test_double_play_tilt_reads_the_runner_on_first_and_the_batter():
    doc = {'cells': {'bip_out|1|0': [['X--X', 0.2], ['---X', 0.5], ['2--X', 0.3]]},
           'tilts': {'double_play_runner': {'beta': -3.0, 'center': 0.5}, 'double_play_batter': {'beta': -3.0, 'center': 0.5}}}
    k = EmpiricalKernel(doc)
    rng = np.random.default_rng(2)
    dp = lambda s1, sb: np.mean([k.apply('bip_out', [runner('r1', s1), None, None], PlayerProfile('b', 'B', speed=sb), 'P1', 0, 0.0, 0.0, rng).outs_added == 2 for _ in range(4000)])
    assert dp(0.9, 0.9) < dp(0.5, 0.5) < dp(0.1, 0.1)


def test_builder_keeps_legal_patterns_and_fits_tilts():
    contact = {'1B|2|1': {'-H-1|0': 300, '-3-1|0': 200, '-1-1|0': 5},      # a runner moving back to first is illegal
               '1B|0|0': {'---1|0': 10}}                                     # too small a cell
    decisions = {'single_from_2nd|1|1': {'H': 30, '3': 70}, 'single_from_2nd|1|4': {'H': 70, '3': 30}, 'single_from_2nd|1|na': {'H': 5}}
    k = tk.build([{'contact': contact, 'decisions': decisions}])
    assert list(k['cells']) == ['single|2|1'] and k['dropped']['illegal'] == 5 and k['plate_appearances_in_small_cells'] == 10
    assert [r[:2] for r in k['cells']['single|2|1']] == [['-H-1', 0.6], ['-3-1', 0.4]]
    t = k['tilts']['single_from_2nd']
    assert t['n'] == 200 and t['center'] == pytest.approx(0.5) and t['beta'] == pytest.approx(np.log(7 / 3 / (3 / 7)) / 0.3, rel=1e-3)
    assert tk.legal('X-HX', 5, 0) and not tk.legal('X-HX', 5, 2) and not tk.legal('11-X', 3, 0) and not tk.legal('---?', 0, 0)


class Provider:
    name = 'test_only'; validation_status = 'synthetic_test_only'

    def probabilities(self, c):
        return dict(zip(('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach'),
                        (.4, .2, .1, .16, .07, .05, .02)))


def matchup():
    def team(side):
        return TeamProfile(side, side, tuple(PlayerProfile(f'{side}{i}', f'{side} Hitter {i}') for i in range(9)),
                           PitcherProfile(side + 'SP', side + ' Starter', role='starter', expected_batters=18),
                           (PitcherProfile(side + 'RP', side + ' Relief'),))
    return GameMatchup(team('away'), team('home'))


def test_whole_games_stay_legal_with_the_kernel():
    cells = {}
    for mask in range(8):
        for outs in range(3):
            # an out with every runner holding, and a double play when first is occupied with fewer than two out
            hold = ''.join(str(b) if (mask >> (b - 1)) & 1 else '-' for b in (1, 2, 3)) + 'X'
            rows = [[hold, 0.8]]
            if mask & 1 and outs < 2:
                rows.append(['X' + hold[1:3] + 'X', 0.2])
            cells[f'bip_out|{mask}|{outs}'] = rows
    k = EmpiricalKernel({'cells': cells, 'tilts': {'double_play_runner': {'beta': -1.0, 'center': 0.5}}})
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    dps = 0
    for seed in range(20):
        r = GameSimulator(Provider(), config, transitions=k).simulate(matchup(), seed, record_events=True)
        assert r.winner in ('away', 'home')
        for e in r.events:
            assert 0 <= e['outs_before'] <= e['outs_after'] <= 3
            dps += e['outs_after'] - e['outs_before'] == 2
    assert dps > 0
