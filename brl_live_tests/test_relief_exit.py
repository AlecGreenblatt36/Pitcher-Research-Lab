"""Fitted reliever exits (research_lab.game_sim.relief_exit, RELIEF-02): the fit and the engine read a decision the same
way, nobody leaves mid-inning before his third batter, the engine with the fitted hazard uses fewer relievers for longer
outings than the hand-set rule, the switch off draws the same games, and production attaches it to a copy."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.manager import ManagerPolicy
from research_lab.game_sim.models import GameMatchup, GameState, PitcherLine, PitcherProfile, PlayerProfile, SimulationConfig, TeamProfile
from research_lab.game_sim.relief_exit import COMMON, MID, ReliefExit, features

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brl_relief_exit', ROOT / 'tools' / 'brl_relief_exit.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
DOC = json.loads((ROOT / 'brl_live' / 'relief_exit.json').read_text())


def test_fit_and_engine_read_a_decision_the_same_way():
    assert DOC['mid']['names'] == list(MID) and DOC['end']['names'] == list(COMMON)
    x = features(4, 4, 1, 9, 2, 0.8, 1.0, runners=2, outs=1)
    assert x['over'] == 0 and x['under'] == 0 and x['runs'] == 1 and x['closer_ninth'] == 1 and x['close'] == 1
    assert x['lead'] == 2 / 6 and x['runners'] == 2 and x['outs'] == 1 and x['third_batter'] == 0
    d = pd.DataFrame({'bf': [4], 'exp': [4], 'runs': [1], 'inning': [9], 'lead': [2], 'ninth': [0.8], 'late': [1.0],
                      'n_on': [2], 'n_outs_when_up': [1]})
    assert tool.design(d, MID)[0].tolist() == [x[n] for n in MID]


def line(bf, runs=0):
    ln = PitcherLine('R1', 'Reliever', 'home', False, 7, 'top')
    ln.batters_faced = bf; ln.batters_since_entry = bf; ln.runs_allowed = runs
    return ln


def test_three_batter_rule_and_inning_end():
    ex = ReliefExit(DOC); rng = np.random.default_rng(0)
    p = PitcherProfile('R1', 'Reliever', 'R', role='setup', expected_batters=4, usage={'ninth': 0.1, 'late': 0.8})
    st = GameState(inning=7, half='top', outs=1, home_score=3, away_score=2)
    assert not any(ex.remove(p, line(2), st, 'home', False, rng) for _ in range(500))
    mid = np.mean([ex.remove(p, line(4), st, 'home', False, rng) for _ in range(4000)])
    end = np.mean([ex.remove(p, line(4), st, 'home', True, rng) for _ in range(4000)])
    assert mid < 0.2 < 0.4 < end
    worse = np.mean([ex.remove(p, line(4, runs=3), st, 'home', False, rng) for _ in range(4000)])
    assert worse > mid + 0.05


class Provider:
    name = 'test_only'; validation_status = 'synthetic_test_only'

    def probabilities(self, c):
        return dict(zip(('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach'),
                        (.45, .22, .1, .14, .045, .035, .01)))


def matchup():
    def team(side):
        pen = tuple(PitcherProfile(f'{side}R{j}', f'{side} Relief {j}', 'L' if j % 3 == 0 else 'R',
                                   role='closer' if j == 0 else 'setup' if j < 3 else 'long' if j == 7 else 'reliever',
                                   leverage=0.9 if j < 3 else 0.5, expected_batters=8 if j == 7 else 4, max_batters=11 if j == 7 else 7)
                    for j in range(8))
        return TeamProfile(side, side, tuple(PlayerProfile(f'{side}{i}', f'{side} {i}') for i in range(9)),
                           PitcherProfile(side + 'SP', side + ' Starter', role='starter', expected_batters=20), pen)
    return GameMatchup(team('away'), team('home'))


def usage(policy, n=150):
    sim = GameSimulator(Provider(), SimulationConfig(max_innings=100, max_plate_appearances=4000), manager_policy=policy)
    used, outing = [], []
    for seed in range(n):
        r = sim.simulate(matchup(), seed)
        assert r.winner in ('away', 'home')
        for side, ids in r.pitcher_appearances.items():
            used.append(len(ids) - 1); outing += [r.pitcher_lines[i]['batters_faced'] for i in ids[1:]]
    return np.mean(used), np.mean(outing)


def test_engine_uses_fewer_relievers_for_longer_outings():
    hand = usage(ManagerPolicy())
    fitted = usage(ManagerPolicy(relief_exit=ReliefExit(DOC)))
    assert fitted[0] < hand[0] - 0.5 and fitted[1] > hand[1] + 0.8


def test_switch_off_and_production_copy():
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    for seed in range(4):
        a = GameSimulator(Provider(), config).simulate(matchup(), seed, record_events=True)
        b = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(relief_exit=None)).simulate(matchup(), seed, record_events=True)
        assert a.to_dict() == b.to_dict()
    from brl_live.boxscore import ADJUST, manager_for
    base = ManagerPolicy()
    m, label = manager_for(base, dict(ADJUST, relief_exit=True, reliever_choice=False))
    assert m is not base and isinstance(m.relief_exit, ReliefExit) and m.reliever_choice is None and base.relief_exit is None
    m, label = manager_for(base, dict(ADJUST, relief_exit=True, reliever_choice=True))
    assert m.relief_exit is not None and m.reliever_choice is not None and ';' in label
