"""Paired streams in the engine (GameSimulator(paired_streams=True)): every draw is tied to a team's plate appearance
number, so two simulators that differ in one part keep drawing the same numbers everywhere else. Off by default, the
engine draws exactly as before (the other engine tests hold that)."""
import json
from pathlib import Path

import numpy as np

from research_lab.game_sim.engine import GameSimulator, _PairedStreams, _Slot
from research_lab.game_sim.manager import ManagerPolicy
from research_lab.game_sim.models import GameMatchup, PitcherProfile, PlayerProfile, SimulationConfig, TeamProfile
from research_lab.game_sim.relief_exit import ReliefExit

ROOT = Path(__file__).resolve().parents[1]
EXIT = ReliefExit(json.loads((ROOT / 'brl_live' / 'relief_exit.json').read_text()))


class Provider:
    name = 'test_only'; validation_status = 'synthetic_test_only'

    def probabilities(self, c):
        k = 0.30 if c.pitcher.player_id.endswith('SP') else 0.20
        return dict(zip(('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach'),
                        (0.67 - k, k, .1, .14, .045, .035, .01)))


def matchup():
    def team(side):
        pen = tuple(PitcherProfile(f'{side}R{j}', f'{side} Relief {j}', 'L' if j % 3 == 0 else 'R',
                                   role='closer' if j == 0 else 'setup' if j < 3 else 'reliever', leverage=0.9 if j < 3 else 0.5,
                                   expected_batters=4, max_batters=7) for j in range(8))
        return TeamProfile(side, side, tuple(PlayerProfile(f'{side}{i}', f'{side} {i}') for i in range(9)),
                           PitcherProfile(side + 'SP', side + ' Starter', role='starter', expected_batters=20), pen)
    return GameMatchup(team('away'), team('home'))


def test_slots_are_fixed_by_team_plate_appearance_and_kind():
    a, b = _PairedStreams(7), _PairedStreams(7)
    b.slot('outcome', 'home', 200)                     # drawing far ahead first changes nothing
    for side in ('away', 'home'):
        for n in (0, 5, 70):
            for kind in _PairedStreams.KINDS:
                sa, sb = a.slot(kind, side, n), b.slot(kind, side, n)
                assert [sa.random() for _ in range(9)] == [sb.random() for _ in range(9)]   # past the fixed ones too
    s = _Slot(np.array([0.05, 0.5, 0.97]), [1, 2, 3, 4])
    assert s.choice(3, p=[0.1, 0.8, 0.1]) == 0 and s.choice(['x', 'y']) == 'y' and s.normal(0, 1) > 1.8


def test_paired_streams_repeat_and_keep_two_simulators_together():
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    hand = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(), paired_streams=True)
    fitted = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(relief_exit=EXIT), paired_streams=True)
    same_to_first_relief = []
    for seed in range(20):
        r1, r2 = hand.simulate(matchup(), seed, record_events=True), hand.simulate(matchup(), seed, record_events=True)
        assert r1.to_dict() == r2.to_dict()
        e1, e2 = r1.events, fitted.simulate(matchup(), seed, record_events=True).events
        # everything before the first reliever's first exit decision is the same game in both
        first = min((i for i, e in enumerate(e1) if not e['pitcher_id'].endswith('SP')), default=len(e1))
        same_to_first_relief.append(e1[:first] == e2[:first])
    assert all(same_to_first_relief)
