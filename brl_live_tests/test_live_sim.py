import numpy as np
import pytest
from brl_live.live_sim import LiveSimulator, LiveStart, PitcherSoFar
from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.models import PlayerProfile, PitcherProfile, TeamProfile, GameMatchup, SimulationConfig


class Provider:
    name = 'test_only'; validation_status = 'synthetic'
    def probabilities(self, c):
        return dict(zip(('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach'), (.42, .22, .09, .15, .06, .04, .02)))


def matchup():
    def team(side):
        return TeamProfile(side, side, tuple(PlayerProfile(f'{side}{i}', f'{side} Hitter {i}') for i in range(9)),
                           PitcherProfile(side + 'SP', side + ' Starter', role='starter', expected_batters=22, max_batters=28),
                           tuple(PitcherProfile(side + f'RP{j}', side + f' Relief {j}', expected_batters=4, max_batters=7) for j in range(5)))
    return GameMatchup(team('away'), team('home'))


def first_pitch_start():
    return LiveStart(inning=1, half='top', outs=0, bases=(None, None, None), away_score=0, home_score=0,
                     away_lineup_index=0, home_lineup_index=0, current_pitcher={'away': 'awaySP', 'home': 'homeSP'},
                     used_pitchers={'away': ['awaySP'], 'home': ['homeSP']})


@pytest.mark.parametrize('seed', range(8))
def test_starting_at_first_pitch_reproduces_the_engine_exactly(seed):
    m = matchup(); config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    a = GameSimulator(Provider(), config).simulate(m, seed, record_events=True)
    b = LiveSimulator(Provider(), config).simulate_from(m, seed, first_pitch_start(), record_events=True)
    assert a.to_dict() == b.to_dict()


def test_mid_game_state_is_honoured():
    m = matchup(); config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    start = LiveStart(inning=7, half='bottom', outs=1, bases=('home3', None, 'home1'), away_score=4, home_score=2,
                      away_lineup_index=5, home_lineup_index=4,
                      current_pitcher={'away': 'awayRP1', 'home': 'homeSP'},
                      used_pitchers={'away': ['awaySP', 'awayRP1'], 'home': ['homeSP']},
                      lines={'awaySP': PitcherSoFar('awaySP', batters_faced=24, outs_recorded=18, runs_allowed=2, strikeouts=5),
                             'awayRP1': PitcherSoFar('awayRP1', batters_faced=2, outs_recorded=1, entry_inning=7, entry_half='bottom', is_starter=False),
                             'homeSP': PitcherSoFar('homeSP', batters_faced=27, outs_recorded=21, runs_allowed=4)})
    sim = LiveSimulator(Provider(), config)
    wins = 0
    for seed in range(300):
        r = sim.simulate_from(m, seed, start)
        assert r.away_score >= 4 and r.home_score >= 2 and r.innings_played >= 7
        assert r.winner in ('away', 'home')
        assert r.pitcher_lines['awaySP']['batters_faced'] == 24 and r.pitcher_lines['awaySP']['exit_inning'] == 7
        assert r.pitcher_lines['homeSP']['batters_faced'] >= 27
        assert 'awaySP' in r.pitcher_appearances['away'] and 'awayRP1' in r.pitcher_appearances['away']
        wins += r.winner == 'home'
    # trailing by two with two on in the seventh: the home team wins sometimes, not usually
    assert 0.1 < wins / 300 < 0.5


def test_runs_already_scored_are_the_floor_and_a_walkoff_ends_it():
    m = matchup(); config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    start = LiveStart(inning=9, half='bottom', outs=2, bases=(None, None, None), away_score=3, home_score=3,
                      away_lineup_index=0, home_lineup_index=0, current_pitcher={'away': 'awayRP2', 'home': 'homeRP1'},
                      used_pitchers={'away': ['awaySP', 'awayRP2'], 'home': ['homeSP', 'homeRP1']})
    sim = LiveSimulator(Provider(), config)
    for seed in range(100):
        r = sim.simulate_from(m, seed, start)
        if r.innings_played == 9:
            assert r.winner == 'home' and r.home_score >= 4 and r.away_score == 3
        else:
            assert r.innings_played >= 10


def test_invalid_starts_are_rejected():
    m = matchup()
    bad = first_pitch_start(); bad.current_pitcher['away'] = 'nobody'
    with pytest.raises(ValueError):
        LiveSimulator(Provider()).simulate_from(m, 1, bad)
    bad = first_pitch_start(); bad.outs = 3
    with pytest.raises(ValueError):
        LiveSimulator(Provider()).simulate_from(m, 1, bad)
