"""Running plays between plate appearances other than steals (research_lab.game_sim.running_events, TRANS-02) and their
builder (tools/brl_running_events.py): patterns move the right runners, the engine scores them and ends games on a
walk-off, box scores still add up, and the switch off draws exactly the same games as before."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.models import BaseRunner, GameMatchup, PitcherProfile, PlayerProfile, SimulationConfig, TeamProfile
from research_lab.game_sim.running_events import RunningEvents

spec = importlib.util.spec_from_file_location('brl_running_events', Path(__file__).resolve().parents[1] / 'tools' / 'brl_running_events.py')
re_tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(re_tool)


def runner(pid):
    return BaseRunner(pid, 'Runner ' + pid, 0.5, 'P1')


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


def every_cell(p=0.3):
    """Every runner moves up one base (a wild pitch), or with a runner on first the runner is picked off."""
    cells = {}
    for mask in range(1, 8):
        for outs in range(3):
            up = ''.join('-' if not (mask >> s) & 1 else ('H' if s == 2 else str(s + 2)) for s in range(3))
            rows = [['wild_pitch', up, 0, 0.8]]
            if mask & 1:
                rows.append(['pickoff', 'X' + up[1:], 1, 0.2])
            cells[f'{mask}|{outs}'] = {'p': p, 'events': rows}
    return {'name': 'test running plays', 'cells': cells}


def test_patterns_move_runners_and_describe_the_play():
    bases = [runner('a'), None, runner('c')]
    new, scored, out = RunningEvents.apply(bases, '2-H')
    assert [b.player_id if b else None for b in new] == [None, 'a', None] and [r.player_id for r in scored] == ['c'] and out == []
    text, lead = RunningEvents.describe('wild_pitch', bases, '2-H')
    assert text == 'Wild pitch: Runner c scores; Runner a to second.' and lead.player_id == 'c'
    text, lead = RunningEvents.describe('pickoff', [runner('a'), None, None], 'X--')
    assert text == 'Runner a picked off first.' and lead.player_id == 'a'
    text, _ = RunningEvents.describe('defensive_indifference', [runner('a'), None, None], '2--')
    assert text == 'Runner a takes second on defensive indifference.'
    m = RunningEvents({'cells': {'1|0': {'p': 0.04, 'events': [['wild_pitch', '2--', 0, 1.0]]}}})
    assert m.rate(1, 0) == 0.04 and m.rate(2, 0) == 0.0
    rng = np.random.default_rng(0)
    draws = [m.draw([runner('a'), None, None], 0, rng) for _ in range(20000)]
    assert abs(np.mean([d is not None for d in draws]) - 0.04) < 0.006 and m.draw([None, None, None], 0, rng) is None


def test_switch_off_draws_the_same_games():
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    for seed in range(5):
        a = GameSimulator(Provider(), config).simulate(matchup(), seed, record_events=True)
        b = GameSimulator(Provider(), config, running_events=None).simulate(matchup(), seed, record_events=True)
        assert a.to_dict() == b.to_dict()


def test_games_with_running_plays_stay_legal_and_score_them():
    events = RunningEvents(every_cell())
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    kinds, runs_on_plays = set(), 0
    for seed in range(30):
        r = GameSimulator(Provider(), config, running_events=events).simulate(matchup(), seed, record_events=True)
        assert r.winner in ('away', 'home')
        last = {'away': 0, 'home': 0}
        for e in r.events:
            assert 0 <= e['outs_before'] <= e['outs_after'] <= 3
            if e['outcome'] in ('wild_pitch', 'pickoff'):
                kinds.add(e['outcome']); runs_on_plays += e['runs_scored']
                assert e['outs_after'] - e['outs_before'] == (1 if e['outcome'] == 'pickoff' else 0)
            assert e['away_score'] >= last['away'] and e['home_score'] >= last['home']
            last = {'away': e['away_score'], 'home': e['home_score']}
        assert (r.away_score, r.home_score) == (last['away'], last['home'])
    assert kinds == {'wild_pitch', 'pickoff'} and runs_on_plays > 0


def test_a_running_play_can_end_the_game_on_a_walk_off():
    events = RunningEvents({'cells': {f'{m}|{o}': {'p': 0.5, 'events': [['wild_pitch', ''.join('-' if not (m >> s) & 1 else ('H' if s == 2 else str(s + 2)) for s in range(3)), 0, 1.0]]}
                                      for m in range(1, 8) for o in range(3)}})
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    walk_offs = 0
    for seed in range(200):
        r = GameSimulator(Provider(), config, running_events=events).simulate(matchup(), seed, record_events=True)
        e = r.events[-1]
        if e['outcome'] == 'wild_pitch' and e['half'] == 'bottom' and e['inning'] >= 9:
            walk_offs += 1
            assert r.home_score == r.away_score + 1 and r.winner == 'home'
    assert walk_offs > 0


@pytest.mark.parametrize('seed', range(6))
def test_box_scores_add_up_with_running_plays(seed):
    from brl_live.boxscore import BookkeepingFit, ObservedSimulator, build_game_box, running_events_for, transitions_for
    from brl_live_tests.test_boxscore import history
    production = running_events_for({'running_events': True})
    assert production is not None and len(production.cells) == 21 and running_events_for({'running_events': False}) is None
    for events in (production, RunningEvents(every_cell(0.4))):
        m = matchup()
        sim = ObservedSimulator(Provider(), SimulationConfig(max_innings=100, max_plate_appearances=4000),
                                transitions=transitions_for({'transitions': True}), running_events=events)
        r = sim.simulate(m, seed, record_events=True)
        box = build_game_box(r, m, BookkeepingFit(history(), '2026-10-06'))
        for side in ('away', 'home'):
            assert sum(x['R'] for x in box['batting'][side]) == getattr(r, side + '_score')
            assert sum(int(v['R']) for v in box['innings'][side].values()) == getattr(r, side + '_score')
            assert sum(x['PA'] for x in box['batting'][side]) == sum(x['AB'] + x['BB'] + x['HBP'] + x['SF'] for x in box['batting'][side])


def test_builder_keeps_non_steal_plays_with_legal_patterns():
    pre = {'1|0': {'none>1--|0|P': 900, 'wild_pitch>2--|0|P': 40, 'stolen_base_2b>2--|0|P': 50, 'pickoff_1b>X--|1|T': 8,
                   'balk>2--|0|P': 2, 'wild_pitch>1--|0|P': 1},
           '5|1': {'none>1-3|0|P': 500, 'wild_pitch>2-H|0|P': 10, 'defensive_indiff>2-3|0|P': 5, 'wild_pitch>3-2|0|P': 1}}
    doc = re_tool.build([{'games': 10, 'pre': pre}])
    c = doc['cells']['1|0']
    assert c['n'] == 1001 and c['p'] == pytest.approx(50 / 1001, abs=1e-6)
    assert [e[:3] for e in c['events']] == [['wild_pitch', '2--', 0], ['pickoff', 'X--', 1], ['balk', '2--', 0]]
    assert doc['dropped'] == {'no movement': 1, 'illegal': 1} and doc['steal_plays_left_to_the_steal_step'] == 50
    assert [e[:3] for e in doc['cells']['5|1']['events']] == [['wild_pitch', '2-H', 0], ['defensive_indifference', '2-3', 0]]
    assert re_tool.legal('23H', 7, 0) and not re_tool.legal('32-', 3, 0) and not re_tool.legal('X--', 1, 0)


def test_production_table_rates_are_plausible():
    doc = json.loads((Path(__file__).resolve().parents[1] / 'brl_live' / 'running_events.json').read_text())
    per = doc['per_team_game']
    assert 0.2 < per['wild_pitch'] < 0.35 and 0.02 < per['passed_ball'] < 0.07 and 0.02 < per['balk'] < 0.06
    assert 0.35 < sum(per.values()) < 0.6
