"""The fitted reliever choice (research_lab.game_sim.reliever_choice, BULLPEN-01): the fit and the simulator read a
reliever the same way, the production table follows roles, rest and the platoon, a pitching change at an inning's end is
scored for the next inning, games with it stay legal, the switch off draws the same games as before, usage facts come
from the right team and window and survive the matchup's trip through the bridge."""
import importlib.util
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.engine_bridge import decode_matchup
from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.manager import ManagerPolicy
from research_lab.game_sim.models import BaseRunner, GameMatchup, GameState, PitcherProfile, PlayerProfile, SimulationConfig, TeamProfile
from research_lab.game_sim.reliever_choice import DEFAULT, RelieverChoice, features, situation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brl_reliever_choice', ROOT / 'tools' / 'brl_reliever_choice.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
DOC = json.loads((ROOT / 'brl_live' / 'reliever_choice.json').read_text())


def facts(**kw):
    out = dict(DEFAULT); out.update(kw); return out


CLOSER = facts(share14=0.14, share30=0.13, days_since=2, ninth=0.9, late=1.0, med_bf=4, k365=85, bf365=260, apps21=7, ninth21=1.0, late21=1.0)
SETUP = facts(share14=0.13, share30=0.13, days_since=2, ninth=0.1, late=0.8, med_bf=4, k365=70, bf365=250, apps21=7, ninth21=0.1, late21=0.9)
LONG = facts(share14=0.10, share30=0.11, days_since=4, ninth=0.0, late=0.05, med_bf=9, k365=45, bf365=300, apps21=3, late21=0.0)
LEFTY = facts(share14=0.07, share30=0.07, days_since=2, ninth=0.0, late=0.4, med_bf=3, k365=55, bf365=190, apps21=8, late21=0.4)
MIDDLE = facts(share14=0.09, share30=0.09, days_since=3, ninth=0.0, late=0.2, med_bf=5, k365=50, bf365=220, apps21=5, late21=0.2)


def pen():
    return (PitcherProfile('CL', 'Closer', 'R', usage=CLOSER), PitcherProfile('SU', 'Setup', 'R', usage=SETUP),
            PitcherProfile('LG', 'Long man', 'R', usage=LONG), PitcherProfile('LH', 'Lefty', 'L', usage=LEFTY),
            PitcherProfile('MD', 'Middle', 'R', usage=MIDDLE))


def test_fit_and_simulator_read_a_reliever_the_same_way():
    cols = features([CLOSER, LONG], [True, False], situation(9, 2, 0, 0), [2.0, 0.0])
    assert set(tool.NAMES) <= set(cols) and DOC['names'] == tool.NAMES and len(DOC['beta']) == len(DOC['names'])
    assert np.allclose(cols['log_share14'], np.log(np.array([0.14, 0.10]) + 0.01))
    q = (np.array([85.0, 45.0]) + 100 * 0.235) / (np.array([260.0, 300.0]) + 100)
    assert np.allclose(cols['quality'], (q - q.mean()) / 0.03)
    assert cols['ninth_x_save'].tolist() == [0.9, 0.0] and cols['same3'].tolist() == [2.0, 0.0]
    assert cols['same3_x_late'].tolist() == [2.0, 0.0] and cols['same3_x_mid'].tolist() == [0.0, 0.0]
    s = situation(7, -1, 1, 2)
    assert s['late_close'] == 1 and s['save'] == 0 and s['mid_inning'] == 1 and s['before8'] == 1


def test_production_table_follows_roles_rest_and_platoon():
    choice = RelieverChoice(DOC)
    p = pen(); ids = [x.player_id for x in p]
    save = choice.probabilities(p, 9, 2, 0, 0, 'R', ['R', 'R', 'L'])
    assert abs(save.sum() - 1) < 1e-9 and ids[int(np.argmax(save))] == 'CL' and save[0] > 0.5
    blowout = choice.probabilities(p, 6, 7, 0, 0, 'R', ['R', 'R', 'R'])
    assert blowout[ids.index('LG')] > blowout[ids.index('CL')]
    tired = list(p); tired[0] = PitcherProfile('CL', 'Closer', 'R', usage=dict(CLOSER, pitched_d1=1, apps_d2=2, apps_d3=2, days_since=1))
    assert choice.probabilities(tired, 9, 2, 0, 0, 'R', ['R', 'R', 'L'])[0] < save[0] / 2
    lefties = choice.probabilities(p, 7, 1, 1, 1, 'L', ['L', 'L', 'L'])
    righties = choice.probabilities(p, 7, 1, 1, 1, 'R', ['R', 'R', 'R'])
    assert lefties[ids.index('LH')] > 1.5 * righties[ids.index('LH')]


class Capture(RelieverChoice):
    def probabilities(self, candidates, inning, lead, outs, runners, batter_hand, upcoming_hands=None):
        self.seen = (inning, lead, outs, runners, batter_hand, upcoming_hands)
        return super().probabilities(candidates, inning, lead, outs, runners, batter_hand, upcoming_hands)


def test_a_change_at_the_inning_end_is_scored_for_the_next_inning():
    choice = Capture(DOC); rng = np.random.default_rng(0)
    state = GameState(inning=8, half='bottom', outs=3, away_score=3, home_score=1)
    hitters = [PlayerProfile('h1', 'One', 'L'), PlayerProfile('h2', 'Two', 'S'), PlayerProfile('h3', 'Three', 'R')]
    choice.select(list(pen()), state, 'away', hitters[0], rng, hitters)
    assert choice.seen == (9, 2, 0, 0, 'L', ['L', 'S', 'R'])
    state = GameState(inning=7, half='top', outs=1, away_score=0, home_score=0)
    state.bases = [None, BaseRunner('r', 'Runner', 0.5, 'P1'), None]
    choice.select(list(pen()), state, 'home', hitters[2], rng, None)
    assert choice.seen == (7, 0, 1, 1, 'R', None)


class Provider:
    name = 'test_only'; validation_status = 'synthetic_test_only'

    def probabilities(self, c):
        return dict(zip(('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach'),
                        (.4, .2, .1, .16, .07, .05, .02)))


def matchup():
    def team(side):
        return TeamProfile(side, side, tuple(PlayerProfile(f'{side}{i}', f'{side} Hitter {i}', 'L' if i % 3 == 0 else 'R') for i in range(9)),
                           PitcherProfile(side + 'SP', side + ' Starter', role='starter', expected_batters=18),
                           tuple(PitcherProfile(side + p.player_id, p.name, p.throws, role='reliever', expected_batters=5, usage=p.usage) for p in pen()))
    return GameMatchup(team('away'), team('home'))


def test_games_with_the_fitted_choice_stay_legal():
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    sim = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(reliever_choice=RelieverChoice(DOC)))
    entered = Counter()
    for seed in range(40):
        r = sim.simulate(matchup(), seed, record_events=True)
        assert r.winner in ('away', 'home') and not r.ended_by_plate_appearance_cap
        for side, ids in r.pitcher_appearances.items():
            assert len(ids) == len(set(ids)) and ids[0] == side + 'SP'
            assert all(i[len(side):] in {'CL', 'SU', 'LG', 'LH', 'MD'} for i in ids[1:])
            entered.update(i[len(side):] for i in ids[1:])
    assert sum(entered.values()) > 40 and len(entered) >= 3


def test_switch_off_draws_the_same_games():
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    for seed in range(5):
        a = GameSimulator(Provider(), config).simulate(matchup(), seed, record_events=True)
        b = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(reliever_choice=None)).simulate(matchup(), seed, record_events=True)
        assert a.to_dict() == b.to_dict()


def test_manager_for_attaches_a_copy():
    from brl_live.boxscore import manager_for, ADJUST
    base = ManagerPolicy()
    m, label = manager_for(base, dict(ADJUST, reliever_choice=True))
    assert m is not base and isinstance(m.reliever_choice, RelieverChoice) and base.reliever_choice is None and label
    m, label = manager_for(base, dict(ADJUST, reliever_choice=False))
    assert m is base and label is None


def test_usage_comes_from_the_team_and_window():
    from brl_live.live_feed import bullpen_usage, reliever_usage
    rows = []
    def add(days, team='AAA', start=False, inning=7, bf=4, k=1, pid=1):
        rows.append({'game_pk': len(rows), 'pitcher': pid, 'date': (pd.Timestamp('2026-06-30') - pd.Timedelta(days=days)).strftime('%Y-%m-%d'),
                     'team': team, 'bf': bf, 'entry_inning': inning, 'throws': 'R', 'start': start, 'finished': False, 'k': k})
    add(1, inning=9); add(2, inning=9); add(5, inning=8, bf=6, k=3); add(20, inning=7); add(40, inning=6, bf=8)
    add(3, team='BBB', inning=9, bf=10, k=5)          # another team's outing
    add(6, start=True, bf=25, k=8)                     # a start
    add(1, pid=2, bf=4, k=0); add(10, pid=2, bf=12, k=2)
    add(400, inning=9, bf=30, k=12)                    # older than a year
    app = pd.DataFrame(rows)
    u = bullpen_usage(app, 'AAA', '2026-06-30', [1, 2])[1]
    assert u['share14'] == (4 + 4 + 6) / (4 + 4 + 6 + 4 + 12) and u['share30'] == (4 + 4 + 6 + 4) / (4 + 4 + 6 + 4 + 4 + 12)
    assert u['pitched_d1'] == 1 and u['apps_d2'] == 2 and u['apps_d3'] == 2 and u['days_since'] == 1
    assert u['ninth'] == 2 / 5 and u['late'] == 3 / 5 and u['med_bf'] == 4 and u['k365'] == 7 and u['bf365'] == 26
    assert u['apps21'] == 4 and u['ninth21'] == 2 / 4 and u['late21'] == 3 / 4
    assert reliever_usage(app, 1, '2026-06-30') == u          # his latest team is AAA (the BBB outing is older)
    assert bullpen_usage(app, 'AAA', '2026-06-30', [99])[99]['days_since'] == 99.0


def test_usage_survives_the_bridge():
    m = matchup()
    back = decode_matchup(json.loads(json.dumps(asdict(m))))
    assert back.home.bullpen[0].usage == m.home.bullpen[0].usage and dict(back.home.bullpen[0].usage)['ninth'] == 0.9
    assert decode_matchup(asdict(m)) == m
