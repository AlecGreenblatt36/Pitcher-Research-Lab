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
    def probabilities(self, candidates, inning, lead, outs, runners, batter_hand, upcoming_hands=None, shocks=None):
        self.seen = (inning, lead, outs, runners, batter_hand, upcoming_hands)
        return super().probabilities(candidates, inning, lead, outs, runners, batter_hand, upcoming_hands, shocks=shocks)


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
    m, label = manager_for(base, dict(ADJUST, reliever_choice=False, relief_exit=False))
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
    # BULLPEN-06: relief batters and appearances over 7 days, and his place in the pool by 14-day share (arm 2 has 16 of 30)
    assert u['bf7'] == 4 + 4 + 6 and u['apps7'] == 3 and u['rank14'] == 1.0
    assert bullpen_usage(app, 'AAA', '2026-06-30', [1, 2])[2]['rank14'] == 0.0
    # BULLPEN-10: starts over 30 days and days since a start count any team; new to the team = no outing for it in a year
    assert u['starts30'] == 1 and u['days_since_start'] == 6 and u['new_arm'] == 0.0
    u2 = bullpen_usage(app, 'AAA', '2026-06-30', [2])[2]; assert u2['starts30'] == 0 and u2['days_since_start'] == 99.0 and u2['new_arm'] == 0.0
    assert bullpen_usage(app, 'BBB', '2026-06-30', [2])[2]['new_arm'] == 1.0 and bullpen_usage(app, 'AAA', '2026-06-30', [77])[77]['new_arm'] == 1.0
    one = reliever_usage(app, 1, '2026-06-30')                # his latest team is AAA (the BBB outing is older)
    assert one['rank14'] == 0.0 and {k: v for k, v in one.items() if k != 'rank14'} == {k: v for k, v in u.items() if k != 'rank14'}
    from brl_live.live_feed import rank_bullpen
    from research_lab.game_sim.models import PitcherProfile
    pen = rank_bullpen((PitcherProfile('1', 'a', 'R', usage=one), PitcherProfile('2', 'b', 'R', usage=reliever_usage(app, 2, '2026-06-30')), PitcherProfile('3', 'c', 'R')))
    assert [dict(p.usage).get('rank14') for p in pen] == [1.0, 0.0, None]
    assert bullpen_usage(app, 'AAA', '2026-06-30', [99])[99]['days_since'] == 99.0


def test_usage_survives_the_bridge():
    m = matchup()
    back = decode_matchup(json.loads(json.dumps(asdict(m))))
    assert back.home.bullpen[0].usage == m.home.bullpen[0].usage and dict(back.home.bullpen[0].usage)['ninth'] == 0.9
    assert decode_matchup(asdict(m)) == m


def test_workload_and_top_two_terms():
    """BULLPEN-06: a heavy week and the pool's two most used arms, read only by fits that name them."""
    from research_lab.game_sim.reliever_choice import features, situation, RelieverChoice
    facts = [{'bf7': 45, 'apps7': 4, 'rank14': 0}, {'bf7': 6, 'apps7': 2, 'rank14': 1}, {'bf7': 0, 'apps7': 0, 'rank14': 5}, {}]
    c = features(facts, [True] * 4, situation(7, 1, 0, 0))
    assert list(c['work7']) == [3.0, 0.6, 0.0, 0.0] and list(c['apps7']) == [4, 2, 0, 0] and list(c['top2_14']) == [1, 1, 0, 0]
    from research_lab.game_sim.models import PitcherProfile
    pen = [PitcherProfile(str(i), str(i), 'R', usage={**f, 'share14': 0.1}) for i, f in enumerate(facts[:3])]
    old = RelieverChoice({'names': ['log_share14'], 'beta': [0.0]})
    assert np.allclose(old.probabilities(pen, 7, 1, 0, 0, 'R'), 1 / 3)            # an older fit ignores the new facts
    new = RelieverChoice({'names': ['work7', 'top2_14'], 'beta': [-1.0, -0.2]})
    p = new.probabilities(pen, 7, 1, 0, 0, 'R')
    assert p[2] > p[1] > p[0]


def test_game_shock_is_off_by_default_fixed_within_a_game_and_spreads_the_bullpen():
    """BULLPEN-07: sigma 0 draws exactly the games of before; with a shock each reliever's term is drawn once per game
    (the same at every entry of that game, different across games), the closer is picked in fewer worlds of a save spot,
    and the live kernel keys the choice by sigma."""
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    plain = RelieverChoice(DOC); shocked = RelieverChoice(DOC, game_sigma=1.25)
    assert plain.game_sigma == 0.0 and 'game shock' in shocked.name and 'game shock' not in plain.name
    for seed in range(5):
        a = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(reliever_choice=plain)).simulate(matchup(), seed, record_events=True)
        b = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(reliever_choice=RelieverChoice(DOC, game_sigma=0.0))).simulate(matchup(), seed, record_events=True)
        assert a.to_dict() == b.to_dict()
    p = pen(); state = GameState(seed=11)
    z1 = shocked.shocks(p, state); z2 = shocked.shocks(p, state); z3 = shocked.shocks(p, GameState(seed=12))
    assert z1 is not None and np.array_equal(z1, z2) and not np.array_equal(z1, z3) and set(state.reliever_shocks) == {x.player_id for x in p}
    assert plain.shocks(p, state) is None
    # over many games, the closer's share of save spots falls and the spread of picks rises
    picks = {0.0: Counter(), 1.25: Counter()}
    for sigma in picks:
        ch = RelieverChoice(DOC, game_sigma=sigma)
        for seed in range(400):
            st = GameState(seed=seed, inning=9, outs=0); st.home_score = 3; st.away_score = 1
            rng = np.random.default_rng(seed)
            picks[sigma][ch.select(list(p), st, 'home', PlayerProfile('b', 'B', 'R'), rng, [PlayerProfile('b', 'B', 'R')] * 3).player_id] += 1
    assert picks[1.25]['CL'] < picks[0.0]['CL'] and len(picks[1.25]) >= len(picks[0.0])
    from brl_live.boxscore import reliever_choice_for, ADJUST
    assert reliever_choice_for(dict(ADJUST, reliever_choice=True)).game_sigma == 0.0
    assert reliever_choice_for(dict(ADJUST, reliever_choice=True, reliever_game_sigma=1.25)).game_sigma == 1.25


def test_temperature_flattens_the_choice_and_is_off_at_one():
    """BULLPEN-09: at temperature 1 the choice is unchanged; above 1 the favorite's chance falls and the others' rise;
    the live kernel keys the choice by it."""
    p = pen()
    plain = RelieverChoice(DOC); warm = RelieverChoice(DOC, temperature=1.3)
    assert 'temperature' not in plain.name and 'temperature 1.3' in warm.name
    a = plain.probabilities(p, 9, 2, 0, 0, 'R', ['R', 'R', 'L']); b = warm.probabilities(p, 9, 2, 0, 0, 'R', ['R', 'R', 'L'])
    assert np.allclose(RelieverChoice(DOC, temperature=1.0).probabilities(p, 9, 2, 0, 0, 'R', ['R', 'R', 'L']), a)
    assert b[int(np.argmax(a))] < a.max() and abs(b.sum() - 1) < 1e-9 and (b[a < a.max()] > a[a < a.max()]).all()
    import pytest
    with pytest.raises(ValueError):
        RelieverChoice(DOC, temperature=-1.0)
    from brl_live.boxscore import reliever_choice_for, ADJUST
    assert reliever_choice_for(dict(ADJUST, reliever_choice=True)).temperature == 1.0
    assert reliever_choice_for(dict(ADJUST, reliever_choice=True, reliever_choice_temperature=1.3)).temperature == 1.3


def test_postseason_temperature_follows_the_game_type():
    """POST-05: reliever_choice_temperature_postseason sets the temperature for F, D, L and W games only; regular-season
    games and missing types keep the base temperature; manager_for hands the game type through."""
    from brl_live.boxscore import ADJUST, reliever_choice_for, choice_temperature_for, manager_for
    s = dict(ADJUST, reliever_choice=True, reliever_choice_temperature=1.0, reliever_choice_temperature_postseason={'F': 0.7, 'D': 0.7, 'L': 0.8})
    assert choice_temperature_for(s, 'R') == 1.0 and choice_temperature_for(s, 'L') == 0.8 and choice_temperature_for(s, 'W') == 1.0
    assert reliever_choice_for(s, 'D').temperature == 0.7 and reliever_choice_for(s, 'R').temperature == 1.0 and reliever_choice_for(s).temperature == 1.0
    assert reliever_choice_for(dict(ADJUST, reliever_choice=True), 'L').temperature == 1.0          # the live setting is inert
    from research_lab.game_sim.manager import ManagerPolicy
    m, label = manager_for(ManagerPolicy(), dict(s, relief_exit=False), ('TB', 'CLE'), 'L')
    assert m.reliever_choice.temperature == 0.8 and 'temperature 0.8' in label


def test_roster_pool_terms_are_read_only_by_fits_that_name_them():
    """BULLPEN-10: a starter in the pen and an arm new to the team get their own terms; older fits ignore them."""
    from research_lab.game_sim.reliever_choice import features, situation, RelieverChoice
    facts = [{'starts30': 3, 'days_since_start': 2}, {'starts30': 1, 'days_since_start': 12}, {'new_arm': 1.0}, {}]
    c = features(facts, [True] * 4, situation(3, -6, 0, 0))           # early, a blowout
    assert list(c['starter_recent']) == [1, 1, 0, 0] and list(c['started_within5']) == [1, 0, 0, 0] and list(c['new_arm']) == [0, 0, 1, 0]
    assert list(c['new_x_blowout']) == [0, 0, 1, 0] and list(c['new_x_early']) == [0, 0, 1, 0] and list(c['starter_x_early']) == [1, 1, 0, 0] and list(c['starter_x_blowout']) == [1, 1, 0, 0]
    c2 = features(facts, [True] * 4, situation(9, 1, 0, 0))
    assert list(c2['new_x_blowout']) == [0, 0, 0, 0] and list(c2['starter_x_early']) == [0, 0, 0, 0]
    from research_lab.game_sim.models import PitcherProfile
    pen = [PitcherProfile(str(i), str(i), 'R', usage={**f, 'share14': 0.1}) for i, f in enumerate(facts)]
    old = RelieverChoice({'names': ['log_share14'], 'beta': [0.0]})
    assert np.allclose(old.probabilities(pen, 3, -6, 0, 0, 'R'), 1 / 4)
    new = RelieverChoice({'names': ['started_within5', 'new_arm'], 'beta': [-3.0, -1.0]})
    p = new.probabilities(pen, 3, -6, 0, 0, 'R')
    assert p[0] < p[2] < p[3] and abs(p[1] - p[3]) < 1e-9
