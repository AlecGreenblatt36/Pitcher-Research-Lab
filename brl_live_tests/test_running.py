import pandas as pd
import pytest

from brl_live.boxscore import BookkeepingFit, ObservedSimulator, build_game_box, steal_model_for
from brl_live.running import StealModel, steal_model
from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.models import GameMatchup, PitcherProfile, PlayerProfile, SimulationConfig, TeamProfile


class Provider:
    name = 'test_only'; validation_status = 'synthetic_test_only'

    def probabilities(self, c):
        return dict(zip(('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach'),
                        (.4, .2, .1, .16, .07, .05, .02)))


def history():
    return pd.DataFrame([{'date_key': '2026-10-01', 'outcome': o, 'pitcher': 1, 'stand': hand, 'terminal_event': event, 'pitch_number': pc}
                         for hand in ('R', 'L') for o, event, pc in [('BIP_OUT', 'field_out', 2), ('K', 'strikeout', 5), ('BB_HBP', 'walk', 6),
                                                                     ('BB_HBP', 'hit_by_pitch', 2), ('1B', 'single', 4), ('2B_3B', 'double', 3),
                                                                     ('HR', 'home_run', 5), ('OTHER_REACH', 'field_error', 2)]])


def doc(sprint=True, sb=30, cs=6):
    runners = {}
    for side in ('away', 'home'):
        for i in range(9):
            r = {'sb': sb if i < 3 else 1, 'cs': cs if i < 3 else 1, 'on1': 60, 'pa': 500}
            if sprint:
                r['sprint'] = 29.5 if i < 3 else 26.5
            runners[f'{side}{i}'] = r
    pitchers = {'awaySP': {'sb': 30, 'cs': 2, 'bf': 700}, 'homeSP': {'sb': 2, 'cs': 6, 'bf': 700}}
    return {'schema': 'brl.running.v1', 'seasons': {'2025': {'runners': runners, 'pitchers': pitchers, 'catchers': {}}}}


def matchup():
    def team(side):
        return TeamProfile(side, side, tuple(PlayerProfile(f'{side}{i}', f'{side} Hitter {i}') for i in range(9)),
                           PitcherProfile(side + 'SP', side + ' Starter', role='starter', expected_batters=18),
                           (PitcherProfile(side + 'RP', side + ' Relief'),))
    return GameMatchup(team('away'), team('home'))


CONFIG = SimulationConfig(max_innings=100, max_plate_appearances=4000)


def test_speeds_are_centered_on_the_league_mean():
    m = StealModel(doc(), 2025)
    assert m.sprint_mean == pytest.approx((6 * 29.5 + 12 * 26.5) / 18)
    assert m.speed('away0') == pytest.approx(0.5 + (29.5 - m.sprint_mean) / 7.5)
    assert m.speed('nobody') == 0.5
    sped = m.with_speeds(matchup())
    assert [p.speed for p in sped.away.lineup][:4] == [m.speed('away0')] * 3 + [m.speed('away3')]
    assert sped.away.starter == matchup().away.starter


def test_attempts_follow_the_runner_and_the_pitcher():
    m = StealModel(doc(), 2025)
    assert m.attempt('away0', 1) > m.attempt('away5', 1)
    assert m.attempt('away0', 2) == pytest.approx(m.attempt('away0', 1) * m.third)
    assert m.attempt('away0', 1, pitcher='awaySP') > m.attempt('away0', 1, pitcher='homeSP')
    assert m.success('away0', 1, pitcher='awaySP') > m.success('away0', 1, pitcher='homeSP')
    assert m.attempt('away0', 1, inning=8, margin=5) == 0.0


def test_no_attempts_means_the_same_games_as_without_the_model():
    """With no sprint speeds and an attempt scale of zero the engine draws exactly the same numbers."""
    quiet = StealModel(doc(sprint=False), 2025, per_pa=0.0)
    for seed in range(8):
        a = GameSimulator(Provider(), CONFIG).simulate(matchup(), seed, record_events=True)
        b = GameSimulator(Provider(), CONFIG, steals=quiet).simulate(matchup(), seed, record_events=True)
        assert a.to_dict() == b.to_dict()


def test_steals_happen_and_the_box_score_balances():
    model = StealModel(doc(), 2025, per_pa=3.0)
    seen = {'stolen_base': 0, 'caught_stealing': 0}
    fit = BookkeepingFit(history(), '2026-10-06')
    for seed in range(30):
        sim = ObservedSimulator(Provider(), CONFIG, steals=model)
        r = sim.simulate(matchup(), seed, record_events=True)
        for e in r.events:
            if e['outcome'] in seen:
                seen[e['outcome']] += 1
                assert e['runs_scored'] == 0
                assert e['outs_after'] - e['outs_before'] == (e['outcome'] == 'caught_stealing')
        box = build_game_box(r, matchup(), fit)
        for side in ('away', 'home'):
            other = 'home' if side == 'away' else 'away'
            assert sum(row['R'] for row in box['batting'][side]) == getattr(r, side + '_score')
            sb = sum(e['outcome'] == 'stolen_base' and e['batting_side'] == side for e in r.events)
            cs = sum(e['outcome'] == 'caught_stealing' and e['batting_side'] == side for e in r.events)
            assert sum(row['SB'] for row in box['batting'][side]) == sb
            assert sum(row['CS'] for row in box['batting'][side]) == cs
            assert sum(row['PA'] for row in box['batting'][side]) == sum(
                e['batting_side'] == side and e['outcome'] not in seen for e in r.events)
            outs = sum(e['outs_after'] - e['outs_before'] for e in r.events if e['batting_side'] == side)
            assert sum(row['outs'] for row in box['pitching'][other]) == outs
    assert seen['stolen_base'] > 10 and seen['caught_stealing'] > 0


def test_switch_and_season_rule():
    m = matchup()
    assert steal_model_for({'steals': False}, m, '2026-10-09') is None
    regular = steal_model_for({'steals': True}, m, '2026-06-01')
    assert regular.through_season == 2025          # the 2026 statistics were fetched after this date
    post = steal_model_for({'steals': True}, GameMatchup(m.away, m.home, game_type='D'), '2026-10-09')
    assert post.through_season == 2026
    assert steal_model(2026) is post
