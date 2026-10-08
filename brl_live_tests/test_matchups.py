"""The hitter-against-pitcher grid: every lineup spot against the starter and the likeliest relievers, in a fixed
situation, through whatever provider the game uses."""
from research_lab.game_sim.models import GameMatchup, PitcherProfile, PlayerProfile, TeamProfile

from brl_live.matchups import matchup_grid, SIM_LABELS


def team(prefix, base):
    lineup = tuple(PlayerProfile(str(base + i), f'{prefix} hitter {i}', 'R' if i % 2 else 'L') for i in range(9))
    starter = PitcherProfile(str(base + 50), f'{prefix} starter', 'R', role='starter')
    pen = tuple(PitcherProfile(str(base + 60 + j), f'{prefix} reliever {j}', 'L' if j % 2 else 'R', role='reliever') for j in range(7))
    return TeamProfile(prefix, prefix, lineup, starter, pen)


class Provider:
    def __init__(self):
        self.calls = []

    def probabilities(self, ctx):
        self.calls.append(ctx)
        k = 0.30 if ctx.pitcher.role == 'starter' else 0.25
        return dict(zip(SIM_LABELS, [0.45, k, 0.09, 0.14, 0.05, 0.03, 1 - 0.45 - k - 0.09 - 0.14 - 0.05 - 0.03]))


def test_grid_covers_every_hitter_against_starter_and_top_relievers():
    m = GameMatchup(away=team('A', 1000), home=team('H', 2000), venue='NYY')
    box = {'teams': {'home': {'pitching': [{'player_id': '2050', 'role': 'starter'}] + [{'player_id': str(2060 + j), 'role': 'reliever', 'appearance_probability': j / 10} for j in range(7)]},
                     'away': {'pitching': []}}}
    prov = Provider()
    out = matchup_grid(prov, m, box, relievers=3)
    away = out['away']
    assert [p['name'] for p in away['pitchers']] == ['H starter', 'H reliever 6', 'H reliever 5', 'H reliever 4']   # likeliest relievers first
    assert len(away['batters']) == 9 and all(len(r) == 4 and len(r[0]) == 7 for r in away['p'])
    assert away['p'][0][0][1] == 0.30 and away['p'][0][1][1] == 0.25
    ctxs = [c for c in prov.calls if c.batting_side == 'away']
    assert all(c.outs == 0 and c.bases == (None, None, None) and c.score_diff == 0 and c.times_through_order == 1 for c in ctxs)
    assert {c.inning for c in ctxs if c.pitcher.role == 'starter'} == {1} and {c.inning for c in ctxs if c.pitcher.role != 'starter'} == {7}
    assert len(out['home']['pitchers']) == 4 and out['home']['batters'][0]['name'] == 'H hitter 0'
