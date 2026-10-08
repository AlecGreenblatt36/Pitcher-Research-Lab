"""Hitter-against-pitcher grid for a game: the model's chances for one plate appearance of every hitter in the lineup
against the opposing starter and the relievers most likely to pitch.

The probabilities come from the same adjusted provider the simulator uses for the game (PA model, context offsets,
run environment and the other enabled layers), in a fixed situation so the cells compare like with like: bases
empty, none out, score tied; the starter seeing the hitter the first time in the first inning, a reliever in the
seventh. Only model outputs and player names leave the runner.
"""
from __future__ import annotations

from research_lab.game_sim.models import PAContext

SIM_LABELS = ('bip_out', 'strikeout', 'bb_hbp', 'single', 'double_triple', 'home_run', 'other_reach')
RELIEVERS = 5


def _context(batter, pitcher, batting_side: str, spot: int, inning: int) -> PAContext:
    return PAContext(batter=batter, pitcher=pitcher, batting_side=batting_side, inning=inning,
                     half='top' if batting_side == 'away' else 'bottom', outs=0, bases=(None, None, None),
                     batting_score=0, fielding_score=0, lineup_position=spot, times_through_order=1,
                     pitcher_batters_faced=0, pitcher_runs_allowed=0, pitcher_fatigue=0.0, park_factor=1.0,
                     weather_run_factor=1.0, batting_team_baserunning=0.5, fielding_team_defense=0.0)


def matchup_grid(provider, matchup, box: dict | None = None, relievers: int = RELIEVERS) -> dict:
    """{'away': grid of the away hitters against the home staff, 'home': ...}; each grid lists batters, pitchers and
    p[batter][pitcher] = seven probabilities in SIM_LABELS order."""
    out = {'schema': 'brl.matchups.v1', 'labels': list(SIM_LABELS),
           'situation': 'bases empty, none out, score tied; starter in the 1st, relievers in the 7th, first time facing the hitter'}
    for bat_side in ('away', 'home'):
        fld_side = 'home' if bat_side == 'away' else 'away'
        batting, fielding = getattr(matchup, bat_side), getattr(matchup, fld_side)
        appear = {}
        for row in (((box or {}).get('teams') or {}).get(fld_side) or {}).get('pitching') or []:
            if row.get('role') != 'starter' and row.get('player_id') is not None:
                appear[str(row['player_id'])] = float(row.get('appearance_probability') or 0.0)
        pen = sorted((p for p in fielding.bullpen if str(p.player_id) != str(fielding.starter.player_id)),
                     key=lambda p: -appear.get(str(p.player_id), 0.0))[:relievers]
        pitchers = [(fielding.starter, 'starter', 1.0, 1)] + [(p, 'reliever', appear.get(str(p.player_id)), 7) for p in pen]
        grid = []
        for spot, b in enumerate(batting.lineup, start=1):
            row = []
            for p, _, _, inning in pitchers:
                probs = provider.probabilities(_context(b, p, bat_side, spot, inning))
                row.append([round(float(probs[k]), 4) for k in SIM_LABELS])
            grid.append(row)
        out[bat_side] = {
            'batters': [{'id': str(b.player_id), 'name': b.name, 'bats': b.bats, 'spot': i} for i, b in enumerate(batting.lineup, start=1)],
            'pitchers': [{'id': str(p.player_id), 'name': p.name, 'throws': p.throws, 'role': role,
                          'appear': None if a is None else round(float(a), 3)} for p, role, a, _ in pitchers],
            'p': grid}
    return out
