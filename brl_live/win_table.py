"""Win chance by game state, from the same simulated games as the box forecast.

Every pregame forecast simulates the game ten thousand times. Each plate appearance of each
simulated game starts in a state (inning, half, outs, runners, home lead), and the share of
those games the home team went on to win is this game's own win chance for that state: it
knows the two lineups, the starters and the bullpens, which a league-wide table would not.
The page reads the table to chart win chance through the real game, play by play, and the
in-game snapshots (full continuations from the current state) remain the exact numbers.

Sparse states are shrunk toward pooled ones: the same inning, half and lead over all outs
and runners, then the lead alone, then the pregame win chance.
"""
from __future__ import annotations

import numpy as np

SHAPE = (9, 2, 3, 8, 13)          # inning (9 is the ninth and later), half, outs, runners, home lead -6..6
DIFF_MIN = -6
K_LEAD, K_INNING, K_STATE = 25.0, 15.0, 10.0
SCHEMA = 'brl.win-table.v1'


def state_index(inning: int, half: str, outs: int, bases, home_lead: int) -> tuple[int, int, int, int, int]:
    """Array index of a state; runners are the three base occupancy flags (first base is bit 1)."""
    i = min(max(int(inning), 1), 9) - 1
    h = 0 if str(half).lower().startswith('top') else 1
    o = min(max(int(outs), 0), 2)
    b = sum(1 << k for k, r in enumerate(tuple(bases or ())[:3]) if r)
    d = min(max(int(home_lead), DIFF_MIN), -DIFF_MIN) - DIFF_MIN
    return i, h, o, b, d


class WinTable:
    def __init__(self):
        self.wins = np.zeros(SHAPE)
        self.n = np.zeros(SHAPE)
        self.games = 0
        self.home_wins = 0

    def add(self, box: dict) -> None:
        """Count every plate appearance's starting state of one simulated game and who won it."""
        home_won = box['score']['home'] > box['score']['away']
        self.games += 1
        self.home_wins += int(home_won)
        for p in box.get('plays') or ():
            runs = int(p.get('runs_scored') or 0)
            lead_after = int(p['home_score']) - int(p['away_score'])
            lead = lead_after - runs if str(p['half']).lower() != 'top' else lead_after + runs
            key = state_index(p['inning'], p['half'], p['outs_before'], p.get('bases_before'), lead)
            self.n[key] += 1
            self.wins[key] += home_won

    def table(self) -> np.ndarray:
        pregame = (self.home_wins + 0.5) / (self.games + 1.0)
        by_lead = (self.wins.sum(axis=(0, 1, 2, 3)) + K_LEAD * pregame) / (self.n.sum(axis=(0, 1, 2, 3)) + K_LEAD)
        by_inning = (self.wins.sum(axis=(2, 3)) + K_LEAD * by_lead[None, None, :]) / (self.n.sum(axis=(2, 3)) + K_LEAD)
        by_outs = (self.wins.sum(axis=3) + K_INNING * by_inning[:, :, None, :]) / (self.n.sum(axis=3) + K_INNING)
        return (self.wins + K_STATE * by_outs[:, :, :, None, :]) / (self.n + K_STATE)

    def finish(self) -> dict:
        if self.games == 0:
            raise ValueError('No simulated games')
        cells = np.rint(self.table() * 100.0).astype(int)
        return {'schema': SCHEMA, 'shape': list(SHAPE), 'lead_min': DIFF_MIN,
                'layout': ['inning 1 to 9 (9 is the ninth and later)', 'half (top, bottom)', 'outs before the plate appearance',
                           'runners before it (first base 1, second 2, third 4, added)', 'home lead before it, -6 to 6'],
                'cells': cells.ravel().tolist(), 'n_games': int(self.games), 'pregame_home': round(self.home_wins / self.games, 4),
                'states_reached': int((self.n > 0).sum()),
                'note': 'Home win share (percent) of the simulated games that reached each state before a plate appearance, '
                        'shrunk toward pooled states. The page charts the real game through it; in-game snapshots stay exact.'}


def lookup(table: dict, inning: int, half: str, outs: int, bases, home_lead: int) -> float:
    """Home win chance (0 to 1) of a state from a finished table (the page's lookup, for tests and tools)."""
    i, h, o, b, d = state_index(inning, half, outs, bases, home_lead)
    shape = table['shape']
    flat = ((((i * shape[1]) + h) * shape[2] + o) * shape[3] + b) * shape[4] + d
    return table['cells'][flat] / 100.0
