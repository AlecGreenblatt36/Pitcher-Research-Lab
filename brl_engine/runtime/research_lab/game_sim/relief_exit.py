"""When a reliever comes out (RELIEF-02): logistic hazards fitted on every real relief decision point, one for the
middle of an inning and one for an inning's end.

The engine's hand-set rule (manager.ManagerPolicy.should_remove for relievers) pulls a reliever in the middle of an
inning at about 57% of the decision points where it may (three batters faced or more) against about 9% for real
managers, so the simulator used 4.6 relievers a game for 3.5 batters each against 3.3 and 4.9 in real games, reaching
deep into the bullpen. After each plate appearance a reliever completes, the hazard uses the batters he has faced, the
runs charged to him, his usual length (median batters per relief outing over the last year), his role (share of entries
in the ninth and in the eighth or later), the inning, the fielding team's lead and, in the middle of an inning, the
runners on base and outs. The three-batter rule stands: in the middle of an inning nobody leaves before his third batter.
Teams differ reliably in how fast they pull relievers (RELIEF-03): an optional per-team offset moves the log-odds.

doc (brl_live/relief_exit.json, from tools/brl_relief_exit.py): {'name': ..., 'mid': {'names', 'intercept', 'beta'},
'end': {'names', 'intercept', 'beta'}}.
"""
from __future__ import annotations

import math

COMMON = ('bf', 'bf2', 'over', 'under', 'runs', 'runs2', 'exp', 'late', 'ninth_inning', 'extra', 'close', 'blowout', 'lead',
          'ninth_role', 'late_role', 'closer_ninth')
MID = COMMON + ('runners', 'outs', 'third_batter')
ROLE_FALLBACK = {'closer': (0.8, 1.0), 'setup': (0.1, 0.8)}


def features(bf: float, exp: float, runs: float, inning: int, lead: float, ninth: float, late: float,
             runners: float = 0.0, outs: float = 0.0) -> dict:
    """Feature values for one decision point; lead is the fielding team's after the plate appearance."""
    bf = float(bf); exp = float(exp); runs = min(max(float(runs), 0.0), 6.0); lead = float(lead)
    return {
        'bf': bf, 'bf2': min(bf, 12.0) ** 2 / 10.0, 'over': max(0.0, bf - exp), 'under': max(0.0, exp - bf),
        'runs': runs, 'runs2': min(runs, 4.0) ** 2, 'exp': exp,
        'late': float(inning >= 7), 'ninth_inning': float(inning >= 9), 'extra': float(inning >= 10),
        'close': float(abs(lead) <= 2), 'blowout': float(abs(lead) >= 5), 'lead': max(-6.0, min(6.0, lead)) / 6.0,
        'ninth_role': float(ninth), 'late_role': float(late), 'closer_ninth': float(ninth >= 0.6 and inning >= 9),
        'runners': float(runners), 'outs': float(outs), 'third_batter': float(bf == 3),
    }


class ReliefExit:
    def __init__(self, doc: dict):
        self.name = str(doc.get('name') or 'reliever exits')
        self.parts = {}
        for key in ('mid', 'end'):
            part = doc[key]
            names = [str(n) for n in part['names']]
            beta = [float(b) for b in part['beta']]
            if len(names) != len(beta):
                raise ValueError('relief exit: names and coefficients differ in length')
            self.parts[key] = (float(part['intercept']), list(zip(names, beta)))

    def probability(self, key: str, x: dict) -> float:
        b0, terms = self.parts[key]
        z = b0 + sum(b * x[n] for n, b in terms)
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

    def facts(self, pitcher, line, state, fielding_side, inning_ended) -> tuple[str, dict]:
        usage = dict(getattr(pitcher, 'usage', ()) or ())
        if 'ninth' in usage or 'late' in usage:
            ninth, late = float(usage.get('ninth', 0.0)), float(usage.get('late', 0.0))
        else:
            ninth, late = ROLE_FALLBACK.get(str(getattr(pitcher, 'role', '') or ''), (0.0, 0.2))
        lead = state.score_for(fielding_side) - state.opponent_score_for(fielding_side)
        if inning_ended:
            return 'end', features(line.batters_faced, pitcher.expected_batters, line.runs_allowed, state.inning, lead, ninth, late)
        runners = sum(b is not None for b in state.bases)
        return 'mid', features(line.batters_faced, pitcher.expected_batters, line.runs_allowed, state.inning, lead, ninth, late,
                               runners, min(state.outs, 2))

    def remove(self, pitcher, line, state, fielding_side, inning_ended, rng, offset=None) -> bool:
        """offset: the fielding team's hook tendency {'mid': log-odds, 'end': log-odds} (RELIEF-03), or None."""
        if not inning_ended and line.batters_faced < 3:
            return False
        key, x = self.facts(pitcher, line, state, fielding_side, inning_ended)
        p = self.probability(key, x)
        if offset:
            z = math.log(p / (1.0 - p)) + float(offset.get(key, 0.0))
            p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
        return bool(rng.random() < p)
