"""RELIEF-04's usage check (tools/brl_relief_usage.py) on made-up team-games: the simulated and real slopes of pitchers on
the starter's outs are recovered, and a simulator that adds relievers faster per missing out shows a larger slope."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('brl_relief_usage', Path(__file__).parents[1] / 'tools/brl_relief_usage.py')
ru = importlib.util.module_from_spec(spec); spec.loader.exec_module(ru)


def _games(seed, sim_slope, n=1500):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        st = {}
        for side in ('away', 'home'):
            mu = rng.uniform(10, 20)
            sims = np.clip(np.rint(rng.normal(mu, 3.0, 400)), 0, 27).astype(int)
            real = int(np.clip(np.rint(rng.normal(mu, 3.0)), 0, 27))
            st[side] = {'outs': np.bincount(sims, minlength=28).tolist(), 'actual': {'outs': real},
                        'team_actual': {'pitchers': int(max(1, round(6.5 - 0.15 * real + rng.normal(0, 0.6))))},
                        'team_pitchers_sim_mean': 6.5 - sim_slope * float(sims.mean())}
        out.append({'game_pk': i, 'starters': st})
    return out


def test_slopes_are_recovered_and_a_fast_hook_shows():
    r = ru.season_check(_games(1, 0.15))
    assert abs(r['slope_real'] + 0.15) < 0.02 and abs(r['slope_sim'] + 0.15) < 0.005 and abs(r['slope_gap']) < 0.02
    fast = ru.season_check(_games(2, 0.19))
    assert fast['slope_gap'] < -0.02 and fast['team_games'] == 3000
    assert set(fast['by_starter_outs']) == {'0-11', '12-14', '15-17', '18-20', '21-27'}
