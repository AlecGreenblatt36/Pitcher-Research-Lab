"""STAB-01: the shrinkage constant comes out right on made-up seasons where it is known: per-PA strikeouts with true
rates spread sd 0.05 around 0.22 give k = 0.22 x 0.78 / 0.05^2 = 68.6 plate appearances."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('brl_stability', Path(__file__).resolve().parents[1] / 'tools' / 'brl_stability.py')
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)


def _seasons(seed=7, players=400, pa=500):
    rng = np.random.default_rng(seed)
    true = np.clip(rng.normal(0.22, 0.05, players), 0.02, 0.6)
    rows = []
    for season in (2017, 2018):
        for i, p in enumerate(true):
            k = rng.random(pa) < p
            rows.append(pd.DataFrame({'season': season, 'batter': i, 'pitcher': 10_000, 'outcome': np.where(k, 'K', 'BIP_OUT'),
                                      'n': 4, 'sw': 2, 'wh': 0, 'oz': 1, 'ch': 0, 'iz': 3, 'izs': 2, 'izc': 2, 'fb': 2, 'velo': 190.0,
                                      'ev': np.nan, 'la': np.nan}))
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=1).sort_values('season', kind='stable').reset_index(drop=True)


def test_split_half_and_next_season_recover_the_known_k():
    pa = _seasons()
    u = st.unit_frame(pa, 'K per PA')
    sh = st._pool(st.split_half(u, 150), range(2015, 2027), 'players')
    ns = st._pool(st.next_season(u, 300), range(2015, 2027), 'pairs')
    assert sh['players'] == 800 and 45 < sh['k'] < 100
    assert ns['pairs'] == 400 and 50 < ns['k'] < 95 and abs(ns['n'] - 500) < 1


def test_run_reports_every_stat_and_era():
    out = st.run(_seasons(players=60, pa=320))
    assert set(out['stats']) == set(st.STATS) and out['plate_appearances_by_season'] == {2017: 19200, 2018: 19200}
    k = out['stats']['K per PA']
    assert set(k['split_half']) == {'2015-2019', '2021-2026', 'all'} and k['split_half']['2021-2026'] == {'seasons': 0}
