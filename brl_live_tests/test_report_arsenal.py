"""Pitcher cards list the arsenal by pitch type (share, speed, spin, movement with arm side positive, misses, chases,
strikes, put-aways), on a made-up pitcher."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_arsenal_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pitcher_card_arsenal_by_pitch_type():
    B = _report()
    rng = np.random.default_rng(5)
    n = 400
    sub = np.where(np.arange(n) % 4 == 0, 4, 0)                       # a quarter sliders (SL), the rest four-seamers (FF)
    group = np.where(sub == 4, 3, 0)
    T = {'group': group, 'sub': sub, 'v0': np.where(sub == 4, 86.0, 96.0) + rng.normal(0, 0.5, n),
         'spin': np.where(sub == 4, 2600.0, 2350.0), 'pfx_x': np.where(sub == 4, 5.0, -7.0), 'pfx_z': np.where(sub == 4, 1.0, 9.0),
         'throw_r': np.ones(n, int), 'stand_r': np.resize([0, 1], n), 'strikes': np.resize([0, 1, 2], n), 'balls': np.zeros(n, int),
         'last_in_pa': np.resize([0, 0, 1], n), 'out7': np.resize([1, 0, 3], n), 'zone': np.resize([5, 14], n)}
    swing = (np.arange(n) % 2 == 0).astype(float); whiff = (np.arange(n) % 6 == 0).astype(float)
    fake = SimpleNamespace(T=T, gp={77: np.arange(n)}, swing=swing, whiff=whiff, outside=np.resize([False, True], n),
                           xt=np.zeros(n), zt=np.full(n, 2.5), looks_in_ends_out=np.zeros(n, bool))
    card = B.Fitted.pitcher_card(fake, 77)
    ars = card['arsenal']
    assert [a['type'] for a in ars] == ['FF', 'SL']                    # by usage
    ff, sl = ars
    assert abs(ff['share'] - 0.75) < 0.01 and abs(ff['speed'] - 96.0) < 0.2 and ff['spin'] == 2350
    assert ff['h_arm'] == 7.0 and sl['h_arm'] == -5.0                   # a righty's four-seamer runs to his arm side, his slider to his glove side
    assert ff['v_mov'] == 9.0 and sl['v_mov'] == 1.0
