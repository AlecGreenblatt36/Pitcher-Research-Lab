"""PTAGS-01's machinery on made-up seasons: pitchers whose habits carry over keep their tags; redrawn habits do not."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_ptags_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _season(rng, tr, season, n_pitch=1500):
    npit = len(tr['zone'])
    pit = np.repeat(np.arange(npit), n_pitch); n = len(pit)
    balls = rng.integers(0, 4, n); strikes = rng.integers(0, 3, n)
    first = rng.random(n) < 0.25; balls[first] = 0; strikes[first] = 0
    inz = rng.random(n) < np.clip(tr['zone'][pit], 0.2, 0.8)
    zone = np.where(inz, rng.integers(1, 10, n), rng.integers(11, 15, n))
    behind = balls > strikes
    fb = rng.random(n) < np.clip(np.where(behind, tr['fbb'][pit], 0.55), 0.05, 0.95)
    sw = rng.random(n) < np.where(inz, 0.66, np.clip(tr['chase'][pit], 0.05, 0.6))
    wh = sw & (rng.random(n) < 0.25)
    call = np.where(~sw, 0, np.where(wh, 2, 1)); cs = np.where((call == 0) & inz, 1, 0)
    return {'pitcher': pit + 500, 'season': np.full(n, season), 'day': np.full(n, 700000 + season * 400), 'post': np.zeros(n, int),
            'group': np.where(fb, 0, 3), 'call': call, 'cs': cs, 'balls': balls, 'strikes': strikes, 'zone': zone}


def _traits(rng, k):
    return {'zone': rng.normal(0.5, 0.06, k), 'fbb': rng.normal(0.6, 0.15, k), 'chase': rng.normal(0.28, 0.05, k)}


def _world(persist, seed=5, k=150):
    rng = np.random.default_rng(seed)
    t1 = _traits(rng, k); t2 = t1 if persist else _traits(rng, k)
    parts = [_season(rng, t1, 1), _season(rng, t1, 2), _season(rng, t2, 3)]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def test_pitcher_tags_hold_when_habits_carry_over():
    B = _report()
    spec = {'splits': [{'train': [1, 2], 'test': 3}], 'min_pitches': 500, 'min_test': 300, 'boot': 300}
    held = B.pitcher_tag_study(_world(True), lambda *a: None, spec)['splits'][0]
    for name in ('Lives in the zone', 'Works off the plate', 'Fastballs when behind', 'Spins it when behind', 'Gets chases'):
        t = held['tags'][name]
        assert t['pitchers'] >= 5 and t['held'], (name, t)
    assert held['corr']['zone']['r'] > 0.6 and held['corr']['fb_behind']['r'] > 0.6
    null = B.pitcher_tag_study(_world(False), lambda *a: None, spec)['splits'][0]
    for name in ('Lives in the zone', 'Fastballs when behind'):
        assert not null['tags'][name]['held'], (name, null['tags'][name])
    assert abs(null['corr']['zone']['r']) < 0.25
