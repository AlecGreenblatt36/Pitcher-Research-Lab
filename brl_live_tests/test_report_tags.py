"""TAGS-01's machinery on made-up seasons: hitters whose habits carry over keep their tags on later pitches; hitters whose
habits are redrawn each season do not."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_tags_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _season(rng, traits, season, n_pitch=1500):
    nh = len(traits['chase'])
    bat = np.repeat(np.arange(nh), n_pitch)
    n = len(bat)
    balls = rng.integers(0, 4, n); strikes = rng.integers(0, 3, n)
    first = rng.random(n) < 0.25
    balls[first] = 0; strikes[first] = 0
    px = rng.normal(0, 0.9, n); pz = rng.normal(2.5, 0.9, n)
    outside = (np.abs(px) > 0.83) | (pz > 3.5) | (pz < 1.5)
    col = np.clip(((px + 0.83) / (1.66 / 3)).astype(int), 0, 2); row = np.clip(((3.5 - pz) / (2.0 / 3)).astype(int), 0, 2)
    zone = np.where(outside, np.where(pz >= 2.5, np.where(px < 0, 11, 12), np.where(px < 0, 13, 14)), 1 + col + 3 * row)
    two = strikes == 2
    p_sw = np.where(outside, traits['chase'][bat] + np.where(two, 0.12 + traits['two'][bat], 0.0), 0.66)
    p_sw = np.where(first & ~outside, 0.66 + traits['first'][bat], p_sw)
    p_sw = np.clip(p_sw, 0.01, 0.99)
    sw = rng.random(n) < p_sw
    group = rng.choice(np.array([0, 0, 3, 5]), n)
    v0 = np.where(group == 0, rng.choice(np.array([91.0, 91.0, 97.5]), n), 85.0)
    wh = sw & (rng.random(n) < np.clip(traits['whiff'][bat] + np.where(group == 3, traits['brk'][bat], 0.0) + np.where(v0 >= 96, traits['velo'][bat], 0.0), 0.01, 0.9))
    call = np.where(~sw, 0, np.where(wh, 2, 1))
    inplay = call == 1
    side_r = (np.arange(n) // n_pitch) % 2 == 1
    pull_p = np.clip(traits['pull'][bat], 0.05, 0.95)
    pulled = rng.random(n) < pull_p
    spray = np.where(inplay, np.where(pulled, np.where(side_r, -30.0, 30.0), np.where(rng.random(n) < 0.5, 0.0, np.where(side_r, 30.0, -30.0))), np.nan)
    traj = np.where(inplay, rng.choice(np.array([0, 0, 1, 2]), n), -1)
    day = 700000 + season * 400 + rng.integers(0, 180, n)
    return {'batter': bat + 1000, 'season': np.full(n, season), 'day': day, 'post': np.zeros(n, int), 'group': group, 'call': call,
            'balls': balls, 'strikes': strikes, 'px': px.astype(np.float32), 'pz': pz.astype(np.float32), 'zone': zone, 'stand_r': (np.arange(n) // n_pitch) % 2,
            'bunt_pa': np.zeros(n, int), 'last_in_pa': np.zeros(n, int), 'v0': v0.astype(np.float32), 'spray': spray.astype(np.float32), 'traj': traj}


def _traits(rng, nh):
    return {'chase': rng.normal(0.28, 0.07, nh), 'whiff': rng.normal(0.24, 0.06, nh), 'first': rng.normal(-0.36, 0.2, nh), 'two': rng.normal(0.0, 0.07, nh),
            'brk': rng.normal(0.08, 0.08, nh), 'velo': rng.normal(0.05, 0.08, nh), 'pull': rng.normal(0.4, 0.12, nh)}


def _world(persist, seed=3, nh=160, velo=False):
    rng = np.random.default_rng(seed)
    t1 = _traits(rng, nh); t2 = t1 if persist else _traits(rng, nh)
    if not velo:
        t1['velo'] = t1['velo'] * 0.0; t2['velo'] = t2['velo'] * 0.0
    parts = [_season(rng, t1, 1), _season(rng, t1, 2), _season(rng, t2, 3)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def test_tags_hold_when_habits_carry_over_and_not_when_redrawn():
    B = _report()
    spec = {'splits': [{'train': [1, 2], 'test': 3}], 'min_pitches': 300, 'min_test': 200, 'boot': 300}
    held = B.tag_study(_world(True), lambda *a: None, spec)['splits'][0]
    for name in ('Chases a lot', 'Patient', 'Swings and misses', 'Puts the bat on the ball', 'Takes the first pitch', 'Swings at the first pitch'):
        t = held['tags'][name]
        assert t['hitters'] >= 5 and t['held'], (name, t)
        assert abs(t['test_diff'] - t['train_diff']) < 0.03, (name, t)
    fam = B.tag_study(_world(True), lambda *a: None, dict(spec, families=True))['splits'][0]['relative_family_whiff']
    assert set(fam) == {'breaking', 'offspeed'} and fam['breaking']['hitters'] > 50 and fam['breaking']['corr_train_test'] > 0.4
    assert abs(fam['offspeed']['corr_train_test']) < 0.3                       # nothing planted on offspeed
    pull = B.tag_study(_world(True), lambda *a: None, dict(spec, spray=True, gb_share=True))['splits'][0]['pull']
    assert 'gb_share' in pull and pull['gb_share']['hitters'] > 50
    assert pull['pull_gb']['corr_train_test'] > 0.5 and pull['pull_gb']['tags']['pulls']['test_diff'] > 0.05
    vel = B.tag_study(_world(True, velo=True), lambda *a: None, dict(spec, velocity=True))['splits'][0]['relative_velocity_whiff']
    assert vel['hitters'] > 50 and vel['corr_train_test'] > 0.3 and vel['tags']['late_on_velocity']['test_diff'] > 0.03
    rel = held['relative_two_strike']
    assert rel['corr_train_test'] > 0.5 and rel['tags']['expands_relative']['test_diff'] > 0.03
    null = B.tag_study(_world(False), lambda *a: None, spec)['splits'][0]
    for name in ('Chases a lot', 'Patient', 'Swings and misses', 'Puts the bat on the ball'):
        t = null['tags'][name]
        assert not t['held'] and abs(t['test_diff']) < 0.045, (name, t)
    assert abs(null['relative_two_strike']['corr_train_test']) < 0.25
