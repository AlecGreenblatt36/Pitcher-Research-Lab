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
    pz = np.where(fb, rng.normal(2.5 + tr['fbh'][pit], 0.7), rng.normal(1.9 + tr['brh'][pit], 0.7))
    inplay = call == 1
    gb = rng.random(n) < np.clip(tr['gb'][pit], 0.05, 0.95)
    traj = np.where(inplay, np.where(gb, 0, rng.choice(np.array([1, 2, 3]), n)), -1)
    spray = np.where(inplay, rng.normal(0, 20, n), np.nan)
    # across the plate: a righty batter stands at negative x (catcher's view); 'ins' moves fastballs toward the batter;
    # lefty pitchers get more lefty batters, so the matchup mix differs by pitcher
    throw_r = (tr['rhp'][pit] > 0).astype(int)
    stand_r = (rng.random(n) < np.where(throw_r == 1, 0.6, 0.35)).astype(int)
    toward = np.where(stand_r == 1, -1.0, 1.0)
    px = np.where(fb, toward * (tr['ins'][pit] + np.where(stand_r == throw_r, 0.1, -0.1)), 0.0) + rng.normal(0, 0.55, n)
    return {'pitcher': pit + 500, 'season': np.full(n, season), 'day': np.full(n, 700000 + season * 400), 'post': np.zeros(n, int),
            'group': np.where(fb, 0, 3), 'call': call, 'cs': cs, 'balls': balls, 'strikes': strikes, 'zone': zone, 'pz': pz.astype(np.float32),
            'traj': traj, 'spray': spray.astype(np.float32), 'px': px.astype(np.float32), 'stand_r': stand_r, 'throw_r': throw_r}


def _traits(rng, k):
    return {'zone': rng.normal(0.5, 0.06, k), 'fbb': rng.normal(0.6, 0.15, k), 'chase': rng.normal(0.28, 0.05, k),
            'fbh': rng.normal(0.0, 0.25, k), 'brh': rng.normal(0.0, 0.25, k), 'gb': rng.normal(0.43, 0.07, k),
            'ins': rng.normal(0.0, 0.2, k), 'rhp': rng.normal(0.3, 1.0, k)}


def _world(persist, seed=5, k=150):
    rng = np.random.default_rng(seed)
    t1 = _traits(rng, k); t2 = t1 if persist else dict(_traits(rng, k), rhp=t1['rhp'])      # a pitcher keeps his hand
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


def test_location_tags_hold_when_habits_carry_over():
    # PTAGS-02: fastballs up or down and breaking balls below the zone, by height
    B = _report()
    spec = {'splits': [{'train': [1, 2], 'test': 3}], 'min_pitches': 500, 'min_test': 300, 'boot': 300, 'location': True}
    held = B.pitcher_tag_study(_world(True), lambda *a: None, spec)['splits'][0]
    for name in ('Elevates fastballs', 'Keeps fastballs down', 'Buries breaking balls'):
        t = held['tags'][name]
        assert t['pitchers'] >= 5 and t['held'], (name, t)
    assert held['corr']['fb_up']['r'] > 0.6 and held['corr']['br_dn']['r'] > 0.6
    null = B.pitcher_tag_study(_world(False), lambda *a: None, spec)['splits'][0]
    assert not null['tags']['Elevates fastballs']['held'] and abs(null['corr']['fb_up']['r']) < 0.25
    plain = B.pitcher_tag_study(_world(True), lambda *a: None, dict(spec, location=False))['splits'][0]
    assert 'Elevates fastballs' not in plain['tags'] and 'fb_up' not in plain['corr']


def test_ground_ball_tags_hold_when_habits_carry_over():
    # PTAGS-03: ground balls among batted balls against him
    B = _report()
    spec = {'splits': [{'train': [1, 2], 'test': 3}], 'min_pitches': 500, 'min_test': 300, 'boot': 300, 'batted': True}
    held = B.pitcher_tag_study(_world(True), lambda *a: None, spec)['splits'][0]
    for name in ('Ground-ball pitcher', 'Fly-ball pitcher'):
        t = held['tags'][name]
        assert t['pitchers'] >= 5 and t['held'], (name, t)
    assert held['corr']['gb_against']['r'] > 0.6
    null = B.pitcher_tag_study(_world(False), lambda *a: None, spec)['splits'][0]
    assert not null['tags']['Ground-ball pitcher']['held'] and abs(null['corr']['gb_against']['r']) < 0.25


def test_side_tags_hold_when_habits_carry_over_and_read_against_the_matchups():
    # PTAGS-04: fastballs inside or away, against the league's rate for each pitcher hand and batter side he faced
    B = _report()
    spec = {'splits': [{'train': [1, 2], 'test': 3}], 'min_pitches': 500, 'min_test': 300, 'boot': 300, 'side': True}
    held = B.pitcher_tag_study(_world(True), lambda *a: None, spec)['splits'][0]
    for name in ('Works inside with the fastball', 'Works away with the fastball'):
        t = held['tags'][name]
        assert t['pitchers'] >= 5 and t['held'], (name, t)
    assert held['corr']['fb_in']['r'] > 0.6 and held['corr']['fb_away']['r'] > 0.6
    null = B.pitcher_tag_study(_world(False), lambda *a: None, spec)['splits'][0]
    assert not null['tags']['Works inside with the fastball']['held'] and abs(null['corr']['fb_in']['r']) < 0.3
    # the adjusted numerators keep the window's overall count
    T = _world(True); m = T['season'] == 3
    pc = B.pitcher_tag_counts(T, m)
    k_, n_ = pc['fb_in']
    fb = (T['group'][m] == 0) & np.isfinite(T['px'][m])
    assert abs(k_.sum() - n_.sum() * (k_.sum() / n_.sum())) < 1e-6 and abs(n_.sum() - fb.sum()) < 1e-9
