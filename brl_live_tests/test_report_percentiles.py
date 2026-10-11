"""Player lines and percentile ranks (PROD-08): counts from the pitch rows, ranks among players with enough plate appearances."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_pct_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_player_lines_count_each_hitter_and_pitcher():
    B = _report()
    ff, si, sl = B.D.SUBTYPES.index('FF'), B.D.SUBTYPES.index('SI'), B.D.SUBTYPES.index('SL')
    #        batter pitcher call last out7 outside spray  ls    sub v0
    rows = [(1, 10, 0, 0, -1, 1, np.nan, np.nan, ff, 95.0),    # a ball out of the zone, taken
            (1, 10, 2, 0, -1, 1, np.nan, np.nan, sl, 85.0),    # a chase, missed
            (1, 10, 1, 1, 3, 0, 5.0, 101.0, ff, 96.0),         # a hard single
            (1, 10, 2, 1, 1, 0, np.nan, np.nan, sl, 86.0),     # strike three (another plate appearance)
            (2, 10, 0, 1, 2, 1, np.nan, np.nan, ff, 97.0),     # ball four
            (2, 11, 1, 1, 5, 0, -20.0, 104.0, si, 93.0),       # a home run off a sinker
            (2, 11, 1, 1, 0, 0, 30.0, 80.0, si, 92.0)]         # a soft out
    a = np.array(rows, dtype=float)
    T = {'batter': a[:, 0].astype(int), 'pitcher': a[:, 1].astype(int), 'call': a[:, 2].astype(int), 'last_in_pa': a[:, 3].astype(int),
         'out7': a[:, 4].astype(int), 'spray': a[:, 6], 'ls': a[:, 7], 'sub': a[:, 8].astype(int), 'v0': a[:, 9]}
    outside = a[:, 5] == 1
    swing = ((T['call'] == 1) | (T['call'] == 2)).astype(float); whiff = (T['call'] == 2).astype(float)
    H = B.player_lines(T, np.ones(len(rows), bool), swing, whiff, outside, 'batter')
    assert H[1] == {'pa': 2, 'ab': 2, 'h': 1, 'tb': 1, 'hr': 0, 'k': 1, 'bb': 0, 'pitches': 4, 'swings': 3, 'misses': 2,
                    'outside': 2, 'chases': 1, 'bbe': 1, 'hard': 1}
    assert H[2]['pa'] == 3 and H[2]['ab'] == 2 and H[2]['bb'] == 1 and H[2]['hr'] == 1 and H[2]['tb'] == 4 and H[2]['bbe'] == 2 and H[2]['hard'] == 1
    r = B.line_rates(H[2])
    assert r['avg'] == 0.5 and r['slg'] == 2.0 and abs(r['bb'] - 1 / 3) < 1e-9 and r['hard'] == 0.5 and r['chase'] == 0.0
    P = B.player_lines(T, np.ones(len(rows), bool), swing, whiff, outside, 'pitcher')
    assert P[10]['pa'] == 3 and P[10]['k'] == 1 and P[10]['bb'] == 1 and 'velo' not in P[10]     # under 50 fastballs: no speed
    assert B.player_lines(T, np.zeros(len(rows), bool), swing, whiff, outside, 'batter') == {}


def test_main_fastball_speed_needs_fifty_and_takes_the_one_he_throws_more():
    B = _report()
    ff, si = B.D.SUBTYPES.index('FF'), B.D.SUBTYPES.index('SI')
    n = 120
    sub = np.array([ff] * 70 + [si] * 50); v0 = np.array([96.0] * 70 + [93.0] * 50)
    T = {'batter': np.arange(n) % 7, 'pitcher': np.full(n, 5), 'call': np.zeros(n, int), 'last_in_pa': np.zeros(n, int), 'out7': np.full(n, -1),
         'spray': np.full(n, np.nan), 'ls': np.full(n, np.nan), 'sub': sub, 'v0': v0}
    z = np.zeros(n)
    P = B.player_lines(T, np.ones(n, bool), z, z, np.zeros(n, bool), 'pitcher')
    assert P[5]['velo'] == 96.0
    T['sub'] = np.array([ff] * 40 + [si] * 80); T['v0'] = np.array([96.0] * 40 + [93.0] * 80)
    assert B.player_lines(T, np.ones(n, bool), z, z, np.zeros(n, bool), 'pitcher')[5]['velo'] == 93.0


def test_percentile_ranks_only_for_players_with_enough_plate_appearances():
    B = _report()
    lines = {}
    for i in range(100):           # 100 ranked hitters, strikeouts 10 to 109 per 400
        lines[i] = {'pa': 400, 'ab': 360, 'h': 90, 'tb': 140, 'hr': 10, 'k': 10 + i, 'bb': 40, 'pitches': 1600, 'swings': 750,
                    'misses': 150, 'outside': 700, 'chases': 200, 'bbe': 250, 'hard': 100}
    lines[500] = dict(lines[0], pa=120)      # too few to be ranked, and not in the reference
    ref = B.pct_reference(lines, B.PCT_HITTER)
    assert len(ref['k']) == 100
    few = B.pct_ranks(lines[0], ref, B.PCT_HITTER)
    many = B.pct_ranks(lines[99], ref, B.PCT_HITTER)
    assert few['k'] == 100 and many['k'] == 1                 # fewest strikeouts ranks best
    assert few['avg'] == 50                                   # everyone ties: the middle
    assert B.pct_ranks(lines[500], ref, B.PCT_HITTER) is None
    assert B.pct_ranks(None, ref, B.PCT_HITTER) is None
    p_lines = {i: dict(lines[i], velo=90.0 + i / 10) for i in range(100)}
    pref = B.pct_reference(p_lines, B.PCT_PITCHER)
    top = B.pct_ranks(p_lines[99], pref, B.PCT_PITCHER)
    assert top['velo'] == 100 and top['k'] == 100            # a pitcher's strikeouts rank higher as better


def test_head_to_head_counts_each_pair_on_training_pitches():
    from types import SimpleNamespace
    B = _report()
    #        batter pitcher last out7 train
    rows = [(1, 10, 0, -1, 1),   # a pitch mid at-bat: not counted
            (1, 10, 1, 5, 1),    # a home run
            (1, 10, 1, 1, 1),    # a strikeout
            (1, 10, 1, 2, 1),    # a walk
            (1, 11, 1, 3, 1),    # a single off another arm
            (2, 10, 1, 0, 1),    # an out
            (1, 10, 1, 4, 0)]    # after the as-of date: left out
    a = np.array(rows)
    T = {'batter': a[:, 0], 'pitcher': a[:, 1], 'last_in_pa': a[:, 2], 'out7': a[:, 3]}
    fake = SimpleNamespace(T=T, train=a[:, 4] == 1)
    hh = B.Fitted.head_to_head(fake, [1, 2], [10, 11])
    assert hh[(1, 10)] == [3, 2, 1, 4, 1, 1, 1]       # 3 PA, 2 AB, 1 hit (the homer), 4 bases, 1 HR, 1 K, 1 BB
    assert hh[(1, 11)] == [1, 1, 1, 1, 0, 0, 0] and hh[(2, 10)] == [1, 1, 0, 0, 0, 0, 0] and (2, 11) not in hh
    assert B.Fitted.head_to_head(fake, [], [10]) == {} and B.Fitted.head_to_head(fake, [3], [10]) == {}


def test_running_season_waits_for_the_season_to_end():
    from datetime import date
    B = _report()
    y, season = B.running_season(date(2026, 10, 10).toordinal())
    assert y == 2026 and 'runners' in season
    assert B.running_season(date(2026, 7, 1).toordinal())[0] == 2025          # a past plan never shows numbers from after its date
    lg = B.running_league(season)
    assert 0.05 < lg['att_per_on1'] < 0.2 and 0.6 < lg['sb_pct'] < 0.9 and lg['att_per_bf'] > 0
