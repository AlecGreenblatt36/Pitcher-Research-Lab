"""build_report for a game still to come, end to end on a small made-up league: the lineup the simulator would project
(by the opposing starter's hand), the bench, and the bullpen (the simulator's likeliest arms, else the most used)."""
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DAY = '2026-10-10'
D0 = date.fromisoformat(DAY).toordinal()
VS_R = [201, 202, 203, 204, 205, 206, 207, 208, 209]      # the away team against righties
VS_L = [201, 202, 203, 204, 205, 206, 217, 218, 219]      # and against lefties
HOME = [101, 102, 103, 104, 105, 106, 107, 108, 109]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_build_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table():
    """Six earlier games between the two clubs: the away club (2) bats in the top half, the home club (1) fields with
    starter 900 or 901 and relievers; the away club's starter is a righty (800) or a lefty (801)."""
    rows, sched = [], []
    plan = [(1, 'R', 800, [950, 951]), (2, 'L', 801, [951]), (3, 'R', 800, [952, 951]), (4, 'R', 800, [953]), (5, 'L', 801, [951, 952]), (6, 'R', 800, [954])]
    for k, (ago, hand, away_sp, pen) in enumerate(plan):
        pk = 5000 + k; day = D0 - ago
        sched.append({'date': date.fromordinal(day).isoformat(), 'games': [{'gamePk': pk, 'status': {'abstractGameState': 'Final'}, 'gameType': 'D',
                      'teams': {'away': {'team': {'id': 2}}, 'home': {'team': {'id': 1}}}}]})
        home_sp_hand = 1                                                   # the home club's starter throws right, so the away
        for ab, bat in enumerate(VS_R * 2, start=1):                      # club ran out its lineup against righties every game
            pit = 900 if ab <= 12 else pen[(ab - 13) % len(pen)]
            for pn in range(2):
                rows.append((pk, day, 0, bat, pit, ab, pn, home_sp_hand, 2026))
        bottom_hand = 1 if hand == 'R' else 0
        for ab, bat in enumerate(HOME * 2, start=1):
            for pn in range(2):
                rows.append((pk, day, 1, bat, away_sp, 100 + ab, pn, bottom_hand, 2026))
    a = np.array(rows)
    T = {'game': a[:, 0], 'day': a[:, 1], 'half': a[:, 2], 'batter': a[:, 3], 'pitcher': a[:, 4], 'ab': a[:, 5], 'pitch_no': a[:, 6],
         'throw_r': a[:, 7], 'season': a[:, 8]}
    return T, sched


class FakeFit:
    def __init__(self, T):
        self.T = T; self.n_train = len(T['game']); self.PM = None
        self.gp = {int(p): np.flatnonzero(T['pitcher'] == p) for p in np.unique(T['pitcher'])}
        for p in (900, 950, 951, 952, 953, 954):                          # enough pitches for a plan
            self.gp[p] = np.resize(self.gp[p], 400)
        self.maps_s = {int(b): 1 for b in np.unique(T['batter'])}

    def pair(self, h, p):
        return {'aim': {'cells': {}}}

    def hitter_card(self, pid):
        return None

    def pitcher_card(self, pid):
        return None


def _wire(B, sched):
    game = {'gamePk': 6000, 'gameType': 'D', 'gameDate': DAY + 'T23:00:00Z', 'status': {'detailedState': 'Scheduled', 'abstractGameState': 'Preview'},
            'teams': {'away': {'team': {'id': 2, 'name': 'Away', 'abbreviation': 'AWY'}, 'probablePitcher': {'id': 800}},
                      'home': {'team': {'id': 1, 'name': 'Home', 'abbreviation': 'HOM'}, 'probablePitcher': {'id': 900}}}, 'lineups': {}}

    def mlb(url):
        if 'people?' in url:
            return {'people': []}
        if '&date=' in url:
            return {'dates': [{'date': DAY, 'games': [game]}]}
        return {'dates': sched}
    B.mlb = mlb


def test_game_to_come_gets_the_projected_lineup_bench_and_bullpen():
    B = _report()
    T, sched = _table()
    _wire(B, sched)
    fit = FakeFit(T)
    rep = B.build_report(fit, T, DAY, DAY, lambda *a: None, max_relievers=2)
    se = rep['games']['6000']['sides']
    away = se['away']                                    # the away club bats against the home starter, a righty
    assert away['lineup_source'] == 'last game against a righty'
    assert away['hitters'][:9] == VS_R and away['spots']['201'] == 1
    # the home club's relievers by games in relief: 951 (four), then 952 (two), then the rest by batters faced and recency
    assert [p['id'] for p in away['pitchers']] == [900, 951, 952]
    # each arm carries his pitches on the five days before the game (the bullpen card's rest columns)
    rec = {p['id']: p['recent'] for p in away['pitchers']}
    assert all(len(v) == 5 and all(isinstance(n, int) for n in v) for v in rec.values())
    assert rec[900][0] > 0 and rec[952][0] == 0 and rec[952][2] > 0        # 900 started yesterday; 952 last pitched three days ago
    # with a simulator box, its likeliest arms come first instead
    rep2 = B.build_report(fit, T, DAY, DAY, lambda *a: None, max_relievers=2, sim_pens={6000: {'home': [954, 950, 951]}})
    assert [p['id'] for p in rep2['games']['6000']['sides']['away']['pitchers']] == [900, 954, 950]
