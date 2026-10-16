"""LEASH-02's diagnosis (tools/brl_research.py hazard_check) end to end on made-up games: a hazard bundle fitted on them is
scored back on every removal decision, by season, inning, whether the inning ended and the score, with the starts by the
outs recorded at the exit; the summed chance tracks the removals."""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
spec = importlib.util.spec_from_file_location('brl_research', ROOT / 'tools/brl_research.py')
br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)


def _games(rng, n=160):
    rows = []
    for g in range(n):
        date = '%d-%02d-%02d' % (2025 + g % 2, 4 + (g // 60) % 5, 1 + g % 28)
        score = {'top': 0, 'bot': 0}
        ab = 0
        exit_inning = {'top': int(rng.choice([4, 5, 5, 6, 6, 7])), 'bot': int(rng.choice([4, 5, 5, 6, 6, 7]))}
        for inning in range(1, 10):
            for half in ('top', 'bot'):
                starter = (100 + g % 7) if half == 'top' else (300 + g % 5)
                pitcher = starter if inning <= exit_inning[half] else (starter + 1000 + g)
                outs, r1 = 0, 0
                while outs < 3:
                    ab += 1
                    bat = score['top'] if half == 'top' else score['bot']
                    fld = score['bot'] if half == 'top' else score['top']
                    rows.append({'date_key': date, 'game_pk': g + 1, 'at_bat_number': ab, 'inning_topbot': 'Top' if half == 'top' else 'Bot',
                                 'home_team': 'H%d' % (g % 3), 'away_team': 'A%d' % (g % 4), 'pitcher': pitcher, 'batter': 5000 + ab % 9, 'inning': inning,
                                 'outs_when_up': outs, 'bat_score': bat, 'fld_score': fld, 'runner_1b': r1, 'runner_2b': 0, 'runner_3b': 0,
                                 'outcome': 'K', 'game_type': 'R'})
                    if rng.random() < 0.3:
                        r1 = 1
                        if rng.random() < 0.3:
                            score[half] += 1; r1 = 0
                    else:
                        outs += 1
    return pd.DataFrame(rows)


def test_hazard_check_scores_real_decisions_by_inning(tmp_path):
    from research_lab.game_sim import starter_hazard as sh
    rng = np.random.default_rng(4)
    pa = _games(rng)
    pa.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    from brl_replay.games import load_history
    h = load_history(tmp_path / 'plate_appearances.csv.gz')
    d, starts = sh.build_decisions(h)
    league = float(starts['bf_total'].mean())
    d = sh.add_tendencies(d, starts, league)
    joblib.dump({'model': sh.fit_hazard(d, 1.0), 'league_mean_bf': league}, tmp_path / 'starter_hazard.joblib')
    out = br.hazard_check(tmp_path)
    assert out['decisions'] == len(d)
    n = sum(c[4] for c in out['cells']); y = sum(c[5] for c in out['cells']); p = sum(c[6] for c in out['cells'])
    assert n == len(d) and y == int(d['y'].sum()) and abs(p - y) < 0.05 * y + 2       # a logistic fit's chances sum to its removals
    ex = out['exit_outs']
    assert set(ex) == {'2025', '2026'} and sum(sum(v) for v in ex.values()) == y
    assert all(len(v) == 28 for v in ex.values())
    # every exit here comes at an inning's end (exit after the 4th to 7th): 12 to 21 outs
    assert sum(sum(v[12:22]) for v in ex.values()) == y


def test_hazard_fix_recovers_a_planted_inning_habit_and_the_policy_applies_it(tmp_path):
    """LEASH-02: the fit runs end to end on made-up games and reports each season in the terms' order; the Newton fit with
    the hazard's log-odds as the offset recovers planted terms; the policy's chance with the correction is the plain chance
    moved by the term for the state it sees (after the 4th with a lead), and unchanged where the term is zero."""
    from research_lab.game_sim import starter_hazard as sh
    rng = np.random.default_rng(8)
    pa = _games(rng, n=400)
    pa.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    from brl_replay.games import load_history
    h = load_history(tmp_path / 'plate_appearances.csv.gz')
    d, starts = sh.build_decisions(h)
    league = float(starts['bf_total'].mean())
    d = sh.add_tendencies(d, starts, league)
    model = sh.fit_hazard(d, 1.0)
    joblib.dump({'model': model, 'league_mean_bf': league}, tmp_path / 'starter_hazard.joblib')
    # the fix itself on real-shaped data: fits, reports both seasons, keeps the terms' order
    out = br.hazard_fix(tmp_path, train=(2025,))
    assert out['terms'] == list(sh.CORRECTION_TERMS) and len(out['coef']) == 32
    assert set(out['by_season']) == {'2025', '2026'}
    # the Newton fit with an offset recovers a planted term
    n = 20000
    off = rng.normal(-2.0, 1.0, n); Z = np.zeros((n, 3)); Z[np.arange(n), rng.integers(0, 3, n)] = 1
    truth = np.array([-0.6, 0.0, 0.4])
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + Z @ truth)))).astype(float)
    g = br.ridge_logit_offset(Z, y, off, lam=1.0)
    assert np.allclose(g, truth, atol=0.12)
    # the policy adds the term for the state it sees
    corr = {'terms': list(sh.CORRECTION_TERMS), 'coef': [0.0] * 32}
    corr['coef'][sh.CORRECTION_TERMS.index('i4e_ahead')] = -0.6
    plain = sh.FittedStarterPolicy({'model': model, 'league_mean_bf': league}, {}, {}, {})
    fixed = sh.FittedStarterPolicy({'model': model, 'league_mean_bf': league}, {}, {}, {}, correction=corr)
    a = plain._prob(14.0, 1.0, 4.0, 1.0, 3.0, 0.0, 2.0, 22.0, 22.0); b = fixed._prob(14.0, 1.0, 4.0, 1.0, 3.0, 0.0, 2.0, 22.0, 22.0)
    lo = lambda p: np.log(p / (1 - p))
    assert abs((lo(b) - lo(a)) + 0.6) < 1e-9
    assert fixed._prob(14.0, 1.0, 5.0, 1.0, 3.0, 0.0, 2.0, 22.0, 22.0) == plain._prob(14.0, 1.0, 5.0, 1.0, 3.0, 0.0, 2.0, 22.0, 22.0)
    z = sh.correction_matrix(d.head(500))
    zz = np.array([sh.correction_terms(r.inning, r.inning_ended, r.lead, r.runs) for r in d.head(500).itertuples()])
    assert np.array_equal(z, zz)


def test_live_correction_is_regular_season_only():
    """On since LEASH-02 held (October 11, 2026): regular-season games get the correction, postseason games the plain hazard."""
    from brl_live import boxscore as bx
    from research_lab.game_sim import starter_hazard as sh
    assert bx.ADJUST['hazard_correction'] is True
    assert bx.hazard_correction_for(dict(bx.ADJUST, hazard_correction=False), 'R') is None
    on = bx.ADJUST
    # LEASH-04 (on from October 16, 2026): with the chain pitch draws the correction is LEASH-03's refit with pitch terms
    assert on['pitch_counts'] == 'chain'
    c = bx.hazard_correction_for(on, 'R')
    assert tuple(c['terms']) == sh.CORRECTION_TERMS_V2 and len(c['coef']) == len(sh.CORRECTION_TERMS_V2) == 44
    assert bx.hazard_correction_for(on, 'L') is None and bx.hazard_correction_for(on, 'W') is None
    plain = bx.hazard_correction_for(dict(on, pitch_counts=None), 'R')
    assert tuple(plain['terms']) == sh.CORRECTION_TERMS and len(plain['coef']) == 32
    import json
    live = json.loads((ROOT / 'brl_live/hazard_correction.json').read_text()); rep = json.loads((ROOT / 'brl_replay/hazard_correction.json').read_text())
    assert live['coef'] == rep['coef']                                       # the live tables are the ones the replays test
    live2 = json.loads((ROOT / 'brl_live/hazard_correction_v2.json').read_text()); rep2 = json.loads((ROOT / 'brl_replay/hazard_correction_v2.json').read_text())
    assert live2['coef'] == rep2['coef']


def test_hazard_pitches_sums_each_starters_pitches_and_scores_both_hazards(tmp_path):
    """LEASH-03's diagnosis on made-up games with a pitch count on every plate appearance: the bands cover every decision
    and each hazard's chances are summed per band."""
    from research_lab.game_sim import starter_hazard as sh
    rng = np.random.default_rng(12)
    pa = _games(rng, n=200)
    pa['pitch_number'] = rng.integers(1, 8, len(pa))
    pa.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    from brl_replay.games import load_history
    h = load_history(tmp_path / 'plate_appearances.csv.gz')
    d, starts = sh.build_decisions(h)
    league = float(starts['bf_total'].mean())
    d = sh.add_tendencies(d, starts, league)
    joblib.dump({'model': sh.fit_hazard(d, 1.0), 'league_mean_bf': league}, tmp_path / 'starter_hazard.joblib')
    out = br.hazard_pitches(tmp_path)
    assert out['with_pitches'] == out['decisions'] == len(d)
    assert sum(c[3] for c in out['cells']) == len(d) and sum(c[4] for c in out['cells']) == int(d['y'].sum())
    assert {c[1] for c in out['cells']} <= set(range(6))
    no = pa.drop(columns=['pitch_number']); no.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    assert br.hazard_pitches(tmp_path)['missing'] == 'pitch_number'
    # from the physics tables' pitches per plate appearance instead: the same counts give the same cells
    phys = pa[['game_pk', 'at_bat_number', 'pitch_number']].rename(columns={'pitch_number': 'n'})
    assert br.hazard_pitches(tmp_path, phys)['cells'] == out['cells']


def test_relief_pitches_scores_every_relief_decision_by_pitches_in_the_outing(tmp_path):
    rng = np.random.default_rng(14)
    pa = _games(rng, n=120)
    pa.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    phys = pa[['game_pk', 'at_bat_number']].assign(n=rng.integers(1, 8, len(pa)))
    out = br.relief_pitches(tmp_path, phys)
    assert out['decisions'] > 0 and sum(c[3] for c in out['cells']) == out['decisions']
    assert all(0 <= c[1] <= 4 and c[4] <= c[3] for c in out['cells'])


def test_pitch_efficiency_finds_planted_pitcher_factors_and_their_carry(tmp_path):
    rng = np.random.default_rng(15)
    pa = _games(rng, n=400)
    pa.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    fac = {p: f for p, f in zip(sorted(pa['pitcher'].unique()), rng.uniform(0.8, 1.2, pa['pitcher'].nunique()))}
    phys = pa[['game_pk', 'at_bat_number']].assign(n=np.maximum(1, np.rint(4 * pa['pitcher'].map(fac) + rng.normal(0, 1, len(pa)))).astype(int))
    out = br.pitch_efficiency(tmp_path, phys, min_bf=50)
    assert out['pitcher_seasons'] > 0 and out['factor_sd'] > 0.05
    assert out['carry']['2025-2026']['corr'] > 0.5            # the same pitcher's factor in both seasons


def test_relief_entry_counts_every_relief_decision_and_real_relievers_per_game(tmp_path):
    """The relievers' per-starter-out diagnosis on made-up games: every legal relief decision lands in one cell with its
    removal and the hazard's chance, and the real relievers per team-game follow the starter's outs."""
    rng = np.random.default_rng(31)
    pa = _games(rng, n=120)
    late = pa['inning'] >= 8
    pa.loc[late, 'pitcher'] = pa.loc[late, 'pitcher'] + 50000              # a second reliever from the 8th
    pa.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    out = br.relief_entry(tmp_path)
    from brl_replay.games import load_history
    from brl_replay.relief_decisions import decisions, fitted
    d = decisions(load_history(tmp_path / 'plate_appearances.csv.gz'))
    p = fitted(d, json.loads((ROOT / 'brl_live/relief_exit.json').read_text()))
    ok = np.isfinite(p)
    assert out['decisions'] == int(ok.sum()) == sum(c[5] for c in out['cells'])
    assert sum(c[6] for c in out['cells']) == int(d.loc[ok, 'removed'].sum()) > 0
    assert abs(sum(c[7] for c in out['cells']) - float(p[ok].sum())) < 1.0
    assert {c[1] for c in out['cells']} <= set(range(7)) and {c[3] for c in out['cells']} <= set(range(4))
    for v in out['team_games'].values():
        assert 2.0 <= v['pitchers'] <= 3.0 and v['slope_per_starter_out'] < 0      # a starter through the 7th leaves one reliever


def test_relief_fix_fits_each_table_and_scores_the_held_out_season(tmp_path):
    """RELIEF-04's stage 1 on made-up games with relievers who enter early and stay: each table gets 49 coefficients, the
    held-out season's log loss and its removals by entry inning are reported, and a planted stay-in shows as a negative term."""
    from research_lab.game_sim.relief_exit import CORRECTION_TERMS
    rng = np.random.default_rng(32)
    pa = _games(rng, n=160)
    late = pa['inning'] >= 8
    pa.loc[late, 'pitcher'] = pa.loc[late, 'pitcher'] + 50000
    pa.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    out = br.relief_fix(tmp_path)
    assert out['terms'] == list(CORRECTION_TERMS) and set(out['fits']) == {'2025', '2026', 'live'}
    for lab in ('2025', '2026'):
        f = out['fits'][lab]
        assert len(f['coef']) == 49 and f['held_out']['season'] == int(lab)
        assert sum(v[0] for v in f['held_out']['by_entry'].values()) == f['held_out']['decisions']
        assert f['held_out']['log_loss_change'] < 0                       # the made-up relievers stay far longer than the real hazard expects
    assert 'held_out' not in out['fits']['live']
