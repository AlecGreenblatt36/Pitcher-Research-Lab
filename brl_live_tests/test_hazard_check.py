"""LEASH-02's diagnosis (tools/brl_research.py hazard_check) end to end on made-up games: a hazard bundle fitted on them is
scored back on every removal decision, by season, inning, whether the inning ended and the score, with the starts by the
outs recorded at the exit; the summed chance tracks the removals."""
import importlib.util
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
