"""LONG-01: compare the arms of one fit-arms run (workflow brl-fit-arms) game by game on the 2026 test season.

Each arm's fit receipt (research/fit-<name>-<run>.json on the ledger branch) carries its 2025 tuning-block log loss and,
per 2026 test game, the plate appearances and summed log loss of the blended candidate. The selected arm is the
candidate with the lowest 2025 tuning-block log loss (talent_plus_context); the 2026 test is then read once, against the
control arm, with a bootstrap over games. Metrics only.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def paired(control: dict, arm: dict, column: int = 1, replicates: int = 2000, seed: int = 36) -> dict:
    """Arm minus control log loss per plate appearance (negative = the arm is better), with a 95% interval from
    resampling games. Both receipts must hold the same test games and plate appearances."""
    games = sorted(control)
    if sorted(arm) != games:
        raise ValueError('the arms were tested on different games')
    n = np.array([control[g][0] for g in games], float)
    if not np.array_equal(n, np.array([arm[g][0] for g in games], float)):
        raise ValueError('the arms were tested on different plate appearances')
    diff = np.array([arm[g][column] - control[g][column] for g in games], float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(games), size=(replicates, len(games)))
    values = diff[draws].sum(axis=1) / n[draws].sum(axis=1)
    return {'difference_nats_per_pa': float(diff.sum() / n.sum()), 'ci_95_low': float(np.quantile(values, 0.025)),
            'ci_95_high': float(np.quantile(values, 0.975)), 'games': len(games), 'plate_appearances': int(n.sum()),
            'arm_log_loss': float(sum(arm[g][column] for g in games) / n.sum()), 'control_log_loss': float(sum(control[g][column] for g in games) / n.sum())}


def decide(arms: list[dict], receipts: dict) -> dict:
    control = next(a['name'] for a in arms if a.get('role') == 'control')
    out = {'control': control, 'arms': {}, 'selection_rule': 'lowest 2025 tuning-block log loss (talent_plus_context) among candidate arms'}
    for a in arms:
        r = receipts.get(a['name']) or {}
        out['arms'][a['name']] = {'role': a.get('role'), 'status': r.get('status', 'missing'), 'tune_2025': (r.get('tune_2025') or {}).get('talent_plus_context'),
                                  'test_2026': r.get('test_2026'), 'train_years': (r.get('config') or {}).get('train_years'),
                                  'history_half_life_days': (r.get('config') or {}).get('history_half_life_days', 0.0),
                                  'pa_rows_by_season': r.get('pa_rows_by_season'), 'seconds': r.get('seconds'), 'error': r.get('error')}
    if (receipts.get(control) or {}).get('status') != 'completed':
        out['verdict'] = 'incomplete: the control arm did not finish'
        return out
    ok = [a['name'] for a in arms if a.get('role') == 'candidate' and (receipts.get(a['name']) or {}).get('status') == 'completed'
          and out['arms'][a['name']]['tune_2025'] is not None]
    missing = [a['name'] for a in arms if a.get('role') == 'candidate' and a['name'] not in ok]
    for name in ok:
        try:
            out['arms'][name]['vs_control_2026'] = paired(receipts[control]['test_game_losses'], receipts[name]['test_game_losses'])
        except ValueError as exc:
            # RETRAIN-04: an arm tested on another block (blend_as_test) pairs only on the games both hold
            c, a = receipts[control]['test_game_losses'], receipts[name]['test_game_losses']
            common = sorted(set(c) & set(a))
            if len(common) < 50:
                out['arms'][name]['vs_control_2026'] = {'error': str(exc)}
                continue
            r = paired({g: c[g] for g in common}, {g: a[g] for g in common})
            r['paired_on'] = 'the games both arms hold'; out['arms'][name]['vs_control_2026'] = r
    if missing:
        out['missing_candidates'] = missing
    if not ok:
        out['verdict'] = 'incomplete: no candidate arm finished'
        return out
    ok = [n for n in ok if 'error' not in out['arms'][n]['vs_control_2026']]
    if not ok:
        out['verdict'] = 'incomplete: no candidate arm shares test games with the control'
        return out
    selected = min(ok, key=lambda n: out['arms'][n]['tune_2025'])
    test = out['arms'][selected]['vs_control_2026']
    out['selected'] = selected
    promo = receipts[selected].get('promotion') or {}
    calibrated = promo.get('max_absolute_class_calibration_gap_below_0_015')
    out['gate'] = {'rule': 'the selected arm beats the control on 2026 log loss with the 95% game-bootstrap interval below zero, '
                           'and its largest class calibration gap on 2026 stays below 0.015',
                   'difference_nats_per_pa': test['difference_nats_per_pa'], 'ci_95_high': test['ci_95_high'],
                   'max_absolute_class_calibration_gap': promo.get('max_absolute_class_calibration_gap'),
                   'passed': bool(test['ci_95_high'] < 0 and calibrated is not False)}
    out['verdict'] = 'passed' if out['gate']['passed'] else 'failed'
    return out


def main():
    from brl_fit_model import read_blob, put_bytes
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data'); run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    plan = json.loads((ROOT / 'tools' / 'fit_arms.json').read_text())
    receipts = {}
    for a in plan['arms']:
        raw = read_blob(repo, token, f"research/fit-{a['name']}-{run_id}.json", branch)
        receipts[a['name']] = json.loads(raw) if raw else {}
    result = {'schema': 'brl.fit-arms-compare.v1', 'experiment': plan.get('experiment'), 'run_id': run_id, **decide(plan['arms'], receipts),
              'compared_at': datetime.now(timezone.utc).isoformat()}
    text = json.dumps(result, indent=1, default=float)
    put_bytes(repo, token, f"research/{str(plan.get('experiment', 'arms')).lower()}-compare-{run_id}.json", text.encode(), branch, 'BRL: fit arms comparison')
    print(json.dumps({k: result.get(k) for k in ('selected', 'gate', 'verdict')}, default=float))


if __name__ == '__main__':
    main()
