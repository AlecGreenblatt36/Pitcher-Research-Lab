"""LONG-01: the arms are compared game by game; the arm with the best 2025 tuning score is the one tested on 2026."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('brl_fit_compare', Path(__file__).parents[1] / 'tools' / 'brl_fit_compare.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


def games(shift, seed=0, n=300):
    rng = np.random.default_rng(seed)
    out = {}
    for g in range(n):
        pa = int(rng.integers(60, 90)); base = float(rng.normal(1.46, 0.05)) * pa
        out[str(1000 + g)] = [pa, base + shift * pa + float(rng.normal(0, 0.02)) * np.sqrt(pa), base]
    return out


def test_paired_difference_and_interval():
    control = games(0.0)
    better = {g: [v[0], v[1] - 0.002 * v[0], v[2]] for g, v in control.items()}
    r = tool.paired(control, better)
    assert abs(r['difference_nats_per_pa'] + 0.002) < 1e-9 and r['ci_95_high'] < 0 and r['games'] == 300
    with pytest.raises(ValueError):
        tool.paired(control, {**better, '1': [70, 100.0, 100.0]})


def test_selection_uses_the_tuning_season_not_the_test():
    control = games(0.0)
    arms = [{'name': 'c', 'role': 'control'}, {'name': 'x', 'role': 'candidate'}, {'name': 'y', 'role': 'candidate'}, {'name': 'z', 'role': 'candidate'}]
    rec = lambda tune, shift: {'status': 'completed', 'tune_2025': {'talent_plus_context': tune}, 'config': {}, 'test_game_losses': {g: [v[0], v[1] + shift * v[0], v[2]] for g, v in control.items()}}
    receipts = {'c': rec(1.47, 0.0), 'x': rec(1.460, -0.001), 'y': rec(1.465, -0.010), 'z': {'status': 'failed', 'error': 'MemoryError'}}
    out = tool.decide(arms, receipts)
    # y is better on the test season but x had the better tuning score: x is the one tested
    assert out['selected'] == 'x' and out['gate']['passed'] and out['missing_candidates'] == ['z']
    assert out['arms']['y']['vs_control_2026']['difference_nats_per_pa'] < out['arms']['x']['vs_control_2026']['difference_nats_per_pa']
    receipts['x'] = rec(1.460, 0.0005)
    assert tool.decide(arms, receipts)['verdict'] == 'failed'
    assert tool.decide(arms, {'x': receipts['x']})['verdict'].startswith('incomplete')


def test_the_arms_file_has_one_control_and_unique_names():
    plan = json.loads((Path(__file__).parents[1] / 'tools' / 'fit_arms.json').read_text())
    names = [a['name'] for a in plan['arms']]
    assert len(names) == len(set(names)) and sum(a['role'] == 'control' for a in plan['arms']) == 1
    assert all(a['settings']['name'] == a['name'] and a['settings'].get('manifest_branch') == 'ledger' for a in plan['arms'])
