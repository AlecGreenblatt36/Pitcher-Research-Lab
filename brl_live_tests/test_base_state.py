"""Base-state offsets (brl_live.provider_adjust.BaseStateAdjust, RUNS-02): cells from bases and outs, probabilities
multiplied and renormalized, the production table loads with every cell, and the builder keeps the shape by situation
while leaving the season's level out."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from brl_live.provider_adjust import BaseStateAdjust, SIM_LABELS, base_cell, load_base_state

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brl_base_state', ROOT / 'tools' / 'brl_base_state.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


class Flat:
    name = 'flat'; validation_status = 'synthetic'

    def probabilities(self, ctx):
        return dict(zip(SIM_LABELS, (.45, .22, .1, .14, .05, .03, .01)))


def ctx(bases, outs):
    return SimpleNamespace(bases=bases, outs=outs)


def test_cells_from_bases_and_outs():
    r = object()
    assert base_cell(ctx((None, None, None), 0)) == '0|0' and base_cell(ctx((r, None, None), 2)) == '1|2'
    assert base_cell(ctx((None, r, None), 1)) == '2|1' and base_cell(ctx((r, None, r), 1)) == '2|1'
    assert base_cell(ctx((True, None, None), 1)) == '1|1' and base_cell(ctx((None, None, None), 3)) == '0|2'


def test_probabilities_are_shaped_and_renormalized():
    doc = load_base_state()
    assert set(doc['table']) == {f'{b}|{o}' for b in range(3) for o in range(3)}
    adj = BaseStateAdjust(Flat(), doc)
    for bases in ((None, None, None), (True, None, None), (None, True, True)):
        for outs in range(3):
            p = adj.probabilities(ctx(bases, outs))
            assert abs(sum(p.values()) - 1) < 1e-12
            raw = np.array([.45, .22, .1, .14, .05, .03, .01]) * doc['table'][base_cell(ctx(bases, outs))]
            assert np.allclose([p[k] for k in SIM_LABELS], raw / raw.sum())
    # walks with a runner on first are the clearest miss of the stack in both seasons
    walks = {c: doc['table'][c][2] for c in doc['table']}
    assert walks['1|0'] > 1.05 and walks['0|1'] < 0.97


def test_builder_keeps_shape_and_drops_level():
    rng = np.random.default_rng(5)
    pred_share = np.array([.45, .22, .1, .14, .05, .03, .01])
    level = np.array([1.0, 1.03, 0.95, 1.0, 1.02, 0.97, 1.0])
    planted = {(1, 0): np.array([1.0, 0.97, 1.10, 1.0, 0.9, 0.9, 1.2])}
    c = {}
    for b in range(3):
        for o in range(3):
            n = 400000
            pred = pred_share * n
            shape_ = planted.get((b, o), np.ones(7))
            obs = rng.poisson(pred * level * shape_).astype(float)
            c[(b, o)] = [n, obs, pred]
    off, _ = tool.shape(c, k=2000.0)
    got = np.exp(off[(1, 0)])
    assert abs(got[2] - 1.10) < 0.03 and abs(got[1] - 0.97) < 0.03
    flat = np.exp(off[(0, 1)])
    assert np.all(np.abs(flat[:4] - 1) < 0.03)        # no shape planted: no offset, whatever the season's level
