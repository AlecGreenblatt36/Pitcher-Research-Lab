"""Day form (brl_live.provider_adjust.DayForm, DISP-01): one shock per batting team and world, reproducible by seed,
centered so the average chance of reaching base is unchanged, and off at sd 0."""
from types import SimpleNamespace

import numpy as np

from brl_live.provider_adjust import DayForm, SIM_LABELS


class Flat:
    name = 'flat'; validation_status = 'synthetic'

    def probabilities(self, ctx):
        return dict(zip(SIM_LABELS, (.45, .22, .1, .14, .05, .03, .01)))


def reach(p):
    return 1 - p['bip_out'] - p['strikeout']


def test_off_and_reproducible():
    home, away = SimpleNamespace(batting_side='home'), SimpleNamespace(batting_side='away')
    d0 = DayForm(Flat(), 0.0); d0.new_world(5)
    assert d0.probabilities(home) == Flat().probabilities(home)
    d = DayForm(Flat(), 0.12); d.new_world(11); a = (d.probabilities(home), d.probabilities(away))
    d.new_world(12); d.new_world(11); b = (d.probabilities(home), d.probabilities(away))
    assert a == b and a[0] != a[1] and abs(sum(a[0].values()) - 1) < 1e-12


def test_centered_and_spread():
    d = DayForm(Flat(), 0.12); home = SimpleNamespace(batting_side='home')
    r = []
    for seed in range(4000):
        d.new_world(seed); r.append(reach(d.probabilities(home)))
    base = reach(Flat().probabilities(home))
    assert abs(np.mean(r) - base) < 0.004 and 0.025 < np.std(r) < 0.05


def test_production_hook():
    from brl_live.boxscore import ADJUST, adjusted_provider
    p, hook, label = adjusted_provider(Flat(), None, '2026-06-01', dict(ADJUST, context_offsets=False, environment=False, team_offsets=False, day_form_sigma=0.1))
    assert hook is not None and any('day form' in x for x in label)
    p, hook, label = adjusted_provider(Flat(), None, '2026-06-01', dict(ADJUST, context_offsets=False, environment=False, team_offsets=False))
    assert hook is None
