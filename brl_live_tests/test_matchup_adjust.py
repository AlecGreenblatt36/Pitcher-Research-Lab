"""Decision-moment matchup layer (brl_live.provider_adjust.MatchupAdjust): strikeouts and walks move by the fitted
log-odds per point of the pair's chase and zone-swing deviations, the other outcomes keep their proportions, and pairs
outside the table are untouched."""
import math
from types import SimpleNamespace

from brl_live.provider_adjust import MatchupAdjust, SIM_LABELS


class Flat:
    name = 'flat'; validation_status = 'test'

    def probabilities(self, ctx):
        return dict(zip(SIM_LABELS, (0.45, 0.22, 0.09, 0.14, 0.05, 0.03, 0.02)))


def ctx(b, p):
    return SimpleNamespace(batter=SimpleNamespace(player_id=b), pitcher=SimpleNamespace(player_id=p))


def test_matchup_moves_strikeouts_and_walks_only_for_listed_pairs():
    m = MatchupAdjust(Flat(), {('1', '9'): (3.0, -1.0)}, {'K': (0.02, 0.0), 'BB': (-0.03, 0.0)})
    base = Flat().probabilities(None)
    assert m.probabilities(ctx('2', '9')) == base
    q = m.probabilities(ctx('1', '9'))
    lo = lambda x: math.log(x / (1 - x))
    assert abs(lo(q['strikeout']) - lo(base['strikeout']) - 0.06) < 1e-9
    assert abs(lo(q['bb_hbp']) - lo(base['bb_hbp']) + 0.09) < 1e-9
    assert abs(sum(q.values()) - 1) < 1e-12
    r = q['single'] / q['home_run']; assert abs(r - base['single'] / base['home_run']) < 1e-12
