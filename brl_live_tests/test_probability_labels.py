"""normalize_probabilities refuses labels it does not know (REVIEW-01 follow-up): a provider that returned the model's labels
or a misspelt one used to lose that mass silently and renormalize the rest."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
from research_lab.game_sim.probability import normalize_probabilities  # noqa: E402
from research_lab.game_sim.models import OUTCOME_LABELS  # noqa: E402


def test_known_labels_normalize():
    p = normalize_probabilities({label: 1.0 for label in OUTCOME_LABELS})
    assert abs(sum(p.values()) - 1) < 1e-12 and set(p) == set(OUTCOME_LABELS)
    q = normalize_probabilities({'bip_out': 3.0, 'strikeout': 1.0})          # missing labels count as zero
    assert q['bip_out'] > q['strikeout'] > q['home_run']


def test_unknown_labels_refused():
    with pytest.raises(ValueError, match='unknown outcome labels'):
        normalize_probabilities({'bip_out': 0.5, 'K': 0.2, 'strikeout': 0.3})
    with pytest.raises(ValueError, match='unknown outcome labels'):
        normalize_probabilities({'BIP_OUT': 1.0})
