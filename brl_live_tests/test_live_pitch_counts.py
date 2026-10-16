"""LEASH-04 in the live runtime (off by default): the setting selects the pitch-term correction and the matchup pitch
draws for every simulator; without the setting nothing changes."""
import pytest

from brl_live import boxscore


def test_off_by_default_and_the_correction_follows_the_setting():
    assert boxscore.ADJUST['pitch_counts'] is None
    assert boxscore.pitch_counts_for(boxscore.ADJUST, object()) is None
    base = boxscore.hazard_correction_for(dict(boxscore.ADJUST, hazard_correction=True), 'R')
    assert base is None or len(base['terms']) == 32
    on = dict(boxscore.ADJUST, hazard_correction=True, pitch_counts='chain')
    v2 = boxscore.hazard_correction_for(on, 'R')
    assert len(v2['terms']) == 44 and len(v2['coef']) == 44 and boxscore.hazard_correction_for(on, 'D') is None


def test_chain_pitch_counts_need_a_chain_model():
    on = dict(boxscore.ADJUST, pitch_counts='chain')
    class NoChain:
        chain = None
    with pytest.raises(ValueError):
        boxscore.pitch_counts_for(on, NoChain())
    class WithChain:
        chain = object(); park = 'NYY'; game_date = '2026-05-01'
    pc = boxscore.pitch_counts_for(on, WithChain())
    assert pc.wants_context and pc.fallback.mean('strikeout') > 4
