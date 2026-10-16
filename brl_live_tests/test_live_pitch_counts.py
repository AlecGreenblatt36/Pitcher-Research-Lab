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


def test_live_manifest_is_read_from_the_repository_not_the_work_folder():
    """The live run resolved the model manifest under the work folder, so a chain model's pitch tables were never
    assembled (October 16, first run after the switch); the manifest lives in the repository."""
    import re
    from pathlib import Path
    src = (Path(__file__).parents[1] / 'brl_live' / 'box_runner.py').read_text()
    assert "manifest_path=Path(__file__).resolve().parents[1]/'brl_engine'/'models'" in src
    assert "ROOT/'brl_engine'/'models'" not in src
    assert (Path(__file__).parents[1] / 'brl_engine' / 'models' / 'pitch02b-v2h120.json').exists()
