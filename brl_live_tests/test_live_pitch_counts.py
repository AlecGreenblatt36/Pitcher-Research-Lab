"""LEASH-04 in the live runtime (off by default): the setting selects the pitch-term correction and the matchup pitch
draws for every simulator; without the setting nothing changes."""
import pytest

from brl_live import boxscore


def test_on_since_leash04_and_the_correction_follows_the_setting():
    """LEASH-04 (October 16, 2026): the live simulators draw pitches from the chain; with the setting off the plain
    league table and LEASH-02's 32-term correction come back."""
    assert boxscore.ADJUST['pitch_counts'] == 'chain'
    off = dict(boxscore.ADJUST, pitch_counts=None)
    assert boxscore.pitch_counts_for(off, object()) is None
    base = boxscore.hazard_correction_for(dict(off, hazard_correction=True), 'R')
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


def test_settings_change_is_a_new_forecast():
    """A simulator settings change (ADJUST) is part of the forecast fingerprint, so pending games get a new version at
    the next run instead of waiting for a lineup or history change (POST-05 would otherwise have reached tonight's game
    only when its lineups posted)."""
    from pathlib import Path
    from brl_live.boxscore import ADJUST, settings_fingerprint
    a = settings_fingerprint(ADJUST)
    assert a == settings_fingerprint(dict(ADJUST)) and len(a) == 64
    changed = dict(ADJUST, postseason_exit_offset=dict(ADJUST['postseason_exit_offset'], L=0.4))
    assert settings_fingerprint(changed) != a
    src = (Path(__file__).parents[1] / 'brl_live' / 'box_runner.py').read_text()
    assert "'settings':settings_fingerprint(ADJUST)" in src


def test_refitted_table_is_a_new_forecast(tmp_path):
    """A table refitted under the same file name changes the fingerprint (REVIEW-01 follow-up); the same tables do not."""
    from brl_live.boxscore import ADJUST, settings_fingerprint, table_hashes
    (tmp_path / 'leash.json').write_text('{"base": 0.6}')
    (tmp_path / 'notes.txt').write_text('ignored')
    t1 = table_hashes(tmp_path)
    assert list(t1) == ['leash.json']
    (tmp_path / 'leash.json').write_text('{"base": 0.61}')
    t2 = table_hashes(tmp_path)
    assert settings_fingerprint(ADJUST, t1) != settings_fingerprint(ADJUST, t2)
    assert settings_fingerprint(ADJUST, t1) == settings_fingerprint(ADJUST, dict(t1))
    assert 'reliever_choice10.json' in table_hashes() and 'transitions.json' in table_hashes()
