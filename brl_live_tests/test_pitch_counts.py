"""LEASH-03: pitches per plate appearance in the engine (research_lab.game_sim.pitch_counts) and the pitch terms of the
starter-exit correction. Off, the engine draws exactly as before; on, with paired streams, the pitches come from their
own stream, so a simulator whose policy does not read them plays the same games; the stage-1 fit runs end to end."""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from research_lab.game_sim import starter_hazard as sh
from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.manager import ManagerPolicy
from research_lab.game_sim.models import SimulationConfig
from research_lab.game_sim.pitch_counts import PitchCounts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from test_paired_streams import Provider, matchup  # noqa: E402

DOC = {'labels': ['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH'],
       'counts': {'BIP_OUT': [0, 30, 30, 20, 10, 10], 'K': [0, 0, 0, 30, 40, 30], 'BB_HBP': [5, 0, 0, 0, 40, 55], '1B': [0, 30, 30, 20, 10, 10],
                  '2B_3B': [0, 30, 30, 20, 10, 10], 'HR': [0, 30, 30, 20, 10, 10], 'OTHER_REACH': [0, 30, 30, 20, 10, 10]}}


def test_draws_follow_the_outcome_table():
    pc = PitchCounts(DOC)
    assert pc.draw('strikeout', 0.0) == 3 and pc.draw('strikeout', 0.31) == 4 and pc.draw('strikeout', 0.999) == 5
    assert pc.draw('bb_hbp', 0.01) == 0 and abs(pc.mean('strikeout') - 4.0) < 1e-9


def test_engine_pitches_from_their_own_stream_leave_the_games_alone():
    config = SimulationConfig(max_innings=100, max_plate_appearances=4000)
    plain = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(), paired_streams=True)
    counted = GameSimulator(Provider(), config, manager_policy=ManagerPolicy(), paired_streams=True, pitch_counts=PitchCounts(DOC))
    for seed in range(12):
        a, b = plain.simulate(matchup(), seed), counted.simulate(matchup(), seed)
        assert (a.away_score, a.home_score) == (b.away_score, b.home_score)
        for pid, ln in b.pitcher_lines.items():
            assert ln['batters_faced'] == a.pitcher_lines[pid]['batters_faced'] and a.pitcher_lines[pid]['pitches'] == 0
            assert ln['batters_faced'] <= ln['pitches'] <= 5 * ln['batters_faced']
        assert b.to_dict() == counted.simulate(matchup(), seed).to_dict()          # the same seed, the same pitches
    # a single stream (the live engine) draws exactly as before when the counts are off
    one = GameSimulator(Provider(), config, manager_policy=ManagerPolicy())
    assert one.simulate(matchup(), 3).to_dict() == one.simulate(matchup(), 3).to_dict()


def test_pitch_terms_follow_the_band_and_the_inning_end():
    z = sh.correction_terms_v2(5, True, 1, 0, 97)
    names = [n for n, v in zip(sh.CORRECTION_TERMS_V2, z) if v]
    assert names == ['i5e_ahead', 'p95_104e']
    assert [n for n, v in zip(sh.CORRECTION_TERMS_V2, sh.correction_terms_v2(2, False, 0, 3, 41)) if v] == ['plt60m']
    assert sh.pitch_band(105) == 5 and sh.pitch_band(104.9) == 4 and sh.pitch_band(59) == 0


def test_stage_one_fit_runs_end_to_end(tmp_path):
    spec = importlib.util.spec_from_file_location('brl_research', ROOT / 'tools/brl_research.py')
    br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)
    from test_hazard_check import _games
    rng = np.random.default_rng(21)
    pa = _games(rng, n=300)
    pa['outcome'] = rng.choice(['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH'], len(pa))
    pa.to_csv(tmp_path / 'plate_appearances.csv.gz', index=False, compression='gzip')
    from brl_replay.games import load_history
    h = load_history(tmp_path / 'plate_appearances.csv.gz')
    d, starts = sh.build_decisions(h)
    league = float(starts['bf_total'].mean())
    d = sh.add_tendencies(d, starts, league)
    joblib.dump({'model': sh.fit_hazard(d, 1.0), 'league_mean_bf': league}, tmp_path / 'starter_hazard.joblib')
    phys = pa[['game_pk', 'at_bat_number']].assign(n=rng.integers(1, 8, len(pa)))
    out = br.pitch_counts_fit(tmp_path, phys, train=(2025,))
    assert out['terms'] == list(sh.CORRECTION_TERMS_V2) and len(out['coef']) == 44
    assert set(out['by_season']) == {'2025', '2026'} and len(out['table']['counts']['K']) == 21
    PitchCounts(out['table'])                                                      # the table the engine reads


def test_live_correction_file_must_be_leash02s(tmp_path, monkeypatch):
    """LEASH-03 failed, so the live correction keeps LEASH-02's 32 terms; a file with pitch terms is refused, since the box
    worlds, the market worlds and live updates build their simulators without the pitch draws those terms read."""
    from brl_live import boxscore as bx
    rep = json.loads((ROOT / 'brl_replay/hazard_correction_v2.json').read_text())
    assert len(rep['terms']) == 44
    f = tmp_path / 'hazard_correction.json'; f.write_text(json.dumps(rep))
    monkeypatch.setattr(bx, 'HAZARD_CORRECTION_PATH', f)
    monkeypatch.delitem(bx._KERNEL, 'hc', raising=False)
    import pytest
    plain = dict(bx.ADJUST, pitch_counts=None)                                  # LEASH-04 off: the simulators draw no pitches
    with pytest.raises(ValueError, match='32 terms'):
        bx.hazard_correction_for(plain, 'R')
    assert bx.hazard_correction_for(plain, 'F') is None                        # postseason games never read the file
    monkeypatch.delitem(bx._KERNEL, 'hc', raising=False)
    assert len(json.loads((ROOT / 'brl_live/hazard_correction.json').read_text())['terms']) == 32
