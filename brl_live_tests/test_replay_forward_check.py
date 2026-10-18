"""The replay lane names every input fitted on the season it replays (LEG-01; the BULLPEN-10 and -11 replays used the
replayed season's own leash table without saying so)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('brl_replay_fwd_t', ROOT / 'tools' / 'brl_replay.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_fit_seasons_reads_the_fields_the_fitting_tools_write():
    r = load()
    assert r.fit_seasons({'replays': "['v2-prod-trans-ctx-1000-2025']"}) == {2025}
    assert r.fit_seasons({'fitted_on': [2023, 2024, 2026]}) == {2023, 2024, 2026}
    assert r.fit_seasons({'config': {'train_years': [2023, 2024, 2025], 'validation_years': [2026]}}) == {2023, 2024, 2025, 2026}
    assert r.fit_seasons({'estimated_through': '2025-09-28'}) == {2025}
    assert r.fit_seasons({'name': 'no seasons stated'}) == set()


def test_the_bullpen_replays_of_october_17_would_have_been_flagged():
    r = load()
    params = {'season': 2026, 'leash': 'brl_replay/leash_fit_2026.json', 'offsets': 'brl_replay/context_retrain04_fit2025.json',
              'reliever_choice': 'brl_replay/reliever_choice10_fit_2023_2025.json'}
    manifest = json.loads((ROOT / 'brl_engine' / 'models' / 'retrain04-rt.json').read_text())
    found = {x['input'] for x in r.in_sample_inputs(params, 2026, manifest)}
    assert found == {'leash', 'model'}
    clean = dict(params, leash='brl_replay/leash_fit_2025.json')
    manifest67 = json.loads((ROOT / 'brl_engine' / 'models' / 'chain67-both.json').read_text())
    assert r.in_sample_inputs(clean, 2026, manifest67) == []
