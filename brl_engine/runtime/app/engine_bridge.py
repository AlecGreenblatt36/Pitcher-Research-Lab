"""Paths to the private data files and the matchup decoder.

BRL_DATA_ROOT points at the decrypted data directory (the inherited runtime's layout):
  pa_model_reference/model_runs/pa_locked_2026/plate_appearances.csv.gz   locked PA history (seed)
  pa_model_reference/model_runs/pa_locked_2026/artifacts/pa_model.joblib  locked PA model
  model_runs/starter_hazard_v1/starter_hazard.joblib                       fitted starter hazard
  pa_model_reference/chadwick_mlbam_names.csv.gz                           names and birthdates
  data/official_team_results/team_results_YYYY.json.gz                    regular-season finals
"""
from __future__ import annotations
import os
from pathlib import Path

from research_lab.game_sim.models import GameMatchup, PlayerProfile, PitcherProfile, TeamProfile

DATA_ROOT = Path(os.environ.get('BRL_DATA_ROOT') or (Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'brl-data'))
HISTORY = DATA_ROOT / 'pa_model_reference/model_runs/pa_locked_2026/plate_appearances.csv.gz'
MODEL = Path(os.environ['BRL_MODEL_PATH']) if os.environ.get('BRL_MODEL_PATH') else DATA_ROOT / 'pa_model_reference/model_runs/pa_locked_2026/artifacts/pa_model.joblib'
# The locked 2026 model's hash unless the entrypoint selected another sealed model (brl_engine/model.json).
MODEL_SHA256 = os.environ.get('BRL_MODEL_SHA256') or '3c87e4deedfb5253ac81252ad7fa2f117b16457f2c383c465670e2c3ee2fa095'
MODEL_NAME = os.environ.get('BRL_MODEL_NAME') or 'locked-pa-2026-v1'
HAZARD = DATA_ROOT / 'model_runs/starter_hazard_v1/starter_hazard.joblib'
NAMES = DATA_ROOT / 'pa_model_reference/chadwick_mlbam_names.csv.gz'
TEAM_RESULTS = DATA_ROOT / 'data/official_team_results'


def _player(d: dict) -> PlayerProfile:
    return PlayerProfile(str(d['player_id']), str(d['name']), str(d.get('bats', 'R')), float(d.get('contact', 0.0)),
                         float(d.get('power', 0.0)), float(d.get('discipline', 0.0)), float(d.get('speed', 0.5)))


def _pitcher(d: dict) -> PitcherProfile:
    return PitcherProfile(str(d['player_id']), str(d['name']), str(d.get('throws', 'R')), str(d.get('role', 'reliever')),
                          float(d.get('stuff', 0.0)), float(d.get('command', 0.0)), float(d.get('contact_management', 0.0)),
                          float(d.get('stamina', 0.5)), float(d.get('leverage', 0.5)), float(d.get('rest', 1.0)),
                          int(d.get('expected_batters', 6)), int(d.get('max_batters', 10)), bool(d.get('available', True)),
                          tuple((str(k), float(v)) for k, v in (d.get('usage') or ())))


def team(t: dict) -> TeamProfile:
    return TeamProfile(str(t['team_id']), str(t['name']), tuple(_player(p) for p in t['lineup']), _pitcher(t['starter']),
                       tuple(_pitcher(p) for p in t.get('bullpen', ())), float(t.get('defense', 0.0)), float(t.get('baserunning', 0.0)))


def decode_matchup(data: dict) -> GameMatchup:
    """Inverse of dataclasses.asdict(GameMatchup)."""
    return GameMatchup(away=team(data['away']), home=team(data['home']), venue=str(data.get('venue', 'Unknown Park')),
                       park_factor=float(data.get('park_factor', 1.0)), weather_run_factor=float(data.get('weather_run_factor', 1.0)),
                       game_type=str(data.get('game_type', 'R')))
