from __future__ import annotations

from datetime import datetime, timezone

import pytest
from flask import Flask

from research_lab.game_sim.mlb_pregame import parse_live_feed, parse_schedule
from research_lab.game_sim.probability import RatingsProbabilityProvider
from research_lab.game_sim.routes import GameSimulatorUnavailable, _provider
from research_lab.game_sim.rules import (
    automatic_runner_allowed,
    effective_weather_run_factor,
    resolve_rules,
)


def test_automatic_runner_is_regular_season_only() -> None:
    assert automatic_runner_allowed("R") is True
    for game_type in ("F", "D", "L", "W", "P", "", "unknown"):
        assert automatic_runner_allowed(game_type) is False


def test_closed_roof_neutralizes_weather() -> None:
    assert effective_weather_run_factor(1.18, "closed") == 1.0
    assert effective_weather_run_factor(0.88, "Retractable Roof Closed") == 1.0
    assert effective_weather_run_factor(1.18, "open") == pytest.approx(1.18)


def test_rule_resolution_disables_postseason_runner() -> None:
    resolved = resolve_rules(
        game_type="P",
        automatic_runner_requested=True,
        roof_status="closed",
        weather_run_factor=1.14,
    )
    assert resolved.automatic_runner_in_extras is False
    assert resolved.effective_weather_run_factor == 1.0


def test_schedule_parser_builds_game_picker_rows() -> None:
    payload = {
        "dates": [
            {
                "date": "2026-10-05",
                "games": [
                    {
                        "gamePk": 900001,
                        "gameType": "D",
                        "gameDate": "2026-10-05T20:00:00Z",
                        "officialDate": "2026-10-05",
                        "status": {
                            "codedGameState": "P",
                            "detailedState": "Pre-Game",
                        },
                        "teams": {
                            "away": {
                                "team": {"id": 1, "name": "Away Club"},
                                "probablePitcher": {"id": 11, "fullName": "Away Starter"},
                            },
                            "home": {
                                "team": {"id": 2, "name": "Home Club"},
                                "probablePitcher": {"id": 22, "fullName": "Home Starter"},
                            },
                        },
                        "venue": {"id": 99, "name": "Test Park"},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                    }
                ],
            }
        ]
    }
    rows = parse_schedule(payload)
    assert len(rows) == 1
    row = rows[0]
    assert row.game_pk == 900001
    assert row.game_type == "D"
    assert row.away_probable_pitcher == {"id": 11, "name": "Away Starter"}
    assert row.home_team == "Home Club"


def _live_feed() -> dict:
    return {
        "gamePk": 900001,
        "gameData": {
            "game": {"pk": 900001, "type": "D"},
            "datetime": {
                "dateTime": "2026-10-05T20:00:00Z",
                "officialDate": "2026-10-05",
            },
            "status": {
                "abstractGameState": "Preview",
                "codedGameState": "P",
                "detailedState": "Pre-Game",
            },
            "teams": {
                "away": {"id": 1, "name": "Away Club", "abbreviation": "AWY"},
                "home": {"id": 2, "name": "Home Club", "abbreviation": "HOM"},
            },
            "probablePitchers": {
                "away": {"id": 11, "fullName": "Away Starter"},
                "home": {"id": 22, "fullName": "Home Starter"},
            },
            "venue": {
                "id": 99,
                "name": "Test Park",
                "fieldInfo": {
                    "roofType": "Retractable Roof Closed",
                    "turfType": "Grass",
                    "leftLine": 330,
                    "center": 400,
                    "rightLine": 330,
                },
            },
            "weather": {"condition": "Roof Closed", "temp": 72, "wind": "0 mph"},
        },
        "liveData": {
            "boxscore": {
                "officials": [
                    {
                        "officialType": "Home Plate",
                        "official": {"id": 500, "fullName": "Plate Umpire"},
                    }
                ],
                "teams": {
                    "away": {
                        "players": {
                            "ID1": {
                                "person": {"id": 101, "fullName": "Second Batter"},
                                "battingOrder": "200",
                                "position": {"abbreviation": "SS"},
                                "stats": {"batting": {}},
                            },
                            "ID2": {
                                "person": {"id": 100, "fullName": "Leadoff Batter"},
                                "battingOrder": "100",
                                "position": {"abbreviation": "CF"},
                                "stats": {"batting": {}},
                            },
                        }
                    },
                    "home": {"players": {}},
                },
            }
        },
    }


def test_live_feed_snapshot_preserves_capture_cutoff() -> None:
    snapshot = parse_live_feed(
        _live_feed(),
        captured_at_utc=datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc),
    )
    assert snapshot.forecast_valid_at_capture is True
    assert snapshot.game_type == "D"
    assert snapshot.roof_status == "Retractable Roof Closed"
    assert snapshot.plate_umpire == {"id": 500, "name": "Plate Umpire"}
    assert [player["player_id"] for player in snapshot.away["lineup"]] == [100, 101]
    assert snapshot.away["lineup_posted"] is True


def test_post_first_pitch_snapshot_is_not_forecast_valid() -> None:
    snapshot = parse_live_feed(
        _live_feed(),
        captured_at_utc=datetime(2026, 10, 5, 21, 0, tzinfo=timezone.utc),
    )
    assert snapshot.forecast_valid_at_capture is False


def test_probability_provider_fails_closed_without_configuration() -> None:
    app = Flask(__name__)
    with app.app_context():
        with pytest.raises(GameSimulatorUnavailable):
            _provider()
        app.config["GAME_SIM_ALLOW_DEV_PROVIDER"] = True
        assert isinstance(_provider(), RatingsProbabilityProvider)
