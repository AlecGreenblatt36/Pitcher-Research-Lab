from __future__ import annotations

from typing import Any, Mapping

from .models import GameMatchup, PitcherProfile, PlayerProfile, TeamProfile


def _player(payload: Mapping[str, Any], fallback_id: str) -> PlayerProfile:
    return PlayerProfile(
        player_id=str(payload.get("player_id") or fallback_id),
        name=str(payload.get("name") or fallback_id),
        bats=str(payload.get("bats") or "R"),
        contact=float(payload.get("contact", 0.0)),
        power=float(payload.get("power", 0.0)),
        discipline=float(payload.get("discipline", 0.0)),
        speed=float(payload.get("speed", 0.5)),
    )


def _pitcher(payload: Mapping[str, Any], fallback_id: str, role: str) -> PitcherProfile:
    is_starter = role == "starter"
    return PitcherProfile(
        player_id=str(payload.get("player_id") or fallback_id),
        name=str(payload.get("name") or fallback_id),
        throws=str(payload.get("throws") or "R"),
        role=str(payload.get("role") or role),
        stuff=float(payload.get("stuff", 0.0)),
        command=float(payload.get("command", 0.0)),
        contact_management=float(payload.get("contact_management", 0.0)),
        stamina=float(payload.get("stamina", 0.78 if is_starter else 0.35)),
        leverage=float(payload.get("leverage", 0.45 if is_starter else 0.55)),
        rest=float(payload.get("rest", 1.0)),
        expected_batters=int(payload.get("expected_batters", 22 if is_starter else 4)),
        max_batters=int(payload.get("max_batters", 30 if is_starter else 8)),
        available=bool(payload.get("available", True)),
    )


def _team(payload: Mapping[str, Any], side: str) -> TeamProfile:
    lineup_payload = payload.get("lineup")
    if not isinstance(lineup_payload, list) or len(lineup_payload) != 9:
        raise ValueError(f"{side} lineup must contain exactly nine hitters")
    starter_payload = payload.get("starter")
    if not isinstance(starter_payload, Mapping):
        raise ValueError(f"{side} starter is required")
    bullpen_payload = payload.get("bullpen", [])
    if not isinstance(bullpen_payload, list):
        raise ValueError(f"{side} bullpen must be a list")

    team_id = str(payload.get("team_id") or side)
    return TeamProfile(
        team_id=team_id,
        name=str(payload.get("name") or side.title()),
        lineup=tuple(
            _player(player, f"{team_id}-hitter-{index + 1}")
            for index, player in enumerate(lineup_payload)
        ),
        starter=_pitcher(starter_payload, f"{team_id}-starter", "starter"),
        bullpen=tuple(
            _pitcher(pitcher, f"{team_id}-reliever-{index + 1}", "reliever")
            for index, pitcher in enumerate(bullpen_payload)
        ),
        defense=float(payload.get("defense", 0.0)),
        baserunning=float(payload.get("baserunning", 0.0)),
    )


def matchup_from_payload(payload: Mapping[str, Any]) -> GameMatchup:
    away = payload.get("away")
    home = payload.get("home")
    if not isinstance(away, Mapping) or not isinstance(home, Mapping):
        raise ValueError("payload must include away and home team objects")
    return GameMatchup(
        away=_team(away, "away"),
        home=_team(home, "home"),
        venue=str(payload.get("venue") or "Unknown Park"),
        park_factor=float(payload.get("park_factor", 1.0)),
        weather_run_factor=float(payload.get("weather_run_factor", 1.0)),
        game_type=str(payload.get("game_type") or "R"),
    )


def example_payload() -> dict[str, Any]:
    def lineup(prefix: str) -> list[dict[str, Any]]:
        return [
            {
                "player_id": f"{prefix}-h{index}",
                "name": f"{prefix} Hitter {index}",
                "bats": "L" if index in {1, 4, 7} else "R",
                "contact": round((index - 5) * 0.05, 2),
                "power": round((5 - index) * 0.04, 2),
                "discipline": 0.0,
                "speed": round(0.42 + (index % 4) * 0.10, 2),
            }
            for index in range(1, 10)
        ]

    def bullpen(prefix: str) -> list[dict[str, Any]]:
        roles = ["long", "middle", "middle", "setup", "setup", "closer"]
        return [
            {
                "player_id": f"{prefix}-p{index}",
                "name": f"{prefix} Reliever {index}",
                "role": role,
                "throws": "L" if index in {2, 5} else "R",
                "stuff": 0.10 + 0.08 * index,
                "command": 0.05 + 0.04 * index,
                "contact_management": 0.04 + 0.05 * index,
                "stamina": 0.30,
                "leverage": 0.35 + 0.09 * index,
                "rest": 1.0,
                "expected_batters": 5 if role == "long" else 4,
                "max_batters": 10 if role == "long" else 7,
            }
            for index, role in enumerate(roles, start=1)
        ]

    return {
        "away": {
            "team_id": "sd-demo",
            "name": "San Diego (demo ratings)",
            "lineup": lineup("SD"),
            "starter": {
                "player_id": "sd-sp",
                "name": "San Diego Starter",
                "role": "starter",
                "throws": "R",
                "stuff": 0.35,
                "command": 0.20,
                "contact_management": 0.18,
                "stamina": 0.80,
                "expected_batters": 22,
                "max_batters": 29,
            },
            "bullpen": bullpen("SD"),
            "defense": 0.10,
            "baserunning": 0.12,
        },
        "home": {
            "team_id": "mil-demo",
            "name": "Milwaukee (demo ratings)",
            "lineup": lineup("MIL"),
            "starter": {
                "player_id": "mil-sp",
                "name": "Milwaukee Starter",
                "role": "starter",
                "throws": "R",
                "stuff": 0.42,
                "command": 0.12,
                "contact_management": 0.20,
                "stamina": 0.82,
                "expected_batters": 23,
                "max_batters": 30,
            },
            "bullpen": bullpen("MIL"),
            "defense": 0.16,
            "baserunning": 0.06,
        },
        "venue": "Demo Park",
        "park_factor": 1.00,
        "weather_run_factor": 1.00,
        "game_type": "R",
        "simulations": 1000,
        "seed": 20261003,
        "data_status": "Synthetic demo ratings; replace with live MLB/Statcast adapter.",
    }
