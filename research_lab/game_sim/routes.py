from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from .engine import GameSimulator
from .mlb_pregame import (
    PregameDataError,
    archive_snapshot,
    fetch_pregame_snapshot,
    list_games,
)
from .models import SimulationConfig
from .monte_carlo import simulate_many
from .payload import example_payload, matchup_from_payload
from .probability import RatingsProbabilityProvider
from .rules import resolve_rules


game_sim_bp = Blueprint("game_sim", __name__)


class GameSimulatorUnavailable(RuntimeError):
    pass


def _provider():
    configured = current_app.extensions.get("game_sim_probability_provider")
    if configured is not None:
        return configured
    if bool(current_app.config.get("GAME_SIM_ALLOW_DEV_PROVIDER", False)):
        return RatingsProbabilityProvider()
    raise GameSimulatorUnavailable(
        "No locked game-simulator probability provider is configured. "
        "The development ratings provider is disabled."
    )


@game_sim_bp.route("/game-simulator")
def game_simulator_page():
    return render_template("game_simulator.html")


@game_sim_bp.route("/api/game-simulator/example")
def game_simulator_example():
    return jsonify(example_payload())


@game_sim_bp.route("/api/game-simulator/games")
def game_simulator_games():
    """Official MLB schedule rows for the Game Room picker."""

    game_date = str(request.args.get("date") or datetime.now(timezone.utc).date())[:10]
    try:
        rows = list_games(game_date)
    except Exception as exc:  # requests errors are surfaced as a gateway failure.
        current_app.logger.exception("MLB schedule request failed")
        return jsonify({"error": f"MLB schedule unavailable: {exc}"}), 502
    return jsonify(
        {
            "date": game_date,
            "games": [asdict(row) for row in rows],
            "source": "MLB Stats API",
        }
    )


@game_sim_bp.route("/api/game-simulator/game/<int:game_pk>/pregame")
def game_simulator_pregame(game_pk: int):
    """Capture the lineup/starter/roof/umpire state available at request time."""

    try:
        snapshot = fetch_pregame_snapshot(game_pk)
        archived_path: str | None = None
        if str(request.args.get("archive") or "").lower() in {"1", "true", "yes"}:
            if not snapshot.forecast_valid_at_capture:
                return (
                    jsonify(
                        {
                            "error": "Snapshot was captured after first pitch and cannot be archived as a pregame forecast input.",
                            "snapshot": asdict(snapshot),
                        }
                    ),
                    409,
                )
            root = Path(
                current_app.config.get(
                    "GAME_SIM_PREGAME_ARCHIVE_ROOT",
                    "private_data/pregame_snapshots",
                )
            )
            archived_path = str(archive_snapshot(snapshot, root))
    except PregameDataError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("MLB pregame snapshot request failed")
        return jsonify({"error": f"MLB pregame snapshot unavailable: {exc}"}), 502
    payload = asdict(snapshot)
    payload["archived_path"] = archived_path
    return jsonify(payload)


@game_sim_bp.route("/api/game-simulator/health")
def game_simulator_health():
    try:
        provider = _provider()
    except GameSimulatorUnavailable as exc:
        return (
            jsonify(
                {
                    "status": "unavailable",
                    "engine": "sequential-monte-carlo-v1",
                    "error": str(exc),
                    "game_engine_validation": "not-yet-game-level-validated",
                }
            ),
            503,
        )
    return jsonify(
        {
            "status": "ok",
            "engine": "sequential-monte-carlo-v1",
            "provider": getattr(provider, "name", type(provider).__name__),
            "provider_validation": getattr(provider, "validation_status", "unknown"),
            "game_engine_validation": "not-yet-game-level-validated",
        }
    )


@game_sim_bp.route("/api/game-simulator/simulate", methods=["POST"])
def game_simulator_simulate():
    incoming = request.get_json(silent=True)
    if not isinstance(incoming, dict):
        return jsonify({"error": "A JSON game payload is required."}), 400
    payload = dict(incoming)
    try:
        rules = resolve_rules(
            game_type=str(payload.get("game_type") or "R"),
            automatic_runner_requested=bool(
                payload.get("automatic_runner_in_extras", True)
            ),
            roof_status=str(payload.get("roof_status") or "unknown"),
            weather_run_factor=payload.get("weather_run_factor", 1.0),
        )
        payload["weather_run_factor"] = rules.effective_weather_run_factor
        matchup = matchup_from_payload(payload)
        simulations = int(payload.get("simulations", 1000))
        seed = int(payload.get("seed", 20261003))
        if simulations > 5000:
            raise ValueError("cloud-browser requests are limited to 5,000 simulations")
        config = SimulationConfig(
            regulation_innings=int(payload.get("regulation_innings", 9)),
            max_innings=int(payload.get("max_innings", 15)),
            automatic_runner_in_extras=rules.automatic_runner_in_extras,
            three_batter_minimum=bool(payload.get("three_batter_minimum", True)),
            record_events=False,
        )
        simulator = GameSimulator(_provider(), config=config)
        result = simulate_many(
            simulator,
            matchup,
            simulations=simulations,
            seed=seed,
        )
        result["resolved_rules"] = asdict(rules)
    except GameSimulatorUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)
