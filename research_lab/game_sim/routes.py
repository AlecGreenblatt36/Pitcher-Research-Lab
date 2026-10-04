from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from .engine import GameSimulator
from .models import SimulationConfig
from .monte_carlo import simulate_many
from .payload import example_payload, matchup_from_payload
from .probability import RatingsProbabilityProvider


game_sim_bp = Blueprint("game_sim", __name__)


def _provider():
    configured = current_app.extensions.get("game_sim_probability_provider")
    return configured if configured is not None else RatingsProbabilityProvider()


@game_sim_bp.route("/game-simulator")
def game_simulator_page():
    return render_template("game_simulator.html")


@game_sim_bp.route("/api/game-simulator/example")
def game_simulator_example():
    return jsonify(example_payload())


@game_sim_bp.route("/api/game-simulator/health")
def game_simulator_health():
    provider = _provider()
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
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON game payload is required."}), 400
    try:
        matchup = matchup_from_payload(payload)
        simulations = int(payload.get("simulations", 1000))
        seed = int(payload.get("seed", 20261003))
        if simulations > 5000:
            raise ValueError("cloud-browser requests are limited to 5,000 simulations")
        config = SimulationConfig(
            regulation_innings=int(payload.get("regulation_innings", 9)),
            max_innings=int(payload.get("max_innings", 15)),
            automatic_runner_in_extras=bool(
                payload.get("automatic_runner_in_extras", True)
            ),
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
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)
