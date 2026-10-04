from __future__ import annotations

import os

from flask import redirect

from app import app
from research_lab.game_sim.routes import game_sim_bp


if "game_sim" not in app.blueprints:
    app.register_blueprint(game_sim_bp)

_original_dashboard = app.view_functions.get("dashboard")
if _original_dashboard is not None and "pitcher_lab_dashboard" not in app.view_functions:
    app.add_url_rule(
        "/pitcher-lab",
        endpoint="pitcher_lab_dashboard",
        view_func=_original_dashboard,
    )
    app.view_functions["dashboard"] = lambda: redirect("/game-simulator")

application = app


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", os.environ.get("PRL_PORT", "5050"))),
        debug=False,
    )
