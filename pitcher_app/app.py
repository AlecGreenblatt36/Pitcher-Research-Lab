from flask import Flask, jsonify, request, render_template
import sqlite3
import pandas as pd
import os
import socket
import threading
import webbrowser
from pathlib import Path
from change_analysis import compare_metrics, sustained_change
from comparison import ComparisonError, resolve_comparison
from research_routes import research_bp
from location_routes import location_bp
from performance_routes import performance_bp
from career_routes import career_bp
from pitcher_routes import pitcher_bp
from pitcher_core import DATABASE_FILE, current_research_season, default_baseline_seasons
# ==================================================
# Project setup
# ==================================================

project_folder = Path(__file__).resolve().parent
database_file = DATABASE_FILE

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def prevent_stale_frontend_assets(response):
    if request.path.startswith("/static/") or request.path == "/":
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response
app.register_blueprint(
    research_bp
)
app.register_blueprint(
    location_bp
)
app.register_blueprint(
    performance_bp
)
app.register_blueprint(
    career_bp
)
app.register_blueprint(
    pitcher_bp
)
# ==================================================
# Dashboard
# ==================================================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ==================================================
# Helper:
# Find first sustained change
# ==================================================

def find_first_sustained_change(current_outings, baseline_mean, baseline_std):
    return sustained_change(current_outings, baseline_mean, baseline_std)


# ==================================================
# Generalized change detection API
# ==================================================

@app.route("/api/pitchers/<int:pitcher_id>/changes")
def pitcher_changes(pitcher_id):
    season = request.args.get("season", type=int) or current_research_season(pitcher_id)
    if season is None:
        return jsonify([])
    try:
        comparison = resolve_comparison(request.args, pitcher_id, season)
    except ComparisonError as exc:
        return jsonify({"error": str(exc), "code": "invalid_comparison_periods"}), 400
    return jsonify(compare_metrics(database_file, pitcher_id, season, comparison))


@app.route("/api/pitchers/<int:pitcher_id>/release")
def pitcher_release(pitcher_id):
    season = request.args.get("season", type=int) or current_research_season(pitcher_id)
    if season is None:
        return jsonify({"error": "No regular-season Statcast data found."}), 404
    try:
        comparison = resolve_comparison(request.args, pitcher_id, season)
    except ComparisonError as exc:
        return jsonify({"error": str(exc), "code": "invalid_comparison_periods"}), 400
    pitch = request.args.get("pitch", "").upper()
    if not pitch:
        return jsonify({"error": "Select a pitch for release measurements."}), 400
    return jsonify(
        {
            "pitch_type": pitch,
            "comparison_periods": comparison.payload(),
            "measurements": compare_metrics(
                database_file, pitcher_id, season, comparison, pitch=pitch, release=True
            ),
        }
    )


# ==================================================
# Generalized synchronized outing timeline
# ==================================================

@app.route("/api/pitchers/<int:pitcher_id>/timeline")
def pitcher_timeline(pitcher_id):

    season = request.args.get("season", type=int)
    if season is None:
        season = current_research_season(pitcher_id)
    if season is None:
        return jsonify([])

    connection = sqlite3.connect(database_file)
    query = """
    SELECT
        game_date, game_pk, pitch_type, release_speed,
        release_spin_rate, release_extension, pfx_x, pfx_z,
        description, delta_run_exp
    FROM pitches
    WHERE game_type = 'R'
      AND CAST(pitcher AS INTEGER) = ?
      AND CAST(season AS INTEGER) = ?
      AND pitch_type IS NOT NULL;
    """
    data = pd.read_sql_query(
        query, connection, params=(int(pitcher_id), int(season))
    )
    connection.close()

    if data.empty:
        return jsonify([])

    numeric_columns = [
        "release_speed", "release_spin_rate", "release_extension",
        "pfx_x", "pfx_z", "delta_run_exp"
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    swing_descriptions = [
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "hit_into_play", "foul_bunt",
        "missed_bunt", "swinging_pitchout"
    ]
    whiff_descriptions = [
        "swinging_strike", "swinging_strike_blocked",
        "missed_bunt", "swinging_pitchout"
    ]
    data["is_swing"] = data["description"].isin(swing_descriptions).astype(int)
    data["is_whiff"] = data["description"].isin(whiff_descriptions).astype(int)
    data["pitcher_run_value"] = -data["delta_run_exp"]

    outing_totals = (
        data.groupby(["game_date", "game_pk"]).size()
        .reset_index(name="total_pitches")
    )
    timeline = (
        data.groupby(["game_date", "game_pk", "pitch_type"])
        .agg(
            pitch_count=("pitch_type", "size"),
            avg_velocity=("release_speed", "mean"),
            avg_spin=("release_spin_rate", "mean"),
            avg_extension=("release_extension", "mean"),
            avg_horizontal_movement=("pfx_x", "mean"),
            avg_vertical_movement=("pfx_z", "mean"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
            run_value=("pitcher_run_value", lambda values: values.sum(min_count=1)),
            run_value_pitches=("pitcher_run_value", "count"),
        )
        .reset_index()
    )
    timeline = timeline.merge(outing_totals, on=["game_date", "game_pk"], how="left")
    timeline["usage_pct"] = timeline["pitch_count"] / timeline["total_pitches"] * 100
    timeline["whiff_pct"] = float("nan")
    has_swings = timeline["swings"] > 0
    timeline.loc[has_swings, "whiff_pct"] = (
        timeline.loc[has_swings, "whiffs"]
        / timeline.loc[has_swings, "swings"] * 100
    )
    timeline["run_value_per_100"] = timeline["run_value"] / timeline["run_value_pitches"].replace(0, pd.NA) * 100
    timeline["avg_horizontal_movement"] *= 12
    timeline["avg_vertical_movement"] *= 12
    columns_to_round = [
        "avg_velocity", "avg_spin", "avg_extension",
        "avg_horizontal_movement", "avg_vertical_movement",
        "usage_pct", "whiff_pct", "run_value", "run_value_per_100"
    ]
    timeline[columns_to_round] = timeline[columns_to_round].round(2)
    timeline["season"] = int(season)
    timeline = timeline.sort_values(["game_date", "pitch_type"])
    records = timeline.astype(object).where(pd.notna(timeline), None).to_dict(orient="records")
    return jsonify(records)


# ==================================================
# Dynamic pitch-profile API
# ==================================================

@app.route("/api/pitchers/<int:pitcher_id>/pitch/<pitch_type>")
def pitcher_pitch_profile(pitcher_id, pitch_type):

    pitch_type = pitch_type.upper()

    season = request.args.get(
        "season",
        type=int
    )

    connection = sqlite3.connect(
        database_file
    )

    connection.row_factory = (
        sqlite3.Row
    )

    if season:

        query = """
        SELECT
            season,
            pitch_type,

            COUNT(*) AS pitches,

            ROUND(
                AVG(release_speed),
                2
            ) AS avg_velocity,

            ROUND(
                AVG(release_extension),
                2
            ) AS avg_extension,

            ROUND(
                AVG(release_pos_x),
                2
            ) AS avg_release_x,

            ROUND(
                AVG(release_pos_z),
                2
            ) AS avg_release_z,

            ROUND(
                AVG(release_spin_rate),
                2
            ) AS avg_spin_rate,

            ROUND(
                AVG(pfx_x) * 12,
                2
            ) AS avg_horizontal_movement,

            ROUND(
                AVG(pfx_z) * 12,
                2
            ) AS avg_vertical_movement

        FROM pitches

        WHERE
            game_type = 'R'

            AND CAST(pitcher AS INTEGER) = ?

            AND pitch_type = ?

            AND CAST(season AS INTEGER) <= ?

        GROUP BY
            season,
            pitch_type

        ORDER BY season;
        """

        rows = connection.execute(
            query,
            (
                int(pitcher_id),
                pitch_type,
                season
            )
        ).fetchall()

    else:

        query = """
        SELECT
            season,
            pitch_type,

            COUNT(*) AS pitches,

            ROUND(
                AVG(release_speed),
                2
            ) AS avg_velocity,

            ROUND(
                AVG(release_extension),
                2
            ) AS avg_extension,

            ROUND(
                AVG(release_pos_x),
                2
            ) AS avg_release_x,

            ROUND(
                AVG(release_pos_z),
                2
            ) AS avg_release_z,

            ROUND(
                AVG(release_spin_rate),
                2
            ) AS avg_spin_rate,

            ROUND(
                AVG(pfx_x) * 12,
                2
            ) AS avg_horizontal_movement,

            ROUND(
                AVG(pfx_z) * 12,
                2
            ) AS avg_vertical_movement

        FROM pitches

        WHERE
            game_type = 'R'

            AND CAST(pitcher AS INTEGER) = ?

            AND pitch_type = ?

        GROUP BY
            season,
            pitch_type

        ORDER BY season;
        """

        rows = connection.execute(
            query,
            (
                int(pitcher_id),
                pitch_type,
            )
        ).fetchall()

    connection.close()

    results = [
        dict(row)
        for row in rows
    ]

    if not results:

        return jsonify(
            {
                "error":
                    "No pitches found",

                "pitch_type":
                    pitch_type,

                "season":
                    season
            }
        ), 404

    return jsonify(results)


# ==================================================
# Start Flask
# ==================================================

if __name__ == "__main__":
    host = "127.0.0.1"
    port = int(os.environ.get("PRL_PORT", "5050"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_check:
        try:
            port_check.bind((host, port))
        except OSError as exc:
            raise SystemExit(
                f"Pitcher Research Lab could not start because port {port} is already in use. "
                "Close the other Pitcher Research Lab window and run START_HERE.bat again."
            ) from exc

    if os.environ.get("PRL_OPEN_BROWSER") == "1":
        threading.Timer(1.25, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    app.run(host=host, port=port, debug=False)
