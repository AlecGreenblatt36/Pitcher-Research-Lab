from __future__ import annotations

from flask import Blueprint, jsonify, request
import logging

import sqlite3


from pitcher_core import (
    database_pitcher_summary,
    default_baseline_seasons,
    default_comparison_periods,
    default_transition_window,
    get_pitcher_profile,
    search_pitchers,
    DATABASE_FILE,
)
from pitcher_sync import sync_pitcher_statcast
from official_sync import sync_official_outings

pitcher_bp = Blueprint("pitchers", __name__)
logger = logging.getLogger(__name__)


@pitcher_bp.route("/api/health")
def application_health():
    try:
        connection = sqlite3.connect(DATABASE_FILE)
        pitch_rows = connection.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        pitcher_count = connection.execute(
            "SELECT COUNT(DISTINCT CAST(pitcher AS INTEGER)) FROM pitches"
        ).fetchone()[0]
        failed_ingests = connection.execute(
            "SELECT COUNT(*) FROM ingest_runs WHERE status = 'error'"
        ).fetchone()[0]
        latest_pitch_date = connection.execute(
            "SELECT MAX(game_date) FROM pitches WHERE game_type = 'R'"
        ).fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        connection.close()
        return jsonify(
            {
                "status": "ok" if integrity == "ok" else "warning",
                "database_integrity": integrity,
                "pitch_rows": int(pitch_rows or 0),
                "pitchers_cached": int(pitcher_count or 0),
                "failed_ingest_runs": int(failed_ingests or 0),
                "latest_regular_season_pitch_date": latest_pitch_date,
            }
        )
    except Exception as exc:
        logger.exception("Health check failed")
        return jsonify({"status": "error", "code": "health_check_failed", "error": "The local database health check failed."}), 500


@pitcher_bp.route("/api/pitchers/search")
def pitcher_search():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])
    try:
        return jsonify(search_pitchers(query))
    except Exception as exc:
        logger.exception("Pitcher search failed for query %r", query)
        return jsonify({"code": "pitcher_search_unavailable", "error": "MLB pitcher search is temporarily unavailable."}), 502


@pitcher_bp.route("/api/pitchers/<int:pitcher_id>/meta")
def pitcher_meta(pitcher_id: int):
    try:
        profile = get_pitcher_profile(pitcher_id)
    except Exception:
        # A stored pitcher remains usable if MLB's API is temporarily unavailable.
        profile = {"mlbam_id": pitcher_id, "name": f"MLB Pitcher {pitcher_id}"}

    base_summary = database_pitcher_summary(pitcher_id)
    requested_season = request.args.get("season", type=int)
    available_seasons = base_summary.get("seasons") or []
    target = (
        requested_season
        if requested_season in available_seasons
        else base_summary.get("current_season")
    )
    summary = database_pitcher_summary(pitcher_id, target)
    baseline = default_baseline_seasons(pitcher_id, target)
    start, end = default_transition_window(pitcher_id, target)
    comparison_periods = default_comparison_periods(pitcher_id, target)
    return jsonify(
        {
            "pitcher": profile,
            "database": summary,
            "research_defaults": {
                "target_season": target,
                "baseline_seasons": baseline,
                "transition_start": start,
                "transition_end": end,
                "comparison_periods": comparison_periods,
            },
        }
    )


@pitcher_bp.route("/api/pitchers/<int:pitcher_id>/sync", methods=["POST"])
def pitcher_sync(pitcher_id: int):
    body = request.get_json(silent=True) or {}
    force_full = bool(body.get("force_full", False))
    requested_season = body.get("season")
    try:
        requested_season = int(requested_season) if requested_season is not None else None
    except (TypeError, ValueError):
        return jsonify({"status": "error", "error": "Season must be a valid year."}), 400

    try:
        result = sync_pitcher_statcast(pitcher_id, force_full=force_full)
        try:
            result["official_outings"] = sync_official_outings(pitcher_id, requested_season)
        except Exception as official_error:
            logger.exception("Official outing update failed for pitcher %s", pitcher_id)
            result["official_outings"] = {
                "status": "error",
                "code": "official_outings_unavailable",
                "error": "Statcast updated, but official outing lines could not be refreshed.",
            }
        return jsonify(result)
    except Exception as exc:
        logger.exception("Pitcher sync failed for pitcher %s", pitcher_id)
        return jsonify({
            "status": "error",
            "code": "pitcher_sync_failed",
            "error": "Pitcher data could not be updated. Cached data remains available when present.",
        }), 502
