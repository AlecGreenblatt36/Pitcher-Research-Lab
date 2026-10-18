from __future__ import annotations

from contextlib import closing

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_FOLDER = Path(__file__).resolve().parent
DATABASE_FILE = Path(os.environ.get("PRL_DATABASE", PROJECT_FOLDER / "data" / "pitcher_research.db"))
MLB_API = "https://statsapi.mlb.com/api/v1"


def connect_database() -> sqlite3.Connection:
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_FILE, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def ensure_core_schema(connection: sqlite3.Connection | None = None) -> None:
    owns_connection = connection is None
    if connection is None:
        connection = connect_database()

    pitches_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pitches'"
    ).fetchone()
    if pitches_exists is None:
        schema_file = PROJECT_FOLDER / "schema.sql"
        if not schema_file.exists():
            raise RuntimeError("Pitch cache schema is missing.")
        connection.executescript(schema_file.read_text(encoding="utf-8"))

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pitchers (
            mlbam_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            team_id INTEGER,
            team_name TEXT,
            team_abbreviation TEXT,
            pitch_hand TEXT,
            primary_position TEXT,
            active INTEGER,
            mlb_debut_date TEXT,
            first_statcast_date TEXT,
            last_statcast_date TEXT,
            last_profile_refresh TEXT,
            last_statcast_sync TEXT,
            last_official_sync TEXT
        );
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pitcher_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            fetch_start TEXT,
            fetch_end TEXT,
            rows_fetched INTEGER DEFAULT 0,
            rows_written INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            error TEXT
        );
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS official_outings (
            game_pk INTEGER NOT NULL,
            pitcher_id INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            season INTEGER,
            team TEXT,
            opponent TEXT,
            home_away TEXT,
            innings_pitched TEXT,
            outs_recorded INTEGER,
            hits INTEGER,
            runs INTEGER,
            earned_runs INTEGER,
            walks INTEGER,
            intentional_walks INTEGER,
            hit_by_pitch INTEGER,
            strikeouts INTEGER,
            home_runs INTEGER,
            batters_faced INTEGER,
            pitches INTEGER,
            strikes INTEGER,
            balls INTEGER,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (game_pk, pitcher_id)
        );
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, ?)",
        (datetime.now().astimezone().isoformat(timespec="seconds"),),
    )

    # These indexes matter once the database contains many pitchers.
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pitches_pitcher_date
        ON pitches (pitcher, game_date);
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pitches_pitcher_season_type
        ON pitches (pitcher, season, pitch_type);
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_official_outings_pitcher_date
        ON official_outings (pitcher_id, game_date);
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_pitch_identity
        ON pitches (pitcher, game_pk, at_bat_number, pitch_number)
        WHERE pitcher IS NOT NULL
          AND game_pk IS NOT NULL
          AND at_bat_number IS NOT NULL
          AND pitch_number IS NOT NULL;
        """
    )

    # Register any pitch data imported before the metadata table existed.
    connection.execute(
        """
        INSERT OR IGNORE INTO pitchers (
            mlbam_id, full_name, pitch_hand,
            first_statcast_date, last_statcast_date,
            last_profile_refresh
        )
        SELECT
            CAST(pitcher AS INTEGER),
            COALESCE(MAX(NULLIF(TRIM(player_name), '')), 'MLB Pitcher ' || CAST(pitcher AS INTEGER)),
            MAX(p_throws), MIN(game_date), MAX(game_date), ?
        FROM pitches
        WHERE pitcher IS NOT NULL
        GROUP BY CAST(pitcher AS INTEGER)
        """,
        (datetime.now().astimezone().isoformat(timespec="seconds"),),
    )

    connection.commit()
    if owns_connection:
        connection.close()


def _request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "PitcherResearchLab/2.0",
            "Accept": "application/json",
        },
    )
    response.raise_for_status()
    return response.json()


def search_pitchers(name: str) -> list[dict[str, Any]]:
    """Search MLB's people endpoint by name.

    The MLB Stats API is undocumented but is the same public API already used
    elsewhere in this project for game boxscores.
    """
    query = (name or "").strip()
    if len(query) < 2:
        return []

    payload = _request_json(
        f"{MLB_API}/people/search",
        params={"names": query, "hydrate": "currentTeam"},
    )

    results: list[dict[str, Any]] = []
    pitcher_positions = {"P", "TWP", "Pitcher", "Two-Way Player"}

    for person in payload.get("people", []):
        position = person.get("primaryPosition") or {}
        position_value = position.get("abbreviation") or position.get("name")
        if position_value not in pitcher_positions:
            continue

        pitch_hand = (person.get("pitchHand") or {}).get("code")
        team = person.get("currentTeam") or {}
        results.append(
            {
                "mlbam_id": person.get("id"),
                "name": person.get("fullName"),
                "first_name": person.get("firstName"),
                "last_name": person.get("lastName"),
                "pitch_hand": pitch_hand,
                "position": position_value,
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "active": person.get("active"),
                "mlb_debut_date": person.get("mlbDebutDate"),
            }
        )

    results.sort(
        key=lambda row: (
            0 if row.get("active") else 1,
            row.get("name") or "",
        )
    )
    return results[:12]


def fetch_pitcher_profile(pitcher_id: int) -> dict[str, Any]:
    payload = _request_json(
        f"{MLB_API}/people/{int(pitcher_id)}",
        params={"hydrate": "currentTeam"},
    )
    people = payload.get("people") or []
    if not people:
        raise LookupError(f"MLB player {pitcher_id} was not found.")

    person = people[0]
    team = person.get("currentTeam") or {}
    pitch_hand = person.get("pitchHand") or {}
    position = person.get("primaryPosition") or {}

    profile = {
        "mlbam_id": int(person["id"]),
        "name": person.get("fullName") or f"MLB Player {pitcher_id}",
        "first_name": person.get("firstName"),
        "last_name": person.get("lastName"),
        "team_id": team.get("id"),
        "team_name": team.get("name"),
        "team_abbreviation": team.get("abbreviation") or team.get("teamCode"),
        "pitch_hand": pitch_hand.get("code"),
        "position": position.get("abbreviation") or position.get("name"),
        "active": person.get("active"),
        "mlb_debut_date": person.get("mlbDebutDate"),
    }
    upsert_pitcher(profile)
    return profile


def upsert_pitcher(profile: dict[str, Any]) -> None:
    ensure_core_schema()
    with closing(connect_database()) as connection:
        connection.execute(
            """
            INSERT INTO pitchers (
                mlbam_id, full_name, first_name, last_name,
                team_id, team_name, team_abbreviation,
                pitch_hand, primary_position, active, mlb_debut_date,
                last_profile_refresh
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mlbam_id) DO UPDATE SET
                full_name = excluded.full_name,
                first_name = excluded.first_name,
                last_name = excluded.last_name,
                team_id = excluded.team_id,
                team_name = excluded.team_name,
                team_abbreviation = excluded.team_abbreviation,
                pitch_hand = excluded.pitch_hand,
                primary_position = excluded.primary_position,
                active = excluded.active,
                mlb_debut_date = excluded.mlb_debut_date,
                last_profile_refresh = excluded.last_profile_refresh
            """,
            (
                profile["mlbam_id"],
                profile["name"],
                profile.get("first_name"),
                profile.get("last_name"),
                profile.get("team_id"),
                profile.get("team_name"),
                profile.get("team_abbreviation"),
                profile.get("pitch_hand"),
                profile.get("position"),
                int(bool(profile.get("active"))) if profile.get("active") is not None else None,
                profile.get("mlb_debut_date"),
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()


def get_pitcher_profile(pitcher_id: int, refresh: bool = False) -> dict[str, Any]:
    ensure_core_schema()
    with closing(connect_database()) as connection:
        row = connection.execute(
            "SELECT * FROM pitchers WHERE mlbam_id = ?",
            (int(pitcher_id),),
        ).fetchone()

    if row is not None and not refresh:
        data = dict(row)
        return {
            "mlbam_id": data["mlbam_id"],
            "name": data["full_name"],
            "first_name": data.get("first_name"),
            "last_name": data.get("last_name"),
            "team_id": data.get("team_id"),
            "team_name": data.get("team_name"),
            "team_abbreviation": data.get("team_abbreviation"),
            "pitch_hand": data.get("pitch_hand"),
            "position": data.get("primary_position"),
            "active": bool(data["active"]) if data.get("active") is not None else None,
            "mlb_debut_date": data.get("mlb_debut_date"),
            "first_statcast_date": data.get("first_statcast_date"),
            "last_statcast_date": data.get("last_statcast_date"),
            "last_statcast_sync": data.get("last_statcast_sync"),
            "last_official_sync": data.get("last_official_sync"),
        }

    return fetch_pitcher_profile(int(pitcher_id))


def get_pitcher_seasons(pitcher_id: int) -> list[int]:
    with closing(connect_database()) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT CAST(season AS INTEGER)
            FROM pitches
            WHERE CAST(pitcher AS INTEGER) = ?
              AND game_type = 'R'
              AND season IS NOT NULL
            ORDER BY CAST(season AS INTEGER)
            """,
            (int(pitcher_id),),
        ).fetchall()
    return [int(row[0]) for row in rows if row[0] is not None]


def current_research_season(pitcher_id: int) -> int | None:
    seasons = get_pitcher_seasons(pitcher_id)
    return max(seasons) if seasons else None


def research_date_bounds(pitcher_id: int, season: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    with closing(connect_database()) as connection:
        row = connection.execute(
            """
            SELECT MIN(game_date), MAX(game_date)
            FROM pitches
            WHERE CAST(pitcher AS INTEGER) = ?
              AND game_type = 'R'
              AND CAST(season AS INTEGER) = ?
            """,
            (int(pitcher_id), int(season)),
        ).fetchone()
    if not row or not row[0] or not row[1]:
        return None, None
    return pd.Timestamp(row[0]), pd.Timestamp(row[1])


def research_window_is_within_season(
    pitcher_id: int,
    season: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    first_date, last_date = research_date_bounds(pitcher_id, season)
    return bool(first_date is not None and last_date is not None and first_date <= start <= end <= last_date)


def research_window_is_within_career(
    pitcher_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    with closing(connect_database()) as connection:
        row = connection.execute(
            """
            SELECT MIN(game_date), MAX(game_date)
            FROM pitches
            WHERE CAST(pitcher AS INTEGER) = ? AND game_type = 'R'
            """,
            (int(pitcher_id),),
        ).fetchone()
    if not row or not row[0] or not row[1]:
        return False
    return bool(pd.Timestamp(row[0]) <= start <= end <= pd.Timestamp(row[1]))


def default_baseline_seasons(pitcher_id: int, target_season: int | None = None) -> list[int]:
    seasons = get_pitcher_seasons(pitcher_id)
    if not seasons:
        return []
    target = target_season or max(seasons)
    prior = [season for season in seasons if season < target]
    return prior[-2:]


def default_transition_window(pitcher_id: int, target_season: int | None = None) -> tuple[str | None, str | None]:
    """Return a neutral, data-derived window for generic screens.

    This is intentionally *not* labeled a detected change point. It simply
    divides the target season around the 1/3 and 2/3 outing marks until the
    dedicated change detector supplies a better candidate.
    """
    target = target_season or current_research_season(pitcher_id)
    if target is None:
        return None, None

    with closing(connect_database()) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT game_date
            FROM pitches
            WHERE CAST(pitcher AS INTEGER) = ?
              AND game_type = 'R'
              AND CAST(season AS INTEGER) = ?
            ORDER BY game_date
            """,
            (int(pitcher_id), int(target)),
        ).fetchall()

    dates = [str(row[0]) for row in rows if row[0]]
    if not dates:
        return None, None
    if len(dates) == 1:
        return dates[0], dates[0]
    start_index = max(0, min(len(dates) - 1, len(dates) // 3))
    end_index = max(start_index, min(len(dates) - 1, (2 * len(dates)) // 3))
    return dates[start_index], dates[end_index]


def default_comparison_periods(pitcher_id: int, target_season: int | None = None) -> dict[str, str | None]:
    """Return non-overlapping pitcher-specific baseline and comparison periods."""
    target = target_season or current_research_season(pitcher_id)
    if target is None:
        return {
            "baseline_start": None,
            "baseline_end": None,
            "comparison_start": None,
            "comparison_end": None,
            "source": "unavailable",
        }

    baseline_seasons = default_baseline_seasons(pitcher_id, target)
    comparison_start, comparison_end = research_date_bounds(pitcher_id, target)
    if comparison_start is None or comparison_end is None:
        return {
            "baseline_start": None,
            "baseline_end": None,
            "comparison_start": None,
            "comparison_end": None,
            "source": "unavailable",
        }

    if baseline_seasons:
        with closing(connect_database()) as connection:
            row = connection.execute(
                """
                SELECT MIN(game_date), MAX(game_date)
                FROM pitches
                WHERE CAST(pitcher AS INTEGER) = ?
                  AND game_type = 'R'
                  AND CAST(season AS INTEGER) BETWEEN ? AND ?
                """,
                (int(pitcher_id), min(baseline_seasons), max(baseline_seasons)),
            ).fetchone()
        if row and row[0] and row[1]:
            return {
                "baseline_start": str(row[0]),
                "baseline_end": str(row[1]),
                "comparison_start": comparison_start.strftime("%Y-%m-%d"),
                "comparison_end": comparison_end.strftime("%Y-%m-%d"),
                "source": "historical_baseline",
            }

    with closing(connect_database()) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT game_date
            FROM pitches
            WHERE CAST(pitcher AS INTEGER) = ?
              AND game_type = 'R'
              AND CAST(season AS INTEGER) = ?
            ORDER BY game_date
            """,
            (int(pitcher_id), int(target)),
        ).fetchall()
    dates = [str(row[0]) for row in rows if row[0]]
    if len(dates) < 2:
        only = dates[0] if dates else None
        return {
            "baseline_start": only,
            "baseline_end": only,
            "comparison_start": None,
            "comparison_end": None,
            "source": "insufficient_outings",
        }

    segment = max(1, len(dates) // 3)
    comparison_index = max(segment, len(dates) - segment)
    comparison_index = min(comparison_index, len(dates) - 1)
    return {
        "baseline_start": dates[0],
        "baseline_end": dates[segment - 1],
        "comparison_start": dates[comparison_index],
        "comparison_end": dates[-1],
        "source": "rookie_outing_split",
    }


def comparison_periods_are_valid(
    pitcher_id: int,
    baseline_start: pd.Timestamp,
    baseline_end: pd.Timestamp,
    comparison_start: pd.Timestamp,
    comparison_end: pd.Timestamp,
) -> bool:
    """Validate ordered, non-overlapping periods against cached career coverage."""
    if not (
        baseline_start <= baseline_end
        and comparison_start <= comparison_end
        and baseline_end < comparison_start
    ):
        return False
    return research_window_is_within_career(
        pitcher_id,
        baseline_start,
        comparison_end,
    )


def database_pitcher_summary(pitcher_id: int, arsenal_season: int | None = None) -> dict[str, Any]:
    with closing(connect_database()) as connection:
        row = connection.execute(
            """
            SELECT
                MIN(game_date),
                MAX(game_date),
                COUNT(*),
                COUNT(DISTINCT game_pk),
                SUM(CASE WHEN release_speed IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN release_pos_x IS NOT NULL AND release_pos_z IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN pfx_x IS NOT NULL AND pfx_z IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN release_spin_rate IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN plate_x IS NOT NULL AND plate_z IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN launch_speed IS NOT NULL THEN 1 ELSE 0 END)
            FROM pitches
            WHERE CAST(pitcher AS INTEGER) = ?
            """,
            (int(pitcher_id),),
        ).fetchone()
    seasons = get_pitcher_seasons(pitcher_id)
    current_season = max(seasons) if seasons else None
    selected_season = (
        int(arsenal_season)
        if arsenal_season is not None and int(arsenal_season) in seasons
        else current_season
    )
    arsenal = []
    official_outing_count = 0
    selected_season_first_game_date = None
    selected_season_last_game_date = None
    if selected_season is not None:
        with closing(connect_database()) as connection:
            season_bounds = connection.execute(
                """
                SELECT MIN(game_date), MAX(game_date)
                FROM pitches
                WHERE CAST(pitcher AS INTEGER) = ?
                  AND game_type = 'R'
                  AND CAST(season AS INTEGER) = ?
                """,
                (int(pitcher_id), int(selected_season)),
            ).fetchone()
            selected_season_first_game_date = season_bounds[0] if season_bounds else None
            selected_season_last_game_date = season_bounds[1] if season_bounds else None
            arsenal_rows = connection.execute(
                """
                SELECT pitch_type, MAX(pitch_name) AS pitch_name, COUNT(*) AS pitch_count
                FROM pitches
                WHERE CAST(pitcher AS INTEGER) = ?
                  AND game_type = 'R'
                  AND CAST(season AS INTEGER) = ?
                  AND pitch_type IS NOT NULL
                GROUP BY pitch_type
                ORDER BY pitch_count DESC
                """,
                (int(pitcher_id), int(selected_season)),
            ).fetchall()
            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='official_outings'"
            ).fetchone()
            if table_exists:
                official_outing_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM official_outings
                        WHERE pitcher_id = ? AND CAST(season AS INTEGER) = ?
                        """,
                        (int(pitcher_id), int(selected_season)),
                    ).fetchone()[0]
                    or 0
                )
        total = sum(int(item[2]) for item in arsenal_rows)
        arsenal = [
            {
                "pitch_type": item[0],
                "pitch_name": item[1] or item[0],
                "pitch_count": int(item[2]),
                "usage_pct": round(int(item[2]) / total * 100.0, 1) if total else 0.0,
            }
            for item in arsenal_rows
        ]

    pitch_rows = int(row[2] if row else 0)

    def coverage(index: int) -> float | None:
        if not row or not pitch_rows or row[index] is None:
            return None
        return round(float(row[index]) / pitch_rows * 100.0, 1)

    return {
        "first_game_date": row[0] if row else None,
        "last_game_date": row[1] if row else None,
        "pitch_rows": pitch_rows,
        "outing_count": int(row[3] if row and row[3] is not None else 0),
        "seasons": seasons,
        "current_season": current_season,
        "selected_season": selected_season,
        "selected_season_first_game_date": selected_season_first_game_date,
        "selected_season_last_game_date": selected_season_last_game_date,
        "official_outing_count": official_outing_count,
        "arsenal": arsenal,
        "data_coverage": {
            "velocity_pct": coverage(4),
            "release_position_pct": coverage(5),
            "movement_pct": coverage(6),
            "spin_pct": coverage(7),
            "location_pct": coverage(8),
            "batted_ball_exit_velocity_pct": coverage(9),
        },
    }


def update_sync_metadata(pitcher_id: int, *, statcast: bool = False, official: bool = False) -> None:
    ensure_core_schema()
    summary = database_pitcher_summary(pitcher_id)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    assignments = ["first_statcast_date = ?", "last_statcast_date = ?"]
    values: list[Any] = [summary["first_game_date"], summary["last_game_date"]]
    if statcast:
        assignments.append("last_statcast_sync = ?")
        values.append(now)
    if official:
        assignments.append("last_official_sync = ?")
        values.append(now)
    values.append(int(pitcher_id))
    with closing(connect_database()) as connection:
        connection.execute(
            f"UPDATE pitchers SET {', '.join(assignments)} WHERE mlbam_id = ?",
            values,
        )
        connection.commit()


# Initialize schema whenever the app imports this module.
ensure_core_schema()
