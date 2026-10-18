from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

from pitcher_core import connect_database, current_research_season, update_sync_metadata

BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
REFRESH_DAYS = 7


def ensure_table(connection: sqlite3.Connection) -> None:
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
    connection.commit()


def innings_to_outs(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    whole, _, fraction = text.partition(".")
    try:
        outs = int(whole) * 3
    except ValueError:
        return None
    if fraction.startswith("1"):
        outs += 1
    elif fraction.startswith("2"):
        outs += 2
    return outs


def fetch_boxscore(game_pk: int) -> dict[str, Any]:
    response = requests.get(
        BOXSCORE_URL.format(game_pk=int(game_pk)),
        timeout=30,
        headers={"User-Agent": "PitcherResearchLab/2.0", "Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def extract_line(payload: dict[str, Any], game: dict[str, Any], pitcher_id: int) -> dict[str, Any]:
    player_key = f"ID{int(pitcher_id)}"
    found_side = None
    player = None
    for side in ("home", "away"):
        candidate = (
            payload.get("teams", {})
            .get(side, {})
            .get("players", {})
            .get(player_key)
        )
        if candidate and candidate.get("stats", {}).get("pitching"):
            found_side = side
            player = candidate
            break
    if not found_side or not player:
        raise RuntimeError(f"No pitching line found for game {game['game_pk']}.")

    pitching = player["stats"]["pitching"]
    if found_side == "home":
        team, opponent, home_away = game["home_team"], game["away_team"], "Home"
    else:
        team, opponent, home_away = game["away_team"], game["home_team"], "Away"

    innings = pitching.get("inningsPitched")
    outs = pitching.get("outs")
    if outs is None:
        outs = innings_to_outs(innings)

    return {
        "game_pk": int(game["game_pk"]),
        "pitcher_id": int(pitcher_id),
        "game_date": str(game["game_date"]),
        "season": int(game["season"]),
        "team": team,
        "opponent": opponent,
        "home_away": home_away,
        "innings_pitched": innings,
        "outs_recorded": outs,
        "hits": pitching.get("hits"),
        "runs": pitching.get("runs"),
        "earned_runs": pitching.get("earnedRuns"),
        "walks": pitching.get("baseOnBalls"),
        "intentional_walks": pitching.get("intentionalWalks"),
        "hit_by_pitch": pitching.get("hitByPitch", pitching.get("hitBatsmen")),
        "strikeouts": pitching.get("strikeOuts"),
        "home_runs": pitching.get("homeRuns"),
        "batters_faced": pitching.get("battersFaced"),
        "pitches": pitching.get("numberOfPitches", pitching.get("pitchesThrown")),
        "strikes": pitching.get("strikes"),
        "balls": pitching.get("balls"),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def upsert(connection: sqlite3.Connection, row: dict[str, Any]) -> None:
    columns = list(row.keys())
    placeholders = ", ".join(f":{column}" for column in columns)
    update_columns = [column for column in columns if column not in {"game_pk", "pitcher_id"}]
    updates = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
    connection.execute(
        f"""
        INSERT INTO official_outings ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(game_pk, pitcher_id) DO UPDATE SET {updates};
        """,
        row,
    )


def sync_official_outings(pitcher_id: int, season: int | None = None) -> dict[str, Any]:
    pitcher_id = int(pitcher_id)
    season = int(season) if season is not None else current_research_season(pitcher_id)
    if season is None:
        return {"status": "no_data", "season": None, "games_synced": 0, "failures": []}

    connection = connect_database()
    try:
        ensure_table(connection)
        games = [
            dict(row)
            for row in connection.execute(
                """
                SELECT game_pk, MIN(game_date) AS game_date, MAX(season) AS season,
                       MAX(home_team) AS home_team, MAX(away_team) AS away_team
                FROM pitches
                WHERE game_type='R'
                  AND CAST(pitcher AS INTEGER)=?
                  AND CAST(season AS INTEGER)=?
                  AND game_pk IS NOT NULL
                GROUP BY game_pk
                ORDER BY game_date
                """,
                (pitcher_id, int(season)),
            ).fetchall()
        ]
        cached = {
            int(row[0])
            for row in connection.execute(
                "SELECT game_pk FROM official_outings WHERE pitcher_id=?",
                (pitcher_id,),
            ).fetchall()
        }
        cutoff = date.today() - timedelta(days=REFRESH_DAYS)
        targets = [
            game
            for game in games
            if int(game["game_pk"]) not in cached
            or date.fromisoformat(str(game["game_date"])[:10]) >= cutoff
        ]

        failures: list[dict[str, Any]] = []
        synced = 0
        for game in targets:
            try:
                payload = fetch_boxscore(int(game["game_pk"]))
                row = extract_line(payload, game, pitcher_id)
                upsert(connection, row)
                connection.commit()
                synced += 1
            except Exception as exc:
                connection.rollback()
                failures.append({"game_pk": int(game["game_pk"]), "error": str(exc)})
            time.sleep(0.08)

        total = connection.execute(
            "SELECT COUNT(*) FROM official_outings WHERE pitcher_id=?",
            (pitcher_id,),
        ).fetchone()[0]
    finally:
        connection.close()

    update_sync_metadata(pitcher_id, official=True)
    return {
        "status": "success" if not failures else "partial",
        "season": int(season),
        "games_considered": len(games),
        "games_requested": len(targets),
        "games_synced": synced,
        "official_outings_cached": int(total),
        "failures": failures,
    }


def sync_current_season_official_outings(pitcher_id: int) -> dict[str, Any]:
    return sync_official_outings(pitcher_id)
