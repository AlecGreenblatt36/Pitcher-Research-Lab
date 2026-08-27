from __future__ import annotations

from contextlib import closing

import io
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pitcher_core import (
    DATABASE_FILE,
    connect_database,
    database_pitcher_summary,
    ensure_core_schema,
    get_pitcher_profile,
    update_sync_metadata,
)

BASEBALL_SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
OVERLAP_DAYS = 7
STATCAST_START_YEAR = 2015
STATCAST_ATTEMPTS = 3


def _database_columns(connection: sqlite3.Connection) -> list[str]:
    return [row[1] for row in connection.execute("PRAGMA table_info(pitches)").fetchall()]


def _fetch_statcast(pitcher_id: int, start_date: date, end_date: date) -> pd.DataFrame:
    params = {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "pitchers_lookup[]": str(int(pitcher_id)),
        "game_date_gt": start_date.strftime("%Y-%m-%d"),
        "game_date_lt": end_date.strftime("%Y-%m-%d"),
        "hfGT": "R|",
        "hfPT": "",
        "hfAB": "",
        "hfBBT": "",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfSea": "",
        "hfSit": "",
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "team": "",
        "position": "",
        "hfRO": "",
        "home_road": "",
        "hfFlag": "",
        "metric_1": "",
        "hfInn": "",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "h_launch_speed",
        "sort_order": "desc",
        "min_abs": "0",
    }
    response = None
    for attempt in range(STATCAST_ATTEMPTS):
        try:
            response = requests.get(
                BASEBALL_SAVANT_URL,
                params=params,
                timeout=120,
                headers={
                    "User-Agent": "Mozilla/5.0 PitcherResearchLab/2.0",
                    "Accept": "text/csv,text/plain,*/*",
                },
            )
            response.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt + 1 == STATCAST_ATTEMPTS:
                raise
            time.sleep(0.5 * (2**attempt))
        except requests.HTTPError:
            if response is None or response.status_code < 500 or attempt + 1 == STATCAST_ATTEMPTS:
                raise
            time.sleep(0.5 * (2**attempt))
    if response is None:
        raise RuntimeError("Baseball Savant request did not return a response.")
    text = response.text
    if "<html" in text[:1000].lower():
        raise RuntimeError("Baseball Savant returned HTML instead of CSV data.")
    if not text.strip():
        return pd.DataFrame()
    data = pd.read_csv(io.StringIO(text), low_memory=False)
    data.columns = data.columns.astype(str).str.strip()
    return data


def _prepare(data: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
    if data.empty:
        return data
    required = ["game_date", "game_pk", "at_bat_number", "pitch_number", "pitcher"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError("Savant response missing required fields: " + ", ".join(missing))

    data = data.copy()
    data["pitcher"] = pd.to_numeric(data["pitcher"], errors="coerce")
    data = data[data["pitcher"] == int(pitcher_id)].copy()
    if "game_type" in data.columns:
        data = data[data["game_type"] == "R"].copy()

    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    data = data[data["game_date"].notna()].copy()
    data["season"] = data["game_date"].dt.year
    data["game_date"] = data["game_date"].dt.strftime("%Y-%m-%d")

    if "release_spin" in data.columns and "release_spin_rate" not in data.columns:
        data["release_spin_rate"] = data["release_spin"]
    if "release_spin_rate" in data.columns and "release_spin" not in data.columns:
        data["release_spin"] = data["release_spin_rate"]

    identities = ["game_pk", "at_bat_number", "pitch_number", "pitcher"]
    for column in identities:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=identities).copy()
    for column in identities:
        data[column] = data[column].astype(int)
    return data.drop_duplicates(
        subset=["pitcher", "game_pk", "at_bat_number", "pitch_number"],
        keep="last",
    )


def _align(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # reindex adds missing database columns in one operation and avoids
    # fragmenting the DataFrame when the SQLite schema is wide.
    aligned = data.reindex(columns=columns).copy()
    return aligned.astype(object).where(pd.notna(aligned), None)


def _write_batch(connection: sqlite3.Connection, data: pd.DataFrame) -> int:
    if data.empty:
        return 0
    columns = _database_columns(connection)
    aligned = _align(data, columns)
    identity_rows = [
        (int(row.pitcher), int(row.game_pk), int(row.at_bat_number), int(row.pitch_number))
        for row in data[["pitcher", "game_pk", "at_bat_number", "pitch_number"]].itertuples(index=False)
    ]
    connection.executemany(
        """
        DELETE FROM pitches
        WHERE CAST(pitcher AS INTEGER)=?
          AND CAST(game_pk AS INTEGER)=?
          AND CAST(at_bat_number AS INTEGER)=?
          AND CAST(pitch_number AS INTEGER)=?
        """,
        identity_rows,
    )
    aligned.to_sql("pitches", connection, if_exists="append", index=False)
    return len(aligned)


def _initial_start_date(profile: dict[str, Any]) -> date:
    debut = profile.get("mlb_debut_date")
    if debut:
        try:
            parsed = pd.Timestamp(debut).date()
            return max(parsed, date(STATCAST_START_YEAR, 1, 1))
        except Exception:
            pass
    return date(max(STATCAST_START_YEAR, date.today().year - 2), 1, 1)


def _windows(start_date: date, end_date: date) -> list[tuple[date, date]]:
    # First loads are split by calendar year to keep Savant responses manageable.
    windows: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        year_end = date(cursor.year, 12, 31)
        window_end = min(year_end, end_date)
        windows.append((cursor, window_end))
        cursor = date(cursor.year + 1, 1, 1)
    return windows


def sync_pitcher_statcast(pitcher_id: int, force_full: bool = False) -> dict[str, Any]:
    ensure_core_schema()
    pitcher_id = int(pitcher_id)
    try:
        profile = get_pitcher_profile(pitcher_id, refresh=True)
    except Exception:
        # A temporary MLB profile API failure should not prevent incremental
        # Savant updates for a pitcher we already know about.
        profile = get_pitcher_profile(pitcher_id, refresh=False)
    summary_before = database_pitcher_summary(pitcher_id)
    today = date.today()

    if (
        not force_full
        and profile.get("active") is False
        and summary_before["last_game_date"]
    ):
        update_sync_metadata(pitcher_id, statcast=True)
        return {
            "status": "success",
            "mode": "inactive_noop",
            "pitcher": profile,
            "fetch_start": None,
            "fetch_end": None,
            "rows_fetched": 0,
            "rows_written": 0,
            **summary_before,
        }

    if force_full or not summary_before["last_game_date"]:
        start_date = _initial_start_date(profile)
        mode = "initial"
    else:
        latest = pd.Timestamp(summary_before["last_game_date"]).date()
        start_date = latest - timedelta(days=OVERLAP_DAYS)
        mode = "incremental"

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with closing(connect_database()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO ingest_runs (pitcher_id, source, started_at, fetch_start, fetch_end, status)
            VALUES (?, 'statcast', ?, ?, ?, 'running')
            """,
            (pitcher_id, started_at, str(start_date), str(today)),
        )
        run_id = cursor.lastrowid
        connection.commit()

    rows_fetched = 0
    rows_written = 0
    try:
        prepared_batches = []
        for window_start, window_end in _windows(start_date, today):
            raw = _fetch_statcast(pitcher_id, window_start, window_end)
            rows_fetched += len(raw)
            prepared_batches.append(_prepare(raw, pitcher_id))

        with closing(connect_database()) as connection:
            connection.execute("BEGIN")
            for incoming in prepared_batches:
                rows_written += _write_batch(connection, incoming)
            connection.commit()

        update_sync_metadata(pitcher_id, statcast=True)
        summary_after = database_pitcher_summary(pitcher_id)
        completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with closing(connect_database()) as connection:
            connection.execute(
                """
                UPDATE ingest_runs
                SET completed_at=?, rows_fetched=?, rows_written=?, status='success'
                WHERE id=?
                """,
                (completed_at, rows_fetched, rows_written, run_id),
            )
            connection.commit()
        return {
            "status": "success",
            "mode": mode,
            "pitcher": profile,
            "fetch_start": str(start_date),
            "fetch_end": str(today),
            "rows_fetched": rows_fetched,
            "rows_written": rows_written,
            **summary_after,
        }
    except Exception as exc:
        completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with closing(connect_database()) as connection:
            connection.execute(
                """
                UPDATE ingest_runs
                SET completed_at=?, rows_fetched=?, rows_written=?, status='error', error=?
                WHERE id=?
                """,
                (completed_at, rows_fetched, rows_written, str(exc), run_id),
            )
            connection.commit()
        raise
