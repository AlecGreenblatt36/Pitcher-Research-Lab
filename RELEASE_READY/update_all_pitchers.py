from __future__ import annotations

from contextlib import closing

from datetime import datetime

from official_sync import sync_current_season_official_outings
from pitcher_core import connect_database, ensure_core_schema
from pitcher_sync import sync_pitcher_statcast


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def cached_pitcher_ids() -> list[int]:
    ensure_core_schema()
    with closing(connect_database()) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT CAST(pitcher AS INTEGER)
            FROM pitches
            WHERE pitcher IS NOT NULL
            ORDER BY CAST(pitcher AS INTEGER)
            """
        ).fetchall()
    return [int(row[0]) for row in rows if row[0] is not None]


def main() -> None:
    pitcher_ids = cached_pitcher_ids()
    log(f"Refreshing {len(pitcher_ids)} cached pitcher(s).")
    failures: list[tuple[int, str]] = []

    for index, pitcher_id in enumerate(pitcher_ids, start=1):
        log(f"[{index}/{len(pitcher_ids)}] MLBAM {pitcher_id}: Statcast")
        try:
            result = sync_pitcher_statcast(pitcher_id)
            log(
                f"  {result['mode']} sync complete: "
                f"{result['rows_written']:,} rows refreshed, "
                f"{result['pitch_rows']:,} cached."
            )

            official = sync_current_season_official_outings(pitcher_id)
            log(
                f"  Official outings: {official.get('games_synced', 0)} refreshed "
                f"({official.get('official_outings_cached', 0)} cached)."
            )
        except Exception as exc:
            failures.append((pitcher_id, str(exc)))
            log(f"  FAILED: {exc}")

    if failures:
        log(f"Finished with {len(failures)} failure(s):")
        for pitcher_id, error in failures:
            log(f"  {pitcher_id}: {error}")
        raise SystemExit(1)

    log("All cached pitchers are current.")


if __name__ == "__main__":
    main()
