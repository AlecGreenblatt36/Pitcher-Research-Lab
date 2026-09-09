from __future__ import annotations

import sqlite3
import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

import _test_environment
import pitcher_sync


class IngestionTests(unittest.TestCase):
    PITCHER_ID = 110001

    def setUp(self):
        connection = sqlite3.connect(_test_environment.TEST_DATABASE)
        connection.execute(
            "DELETE FROM pitches WHERE CAST(pitcher AS INTEGER) = ?", (self.PITCHER_ID,)
        )
        connection.execute(
            "DELETE FROM pitchers WHERE mlbam_id = ?", (self.PITCHER_ID,)
        )
        connection.execute(
            "DELETE FROM ingest_runs WHERE pitcher_id = ?", (self.PITCHER_ID,)
        )
        connection.execute(
            "INSERT INTO pitchers (mlbam_id, full_name, pitch_hand, primary_position, active) VALUES (?, 'Ingestion Fixture', 'R', 'P', 1)",
            (self.PITCHER_ID,),
        )
        connection.commit()
        connection.close()

    def _frame(self, include_new: bool = False) -> pd.DataFrame:
        game_day = date.today() - timedelta(days=2)
        rows = []
        for pitch_number in range(1, 4 + int(include_new)):
            rows.append(
                {
                    "pitch_type": "FF",
                    "game_date": game_day.isoformat(),
                    "game_pk": 990001,
                    "at_bat_number": pitch_number,
                    "pitch_number": 1,
                    "pitcher": self.PITCHER_ID,
                    "player_name": "Fixture, Ingestion",
                    "game_type": "R",
                    "release_speed": 96.0 + pitch_number / 10,
                    "release_pos_x": -1.8,
                    "release_pos_z": 5.9,
                    "release_extension": 6.5,
                    "release_spin_rate": 2400,
                    "pfx_x": 0.5,
                    "pfx_z": 1.3,
                    "description": "called_strike",
                    "pitch_name": "4-Seam Fastball",
                }
            )
        return pd.DataFrame(rows)

    def test_initial_then_incremental_sync_replaces_overlap_without_duplicates(self):
        profile = {
            "mlbam_id": self.PITCHER_ID,
            "name": "Ingestion Fixture",
            "pitch_hand": "R",
            "active": True,
            "mlb_debut_date": (date.today() - timedelta(days=10)).isoformat(),
        }
        with patch("pitcher_sync.get_pitcher_profile", return_value=profile), patch(
            "pitcher_sync._fetch_statcast", return_value=self._frame(False)
        ) as fetch:
            initial = pitcher_sync.sync_pitcher_statcast(self.PITCHER_ID)
        self.assertEqual(initial["mode"], "initial")
        self.assertEqual(initial["rows_written"], 3)
        self.assertEqual(initial["pitch_rows"], 3)
        self.assertTrue(fetch.called)

        with patch("pitcher_sync.get_pitcher_profile", return_value=profile), patch(
            "pitcher_sync._fetch_statcast", return_value=self._frame(True)
        ) as fetch:
            incremental = pitcher_sync.sync_pitcher_statcast(self.PITCHER_ID)
        self.assertEqual(incremental["mode"], "incremental")
        self.assertEqual(incremental["pitch_rows"], 4)
        self.assertEqual(incremental["rows_written"], 4)
        requested_start = fetch.call_args.args[1]
        self.assertGreaterEqual(requested_start, date.today() - timedelta(days=10))

        connection = sqlite3.connect(_test_environment.TEST_DATABASE)
        duplicates = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT pitcher, game_pk, at_bat_number, pitch_number, COUNT(*) n
                FROM pitches WHERE CAST(pitcher AS INTEGER) = ?
                GROUP BY pitcher, game_pk, at_bat_number, pitch_number HAVING n > 1
            )
            """,
            (self.PITCHER_ID,),
        ).fetchone()[0]
        successful_runs = connection.execute(
            "SELECT COUNT(*) FROM ingest_runs WHERE pitcher_id = ? AND status = 'success'",
            (self.PITCHER_ID,),
        ).fetchone()[0]
        connection.close()
        self.assertEqual(duplicates, 0)
        self.assertEqual(successful_runs, 2)

    def test_later_batch_failure_rolls_back_corrections_and_inserts(self):
        profile = {
            "mlbam_id": self.PITCHER_ID,
            "active": True,
            "mlb_debut_date": "2025-04-01",
        }
        original = pitcher_sync._prepare(self._frame(), self.PITCHER_ID)
        with sqlite3.connect(_test_environment.TEST_DATABASE) as connection:
            pitcher_sync._write_batch(connection, original)
        corrected = self._frame(True)
        corrected["release_speed"] = 80
        writer = pitcher_sync._write_batch
        calls = 0

        def fail_second(connection, frame):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise sqlite3.OperationalError("simulated second-batch failure")
            return writer(connection, frame)

        with patch("pitcher_sync.get_pitcher_profile", return_value=profile), patch(
            "pitcher_sync._fetch_statcast", return_value=corrected
        ), patch(
            "pitcher_sync._windows",
            return_value=[
                (date(2025, 1, 1), date(2025, 12, 31)),
                (date(2026, 1, 1), date(2026, 9, 1)),
            ],
        ), patch(
            "pitcher_sync._write_batch", side_effect=fail_second
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "second-batch"):
                pitcher_sync.sync_pitcher_statcast(self.PITCHER_ID, force_full=True)
        with sqlite3.connect(_test_environment.TEST_DATABASE) as connection:
            values = connection.execute(
                "SELECT release_speed FROM pitches WHERE pitcher=? ORDER BY at_bat_number",
                (self.PITCHER_ID,),
            ).fetchall()
            run = connection.execute(
                "SELECT status, rows_written FROM ingest_runs WHERE pitcher_id=? ORDER BY id DESC LIMIT 1",
                (self.PITCHER_ID,),
            ).fetchone()
        self.assertEqual(values, [(96.1,), (96.2,), (96.3,)])
        self.assertEqual(run, ("error", 0))

    def test_transient_statcast_failure_retries_then_succeeds(self):
        import requests
        from unittest.mock import Mock

        response = Mock(status_code=200, text=self._frame().to_csv(index=False))
        with patch(
            "pitcher_sync.requests.get", side_effect=[requests.Timeout(), response]
        ) as get, patch("pitcher_sync.time.sleep") as sleep:
            result = pitcher_sync._fetch_statcast(
                self.PITCHER_ID, date(2026, 1, 1), date(2026, 9, 1)
            )
        self.assertEqual(len(result), 3)
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once()

    def test_malformed_statcast_response_is_rejected_and_logged(self):
        profile = {
            "mlbam_id": self.PITCHER_ID,
            "name": "Ingestion Fixture",
            "active": True,
            "mlb_debut_date": (date.today() - timedelta(days=10)).isoformat(),
        }
        with patch("pitcher_sync.get_pitcher_profile", return_value=profile), patch(
            "pitcher_sync._fetch_statcast",
            return_value=pd.DataFrame({"unexpected": [1]}),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing required fields"):
                pitcher_sync.sync_pitcher_statcast(self.PITCHER_ID)

        connection = sqlite3.connect(_test_environment.TEST_DATABASE)
        status = connection.execute(
            "SELECT status FROM ingest_runs WHERE pitcher_id = ? ORDER BY id DESC LIMIT 1",
            (self.PITCHER_ID,),
        ).fetchone()[0]
        connection.close()
        self.assertEqual(status, "error")


if __name__ == "__main__":
    unittest.main()
