from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
import os
from pathlib import Path

import _test_environment
import app as app_module
import career_routes
import location_routes
import performance_routes
import pitcher_core
import pitcher_routes
import research_routes
from app import app


DEFAULT_PITCHER_ID = 100001


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = _test_environment.TEST_DATABASE


class MultiPitcherRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = _test_environment.TEST_ROOT
        cls.temp_db = cls.temp_dir / f"multi_pitcher_{os.getpid()}.db"
        shutil.copy2(SOURCE_DB, cls.temp_db)

        cls.original_paths = {
            "pitcher_core": pitcher_core.DATABASE_FILE,
            "app": app_module.database_file,
            "research": research_routes.database_file,
            "location": location_routes.database_file,
            "performance": performance_routes.DATABASE_FILE,
            "career": career_routes.DATABASE_FILE,
            "pitcher_routes": pitcher_routes.DATABASE_FILE,
        }
        pitcher_core.DATABASE_FILE = cls.temp_db
        app_module.database_file = cls.temp_db
        research_routes.database_file = cls.temp_db
        location_routes.database_file = cls.temp_db
        performance_routes.DATABASE_FILE = cls.temp_db
        career_routes.DATABASE_FILE = cls.temp_db
        pitcher_routes.DATABASE_FILE = cls.temp_db

        cls._seed_pitchers()
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        pitcher_core.DATABASE_FILE = cls.original_paths["pitcher_core"]
        app_module.database_file = cls.original_paths["app"]
        research_routes.database_file = cls.original_paths["research"]
        location_routes.database_file = cls.original_paths["location"]
        performance_routes.DATABASE_FILE = cls.original_paths["performance"]
        career_routes.DATABASE_FILE = cls.original_paths["career"]
        pitcher_routes.DATABASE_FILE = cls.original_paths["pitcher_routes"]
        cls.temp_db.unlink(missing_ok=True)

    @classmethod
    def _seed_pitchers(cls):
        connection = sqlite3.connect(cls.temp_db)
        columns = [row[1] for row in connection.execute("PRAGMA table_info(pitches)")]
        quoted_columns = ", ".join(f'"{column}"' for column in columns)

        def clone(pitcher_id: int, name: str, where: str = "1=1", hand: str = "R"):
            expressions = []
            parameters = []
            for column in columns:
                if column == "pitcher":
                    expressions.append("?")
                    parameters.append(pitcher_id)
                elif column == "player_name":
                    expressions.append("?")
                    parameters.append(name)
                elif column == "p_throws":
                    expressions.append("?")
                    parameters.append(hand)
                else:
                    expressions.append(f'"{column}"')
            connection.execute(
                f"""
                INSERT INTO pitches ({quoted_columns})
                SELECT {', '.join(expressions)}
                FROM pitches
                WHERE CAST(pitcher AS INTEGER) = ? AND ({where})
                """,
                (*parameters, DEFAULT_PITCHER_ID),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO pitchers (
                    mlbam_id, full_name, pitch_hand, primary_position, active,
                    last_profile_refresh, last_statcast_sync
                ) VALUES (?, ?, ?, 'P', 1, datetime('now'), datetime('now'))
                """,
                (pitcher_id, name, hand),
            )

        clone(910001, "Veteran Lefty", hand="L")
        clone(910002, "Rookie Pitcher", "CAST(season AS INTEGER) = 2026")
        clone(910003, "Relief Pitcher", "CAST(season AS INTEGER) = 2026")
        connection.execute(
            """
            DELETE FROM pitches
            WHERE CAST(pitcher AS INTEGER) = 910003
              AND rowid NOT IN (
                  SELECT rowid FROM (
                      SELECT rowid,
                             ROW_NUMBER() OVER (PARTITION BY game_date ORDER BY rowid) AS rn
                      FROM pitches
                      WHERE CAST(pitcher AS INTEGER) = 910003
                  )
                  WHERE rn <= 18
              )
            """
        )
        clone(910004, "Sparse Tracking Pitcher", "CAST(season AS INTEGER) = 2026")
        connection.execute(
            """
            UPDATE pitches
            SET release_spin_rate = NULL,
                arm_angle = NULL,
                launch_angle = NULL,
                estimated_woba_using_speedangle = NULL
            WHERE CAST(pitcher AS INTEGER) = 910004
            """
        )
        clone(910005, "One Pitch Pitcher", "CAST(season AS INTEGER) = 2026")
        connection.execute(
            "DELETE FROM pitches WHERE CAST(pitcher AS INTEGER) = 910005 AND pitch_type <> 'FF'"
        )
        clone(
            910006,
            "Short Sample Pitcher",
            "CAST(season AS INTEGER) = 2026 AND game_date IN ("
            "SELECT game_date FROM pitches WHERE CAST(pitcher AS INTEGER) = 100001 "
            "AND CAST(season AS INTEGER) = 2026 GROUP BY game_date ORDER BY game_date LIMIT 2)",
        )
        connection.commit()
        connection.close()

    def _research_window(self, pitcher_id: int):
        meta = self.client.get(f"/api/pitchers/{pitcher_id}/meta").get_json()
        defaults = meta["research_defaults"]
        return meta, defaults["transition_start"], defaults["transition_end"]

    def test_metadata_is_isolated_for_each_pitcher(self):
        ids = [910001, 910002, 910003, 910004, 910005, 910006]
        for pitcher_id in ids:
            with self.subTest(pitcher_id=pitcher_id):
                response = self.client.get(f"/api/pitchers/{pitcher_id}/meta")
                self.assertEqual(response.status_code, 200)
                payload = response.get_json()
                self.assertEqual(payload["pitcher"]["mlbam_id"], pitcher_id)
                self.assertGreater(payload["database"]["pitch_rows"], 0)
                self.assertTrue(payload["database"]["arsenal"])

    def test_main_pages_handle_diverse_pitcher_profiles(self):
        ids = [910001, 910002, 910003, 910004, 910005]
        for pitcher_id in ids:
            meta, start, end = self._research_window(pitcher_id)
            pitch = meta["database"]["arsenal"][0]["pitch_type"]
            paths = [
                f"/api/pitchers/{pitcher_id}/changes",
                f"/api/pitchers/{pitcher_id}/timeline",
                f"/api/pitchers/{pitcher_id}/pitch/{pitch}",
                f"/api/pitchers/{pitcher_id}/research?start={start}&end={end}",
                f"/api/pitchers/{pitcher_id}/location?pitch={pitch}&hand=ALL&start={start}&end={end}",
                f"/api/pitchers/{pitcher_id}/performance?start={start}&end={end}",
                f"/api/pitchers/{pitcher_id}/career",
            ]
            for path in paths:
                with self.subTest(pitcher_id=pitcher_id, path=path):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 200)

    def test_short_sample_fails_gracefully_when_analysis_is_not_supported(self):
        pitcher_id = 910006
        meta, start, end = self._research_window(pitcher_id)
        self.assertTrue(start)
        self.assertTrue(end)
        pitch = meta["database"]["arsenal"][0]["pitch_type"]

        paths = [
            f"/api/pitchers/{pitcher_id}/changes",
            f"/api/pitchers/{pitcher_id}/timeline",
            f"/api/pitchers/{pitcher_id}/pitch/{pitch}",
            f"/api/pitchers/{pitcher_id}/research?start={start}&end={end}",
            f"/api/pitchers/{pitcher_id}/location?pitch={pitch}&hand=ALL&start={start}&end={end}",
            f"/api/pitchers/{pitcher_id}/performance?start={start}&end={end}",
            f"/api/pitchers/{pitcher_id}/career",
        ]
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertIn(response.status_code, {200, 404})
                self.assertNotEqual(response.status_code, 500)

    def test_pitcher_data_does_not_leak_between_ids(self):
        veteran = self.client.get("/api/pitchers/910001/meta").get_json()["database"]
        one_pitch = self.client.get("/api/pitchers/910005/meta").get_json()["database"]
        self.assertGreater(veteran["pitch_rows"], one_pitch["pitch_rows"])
        self.assertEqual(len(one_pitch["arsenal"]), 1)
        self.assertEqual(one_pitch["arsenal"][0]["pitch_type"], "FF")


if __name__ == "__main__":
    unittest.main()
