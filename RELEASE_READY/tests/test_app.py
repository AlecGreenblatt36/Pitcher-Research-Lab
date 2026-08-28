from __future__ import annotations

import re
import unittest
from unittest.mock import patch

import _test_environment
from app import app


PITCHER_ID = 100001


class PitcherResearchLabSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()
        cls.meta = cls.client.get(f"/api/pitchers/{PITCHER_ID}/meta?season=2026").get_json()

    def test_dashboard_renders_neutral_pitcher_selector(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"pitcher-search-input", response.data)
        self.assertIn(b"Select a pitcher to begin", response.data)
        for forbidden in (b"694973", b"Paul Skenes", b"Tarik Skubal"):
            self.assertNotIn(forbidden, response.data)

    def test_every_frontend_dependency_exists_and_serves(self):
        response = self.client.get("/")
        references = set(re.findall(rb'/static/([A-Za-z0-9_.-]+)', response.data))
        expected = {
            b"style.css", b"location_v2.css", b"views.css", b"pitcher_context.js",
            b"dashboard.js", b"research.js", b"location.js", b"overview.js",
            b"navigation.js", b"performance.js", b"career.js", b"release.js",
        }
        self.assertEqual(references, expected)
        dynamic = {b"metric_guide.js", b"metric_guide.css", b"performance.css", b"career.css"}
        for filename in references | dynamic:
            with self.client.get(f"/static/{filename.decode()}") as asset:
                self.assertEqual(asset.status_code, 200, filename)
                self.assertTrue(asset.get_data())

    def test_pitcher_search_route(self):
        result = [{"mlbam_id": PITCHER_ID, "name": "Veteran Starter", "pitch_hand": "R"}]
        with patch("pitcher_routes.search_pitchers", return_value=result) as search:
            response = self.client.get("/api/pitchers/search?q=Veteran")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), result)
        search.assert_called_once_with("Veteran")

    def test_metadata_and_all_primary_apis(self):
        self.assertEqual(self.meta["pitcher"]["mlbam_id"], PITCHER_ID)
        self.assertEqual(self.meta["database"]["selected_season"], 2026)
        pitch = self.meta["database"]["arsenal"][0]["pitch_type"]
        paths = [
            f"/api/pitchers/{PITCHER_ID}/changes?season=2026",
            f"/api/pitchers/{PITCHER_ID}/timeline?season=2026",
            f"/api/pitchers/{PITCHER_ID}/pitch/{pitch}?season=2026",
            f"/api/pitchers/{PITCHER_ID}/research?season=2026",
            f"/api/pitchers/{PITCHER_ID}/location?season=2026&pitch={pitch}&hand=ALL",
            f"/api/pitchers/{PITCHER_ID}/performance?season=2026",
            f"/api/pitchers/{PITCHER_ID}/career?season=2026",
        ]
        for path in paths:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, f"{path}: {response.get_data(as_text=True)[:500]}")

    def test_historical_season_and_custom_periods(self):
        meta = self.client.get(f"/api/pitchers/{PITCHER_ID}/meta?season=2025").get_json()
        self.assertEqual(meta["research_defaults"]["target_season"], 2025)
        pitch = meta["database"]["arsenal"][0]["pitch_type"]
        rows = self.client.get(f"/api/pitchers/{PITCHER_ID}/pitch/{pitch}?season=2025").get_json()
        self.assertLessEqual(max(int(row["season"]) for row in rows), 2025)

        query = (
            "season=2026&baseline_start=2025-04-01&baseline_end=2025-06-03"
            "&comparison_start=2026-04-01&comparison_end=2026-06-03"
        )
        paths = [
            f"/api/pitchers/{PITCHER_ID}/research?{query}",
            f"/api/pitchers/{PITCHER_ID}/location?pitch=FF&hand=ALL&{query}",
            f"/api/pitchers/{PITCHER_ID}/performance?{query}",
        ]
        for path in paths:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True)[:500])
            self.assertEqual(response.get_json()["comparison_periods"]["source"], "custom")

    def test_invalid_parameters_return_explained_400s(self):
        invalid = "baseline_start=2026-08-01&baseline_end=2026-04-01"
        paths = [
            f"/api/pitchers/{PITCHER_ID}/research?{invalid}",
            f"/api/pitchers/{PITCHER_ID}/location?pitch=FF&hand=ALL&{invalid}",
            f"/api/pitchers/{PITCHER_ID}/performance?{invalid}",
            f"/api/pitchers/{PITCHER_ID}/location?pitch=FF&hand=X",
        ]
        for path in paths:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.get_json())

    def test_health_and_no_player_specific_alias(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["database_integrity"], "ok")
        self.assertEqual(self.client.get("/api/skenes/changes").status_code, 404)


if __name__ == "__main__":
    unittest.main()
