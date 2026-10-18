from __future__ import annotations

import unittest

import _test_environment
from app import app


class DataConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_named_fixture_profiles_are_available_together(self):
        for pitcher_id, profile in _test_environment.PROFILES.items():
            response = self.client.get(f"/api/pitchers/{pitcher_id}/meta")
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["pitcher"]["name"], profile[0])
            self.assertGreater(payload["database"]["pitch_rows"], 0)

    def test_one_pitch_and_unusual_arsenals(self):
        one_pitch = self.client.get("/api/pitchers/100007/meta").get_json()["database"]["arsenal"]
        unusual = self.client.get("/api/pitchers/100006/meta").get_json()["database"]["arsenal"]
        self.assertEqual([row["pitch_type"] for row in one_pitch], ["FF"])
        self.assertEqual({row["pitch_type"] for row in unusual}, {"KN", "FS", "CU"})

    def test_sparse_and_short_samples_never_raise_500(self):
        for pitcher_id in (100005, 100008):
            meta = self.client.get(f"/api/pitchers/{pitcher_id}/meta").get_json()
            pitch = meta["database"]["arsenal"][0]["pitch_type"]
            for resource in (
                "changes?season=2026", "timeline?season=2026", "research?season=2026",
                f"location?season=2026&pitch={pitch}&hand=ALL", "performance?season=2026",
                "career?season=2026",
            ):
                response = self.client.get(f"/api/pitchers/{pitcher_id}/{resource}")
                self.assertNotEqual(response.status_code, 500, response.get_data(as_text=True)[:500])
                self.assertIn(response.status_code, {200, 404})

    def test_improvement_deterioration_and_stability_samples(self):
        deltas = {}
        for pitcher_id in (100001, 100002, 100003):
            payload = self.client.get(f"/api/pitchers/{pitcher_id}/research?season=2026").get_json()
            periods = {row["period"]: row for row in payload["overall"]}
            deltas[pitcher_id] = periods["post"]["run_value_per_100"] - periods["early"]["run_value_per_100"]
        self.assertGreater(deltas[100001], 0.5)
        self.assertLess(deltas[100002], -0.5)
        self.assertLess(abs(deltas[100003]), 0.5)

    def test_mixed_results_are_preserved_by_pitch(self):
        payload = self.client.get("/api/pitchers/100006/research?season=2026").get_json()
        values = {}
        for row in payload["pitches"]:
            values.setdefault(row["pitch_type"], {})[row["period"]] = row["run_value_per_100"]
        changes = [
            periods["post"] - periods["early"]
            for periods in values.values()
            if "post" in periods and "early" in periods
        ]
        self.assertTrue(any(value > 0 for value in changes))
        self.assertTrue(any(value < 0 for value in changes))


if __name__ == "__main__":
    unittest.main()
