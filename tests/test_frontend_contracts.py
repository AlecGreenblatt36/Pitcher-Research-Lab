from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTests(unittest.TestCase):
    def test_research_season_selector_is_present(self):
        template = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
        context = (ROOT / "static" / "pitcher_context.js").read_text(encoding="utf-8")
        self.assertIn('id="research-season-select"', template)
        self.assertIn("apiUrl", context)

    def test_pitcher_context_builds_pitcher_scoped_urls(self):
        context = (ROOT / "static" / "pitcher_context.js").read_text(encoding="utf-8")
        self.assertIn("selectedPitcherId()", context)
        self.assertIn("/api/pitchers/${pitcherId}", context)
        self.assertIn("return Number.isFinite(value) && value > 0 ? value : null", context)

    def test_release_profile_is_dynamic_and_conditional(self):
        template = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
        release = (ROOT / "static" / "release.js").read_text(encoding="utf-8")
        self.assertIn("release.js", template)
        for metric in ("release_pos_x", "release_pos_z", "release_extension", "arm_angle"):
            self.assertIn(metric, release)
        self.assertIn("card.hidden = true", release)

    def test_primary_navigation_declares_all_research_views(self):
        template = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
        navigation = (ROOT / "static" / "navigation.js").read_text(encoding="utf-8")
        expected = ("overview", "arsenal", "changes", "release", "performance", "location", "career")
        for view in expected:
            self.assertEqual(template.count(f'data-view="{view}"'), 1, view)
            self.assertIn(f"{view}:", navigation)

    def test_location_and_career_use_distinct_application_views(self):
        navigation = (ROOT / "static" / "navigation.js").read_text(encoding="utf-8")
        self.assertIn('ensureViewPanel("location")', navigation)
        self.assertIn('ensureViewPanel("career")', navigation)
        self.assertIn('`[data-view-panel="${viewName}"]`', navigation)


if __name__ == "__main__":
    unittest.main()
