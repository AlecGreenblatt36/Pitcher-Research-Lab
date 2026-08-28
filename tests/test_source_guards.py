from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceGuardTests(unittest.TestCase):
    def test_frontend_uses_generalized_pitcher_routes(self):
        for path in (ROOT / "static").glob("*.js"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/api/skenes/", text, path.name)

    def test_research_season_selector_is_present(self):
        template = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
        context = (ROOT / "static" / "pitcher_context.js").read_text(encoding="utf-8")
        self.assertIn('id="research-season-select"', template)
        self.assertIn("apiUrl", context)

    def test_frontend_has_no_case_study_date_dependency(self):
        for path in (ROOT / "static").glob("*.js"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("2026-05-06", text, path.name)
            self.assertNotIn("2026-06-09", text, path.name)

    def test_frontend_has_no_default_pitcher(self):
        context = (ROOT / "static" / "pitcher_context.js").read_text(encoding="utf-8")
        self.assertNotIn("694973", context)
        self.assertNotIn("DEFAULT_PITCHER", context)
        self.assertIn("return Number.isFinite(value) && value > 0 ? value : null", context)

    def test_live_ui_has_no_future_work_copy(self):
        sources = [ROOT / "templates" / "dashboard.html", *(ROOT / "static").glob("*.js")]
        forbidden = (
            "next major research layer", "we will analyze", "this page will", "eventually",
            "before publication", "will be sensitivity-tested", "coming soon", "future work",
        )
        for path in sources:
            text = path.read_text(encoding="utf-8").lower()
            for phrase in forbidden:
                self.assertNotIn(phrase, text, f"{phrase!r} in {path.name}")

    def test_release_profile_is_dynamic_and_conditional(self):
        template = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
        release = (ROOT / "static" / "release.js").read_text(encoding="utf-8")
        self.assertIn("release.js", template)
        for metric in ("release_pos_x", "release_pos_z", "release_extension", "arm_angle"):
            self.assertIn(metric, release)
        self.assertIn("card.hidden = true", release)

    def test_primary_navigation_uses_distinct_views(self):
        navigation = (ROOT / "static" / "navigation.js").read_text(encoding="utf-8")
        self.assertIn('ensureViewPanel("location")', navigation)
        self.assertIn('ensureViewPanel("career")', navigation)
        self.assertNotIn('panel:\n            "performance"', navigation)
        self.assertNotIn('panel:\n            "changes"', navigation)
        self.assertIn('`[data-view-panel="${viewName}"]`', navigation)


if __name__ == "__main__":
    unittest.main()
