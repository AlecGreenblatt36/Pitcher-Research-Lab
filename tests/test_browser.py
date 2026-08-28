from __future__ import annotations

import threading
import unittest

import _test_environment
from app import app
from werkzeug.serving import make_server

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - development dependency guard
    sync_playwright = None


@unittest.skipIf(sync_playwright is None, "Playwright is not installed")
class BrowserRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.server = make_server("127.0.0.1", 0, app)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.thread.join(timeout=5)

    def _wait_for_pitcher_ready(self, page):
        page.wait_for_function("() => Boolean(window.pitcherResearchLab?.ready)")
        page.evaluate("async () => { await window.pitcherResearchLab.ready; }")
        overlay = page.locator("#pitcher-loading-overlay")
        if overlay.count():
            overlay.wait_for(state="hidden")

    def _wait_for_release_ready(self, page):
        page.wait_for_function(
            """() => {
                const title = document.getElementById('release-context-title');
                return title && !title.textContent.includes('Select a pitcher');
            }"""
        )

    def _select_fixture(self, page, pitcher_id: int, name: str):
        page.evaluate(
            """([id, name]) => {
                localStorage.setItem('pitcherResearchLab.selectedPitcherId', String(id));
                localStorage.setItem('pitcherResearchLab.selectedPitcherProfile', JSON.stringify({mlbam_id: id, name}));
            }""",
            [pitcher_id, name],
        )
        page.reload(wait_until="domcontentloaded")
        self._wait_for_pitcher_ready(page)
        self.assertEqual(page.locator("#pitcher-name").inner_text(), name)

    def test_neutral_all_views_season_pitch_and_pitcher_switching(self):
        context = self.browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        console_errors = []
        page_errors = []
        failed_responses = []
        failed_requests = []
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("response", lambda response: failed_responses.append(f"{response.status} {response.url}") if response.status >= 400 else None)
        page.on("requestfailed", lambda request: failed_requests.append(request.url))

        page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")
        self.assertEqual(page.locator("#pitcher-name").inner_text(), "Select a pitcher to begin")
        self.assertEqual(page.locator("#database-status").inner_text(), "Waiting for pitcher selection")
        self.assertEqual(page.locator("#pitcher-loading-overlay").count(), 0)

        self._select_fixture(page, 100001, "Veteran Starter")
        page.locator("#career-audit-panel").wait_for(state="attached")

        for view in ("overview", "arsenal", "changes", "release", "performance", "location", "career"):
            button = page.locator(f'[data-view="{view}"]')
            self.assertEqual(button.count(), 1, view)
            button.click()
            self.assertIn("active", button.get_attribute("class"))

            active_panels = page.locator(".app-view.active")
            self.assertEqual(active_panels.count(), 1, view)
            self.assertEqual(
                active_panels.get_attribute("data-view-panel"),
                view,
                f"{view} should have its own application panel",
            )

        page.locator('[data-view="performance"]').click()
        self.assertTrue(page.locator(".performance-page-header").is_visible())
        self.assertFalse(page.locator(".location-lab-v2").is_visible())

        page.locator('[data-view="location"]').click()
        self.assertTrue(page.locator(".location-lab-v2").is_visible())
        self.assertFalse(page.locator(".performance-page-header").is_visible())

        page.locator('[data-view="changes"]').click()
        self.assertTrue(page.locator('[data-view-panel="changes"] .view-page-header').is_visible())
        self.assertFalse(page.locator("#career-audit-panel").is_visible())

        page.locator('[data-view="career"]').click()
        self.assertTrue(page.locator("#career-audit-panel").is_visible())
        self.assertFalse(page.locator('[data-view-panel="changes"] .view-page-header').is_visible())

        release = page.locator('[data-view="release"]')
        release.click()
        self._wait_for_release_ready(page)
        self.assertGreater(page.locator(".release-preview-card:not([hidden]) .release-measurement").count(), 0)
        self.assertNotIn("Select a pitcher", page.locator("#release-context-title").inner_text())

        page.locator('[data-view="overview"]').click()
        page.locator("#research-window-start").fill("2026-04-15")
        page.locator("#research-window-end").fill("2026-04-15")
        with page.expect_navigation(wait_until="domcontentloaded"):
            page.locator("#research-window-apply").click()
        self._wait_for_pitcher_ready(page)
        self.assertIn("Custom mode is active", page.locator("#research-window-note").inner_text())
        with page.expect_navigation(wait_until="domcontentloaded"):
            page.locator("#research-window-reset").click()
        self._wait_for_pitcher_ready(page)
        self.assertIn("Optional", page.locator("#research-window-note").inner_text())

        season = page.locator("#research-season-select")
        self.assertIn("2025", season.locator("option").all_text_contents())
        with page.expect_navigation(wait_until="domcontentloaded"):
            season.select_option("2025")
        self._wait_for_pitcher_ready(page)
        season = page.locator("#research-season-select")
        self.assertEqual(season.input_value(), "2025")

        pitch = page.locator("#pitch-select")
        self.assertTrue(pitch.is_visible(), "Primary pitch selector should be visible after season reload")
        values = pitch.locator("option").evaluate_all("options => options.map(option => option.value)")
        self.assertGreater(len(values), 1)
        pitch.select_option(values[1])

        self._select_fixture(page, 100002, "Veteran Lefty")
        self.assertNotIn("Veteran Starter", page.locator("body").inner_text())
        self._select_fixture(page, 100007, "One Pitch Pitcher")
        self.assertNotIn("Veteran Lefty", page.locator("body").inner_text())
        self.assertEqual(page.locator("#pitch-select option").count(), 1)

        visible_text = page.locator("body").inner_text()
        self.assertNotRegex(visible_text, r"\bundefined\b|\bNaN\b")
        self.assertEqual(console_errors, [])
        self.assertEqual(page_errors, [])
        self.assertEqual(failed_responses, [])
        self.assertEqual(failed_requests, [])
        context.close()

    def test_neutral_mobile_has_no_horizontal_overflow(self):
        context = self.browser.new_context(viewport={"width": 390, "height": 844})
        page = context.new_page()
        page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")
        overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        self.assertLessEqual(overflow, 1)
        self.assertEqual(page.locator("#pitcher-name").inner_text(), "Select a pitcher to begin")
        context.close()


if __name__ == "__main__":
    unittest.main()
