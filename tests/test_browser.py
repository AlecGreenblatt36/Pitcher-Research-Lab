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

    def _select_fixture(self, page, pitcher_id: int, name: str):
        page.evaluate(
            """([id, name]) => {
                localStorage.setItem('pitcherResearchLab.selectedPitcherId', String(id));
                localStorage.setItem('pitcherResearchLab.selectedPitcherProfile', JSON.stringify({mlbam_id: id, name}));
            }""",
            [pitcher_id, name],
        )
        page.reload(wait_until="networkidle")
        page.locator("#pitcher-loading-overlay").wait_for(state="hidden")
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
        for view in ("overview", "arsenal", "changes", "release", "performance", "location", "career"):
            button = page.locator(f'[data-view="{view}"]')
            self.assertEqual(button.count(), 1, view)
            button.click()
            self.assertIn("active", button.get_attribute("class"))

        release = page.locator('[data-view="release"]')
        release.click()
        self.assertGreater(page.locator(".release-preview-card:not([hidden]) .release-measurement").count(), 0)
        self.assertNotIn("Select a pitcher", page.locator("#release-context-title").inner_text())

        page.locator('[data-view="overview"]').click()
        page.locator("#research-window-start").fill("2026-04-15")
        page.locator("#research-window-end").fill("2026-04-15")
        with page.expect_navigation(wait_until="domcontentloaded"):
            page.locator("#research-window-apply").click()
        page.locator("#pitcher-loading-overlay").wait_for(state="hidden")
        self.assertIn("Custom mode is active", page.locator("#research-window-note").inner_text())
        with page.expect_navigation(wait_until="domcontentloaded"):
            page.locator("#research-window-reset").click()
        page.locator("#pitcher-loading-overlay").wait_for(state="hidden")
        self.assertIn("Optional", page.locator("#research-window-note").inner_text())

        season = page.locator("#research-season-select")
        self.assertIn("2025", season.locator("option").all_text_contents())
        with page.expect_navigation(wait_until="domcontentloaded"):
            season.select_option("2025")
        page.locator("#pitcher-loading-overlay").wait_for(state="hidden")
        self.assertEqual(season.input_value(), "2025")

        pitch = page.locator("#pitch-select")
        options = pitch.locator("option").all()
        self.assertGreater(len(options), 1)
        pitch.select_option(options[1].get_attribute("value"))

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
