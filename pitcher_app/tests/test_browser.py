from __future__ import annotations

import threading
import os
from pathlib import Path
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
        page.wait_for_function("""() => {
                const title = document.getElementById('release-context-title');
                return title && !title.textContent.includes('Select a pitcher') && !title.textContent.includes('Loading');
            }""")

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
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on(
            "response",
            lambda response: (
                failed_responses.append(f"{response.status} {response.url}")
                if response.status >= 400
                else None
            ),
        )
        # Requests cut off by the page's own reloads (research window, season) are not failures.
        page.on(
            "requestfailed",
            lambda request: (
                failed_requests.append(request.url)
                if "ERR_ABORTED" not in str(request.failure or "")
                else None
            ),
        )

        page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")
        self.assertEqual(
            page.locator("#pitcher-name").inner_text(), "Select a pitcher to begin"
        )
        self.assertEqual(
            page.locator("#database-status").inner_text(),
            "Waiting for pitcher selection",
        )
        self.assertEqual(page.locator("#pitcher-loading-overlay").count(), 0)

        self._select_fixture(page, 100001, "Veteran Starter")
        page.locator("#career-audit-panel").wait_for(state="attached")

        for view in (
            "overview",
            "arsenal",
            "changes",
            "release",
            "performance",
            "location",
            "career",
        ):
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
        self.assertTrue(
            page.locator('[data-view-panel="changes"] .view-page-header').is_visible()
        )
        self.assertFalse(page.locator("#career-audit-panel").is_visible())

        page.locator('[data-view="career"]').click()
        self.assertTrue(page.locator("#career-audit-panel").is_visible())
        self.assertFalse(
            page.locator('[data-view-panel="changes"] .view-page-header').is_visible()
        )

        release = page.locator('[data-view="release"]')
        release.click()
        self._wait_for_release_ready(page)
        self.assertGreater(
            page.locator(
                ".release-preview-card:not([hidden]) .release-measurement"
            ).count(),
            0,
        )
        self.assertNotIn(
            "Select a pitcher", page.locator("#release-context-title").inner_text()
        )

        page.locator('[data-view="overview"]').click()
        page.locator("#research-baseline-start").fill("2024-04-01")
        page.locator("#research-baseline-end").fill("2025-06-03")
        page.locator("#research-comparison-start").fill("2026-04-01")
        page.locator("#research-comparison-end").fill("2026-06-03")
        with page.expect_navigation(wait_until="domcontentloaded"):
            page.locator("#research-window-apply").click()
        self._wait_for_pitcher_ready(page)
        self.assertIn(
            "Custom mode is active", page.locator("#research-window-note").inner_text()
        )
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

        # The primary pitch selector belongs to the arsenal and release views; the page
        # restores the last view (overview here) after the reload, so open the arsenal view.
        page.locator('[data-view="arsenal"]').click()
        pitch = page.locator("#pitch-select")
        try:
            pitch.wait_for(state="visible", timeout=15000)
        except Exception:
            pass
        self.assertTrue(
            pitch.is_visible(),
            "Primary pitch selector should be visible on the arsenal view after season reload",
        )
        values = pitch.locator("option").evaluate_all(
            "options => options.map(option => option.value)"
        )
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

    def test_release_refreshes_when_primary_pitch_changes(self):
        context = self.browser.new_context()
        page = context.new_page()
        page.goto(f"http://127.0.0.1:{self.port}/")
        self._select_fixture(page, 100001, "Veteran Starter")
        page.locator('[data-view="release"]').click()
        for pitch in ("SL", "FF"):
            page.locator("#pitch-select").select_option(pitch)
            page.wait_for_function(
                "pitch => document.getElementById('release-context-title').textContent.startsWith(pitch + ':')",
                arg=pitch,
            )
            for text in page.locator(
                ".release-preview-card:not([hidden]) .release-preview-detail"
            ).all_text_contents():
                self.assertTrue(text.startswith(pitch + ":"), text)
            self.assertTrue(
                page.locator("#release-x-detail").inner_text().startswith(pitch + ":")
            )
        context.close()

    def test_cached_data_survives_failed_refresh(self):
        context = self.browser.new_context()
        page = context.new_page()

        def stale_meta(route):
            payload = route.fetch().json()
            payload["pitcher"]["last_statcast_sync"] = None
            route.fulfill(json=payload)

        page.route("**/api/pitchers/100001/meta**", stale_meta)
        page.route(
            "**/api/pitchers/100001/sync",
            lambda route: route.fulfill(
                status=502, json={"error": "Refresh unavailable"}
            ),
        )
        page.goto(f"http://127.0.0.1:{self.port}/")
        self._select_fixture(page, 100001, "Veteran Starter")
        self.assertIn(
            "Using cached data through",
            page.locator("#pitcher-sync-warning").inner_text(),
        )
        self.assertTrue(page.get_by_role("button", name="Retry update").is_enabled())
        page.locator('[data-view="release"]').click()
        self._wait_for_release_ready(page)
        self.assertGreater(
            page.locator(".release-preview-card:not([hidden])").count(), 0
        )
        self.assertEqual(
            page.locator("#database-status").get_attribute("data-state"), "warning"
        )
        context.close()

    def test_uncached_failure_offers_retry_and_new_selection(self):
        context = self.browser.new_context()
        page = context.new_page()

        def no_cache(route):
            payload = route.fetch().json()
            payload["database"]["pitch_rows"] = 0
            route.fulfill(json=payload)

        page.route("**/api/pitchers/100001/meta**", no_cache)
        page.route(
            "**/api/pitchers/100001/sync",
            lambda route: route.fulfill(
                status=502, json={"error": "Refresh unavailable"}
            ),
        )
        page.goto(f"http://127.0.0.1:{self.port}/")
        page.evaluate(
            "localStorage.setItem('pitcherResearchLab.selectedPitcherId', '100001')"
        )
        page.reload()
        page.get_by_role("button", name="Choose another pitcher").wait_for(
            state="visible"
        )
        self.assertTrue(
            page.get_by_role("button", name="Retry", exact=True).is_enabled()
        )
        with page.expect_navigation():
            page.get_by_role("button", name="Choose another pitcher").click()
        self._wait_for_pitcher_ready(page)
        self.assertEqual(
            page.locator("#pitcher-name").inner_text(), "Select a pitcher to begin"
        )
        context.close()

    def test_missing_location_metrics_display_unavailable(self):
        context = self.browser.new_context()
        page = context.new_page()

        def missing_metrics(route):
            payload = route.fetch().json()
            for name in ("early", "post"):
                for key in ("whiff_pct", "hard_hit_pct", "run_value_per_100"):
                    payload["periods"][name]["summary"][key] = None
            route.fulfill(json=payload)

        page.route("**/api/pitchers/100001/location?**", missing_metrics)
        page.goto(f"http://127.0.0.1:{self.port}/")
        self._select_fixture(page, 100001, "Veteran Starter")
        page.locator('[data-view="location"]').click()
        page.wait_for_function(
            "() => document.getElementById('location-finding-text').textContent.includes('lack eligible data')"
        )
        for key in ("whiff", "hard-hit", "rv"):
            self.assertEqual(page.locator(f"#location-delta-{key}").inner_text(), "--")
        self.assertNotIn(
            "improved by 0", page.locator("#location-finding-text").inner_text()
        )
        context.close()

    def test_loaded_pages_fit_desktop_and_mobile(self):
        artifacts = os.environ.get("PRL_SCREENSHOT_DIR")
        if artifacts:
            Path(artifacts).mkdir(parents=True, exist_ok=True)
        for width, height in ((1440, 1000), (390, 844)):
            context = self.browser.new_context(
                viewport={"width": width, "height": height}
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")
            self._select_fixture(page, 100001, "Veteran Starter")
            page.wait_for_load_state("networkidle")
            for view in (
                "overview",
                "arsenal",
                "changes",
                "release",
                "performance",
                "location",
                "career",
            ):
                page.locator(f'[data-view="{view}"]').click()
                panel = page.locator(f'.app-view.active[data-view-panel="{view}"]')
                self.assertTrue(panel.is_visible())
                if view == "overview":
                    self.assertIn(
                        "usage was unchanged",
                        page.locator("#overview-arsenal-title").inner_text(),
                    )
                    self.assertNotIn(
                        "redistributed",
                        page.locator("#research-signal-text").inner_text(),
                    )
                self.assertNotRegex(panel.inner_text(), r"\bundefined\b|\bNaN\b")
                if artifacts:
                    page.screenshot(
                        path=str(Path(artifacts) / f"{width}-{view}.png"),
                        full_page=True,
                    )
                overflow = page.evaluate(
                    "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
                )
                if overflow > 1:
                    offenders = page.evaluate(
                        """() => [...document.querySelectorAll('.main-content *')]
                        .filter(el => {const r = el.getBoundingClientRect(); return r.width && r.right > innerWidth;})
                        .slice(0, 15).map(el => ({tag: el.tagName, id: el.id, class: el.className, right: el.getBoundingClientRect().right}))"""
                    )
                else:
                    offenders = []
                self.assertLessEqual(overflow, 1, f"{width}px {view}: {offenders}")
            context.close()

    def test_neutral_mobile_has_no_horizontal_overflow(self):
        context = self.browser.new_context(viewport={"width": 390, "height": 844})
        page = context.new_page()
        page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        self.assertLessEqual(overflow, 1)
        self.assertEqual(
            page.locator("#pitcher-name").inner_text(), "Select a pitcher to begin"
        )
        context.close()


if __name__ == "__main__":
    unittest.main()
