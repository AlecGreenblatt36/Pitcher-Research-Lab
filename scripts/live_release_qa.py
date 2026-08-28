from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


PITCHERS = ("Mason Miller", "Garrett Crochet", "Cade Horton")
VIEWS = ("overview", "arsenal", "changes", "release", "performance", "location", "career")


def wait_for_pitcher(page, name: str) -> None:
    page.wait_for_load_state("domcontentloaded")
    page.locator("#pitcher-name").wait_for(state="visible", timeout=30_000)
    page.locator("#pitcher-loading-overlay").wait_for(state="hidden", timeout=900_000)
    displayed = page.locator("#pitcher-name").inner_text()
    if name.lower() not in displayed.lower():
        raise AssertionError(f"Expected {name!r}; displayed {displayed!r}")
    page.wait_for_function(
        "() => !document.getElementById('overview-primary-finding')?.textContent.includes('Loading')",
        timeout=120_000,
    )


def select_pitcher(page, name: str) -> None:
    search = page.locator("#pitcher-search-input")
    search.fill(name)
    results = page.locator("#pitcher-search-results")
    results.wait_for(state="visible", timeout=30_000)
    match = results.locator("button").filter(has_text=name)
    match.wait_for(state="visible", timeout=30_000)
    if match.count() != 1:
        raise AssertionError(f"Expected one search result for {name}; found {match.count()}")
    match.click()
    wait_for_pitcher(page, name)


def visible_text_has_invalid_values(page) -> bool:
    text = page.locator("body").inner_text()
    return "undefined" in text or "NaN" in text


def run(base_url: str, output: Path, screenshots: Path, pitchers: tuple[str, ...] = PITCHERS) -> dict:
    screenshots.mkdir(parents=True, exist_ok=True)
    report = {"base_url": base_url, "pitchers": [], "console_errors": [], "page_errors": [], "http_errors": [], "request_failures": []}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.on("response", lambda response: report["http_errors"].append({"status": response.status, "url": response.url}) if response.status >= 400 else None)
        page.on("requestfailed", lambda request: report["request_failures"].append(request.url))

        page.goto(base_url, wait_until="networkidle")
        page.evaluate("() => localStorage.clear()")
        page.reload(wait_until="networkidle")
        if page.locator("#pitcher-name").inner_text() != "Select a pitcher to begin":
            raise AssertionError("Fresh browser context did not open in the neutral state.")
        report["neutral_state"] = True
        page.screenshot(path=screenshots / "landing-neutral.png", full_page=True)
        previous_name = None

        for index, name in enumerate(pitchers):
            select_pitcher(page, name)
            body_text = page.locator("body").inner_text()
            if previous_name and previous_name in body_text:
                raise AssertionError(f"Stale pitcher data remained after switching from {previous_name} to {name}.")

            pitcher = {
                "name": page.locator("#pitcher-name").inner_text(),
                "mlbam_id": page.evaluate("() => Number(localStorage.getItem('pitcherResearchLab.selectedPitcherId'))"),
                "views": [],
                "seasons": page.locator("#research-season-select option").all_text_contents(),
                "pitches": page.locator("#pitch-select option").all_text_contents(),
            }

            for view in VIEWS:
                button = page.locator(f'[data-view="{view}"]')
                if button.count() != 1:
                    raise AssertionError(f"Navigation target {view!r} is missing or duplicated.")
                button.click()
                if "active" not in (button.get_attribute("class") or ""):
                    raise AssertionError(f"Navigation target {view!r} did not become active.")
                if visible_text_has_invalid_values(page):
                    raise AssertionError(f"Visible undefined/NaN value on {view!r} for {name}.")
                pitcher["views"].append(view)

            page.locator('[data-view="overview"]').click()
            if index == 0:
                page.screenshot(path=screenshots / "overview-live.png", full_page=True)
                page.locator('[data-view="changes"]').click()
                page.screenshot(path=screenshots / "change-detection-live.png", full_page=True)
                page.locator('[data-view="release"]').click()
                page.wait_for_function(
                    "() => !document.getElementById('release-context-title')?.textContent.includes('Select a pitcher')",
                    timeout=120_000,
                )
                page.screenshot(path=screenshots / "release-profile-live.png", full_page=True)

            if len(pitcher["seasons"]) > 1:
                original_season = page.locator("#research-season-select").input_value()
                alternate = next(value for value in page.locator("#research-season-select option").evaluate_all("options => options.map(option => option.value)") if value != original_season)
                with page.expect_navigation(wait_until="domcontentloaded"):
                    page.locator("#research-season-select").select_option(alternate)
                wait_for_pitcher(page, name)
                selected_season = page.locator("#research-season-select").input_value()
                if selected_season != alternate:
                    raise AssertionError(f"Season switch for {name} selected {selected_season!r}, expected {alternate!r}.")
                pitcher["season_switch"] = {"from": original_season, "to": selected_season}

            pitch_values = page.locator("#pitch-select option").evaluate_all("options => options.map(option => option.value)")
            if len(pitch_values) > 1:
                page.locator("#pitch-select").select_option(pitch_values[1], force=True)
                pitcher["pitch_switch"] = pitch_values[1]

            report["pitchers"].append(pitcher)
            previous_name = name

        context.close()
        browser.close()

    for category in ("console_errors", "page_errors", "http_errors", "request_failures"):
        if report[category]:
            raise AssertionError(f"Live browser QA recorded {category}: {report[category]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live Pitcher Research Lab release browser QA.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5055/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--pitcher", action="append", dest="pitchers")
    args = parser.parse_args()
    pitchers = tuple(args.pitchers) if args.pitchers else PITCHERS
    report = run(args.base_url, args.output, args.screenshots, pitchers)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
