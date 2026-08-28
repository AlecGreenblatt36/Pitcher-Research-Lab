from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATABASE = Path(os.environ.get("PRL_DATABASE", ROOT / "data" / "pitcher_research.db"))


def check_required_structure() -> list[str]:
    required = (
        "app.py",
        "schema.sql",
        "requirements.txt",
        "templates/dashboard.html",
        "static/navigation.js",
        "static/pitcher_context.js",
        "tests/test_browser.py",
    )
    return [f"Missing required project file: {path}" for path in required if not (ROOT / path).exists()]


def check_python() -> list[str]:
    failures = []
    paths = list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    for path in paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"Python syntax: {path.relative_to(ROOT)}: {exc.msg}")
    return failures


def check_javascript() -> list[str]:
    node = shutil.which("node")
    if not node:
        return []

    failures = []
    for path in (ROOT / "static").glob("*.js"):
        result = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            failures.append(f"JavaScript syntax: {path.name}: {result.stderr.strip()}")
    return failures


def check_frontend_dependencies() -> list[str]:
    template = ROOT / "templates" / "dashboard.html"
    if not template.exists():
        return ["Template is missing: templates/dashboard.html"]

    source = template.read_text(encoding="utf-8")
    references = set(
        re.findall(
            r"url_for\(['\"]static['\"],\s*filename=['\"]([^'\"]+)",
            source,
        )
    )
    for path in (ROOT / "static").glob("*.js"):
        text = path.read_text(encoding="utf-8")
        references.update(re.findall(r"/static/([A-Za-z0-9_.-]+)", text))

    static_directory = ROOT / "static"
    actual_names = {
        path.name for path in static_directory.iterdir() if path.is_file()
    } if static_directory.exists() else set()

    failures = []
    for reference in sorted(references):
        if reference in actual_names:
            continue
        case_match = next(
            (name for name in actual_names if name.lower() == reference.lower()),
            None,
        )
        if case_match:
            failures.append(
                f"Static reference capitalization mismatch: {reference} != {case_match}"
            )
        else:
            failures.append(f"Missing static dependency: static/{reference}")
    return failures


def check_secrets() -> list[str]:
    patterns = {
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
        "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    }
    text_suffixes = {
        ".py", ".js", ".html", ".css", ".md", ".txt",
        ".yml", ".yaml", ".json", ".bat", ".sql",
    }
    ignored_directories = {".venv", "__pycache__", ".pytest_cache", ".git"}

    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if any(part in ignored_directories for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in patterns.items():
            if pattern.search(text):
                failures.append(f"Potential {label}: {path.relative_to(ROOT)}")
    return failures


def check_repository_hygiene() -> list[str]:
    failures = []
    ignored_directories = {".venv", "__pycache__", ".pytest_cache", ".git"}

    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored_directories for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in {".exe", ".dll"}:
            failures.append(f"Binary file should not be committed: {relative}")

    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        failures.append(".gitignore is missing.")
    else:
        ignored = gitignore.read_text(encoding="utf-8")
        for required in (".venv/", "__pycache__/", "*.py[cod]", ".pytest_cache/", "data/"):
            if required not in ignored:
                failures.append(f".gitignore is missing {required}")
    return failures


def check_database() -> list[str]:
    if not DATABASE.exists():
        return []

    failures = []
    connection = sqlite3.connect(DATABASE)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            failures.append(f"SQLite integrity check: {integrity}")

        required_tables = {
            "pitches", "pitchers", "official_outings", "ingest_runs", "schema_version"
        }
        found = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required_tables - found
        if missing:
            failures.append("Missing tables: " + ", ".join(sorted(missing)))

        if "pitches" in found:
            duplicate_count = connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT pitcher, game_pk, at_bat_number, pitch_number, COUNT(*) AS n
                    FROM pitches
                    WHERE pitcher IS NOT NULL
                      AND game_pk IS NOT NULL
                      AND at_bat_number IS NOT NULL
                      AND pitch_number IS NOT NULL
                    GROUP BY pitcher, game_pk, at_bat_number, pitch_number
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            if duplicate_count:
                failures.append(f"Duplicate pitch identities: {duplicate_count}")
    finally:
        connection.close()
    return failures


def main() -> int:
    checks = [
        ("Required structure", check_required_structure),
        ("Python syntax", check_python),
        ("JavaScript syntax", check_javascript),
        ("Frontend dependencies", check_frontend_dependencies),
        ("Secret scan", check_secrets),
        ("Repository hygiene", check_repository_hygiene),
        ("Database integrity", check_database),
    ]

    failures = []
    for label, check in checks:
        current = check()
        if current:
            failures.extend(current)
            print(f"[FAIL] {label}")
        else:
            print(f"[PASS] {label}")

    if failures:
        print("\nValidation failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPitcher Research Lab validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
