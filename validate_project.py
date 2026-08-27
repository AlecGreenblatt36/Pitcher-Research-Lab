from __future__ import annotations

import py_compile
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "data" / "pitcher_research.db"


def check_python() -> list[str]:
    failures = []
    paths = list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"Python syntax: {path.name}: {exc.msg}")
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


def check_source_guards() -> list[str]:
    failures = []
    for path in (ROOT / "static").glob("*.js"):
        text = path.read_text(encoding="utf-8")
        if "/api/skenes/" in text:
            failures.append(f"Frontend legacy route found in {path.name}")
        for date_text in ("2026-05-06", "2026-06-09"):
            if date_text in text:
                failures.append(f"Case-study date dependency {date_text} found in {path.name}")
    return failures


def check_repository_hygiene() -> list[str]:
    failures = []
    ignored_directories = {".venv", "__pycache__", ".pytest_cache"}
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
        for required in (".venv/", "__pycache__/", "*.py[cod]", ".pytest_cache/"):
            if required not in ignored:
                failures.append(f".gitignore is missing {required}")
    return failures


def check_database() -> list[str]:
    if not DATABASE.exists():
        # A clean clone creates the cache on first launch. Fresh-schema behavior
        # is covered by the regression suite.
        return []

    failures = []
    connection = sqlite3.connect(DATABASE)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            failures.append(f"SQLite integrity check: {integrity}")

        required_tables = {"pitches", "pitchers", "official_outings", "ingest_runs", "schema_version"}
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
        ("Python syntax", check_python),
        ("JavaScript syntax", check_javascript),
        ("Source guards", check_source_guards),
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
