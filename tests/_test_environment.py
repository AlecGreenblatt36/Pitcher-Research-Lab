from __future__ import annotations

import atexit
import os
import shutil
import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path


TEST_ROOT = Path(tempfile.gettempdir()) / "pitcher-research-lab-tests"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
TEST_DATABASE = TEST_ROOT / f"pitcher_research_{os.getpid()}.db"
os.environ["PRL_DATABASE"] = str(TEST_DATABASE)

import pitcher_core  # noqa: E402


PROFILES = {
    100001: ("Veteran Starter", "R", [2024, 2025, 2026], ["FF", "SL", "CH"], "improvement", 10, 36),
    100002: ("Veteran Lefty", "L", [2024, 2025, 2026], ["FF", "SI", "SL"], "deterioration", 10, 36),
    100003: ("Rookie Pitcher", "R", [2026], ["FF", "ST", "CH"], "stable", 10, 36),
    100004: ("Relief Pitcher", "R", [2025, 2026], ["FF", "SL"], "improvement", 12, 24),
    100005: ("Sparse Tracking Pitcher", "R", [2026], ["FF", "CH"], "stable", 8, 30),
    100006: ("Unusual Arsenal Pitcher", "L", [2025, 2026], ["KN", "FS", "CU"], "mixed", 10, 30),
    100007: ("One Pitch Pitcher", "R", [2025, 2026], ["FF"], "stable", 10, 30),
    100008: ("Short Sample Pitcher", "R", [2026], ["FF", "SL"], "stable", 2, 24),
    100009: ("Mixed Results Pitcher", "R", [2024, 2025, 2026], ["FF", "SI", "CU"], "mixed", 10, 36),
}


def _trend(style: str, season_index: int, pitch_index: int) -> tuple[float, float]:
    if season_index == 0:
        return 0.0, 0.0
    if style == "improvement":
        return 1.1 * season_index, -0.018 * season_index
    if style == "deterioration":
        return -1.1 * season_index, 0.018 * season_index
    if style == "mixed":
        return (0.8 if pitch_index == 0 else -0.6) * season_index, (0.012 if pitch_index == 0 else -0.01) * season_index
    return 0.08 * season_index, 0.0


def build_fixture_database() -> None:
    pitcher_core.ensure_core_schema()
    connection = sqlite3.connect(TEST_DATABASE)
    try:
        connection.execute("DELETE FROM official_outings")
        connection.execute("DELETE FROM pitches")
        connection.execute("DELETE FROM pitchers")
        connection.execute("DELETE FROM ingest_runs")

        pitch_columns = [
            "pitch_type", "game_date", "release_speed", "release_pos_x", "release_pos_z",
            "player_name", "batter", "pitcher", "events", "description", "zone", "game_type",
            "stand", "p_throws", "home_team", "away_team", "type", "balls", "strikes",
            "game_year", "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot",
            "launch_speed", "launch_angle", "release_spin_rate", "release_extension", "game_pk",
            "estimated_woba_using_speedangle", "woba_value", "woba_denom", "at_bat_number",
            "pitch_number", "pitch_name", "delta_run_exp", "arm_angle", "season",
        ]
        insert_pitch = (
            f"INSERT INTO pitches ({', '.join(pitch_columns)}) "
            f"VALUES ({', '.join('?' for _ in pitch_columns)})"
        )

        for pitcher_id, profile in PROFILES.items():
            name, hand, seasons, arsenal, style, outing_count, pitches_per_outing = profile
            connection.execute(
                """
                INSERT INTO pitchers (
                    mlbam_id, full_name, first_name, last_name, team_name,
                    team_abbreviation, pitch_hand, primary_position, active,
                    mlb_debut_date, last_profile_refresh, last_statcast_sync, last_official_sync
                ) VALUES (?, ?, ?, ?, 'Fixture Club', 'FXT', ?, 'P', 1, ?, datetime('now'), datetime('now'), datetime('now'))
                """,
                (pitcher_id, name, name.split()[0], name.split()[-1], hand, f"{seasons[0]}-04-01"),
            )

            for season_index, season in enumerate(seasons):
                season_start = date(season, 4, 1)
                for outing_index in range(outing_count):
                    game_date = season_start + timedelta(days=outing_index * 7)
                    game_pk = pitcher_id * 100000 + season * 100 + outing_index
                    connection.execute(
                        """
                        INSERT INTO official_outings (
                            game_pk, pitcher_id, game_date, season, team, opponent, home_away,
                            innings_pitched, outs_recorded, hits, runs, earned_runs, walks,
                            intentional_walks, hit_by_pitch, strikeouts, home_runs, batters_faced,
                            pitches, strikes, balls, updated_at
                        ) VALUES (?, ?, ?, ?, 'FXT', 'OPP', 'home', '6.0', 18, ?, ?, ?, ?, 0, 0, ?, ?, 24, ?, ?, ?, datetime('now'))
                        """,
                        (
                            game_pk, pitcher_id, game_date.isoformat(), season,
                            4 + outing_index % 3,
                            1 + (outing_index % 2 if style == "deterioration" else 0),
                            1 + (outing_index % 2 if style == "deterioration" else 0),
                            1 + (outing_index % 3 if style == "deterioration" else 0),
                            8 + (2 if style == "improvement" and season_index else 0),
                            outing_index % 2,
                            pitches_per_outing, int(pitches_per_outing * 0.66),
                            pitches_per_outing - int(pitches_per_outing * 0.66),
                        ),
                    )

                    rows = []
                    for pitch_number in range(1, pitches_per_outing + 1):
                        pitch_index = (pitch_number - 1) % len(arsenal)
                        pitch_type = arsenal[pitch_index]
                        velocity_shift, run_shift = _trend(style, season_index, pitch_index)
                        outing_wave = ((outing_index % 5) - 2) * 0.14
                        pitch_wave = ((pitch_number % 7) - 3) * 0.05
                        description_cycle = ["swinging_strike", "foul", "ball", "called_strike", "hit_into_play"]
                        description = description_cycle[pitch_number % len(description_cycle)]
                        event = None
                        launch_speed = None
                        launch_angle = None
                        estimated_woba = None
                        woba_value = None
                        woba_denom = None
                        if description == "hit_into_play":
                            event = "single" if pitch_number % 10 else "home_run"
                            launch_speed = 88.0 + (3.5 if style == "deterioration" and season_index else -2.0 if style == "improvement" and season_index else 0.0)
                            launch_angle = 12.0 + pitch_number % 9
                            estimated_woba = 0.32 + (0.04 if style == "deterioration" and season_index else -0.03 if style == "improvement" and season_index else 0.0)
                            woba_value = 0.9 if event == "single" else 2.0
                            woba_denom = 1.0
                        elif description == "swinging_strike" and pitch_number % 3 == 0:
                            event = "strikeout"
                            woba_value = 0.0
                            woba_denom = 1.0

                        sparse = pitcher_id == 100005
                        rows.append((
                            pitch_type, game_date.isoformat(),
                            95.0 - pitch_index * 5.0 + velocity_shift + outing_wave + pitch_wave,
                            None if sparse else -1.75 + season_index * 0.08 + outing_wave * 0.05,
                            None if sparse else 5.75 + season_index * 0.06 + outing_wave * 0.04,
                            name, 600000 + pitch_number, pitcher_id, event, description,
                            5 if abs(((pitch_number % 11) - 5) / 6) < 0.7 else 14,
                            "R", "L" if pitch_number % 2 else "R", hand, "FXT", "OPP",
                            "S" if description in {"swinging_strike", "called_strike", "foul"} else "B",
                            pitch_number % 4, pitch_number % 3, season,
                            0.55 - pitch_index * 0.12, 1.25 - pitch_index * 0.18,
                            ((pitch_number % 11) - 5) / 6, 1.6 + (pitch_number % 9) * 0.22,
                            3.5, 1.5, launch_speed, launch_angle,
                            None if sparse else 2350 - pitch_index * 180 + season_index * 30,
                            None if sparse else 6.3 + season_index * 0.08,
                            game_pk, estimated_woba, woba_value, woba_denom,
                            pitch_number, 1, {"FF": "4-Seam Fastball", "SL": "Slider", "CH": "Changeup", "SI": "Sinker", "ST": "Sweeper", "FS": "Split-Finger", "CU": "Curveball", "KN": "Knuckle Ball"}.get(pitch_type, pitch_type),
                            -0.012 + run_shift + (0.004 if event in {"single", "home_run"} else -0.003 if description == "swinging_strike" else 0.0),
                            None if sparse else 47.0 + season_index * 0.7,
                            season,
                        ))
                    connection.executemany(insert_pitch, rows)

        connection.commit()
    finally:
        connection.close()


build_fixture_database()
atexit.register(lambda: shutil.rmtree(TEST_ROOT, ignore_errors=True))
