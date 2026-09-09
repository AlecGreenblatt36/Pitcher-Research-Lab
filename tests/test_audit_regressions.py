"""Regression cases for the September 2026 data and comparison audit."""

import shutil
import sqlite3
from urllib.parse import urlencode

import pandas as pd
import pytest

import _test_environment
import app as app_module
import career_routes
import location_routes
import performance_routes
import pitcher_core
import pitcher_routes
import research_routes


@pytest.fixture
def cached_app(tmp_path, monkeypatch):
    database = tmp_path / "audit.db"
    shutil.copy2(_test_environment.TEST_DATABASE, database)
    for module, attribute in (
        (pitcher_core, "DATABASE_FILE"),
        (app_module, "database_file"),
        (career_routes, "DATABASE_FILE"),
        (location_routes, "database_file"),
        (performance_routes, "DATABASE_FILE"),
        (pitcher_routes, "DATABASE_FILE"),
        (research_routes, "database_file"),
    ):
        monkeypatch.setattr(module, attribute, database)
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client(), database


def path(resource, **params):
    return f"/api/pitchers/100001/{resource}?" + urlencode({"season": 2026, **params})


PERIODS = dict(
    baseline_start="2024-04-01",
    baseline_end="2025-06-03",
    comparison_start="2026-04-01",
    comparison_end="2026-06-03",
)


def test_contact_rates_exclude_fouls_and_all_run_value_denominators_agree(cached_app):
    client, database = cached_app
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM pitches WHERE pitcher = 100001 AND rowid NOT IN "
            "(SELECT MIN(rowid) FROM pitches WHERE pitcher = 100001 GROUP BY game_pk, pitch_type)"
        )
        connection.execute(
            "DELETE FROM pitches WHERE pitcher = 100001 AND pitch_type = 'CH'"
        )
        connection.execute(
            "UPDATE pitches SET pitch_type = 'FF', plate_x = 0, plate_z = 2.5, sz_bot = 1.5, sz_top = 3.5, "
            "description = CASE WHEN pitch_type = 'FF' THEN 'foul' ELSE 'hit_into_play' END, "
            "launch_speed = CASE WHEN pitch_type = 'FF' THEN 110 ELSE 100 END, "
            "delta_run_exp = CASE WHEN pitch_type = 'FF' THEN 1 ELSE NULL END WHERE pitcher = 100001"
        )
    research = client.get(path("research", **PERIODS)).get_json()
    for row in research["pitches"] + research["overall"]:
        assert row["hard_hit_pct"] == 100
        assert row["avg_ev"] == 100
        assert row["run_value_per_100"] == -100
    performance = client.get(path("performance", **PERIODS)).get_json()
    for outing in performance["outings"]:
        assert outing["process"]["pitch_value_per_100"] == -100
        assert outing["process"]["run_value_pitches"] == 1
    career = client.get(path("career", **PERIODS)).get_json()
    assert all(
        row["process"]["pitch_value_per_100"] == -100
        for row in career["overall_outings"]
    )
    assert all(row["pitch_value_per_100"] == -100 for row in career["pitch_outings"])
    location = client.get(
        path("location", pitch="FF", hand="ALL", **PERIODS)
    ).get_json()
    for period in ["early", "post"]:
        assert location["periods"][period]["summary"]["run_value_per_100"] == -100


def test_missing_contact_estimates_stay_missing_but_noncontact_values_count(cached_app):
    _, database = cached_app
    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM pitches WHERE pitcher = 100001 LIMIT 2", connection
        )
    frame["events"] = ["home_run", "strikeout"]
    frame["description"] = ["hit_into_play", "swinging_strike"]
    frame["estimated_woba_using_speedangle"] = None
    frame["woba_value"] = [2, 0]
    frame["woba_denom"] = 1
    for flags, summary in [
        (performance_routes.add_pitch_flags, performance_routes.process_summary),
        (career_routes.add_flags, career_routes.overall_process_summary),
    ]:
        data = flags(frame)
        assert pd.isna(data.iloc[0]["xwoba_component"])
        assert data.iloc[1]["xwoba_component"] == 0
        result = summary(data)
        assert result["xwoba_allowed"] == 0
        assert result["xwoba_coverage_pct"] == 50
        assert result["xwoba_missing_contact"] == 1
        assert summary(data.iloc[:1])["xwoba_allowed"] is None
        data["pitcher_run_value"] = float("nan")
        assert summary(data)["pitch_value_per_100"] is None


def test_every_comparison_endpoint_uses_the_same_explicit_dates(cached_app):
    client, _ = cached_app
    payloads = {}
    for resource in [
        "research",
        "performance",
        "location",
        "career",
        "release",
        "changes",
    ]:
        response = client.get(path(resource, pitch="FF", hand="ALL", **PERIODS))
        assert response.status_code == 200, response.get_json()
        payload = response.get_json()
        payloads[resource] = (
            payload[0]["comparison_periods"]
            if resource == "changes"
            else payload["comparison_periods"]
        )
    assert all(value == payloads["research"] for value in payloads.values())
    default = client.get(path("research")).get_json()
    assert {row["period"]: row["outings"] for row in default["overall"]} == {
        "early": 20,
        "post": 10,
    }


@pytest.mark.parametrize(
    "dates",
    [
        {**PERIODS, "baseline_start": "2024-04-01T00:00:00Z"},
        {**PERIODS, "baseline_start": "2024-02-30"},
        {**PERIODS, "baseline_start": "2024-04-02", "baseline_end": "2024-04-03"},
        {**PERIODS, "comparison_start": "2026-04-02", "comparison_end": "2026-04-03"},
    ],
)
def test_bad_or_empty_periods_return_actionable_400(cached_app, dates):
    client, _ = cached_app
    for resource in [
        "research",
        "performance",
        "location",
        "career",
        "release",
        "changes",
    ]:
        response = client.get(path(resource, pitch="FF", **dates))
        assert response.status_code == 400, (resource, response.get_json())
        assert response.get_json()["error"]


def test_release_is_exact_pitch_and_survives_zero_variance_and_screen_ranking(
    cached_app,
):
    client, database = cached_app
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE pitches SET release_pos_x=-1.75, release_pos_z=6.1 WHERE pitcher=100001 AND pitch_type='FF'"
        )
    changes = client.get(path("changes")).get_json()
    assert not any(
        row["pitch_type"] == "FF" and row["metric_key"] == "release_pos_x"
        for row in changes
    )
    release = client.get(path("release", pitch="FF")).get_json()["measurements"]
    assert len(release) == 4
    assert all(row["pitch_type"] == "FF" for row in release)
    horizontal = next(row for row in release if row["metric_key"] == "release_pos_x")
    assert horizontal["current_mean"] == -1.75
    assert horizontal["change"] == 0
    assert horizontal["screen_eligible"] is False
    assert (
        client.get(path("release", pitch="NOT_A_PITCH")).get_json()["measurements"]
        == []
    )
    # A broad arsenal produces more than 30 valid screen rows; measurement data stay complete.
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(pitches)")]
        quoted = ", ".join(f'"{column}"' for column in columns)
        for index, pitch in enumerate(["SI", "ST", "CU", "FC", "FS", "KN"], start=1):
            expressions = ", ".join(
                (
                    "?"
                    if column == "pitch_type"
                    else (
                        f"at_bat_number + {index * 1000}"
                        if column == "at_bat_number"
                        else f'"{column}"'
                    )
                )
                for column in columns
            )
            connection.execute(
                f"INSERT INTO pitches ({quoted}) SELECT {expressions} FROM pitches WHERE pitcher=100001 AND pitch_type='SL'",
                (pitch,),
            )
    assert len(client.get(path("changes")).get_json()) > 30
    assert len(client.get(path("release", pitch="FF")).get_json()["measurements"]) == 4


def test_doubleheaders_remain_separate_outings_in_research_and_screen(cached_app):
    client, database = cached_app
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(pitches)")]
        quoted = ", ".join(f'"{column}"' for column in columns)
        expressions = ", ".join(
            "game_pk + 1" if column == "game_pk" else f'"{column}"'
            for column in columns
        )
        connection.execute(
            f"INSERT INTO pitches ({quoted}) SELECT {expressions} FROM pitches WHERE pitcher=100001 AND game_date='2026-06-03'"
        )
    research = client.get(path("research")).get_json()
    assert (
        next(row for row in research["overall"] if row["period"] == "post")["outings"]
        == 11
    )
    changes = client.get(path("changes")).get_json()
    assert changes and all(row["current_outings"] == 11 for row in changes)
