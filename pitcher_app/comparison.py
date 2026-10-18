from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing

import pandas as pd
import re

from pitcher_core import (
    connect_database,
    comparison_periods_are_valid,
    default_comparison_periods,
    research_window_is_within_career,
    research_window_is_within_season,
)


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class ComparisonContext:
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp
    comparison_start: pd.Timestamp
    comparison_end: pd.Timestamp
    source: str
    scope: str
    legacy_boundaries: bool = False

    def classify(self, game_date) -> str | None:
        value = pd.Timestamp(game_date)
        if self.legacy_boundaries:
            if value < self.baseline_end:
                return "early"
            if value <= self.comparison_start:
                return "transition"
            return "post"
        if self.baseline_start <= value <= self.baseline_end:
            return "early"
        if self.comparison_start <= value <= self.comparison_end:
            return "post"
        if self.baseline_end < value < self.comparison_start:
            return "transition"
        return None

    def payload(self) -> dict:
        baseline_end = (
            self.baseline_end - pd.Timedelta(days=1)
            if self.legacy_boundaries
            else self.baseline_end
        )
        comparison_start = (
            self.comparison_start + pd.Timedelta(days=1)
            if self.legacy_boundaries
            else self.comparison_start
        )

        def period(start, end, label):
            # A legacy boundary at the edge of coverage can leave an empty side.
            return {
                "start": start.strftime("%Y-%m-%d") if start <= end else None,
                "end": end.strftime("%Y-%m-%d") if start <= end else None,
                "label": label,
            }

        return {
            "baseline": period(self.baseline_start, baseline_end, "Baseline"),
            "comparison": period(comparison_start, self.comparison_end, "Comparison"),
            "source": self.source,
            "scope": self.scope,
        }

    def legacy_payload(self) -> dict:
        return {
            "start": self.baseline_end.strftime("%Y-%m-%d"),
            "end": self.comparison_start.strftime("%Y-%m-%d"),
        }


def _timestamp(value, label: str) -> pd.Timestamp:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ComparisonError(
            f"{label} must use YYYY-MM-DD without a time or timezone."
        )
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError("Missing date")
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise ComparisonError(f"{label} must be a valid date.") from exc


def resolve_comparison(args, pitcher_id: int, target_season: int) -> ComparisonContext:
    names = ("baseline_start", "baseline_end", "comparison_start", "comparison_end")
    supplied = [args.get(name) for name in names]
    if any(supplied):
        if not all(supplied):
            raise ComparisonError(
                "Baseline and comparison periods each require a start and end date."
            )
        values = [
            _timestamp(value, name.replace("_", " ").title())
            for name, value in zip(names, supplied)
        ]
        if not comparison_periods_are_valid(pitcher_id, *values):
            raise ComparisonError(
                "Periods must be ordered, must not overlap, and must fall within the pitcher's cached MLB career."
            )
        with closing(connect_database()) as connection:
            for label, start, end in (
                ("Baseline", values[0], values[1]),
                ("Comparison", values[2], values[3]),
            ):
                count = connection.execute(
                    "SELECT COUNT(*) FROM pitches WHERE game_type = 'R' "
                    "AND CAST(pitcher AS INTEGER) = ? AND game_date BETWEEN ? AND ?",
                    (
                        int(pitcher_id),
                        start.strftime("%Y-%m-%d"),
                        end.strftime("%Y-%m-%d"),
                    ),
                ).fetchone()[0]
                if not count:
                    raise ComparisonError(
                        f"{label} period has no cached outings. Choose dates containing an outing."
                    )
        return ComparisonContext(*values, source="custom", scope="career")
    legacy_start = args.get("start")
    legacy_end = args.get("end")
    if legacy_start or legacy_end:
        if not legacy_start or not legacy_end:
            raise ComparisonError(
                "Comparison boundaries require both a start and end date."
            )
        start = _timestamp(legacy_start, "Start date")
        end = _timestamp(legacy_end, "End date")
        scope = str(args.get("scope", "season")).lower()
        if scope not in {"season", "career"}:
            raise ComparisonError("Scope must be season or career.")
        valid = (
            research_window_is_within_career(pitcher_id, start, end)
            if scope == "career"
            else research_window_is_within_season(pitcher_id, target_season, start, end)
        )
        if not valid:
            raise ComparisonError(
                "Comparison dates must fall within the selected scope's available outings."
            )
        with closing(connect_database()) as connection:
            query = "SELECT MIN(game_date), MAX(game_date) FROM pitches WHERE game_type='R' AND CAST(pitcher AS INTEGER)=?"
            params = [int(pitcher_id)]
            if scope == "season":
                query += " AND CAST(season AS INTEGER)=?"
                params.append(int(target_season))
            first, last = connection.execute(query, params).fetchone()
        return ComparisonContext(
            pd.Timestamp(first),
            start,
            end,
            pd.Timestamp(last),
            source="legacy_boundaries",
            scope=scope,
            legacy_boundaries=True,
        )
    defaults = default_comparison_periods(pitcher_id, target_season)
    values = [defaults.get(name) for name in names]
    if not all(values):
        raise ComparisonError(
            "At least two outing dates are needed to define separate comparison periods."
        )
    timestamps = [_timestamp(value, name) for name, value in zip(names, values)]
    return ComparisonContext(
        *timestamps,
        source=str(defaults.get("source") or "automatic"),
        scope="career" if timestamps[0].year < target_season else "season",
    )
