from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pitcher_core import (
    comparison_periods_are_valid,
    default_comparison_periods,
    default_transition_window,
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
        return {
            "baseline": {
                "start": self.baseline_start.strftime("%Y-%m-%d"),
                "end": self.baseline_end.strftime("%Y-%m-%d"),
                "label": "Baseline",
            },
            "comparison": {
                "start": self.comparison_start.strftime("%Y-%m-%d"),
                "end": self.comparison_end.strftime("%Y-%m-%d"),
                "label": "Comparison",
            },
            "source": self.source,
            "scope": self.scope,
        }

    def legacy_payload(self) -> dict:
        return {
            "start": self.baseline_end.strftime("%Y-%m-%d"),
            "end": self.comparison_start.strftime("%Y-%m-%d"),
        }


def _timestamp(value, label: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except Exception as exc:
        raise ComparisonError(f"{label} must be a valid date.") from exc
    if pd.isna(result):
        raise ComparisonError(f"{label} must be a valid date.")
    return result.normalize()


def resolve_comparison(args, pitcher_id: int, target_season: int) -> ComparisonContext:
    names = (
        "baseline_start",
        "baseline_end",
        "comparison_start",
        "comparison_end",
    )
    supplied = [args.get(name) for name in names]
    if any(supplied):
        if not all(supplied):
            raise ComparisonError("Baseline and comparison periods each require a start and end date.")
        values = [_timestamp(value, name.replace("_", " ").title()) for name, value in zip(names, supplied)]
        if not comparison_periods_are_valid(pitcher_id, *values):
            raise ComparisonError(
                "Periods must be ordered, must not overlap, and must fall within the pitcher's cached MLB career."
            )
        return ComparisonContext(*values, source="custom", scope="career")

    legacy_start = args.get("start")
    legacy_end = args.get("end")
    if legacy_start or legacy_end:
        if not legacy_start or not legacy_end:
            raise ComparisonError("Comparison boundaries require both a start and end date.")
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
            raise ComparisonError("Comparison dates must fall within the selected scope's available outings.")
        return ComparisonContext(
            start,
            start,
            end,
            end,
            source="legacy_boundaries",
            scope=scope,
            legacy_boundaries=True,
        )

    defaults = default_comparison_periods(pitcher_id, target_season)
    values = [defaults.get(name) for name in names]
    if not all(values):
        fallback_start, fallback_end = default_transition_window(pitcher_id, target_season)
        if not fallback_start or not fallback_end:
            raise ComparisonError("Not enough outings to define comparison periods.")
        start = _timestamp(fallback_start, "Start date")
        end = _timestamp(fallback_end, "End date")
        return ComparisonContext(
            start,
            start,
            end,
            end,
            source="fallback_boundaries",
            scope="season",
            legacy_boundaries=True,
        )

    timestamps = [_timestamp(value, name) for name, value in zip(names, values)]
    return ComparisonContext(
        *timestamps,
        source=str(defaults.get("source") or "automatic"),
        scope="career" if timestamps[0].year < target_season else "season",
    )
