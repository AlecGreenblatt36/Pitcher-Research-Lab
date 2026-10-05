from __future__ import annotations

from dataclasses import dataclass

# MLB Stats API postseason codes plus the project's generic postseason code.
# Regular-season extra innings use the automatic runner; postseason games do not.
POSTSEASON_GAME_TYPES = frozenset({"F", "D", "L", "W", "P"})
CLOSED_ROOF_LABELS = frozenset(
    {
        "closed",
        "roof closed",
        "retractable roof closed",
        "indoor",
        "indoors",
        "dome",
    }
)


def normalized_game_type(game_type: str | None) -> str:
    return str(game_type or "").strip().upper()


def automatic_runner_allowed(game_type: str | None) -> bool:
    """Return whether MLB's regular-season automatic runner is allowed.

    Fail closed for unknown game types. The simulator should not invent a free
    runner in a game whose competition rules are not known.
    """

    code = normalized_game_type(game_type)
    if code in POSTSEASON_GAME_TYPES:
        return False
    return code == "R"


def roof_is_closed(roof_status: str | None) -> bool:
    text = str(roof_status or "").strip().lower()
    return text in CLOSED_ROOF_LABELS


def effective_weather_run_factor(
    weather_run_factor: float | int | str | None,
    roof_status: str | None,
) -> float:
    """Neutralize outdoor weather effects when a roof is reported closed."""

    if roof_is_closed(roof_status):
        return 1.0
    try:
        factor = float(1.0 if weather_run_factor is None else weather_run_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("weather_run_factor must be numeric") from exc
    return float(min(1.30, max(0.75, factor)))


@dataclass(frozen=True)
class RuleResolution:
    game_type: str
    automatic_runner_in_extras: bool
    roof_status: str
    input_weather_run_factor: float
    effective_weather_run_factor: float


def resolve_rules(
    *,
    game_type: str | None,
    automatic_runner_requested: bool,
    roof_status: str | None,
    weather_run_factor: float | int | str | None,
) -> RuleResolution:
    code = normalized_game_type(game_type)
    input_factor = float(1.0 if weather_run_factor is None else weather_run_factor)
    effective = effective_weather_run_factor(input_factor, roof_status)
    return RuleResolution(
        game_type=code,
        automatic_runner_in_extras=(
            bool(automatic_runner_requested) and automatic_runner_allowed(code)
        ),
        roof_status=str(roof_status or "unknown"),
        input_weather_run_factor=input_factor,
        effective_weather_run_factor=effective,
    )
