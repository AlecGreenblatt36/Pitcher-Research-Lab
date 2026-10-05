from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

SPRINT_SPEED_URLS = (
    "https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={year}&position=&team=&min={minimum}&csv=true",
    "https://baseballsavant.mlb.com/sprint_speed_leaderboard?year={year}&position=&team=&min_run={minimum}&csv=true",
)
OAA_URL = (
    "https://baseballsavant.mlb.com/leaderboard/outs_above_average?"
    "type=Fielder&startYear={year}&endYear={year}&split=no&team=&range=year&"
    "min={minimum}&pos={position}&roles=&viz=hide&csv=true"
)


class TraitDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class TraitSnapshotReceipt:
    trait: str
    season: int
    captured_at_utc: str
    source_url: str
    rows: int
    player_id_column: str
    value_columns: tuple[str, ...]
    replay_policy: str


@dataclass(frozen=True)
class SprintSpeedTrait:
    player_id: int
    sprint_speed_ft_s: float
    competitive_runs: int | None
    speed_percentile: float
    imputed: bool = False


@dataclass(frozen=True)
class OAATrait:
    player_id: int
    position: str
    outs_above_average: float
    attempts: int | None
    team: str


def _normalized_column(value: object) -> str:
    text = str(value).strip().lower()
    for old, new in (("%", "pct"), ("/", "_"), (" ", "_"), ("-", "_"), (".", "")):
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [_normalized_column(column) for column in output.columns]
    return output


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise TraitDataError(
        f"none of the required columns {tuple(candidates)} were present; got {sorted(available)}"
    )


def _download_csv(
    urls: Sequence[str],
    *,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> tuple[pd.DataFrame, str]:
    client = session or requests.Session()
    errors: list[str] = []
    for url in urls:
        try:
            response = client.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "BaseballResearchLab/1.0 (+public research)"},
            )
            response.raise_for_status()
            if not response.content or b"," not in response.content[:4000]:
                raise TraitDataError("response did not look like CSV")
            frame = pd.read_csv(io.BytesIO(response.content), low_memory=False)
            if frame.empty:
                raise TraitDataError("CSV was empty")
            return normalize_columns(frame), url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise TraitDataError("all Baseball Savant CSV endpoints failed: " + " | ".join(errors))


def replay_trait_season(game_year: int) -> int:
    """Use the previous completed season in leakage-controlled historical replay."""

    year = int(game_year)
    if year < 2018:
        raise ValueError("Sprint Speed is not available for this replay period")
    return year - 1


def fetch_sprint_speed(
    season: int,
    *,
    minimum_opportunities: int = 0,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> tuple[dict[int, SprintSpeedTrait], TraitSnapshotReceipt]:
    urls = tuple(
        template.format(year=int(season), minimum=int(minimum_opportunities))
        for template in SPRINT_SPEED_URLS
    )
    frame, used_url = _download_csv(urls, session=session, timeout=timeout)
    id_column = _find_column(
        frame.columns,
        ("player_id", "playerid", "entity_id", "id", "mlbam_id"),
    )
    speed_column = _find_column(
        frame.columns,
        ("sprint_speed", "sprint_speed_ft_sec", "r_sprint_speed_top50percent"),
    )
    opportunity_column = next(
        (
            column
            for column in (
                "competitive_runs",
                "competitive_run_count",
                "r_sprint_speed_count",
                "runs",
            )
            if column in frame.columns
        ),
        None,
    )
    frame[id_column] = pd.to_numeric(frame[id_column], errors="coerce")
    frame[speed_column] = pd.to_numeric(frame[speed_column], errors="coerce")
    frame = frame.dropna(subset=[id_column, speed_column]).copy()
    frame = frame[(frame[speed_column] >= 20.0) & (frame[speed_column] <= 35.0)]
    if frame.empty:
        raise TraitDataError("no plausible Sprint Speed rows remained after validation")
    frame["speed_percentile"] = frame[speed_column].rank(method="average", pct=True)
    traits: dict[int, SprintSpeedTrait] = {}
    for row in frame.itertuples(index=False):
        values = row._asdict()
        player_id = int(values[id_column])
        opportunity_value = values.get(opportunity_column) if opportunity_column else None
        opportunities = None
        if opportunity_value is not None and not pd.isna(opportunity_value):
            opportunities = int(float(opportunity_value))
        traits[player_id] = SprintSpeedTrait(
            player_id=player_id,
            sprint_speed_ft_s=float(values[speed_column]),
            competitive_runs=opportunities,
            speed_percentile=float(values["speed_percentile"]),
        )
    receipt = TraitSnapshotReceipt(
        trait="sprint_speed",
        season=int(season),
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        source_url=used_url,
        rows=len(traits),
        player_id_column=id_column,
        value_columns=tuple(
            column for column in (speed_column, opportunity_column) if column is not None
        ),
        replay_policy=(
            "Historical replay must use a prior-season snapshot or a snapshot captured before first pitch; final same-season leaderboards are not prior-date inputs."
        ),
    )
    return traits, receipt


def fetch_oaa(
    season: int,
    *,
    position: str = "",
    minimum_attempts: str | int = 0,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> tuple[list[OAATrait], TraitSnapshotReceipt]:
    url = OAA_URL.format(
        year=int(season),
        minimum=minimum_attempts,
        position=str(position),
    )
    frame, used_url = _download_csv((url,), session=session, timeout=timeout)
    id_column = _find_column(
        frame.columns,
        ("player_id", "playerid", "entity_id", "id", "mlbam_id"),
    )
    oaa_column = _find_column(
        frame.columns,
        ("outs_above_average", "oaa", "n_outs_above_average"),
    )
    position_column = next(
        (column for column in ("primary_pos_formatted", "position", "pos") if column in frame.columns),
        None,
    )
    attempts_column = next(
        (column for column in ("attempts", "fielding_attempts", "n_opp") if column in frame.columns),
        None,
    )
    team_column = next(
        (column for column in ("team", "team_name", "team_id") if column in frame.columns),
        None,
    )
    frame[id_column] = pd.to_numeric(frame[id_column], errors="coerce")
    frame[oaa_column] = pd.to_numeric(frame[oaa_column], errors="coerce")
    frame = frame.dropna(subset=[id_column, oaa_column]).copy()
    if frame.empty:
        raise TraitDataError("no OAA rows remained after validation")
    traits: list[OAATrait] = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        attempts_value = values.get(attempts_column) if attempts_column else None
        attempts = None
        if attempts_value is not None and not pd.isna(attempts_value):
            attempts = int(float(attempts_value))
        traits.append(
            OAATrait(
                player_id=int(values[id_column]),
                position=str(values.get(position_column) or "") if position_column else "",
                outs_above_average=float(values[oaa_column]),
                attempts=attempts,
                team=str(values.get(team_column) or "") if team_column else "",
            )
        )
    receipt = TraitSnapshotReceipt(
        trait="outs_above_average",
        season=int(season),
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        source_url=used_url,
        rows=len(traits),
        player_id_column=id_column,
        value_columns=tuple(
            column
            for column in (oaa_column, position_column, attempts_column, team_column)
            if column is not None
        ),
        replay_policy=(
            "Historical replay must use prior-season OAA or a timestamped prior-date snapshot. Final same-season OAA is outcome-leaking for earlier games."
        ),
    )
    return traits, receipt


def impute_sprint_speed(
    player_id: int,
    traits: Mapping[int, SprintSpeedTrait],
    *,
    prior_speed_ft_s: float = 27.0,
    prior_percentile: float = 0.5,
) -> SprintSpeedTrait:
    trait = traits.get(int(player_id))
    if trait is not None:
        return trait
    return SprintSpeedTrait(
        player_id=int(player_id),
        sprint_speed_ft_s=float(prior_speed_ft_s),
        competitive_runs=None,
        speed_percentile=float(prior_percentile),
        imputed=True,
    )


def receipt_dict(receipt: TraitSnapshotReceipt) -> dict:
    return asdict(receipt)
