from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

MLB_API_ROOT = "https://statsapi.mlb.com/api"
SCHEDULE_URL = f"{MLB_API_ROOT}/v1/schedule"
LIVE_FEED_URL = f"{MLB_API_ROOT}/v1.1/game/{{game_pk}}/feed/live"


class PregameDataError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _person(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    person_id = payload.get("id")
    name = payload.get("fullName")
    if person_id is None and not name:
        return None
    return {"id": int(person_id) if person_id is not None else None, "name": str(name or "")}


@dataclass(frozen=True)
class GamePickerRow:
    game_pk: int
    game_type: str
    game_date_utc: str
    official_date: str
    status_code: str
    status_detail: str
    away_team_id: int
    away_team: str
    home_team_id: int
    home_team: str
    away_probable_pitcher: dict[str, Any] | None
    home_probable_pitcher: dict[str, Any] | None
    venue_id: int | None
    venue: str
    double_header: str
    game_number: int


@dataclass(frozen=True)
class PregameSnapshot:
    schema: str
    captured_at_utc: str
    forecast_valid_at_capture: bool
    first_pitch_utc: str | None
    game_pk: int
    game_type: str
    official_date: str
    status: dict[str, Any]
    away: dict[str, Any]
    home: dict[str, Any]
    venue: dict[str, Any]
    weather: dict[str, Any]
    roof_status: str
    plate_umpire: dict[str, Any] | None
    source: dict[str, Any]


def parse_schedule(payload: Mapping[str, Any]) -> list[GamePickerRow]:
    rows: list[GamePickerRow] = []
    for date_block in payload.get("dates", []) or []:
        official_date = str(date_block.get("date") or "")
        for game in date_block.get("games", []) or []:
            teams = game.get("teams", {}) or {}
            away = teams.get("away", {}) or {}
            home = teams.get("home", {}) or {}
            away_team = away.get("team", {}) or {}
            home_team = home.get("team", {}) or {}
            venue = game.get("venue", {}) or {}
            status = game.get("status", {}) or {}
            rows.append(
                GamePickerRow(
                    game_pk=int(game["gamePk"]),
                    game_type=str(game.get("gameType") or ""),
                    game_date_utc=str(game.get("gameDate") or ""),
                    official_date=str(game.get("officialDate") or official_date),
                    status_code=str(status.get("codedGameState") or ""),
                    status_detail=str(status.get("detailedState") or ""),
                    away_team_id=int(away_team.get("id") or 0),
                    away_team=str(away_team.get("name") or ""),
                    home_team_id=int(home_team.get("id") or 0),
                    home_team=str(home_team.get("name") or ""),
                    away_probable_pitcher=_person(away.get("probablePitcher")),
                    home_probable_pitcher=_person(home.get("probablePitcher")),
                    venue_id=(int(venue["id"]) if venue.get("id") is not None else None),
                    venue=str(venue.get("name") or ""),
                    double_header=str(game.get("doubleHeader") or "N"),
                    game_number=int(game.get("gameNumber") or 1),
                )
            )
    return sorted(rows, key=lambda row: (row.game_date_utc, row.game_pk))


def _lineup_from_boxscore(team_box: Mapping[str, Any]) -> list[dict[str, Any]]:
    players = team_box.get("players", {}) or {}
    lineup: list[tuple[int, dict[str, Any]]] = []
    for entry in players.values():
        if not isinstance(entry, Mapping):
            continue
        order = entry.get("battingOrder")
        person = entry.get("person", {}) or {}
        if order is None or person.get("id") is None:
            continue
        try:
            numeric_order = int(order)
        except (TypeError, ValueError):
            continue
        position = entry.get("position", {}) or {}
        stats = entry.get("stats", {}) or {}
        batting = stats.get("batting", {}) or {}
        lineup.append(
            (
                numeric_order,
                {
                    "player_id": int(person["id"]),
                    "name": str(person.get("fullName") or ""),
                    "batting_order": numeric_order,
                    "position": str(position.get("abbreviation") or ""),
                    "games_started": int(batting.get("gamesStarted") or 0),
                },
            )
        )
    lineup.sort(key=lambda item: item[0])
    return [item[1] for item in lineup]


def _plate_umpire(boxscore: Mapping[str, Any]) -> dict[str, Any] | None:
    for official in boxscore.get("officials", []) or []:
        if str(official.get("officialType") or "").lower() == "home plate":
            return _person(official.get("official"))
    return None


def parse_live_feed(
    payload: Mapping[str, Any],
    *,
    captured_at_utc: datetime | None = None,
) -> PregameSnapshot:
    captured = (captured_at_utc or _utc_now()).astimezone(timezone.utc)
    game_data = payload.get("gameData", {}) or {}
    live_data = payload.get("liveData", {}) or {}
    game = game_data.get("game", {}) or {}
    datetime_data = game_data.get("datetime", {}) or {}
    status = game_data.get("status", {}) or {}
    teams = game_data.get("teams", {}) or {}
    probable = game_data.get("probablePitchers", {}) or {}
    venue = game_data.get("venue", {}) or {}
    field_info = venue.get("fieldInfo", {}) or {}
    weather = game_data.get("weather", {}) or {}
    boxscore = live_data.get("boxscore", {}) or {}
    box_teams = boxscore.get("teams", {}) or {}

    first_pitch = _parse_utc(datetime_data.get("dateTime"))
    forecast_valid = first_pitch is not None and captured <= first_pitch

    def team(side: str) -> dict[str, Any]:
        team_payload = teams.get(side, {}) or {}
        box = box_teams.get(side, {}) or {}
        return {
            "team_id": int(team_payload.get("id") or 0),
            "name": str(team_payload.get("name") or ""),
            "abbreviation": str(team_payload.get("abbreviation") or ""),
            "probable_pitcher": _person(probable.get(side)),
            "lineup": _lineup_from_boxscore(box),
            "lineup_posted": bool(_lineup_from_boxscore(box)),
        }

    roof_type = str(field_info.get("roofType") or "")
    condition = str(weather.get("condition") or "")
    roof_status = roof_type or ("closed" if "roof closed" in condition.lower() else "unknown")
    game_pk = int(game.get("pk") or payload.get("gamePk") or 0)
    source_url = LIVE_FEED_URL.format(game_pk=game_pk)
    return PregameSnapshot(
        schema="baseball_research_lab.pregame_snapshot.v1",
        captured_at_utc=captured.isoformat(),
        forecast_valid_at_capture=bool(forecast_valid),
        first_pitch_utc=(first_pitch.isoformat() if first_pitch else None),
        game_pk=game_pk,
        game_type=str(game.get("type") or ""),
        official_date=str(datetime_data.get("officialDate") or ""),
        status={
            "abstract": str(status.get("abstractGameState") or ""),
            "coded": str(status.get("codedGameState") or ""),
            "detailed": str(status.get("detailedState") or ""),
        },
        away=team("away"),
        home=team("home"),
        venue={
            "id": int(venue.get("id") or 0),
            "name": str(venue.get("name") or ""),
            "roof_type": roof_type,
            "turf_type": str(field_info.get("turfType") or ""),
            "left_line": field_info.get("leftLine"),
            "left_center": field_info.get("leftCenter"),
            "center": field_info.get("center"),
            "right_center": field_info.get("rightCenter"),
            "right_line": field_info.get("rightLine"),
        },
        weather={
            "condition": condition,
            "temperature_f": weather.get("temp"),
            "wind": str(weather.get("wind") or ""),
        },
        roof_status=roof_status,
        plate_umpire=_plate_umpire(boxscore),
        source={
            "provider": "MLB Stats API",
            "url": source_url,
            "capture_is_prospective": bool(forecast_valid),
            "warning": (
                "Target-game boxscore fields are forecast-valid only when this snapshot was captured before first pitch."
            ),
        },
    )


def _get_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> Mapping[str, Any]:
    client = session or requests.Session()
    response = client.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "BaseballResearchLab/1.0 (+public research)"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise PregameDataError(f"expected JSON object from {url}")
    return payload


def list_games(
    game_date: str,
    *,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> list[GamePickerRow]:
    payload = _get_json(
        SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": str(game_date)[:10],
            "hydrate": "probablePitcher,team,venue",
        },
        session=session,
        timeout=timeout,
    )
    return parse_schedule(payload)


def fetch_pregame_snapshot(
    game_pk: int,
    *,
    session: requests.Session | None = None,
    timeout: int = 30,
    captured_at_utc: datetime | None = None,
) -> PregameSnapshot:
    url = LIVE_FEED_URL.format(game_pk=int(game_pk))
    payload = _get_json(url, session=session, timeout=timeout)
    return parse_live_feed(payload, captured_at_utc=captured_at_utc)


def archive_snapshot(snapshot: PregameSnapshot, root: Path) -> Path:
    """Write an immutable timestamped snapshot without overwriting prior captures."""

    root = Path(root)
    date = snapshot.official_date or "unknown-date"
    folder = root / date / str(snapshot.game_pk)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = snapshot.captured_at_utc.replace(":", "").replace("+00:00", "Z")
    target = folder / f"pregame_{stamp}.json"
    if target.exists():
        raise PregameDataError(f"snapshot already exists: {target}")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(snapshot), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return target
