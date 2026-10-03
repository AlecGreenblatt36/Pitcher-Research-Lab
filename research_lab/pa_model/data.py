from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from .config import PAConfig
from .outcomes import map_event, mapping_report

LOGGER = logging.getLogger(__name__)
SAVANT_ROOT = "https://baseballsavant.mlb.com/statcast_search/csv"

PA_SOURCE_COLUMNS = (
    "game_date", "game_year", "game_pk", "at_bat_number", "pitch_number",
    "batter", "pitcher", "events", "stand", "p_throws", "home_team",
    "away_team", "inning", "inning_topbot", "outs_when_up", "on_1b",
    "on_2b", "on_3b", "home_score", "away_score", "bat_score",
    "fld_score", "bat_score_diff", "n_thruorder_pitcher",
    "batter_days_since_prev_game", "pitcher_days_since_prev_game",
    "age_bat", "age_pit", "fielder_2", "game_type",
)


def _daterange_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("end date must not precede start date")
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _savant_url(start: date, end: date) -> str:
    params = {
        "all": "true", "hfPT": "", "hfAB": "", "hfBBT": "", "hfPR": "",
        "hfZ": "", "stadium": "", "hfBBL": "", "hfNewZones": "",
        "hfGT": "R|", "hfSea": "", "hfSit": "", "player_type": "pitcher",
        "hfOuts": "", "opponent": "", "pitcher_throws": "",
        "batter_stands": "", "hfSA": "", "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(), "team": "", "position": "",
        "hfRO": "", "home_road": "", "hfFlag": "", "metric_1": "",
        "hfInn": "", "min_pitches": "0", "min_results": "0",
        "group_by": "name", "sort_col": "pitches",
        "player_event_sort": "h_launch_speed", "sort_order": "desc",
        "min_abs": "0", "type": "details",
    }
    return f"{SAVANT_ROOT}?{urlencode(params)}"


def _download_one(span: tuple[date, date], cache_dir: Path, config: PAConfig) -> tuple[Path, int]:
    start, end = span
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"statcast_{start.isoformat()}_{end.isoformat()}.csv.gz"
    if target.exists() and target.stat().st_size > 100:
        return target, len(pd.read_csv(target, usecols=["game_pk"]))
    url = _savant_url(start, end)
    headers = {"User-Agent": config.user_agent, "Accept": "text/csv,*/*;q=0.8"}
    last_error = None
    for attempt in range(config.request_retries):
        try:
            response = requests.get(url, headers=headers, timeout=config.request_timeout_seconds)
            response.raise_for_status()
            if not response.content or b"game_date" not in response.content[:2000]:
                raise RuntimeError("Savant returned an empty or non-CSV response")
            frame = pd.read_csv(io.BytesIO(response.content), low_memory=False)
            if "game_pk" not in frame.columns:
                raise RuntimeError("Savant response lacks game_pk")
            keep = [column for column in PA_SOURCE_COLUMNS if column in frame.columns]
            frame.loc[:, keep].to_csv(target, index=False, compression="gzip")
            return target, len(frame)
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Download failed for %s to %s (attempt %s/%s): %s", start, end, attempt + 1, config.request_retries, exc)
            time.sleep(min(60, 2**attempt + 0.25 * (attempt + 1)))
    raise RuntimeError(f"failed to download {start} through {end}: {last_error}")


def download_statcast(start: str, end: str, cache_dir: str | Path, config: PAConfig | None = None) -> list[Path]:
    config = config or PAConfig()
    spans = _daterange_chunks(date.fromisoformat(start), date.fromisoformat(end), config.chunk_days)
    cache_path = Path(cache_dir)
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.request_workers) as executor:
        futures = {executor.submit(_download_one, span, cache_path, config): span for span in spans}
        for future in concurrent.futures.as_completed(futures):
            path, rows = future.result()
            completed.append(path)
            LOGGER.info("cached %s (%s rows)", path.name, rows)
    return sorted(completed)


def _series_or_default(frame: pd.DataFrame, column: str, default: object = np.nan) -> pd.Series:
    return frame[column] if column in frame.columns else pd.Series(default, index=frame.index)


def _safe_numeric(series: pd.Series | object, default: float = np.nan, index=None) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=index)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def build_plate_appearances(pitch_files: Iterable[str | Path]) -> tuple[pd.DataFrame, dict]:
    frames = []
    for file in pitch_files:
        frame = pd.read_csv(file, low_memory=False)
        missing = {"game_date", "game_pk", "at_bat_number", "pitch_number"} - set(frame.columns)
        if missing:
            raise ValueError(f"{file} is missing required columns: {sorted(missing)}")
        frames.append(frame)
    if not frames:
        raise ValueError("no pitch files supplied")
    pitches = pd.concat(frames, ignore_index=True)
    if "game_type" in pitches.columns:
        pitches = pitches[pitches["game_type"].astype(str).eq("R")].copy()
    pitches["game_date"] = pd.to_datetime(pitches["game_date"], errors="coerce")
    pitches = pitches.dropna(subset=["game_date", "game_pk", "at_bat_number"])
    for column in ("game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"):
        if column in pitches:
            pitches[column] = pd.to_numeric(pitches[column], errors="coerce")
    key = ["game_pk", "at_bat_number"]
    pitches = pitches.sort_values(key + ["pitch_number"], kind="mergesort")
    pitches = pitches.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"], keep="last")
    first = pitches.groupby(key, sort=False, observed=True).head(1).copy()
    terminal_candidates = pitches[pitches["events"].notna()].copy()
    terminal = terminal_candidates.groupby(key, sort=False, observed=True).tail(1)[key + ["events"]]
    terminal = terminal.rename(columns={"events": "terminal_event"})
    pa = first.drop(columns=["events"], errors="ignore").merge(terminal, on=key, how="inner", validate="one_to_one")
    pa["outcome"] = pa["terminal_event"].map(map_event)
    event_report = mapping_report(pa["terminal_event"])
    total_terminal = len(pa)
    pa = pa[pa["outcome"].notna()].copy()
    pa["season"] = pa["game_date"].dt.year.astype(int)
    pa["date_key"] = pa["game_date"].dt.normalize()
    pa["batter"] = pd.to_numeric(pa["batter"], errors="coerce").astype("Int64")
    pa["pitcher"] = pd.to_numeric(pa["pitcher"], errors="coerce").astype("Int64")
    pa = pa.dropna(subset=["batter", "pitcher"])
    pa["batter"] = pa["batter"].astype(int)
    pa["pitcher"] = pa["pitcher"].astype(int)
    pa["platoon"] = (_series_or_default(pa, "stand", "U").astype(str).str.upper() == _series_or_default(pa, "p_throws", "U").astype(str).str.upper()).astype(int)
    pa["is_home_batter"] = (_series_or_default(pa, "inning_topbot", "Top").astype(str).str.lower() == "bot").astype(int)
    for base in ("1b", "2b", "3b"):
        pa[f"runner_{base}"] = _series_or_default(pa, f"on_{base}").notna().astype(int)
    if "bat_score_diff" not in pa:
        pa["bat_score_diff"] = _safe_numeric(_series_or_default(pa, "bat_score", 0), 0) - _safe_numeric(_series_or_default(pa, "fld_score", 0), 0)
    pa["bat_score_diff"] = _safe_numeric(pa["bat_score_diff"], 0).clip(-10, 10)
    pa["outs_when_up"] = _safe_numeric(_series_or_default(pa, "outs_when_up", 0), 0).clip(0, 2)
    pa["inning"] = _safe_numeric(_series_or_default(pa, "inning", 1), 1).clip(1, 20)
    pa["n_thruorder_pitcher"] = _safe_numeric(_series_or_default(pa, "n_thruorder_pitcher", 1), 1).clip(1, 8)
    pa["batter_days_since_prev_game"] = _safe_numeric(_series_or_default(pa, "batter_days_since_prev_game")).clip(0, 30)
    pa["pitcher_days_since_prev_game"] = _safe_numeric(_series_or_default(pa, "pitcher_days_since_prev_game")).clip(0, 30)
    pa["age_bat"] = _safe_numeric(_series_or_default(pa, "age_bat")).clip(18, 50)
    pa["age_pit"] = _safe_numeric(_series_or_default(pa, "age_pit")).clip(18, 50)
    pa["park"] = _series_or_default(pa, "home_team", "UNK").fillna("UNK").astype(str)
    pa["matchup_key"] = pa["batter"].astype(str) + "_" + pa["pitcher"].astype(str)
    pa = pa.sort_values(["date_key", "game_pk", "at_bat_number"], kind="mergesort").reset_index(drop=True)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "pitch_rows": int(len(pitches)), "terminal_pa_rows": int(total_terminal),
        "eligible_pa_rows": int(len(pa)), "excluded_terminal_rows": int(total_terminal - len(pa)),
        "date_min": pa["game_date"].min().date().isoformat() if len(pa) else None,
        "date_max": pa["game_date"].max().date().isoformat() if len(pa) else None,
        "games": int(pa["game_pk"].nunique()), "batters": int(pa["batter"].nunique()),
        "pitchers": int(pa["pitcher"].nunique()), **event_report,
    }
    return pa, report


def dataframe_sha256(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame.loc[:, columns]
    return hashlib.sha256(pd.util.hash_pandas_object(selected, index=True).values.tobytes()).hexdigest()


def save_pa_dataset(pa: pd.DataFrame, output: str | Path, report: dict) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pa.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)
    report = dict(report)
    report["dataset_sha256"] = dataframe_sha256(pa, ["date_key", "game_pk", "at_bat_number", "batter", "pitcher", "outcome"])
    path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
