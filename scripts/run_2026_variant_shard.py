"""Run one frozen 2026 full-game replay shard.

Variants
--------
candidate
    Locked seven-outcome PA provider, fitted starter hazard, and the current
    pregame bullpen candidate/selection engine.
flat
    Prior-date flat league PA probabilities through the same downstream game
    engine.  This is the PA signal-survival ablation.
oracle
    Locked PA provider with actual target-game reliever identities/order.  This
    uses postgame information and is a ceiling diagnostic only, never a
    forecast.

Every target game uses ``game_pk`` as its common random seed.  Player history,
bullpen availability, and tendencies are advanced only through dates strictly
before the target game.  Any failed game makes the shard fail after writing a
progress receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_lab.game_sim.locked_pa_provider import LockedPAModelProvider
from research_lab.game_sim.oracle import (
    OracleRelieverOrderPolicy,
    actual_reliever_orders,
    apply_actual_reliever_order,
)
from research_lab.game_sim.replay import (
    FlatLeagueAverageProvider,
    HistoricalGame,
    build_matchup,
    extract_historical_games,
    pitcher_appearances,
    replay_game,
    starter_expectations,
)
from research_lab.game_sim.sequential_history_compat import install_writeability_guard
from research_lab.game_sim.starter_hazard import FittedStarterPolicy

HISTORY_USECOLS = [
    "game_date", "game_year", "game_pk", "at_bat_number", "pitch_number",
    "batter", "pitcher", "stand", "p_throws", "home_team", "away_team",
    "inning", "inning_topbot", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "home_score", "away_score", "bat_score", "fld_score", "bat_score_diff",
    "n_thruorder_pitcher", "batter_days_since_prev_game",
    "pitcher_days_since_prev_game", "age_bat", "age_pit", "game_type",
    "terminal_event", "outcome", "season", "date_key", "runner_1b",
    "runner_2b", "runner_3b", "park",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_environment() -> dict:
    packages = [
        "Flask", "pandas", "numpy", "requests", "joblib",
        "scikit-learn", "scipy",
    ]
    return {
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in packages},
        "github_sha": os.getenv("GITHUB_SHA"),
    }


def load_history(history_path: Path) -> pd.DataFrame:
    history = pd.read_csv(history_path, usecols=HISTORY_USECOLS, low_memory=False)
    history["date_key"] = history["date_key"].astype(str).str[:10]
    return history


def load_overrides(path: str | None) -> dict[int, tuple[int, int] | dict]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    return {int(key): value for key, value in raw.items()}


def run_shard(
    *,
    games: list[HistoricalGame],
    history: pd.DataFrame,
    history_path: Path,
    model_path: Path,
    hazard_path: Path,
    output: Path,
    simulations: int,
    checkpoint_every: int,
    variant: str,
    shard_index: int,
    resume: bool,
    season: int,
) -> None:
    shard_path = output / f"{variant}_shard_{shard_index:02d}.csv.gz"
    progress_path = output / f"{variant}_shard_{shard_index:02d}_progress.json"
    existing = pd.DataFrame()
    completed: set[int] = set()
    if resume and shard_path.exists():
        existing = pd.read_csv(shard_path, low_memory=False)
        completed = set(existing["game_pk"].astype(int).tolist())
    remaining = [game for game in games if game.game_pk not in completed]
    if not remaining:
        return

    apps = pitcher_appearances(history)
    hazard = joblib.load(hazard_path)
    train_seasons = list(hazard.get("train_seasons", hazard.get("protocol", {}).get("train_seasons", [])))
    if train_seasons and max(map(int, train_seasons)) >= season:
        raise RuntimeError("starter hazard was trained on target-season rows")
    oracle_orders = actual_reliever_orders(history, season) if variant == "oracle" else {}
    first = remaining[0]
    locked = LockedPAModelProvider(
        model_path,
        history_path,
        first.game_date,
        first.game_date,
        first.park,
        history_frame=history,
        sequential_history=True,
    )
    provider = FlatLeagueAverageProvider(locked) if variant == "flat" else locked
    current_date = first.game_date
    rows: list[dict] = []
    failures: list[dict] = []
    started = time.time()

    for index, game in enumerate(remaining, start=1):
        try:
            if game.game_date != current_date:
                current_date = game.game_date
                locked.advance_to(current_date, game_date=current_date, park=game.park)
            else:
                locked.set_park(game.park)
            matchup, notes = build_matchup(game, history, apps)
            pitcher_exp, team_exp = starter_expectations(
                apps, game.game_date, float(hazard["league_mean_bf"])
            )
            team_of = {
                int(matchup.away.starter.player_id): game.away_team,
                int(matchup.home.starter.player_id): game.home_team,
            }
            if variant == "oracle":
                matchup = apply_actual_reliever_order(
                    matchup, oracle_orders.get(int(game.game_pk), {})
                )
                manager = OracleRelieverOrderPolicy(
                    hazard, pitcher_exp, team_exp, team_of
                )
                notes["oracle_actual_reliever_order"] = True
                notes["oracle_boundary"] = "postgame target-game information; diagnostic only"
            else:
                manager = FittedStarterPolicy(hazard, pitcher_exp, team_exp, team_of)

            calls_before = int(getattr(provider, "calls", 0))
            hits_before = int(getattr(provider, "cache_hits", 0))
            result = replay_game(
                game, provider, matchup, manager, simulations, int(game.game_pk)
            )
            result["provider_calls"] = int(getattr(provider, "calls", 0)) - calls_before
            result["provider_cache_hits"] = int(getattr(provider, "cache_hits", 0)) - hits_before
            result["score_source"] = game.score_source
            result["common_random_seed"] = int(game.game_pk)
            result["fixture_notes"] = json.dumps(notes, sort_keys=True)
            result["variant"] = variant
            rows.append(result)
        except Exception as exc:  # fail shard after preserving exact failing games
            failures.append({
                "game_pk": int(game.game_pk),
                "game_date": game.game_date,
                "error": f"{type(exc).__name__}: {exc}",
            })

        if index % checkpoint_every == 0 or index == len(remaining):
            frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            if len(frame):
                frame = frame.drop_duplicates("game_pk", keep="last").sort_values(
                    ["game_date", "game_pk"], kind="mergesort"
                )
            frame.to_csv(shard_path, index=False, compression="gzip")
            elapsed = time.time() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(remaining) - index) / rate if rate else None
            progress = {
                "schema": "baseball_research_lab.2026_replay_progress.v1",
                "season": season,
                "variant": variant,
                "shard": shard_index,
                "simulations_per_game": int(simulations),
                "games_completed_this_run": index,
                "games_total_this_run": len(remaining),
                "games_scored_total": int(len(frame)),
                "failures": failures,
                "elapsed_seconds": elapsed,
                "estimated_seconds_remaining": eta,
                "last_game_pk": int(game.game_pk),
                "last_game_date": game.game_date,
            }
            progress_path.write_text(
                json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(
                f"{variant} shard {shard_index}: {index}/{len(remaining)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s" if eta is not None else
                f"{variant} shard {shard_index}: {index}/{len(remaining)}",
                flush=True,
            )

    if failures:
        raise RuntimeError(
            f"{variant} shard {shard_index} failed for {len(failures)} games: {failures[:5]}"
        )
    final = pd.read_csv(shard_path, low_memory=False)
    if len(final) != len(games):
        raise RuntimeError(
            f"{variant} shard {shard_index} incomplete: expected {len(games)}, found {len(final)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("candidate", "flat", "oracle"), required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history", default="pa_model_reference/model_runs/pa_locked_2026/plate_appearances.csv.gz")
    parser.add_argument("--model", default="pa_model_reference/model_runs/pa_locked_2026/artifacts/pa_model.joblib")
    parser.add_argument("--hazard", required=True)
    parser.add_argument("--score-overrides")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.sims <= 1:
        raise SystemExit("--sims must exceed one")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be inside [0, num-shards)")
    if args.max_games < 0:
        raise SystemExit("--max-games may not be negative")

    install_writeability_guard()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    history_path = ROOT / args.history
    model_path = ROOT / args.model
    hazard_path = ROOT / args.hazard
    history = load_history(history_path)
    overrides = load_overrides(args.score_overrides)
    full_games, excluded = extract_historical_games(
        history, args.season, score_overrides=overrides
    )
    full_games = sorted(full_games, key=lambda game: (game.game_date, game.game_pk))
    games = full_games[: args.max_games] if args.max_games else full_games
    chunks = [list(chunk) for chunk in np.array_split(np.asarray(games, dtype=object), args.num_shards)]
    shard_games = chunks[args.shard_index]
    started = time.time()

    if shard_games:
        run_shard(
            games=shard_games,
            history=history,
            history_path=history_path,
            model_path=model_path,
            hazard_path=hazard_path,
            output=output,
            simulations=args.sims,
            checkpoint_every=args.checkpoint_every,
            variant=args.variant,
            shard_index=args.shard_index,
            resume=args.resume,
            season=args.season,
        )
    del history
    gc.collect()

    progress_path = output / f"{args.variant}_shard_{args.shard_index:02d}_progress.json"
    failures: list[dict] = []
    completed = 0
    if shard_games:
        progress = json.loads(progress_path.read_text())
        failures = list(progress.get("failures", []))
        completed = int(progress.get("games_scored_total", 0))

    reason_counts = Counter(str(row.get("reason")) for row in excluded)
    receipt = {
        "schema": "baseball_research_lab.2026_variant_shard.v1",
        "season": int(args.season),
        "variant": args.variant,
        "simulations_per_game": int(args.sims),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "max_games": int(args.max_games),
        "games_available_from_history": int(len(full_games)),
        "games_selected": int(len(games)),
        "games_in_shard": int(len(shard_games)),
        "games_scored": int(completed),
        "failure_count": int(len(failures)),
        "excluded_game_count": int(len(excluded)),
        "excluded_reason_counts": dict(sorted(reason_counts.items())),
        "first_game_pk": int(shard_games[0].game_pk) if shard_games else None,
        "last_game_pk": int(shard_games[-1].game_pk) if shard_games else None,
        "runtime_seconds": float(time.time() - started),
        "common_seed": "game_pk",
        "oracle_boundary": (
            "actual target-game reliever identities/order; postgame diagnostic only"
            if args.variant == "oracle" else None
        ),
        "artifacts": {
            "history_sha256": sha256(history_path),
            "model_sha256": sha256(model_path),
            "starter_hazard_sha256": sha256(hazard_path),
        },
        "environment": replay_environment(),
    }
    receipt_path = output / f"{args.variant}_shard_{args.shard_index:02d}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        raise RuntimeError(f"shard contains {len(failures)} failures")
    if completed != len(shard_games):
        raise RuntimeError(
            f"incomplete shard: expected {len(shard_games)} rows, found {completed}"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
