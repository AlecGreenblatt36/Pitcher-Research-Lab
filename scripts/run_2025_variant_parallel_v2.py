"""Run one 2025 full-season simulator variant at high Monte Carlo precision.

Variants:
- candidate: locked seven-outcome PA model + fitted starter hazard + dev bullpen
- flat: prior-date flat league PA probabilities + same downstream engine
- oracle: locked PA model + fitted starter hazard + actual relievers in actual order

The oracle uses postgame information and is a diagnostic only, never a forecast.
All variants use per-game seed = game_pk and the same path-seed generator.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
from pathlib import Path
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


def load_history(root: Path):
    ref = root / "pa_model_reference/model_runs/pa_locked_2026"
    history_path = ref / "plate_appearances.csv.gz"
    model_path = ref / "artifacts/pa_model.joblib"
    hazard_path = root / "model_runs/starter_hazard_2025_dev/starter_hazard_2025_dev.joblib"
    history = pd.read_csv(history_path, usecols=HISTORY_USECOLS, low_memory=False)
    history["date_key"] = history["date_key"].astype(str).str[:10]
    return history, history_path, model_path, hazard_path


def run_shard(
    shard_id: int,
    games: list[HistoricalGame],
    root_string: str,
    output_string: str,
    simulations: int,
    checkpoint_every: int,
    resume: bool,
    variant: str,
) -> None:
    root = Path(root_string)
    output = Path(output_string)
    shard_path = output / f"{variant}_shard_{shard_id:02d}.csv.gz"
    progress_path = output / f"{variant}_shard_{shard_id:02d}_progress.json"
    existing = pd.DataFrame()
    completed: set[int] = set()
    if resume and shard_path.exists():
        existing = pd.read_csv(shard_path, low_memory=False)
        completed = set(existing["game_pk"].astype(int).tolist())
    remaining = [game for game in games if game.game_pk not in completed]
    if not remaining:
        return

    history, history_path, model_path, hazard_path = load_history(root)
    apps = pitcher_appearances(history)
    hazard = joblib.load(hazard_path)
    oracle_orders = actual_reliever_orders(history, 2025) if variant == "oracle" else {}
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
                apps, game.game_date, hazard["league_mean_bf"]
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
                notes["oracle_boundary"] = "postgame information; diagnostic only"
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
        except Exception as exc:
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
                "variant": variant,
                "shard": shard_id,
                "simulations_per_game": int(simulations),
                "games_completed_this_run": index,
                "games_total_this_run": len(remaining),
                "games_scored_total": int(len(existing) + len(rows)),
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
                f"{variant} shard {shard_id}: {index}/{len(remaining)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s" if eta is not None else
                f"{variant} shard {shard_id}: {index}/{len(remaining)}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("candidate", "flat", "oracle"), required=True)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-games", type=int, default=0)
    args = parser.parse_args()
    if args.sims <= 1:
        raise SystemExit("--sims must exceed one")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    history, _, _, _ = load_history(ROOT)
    override_path = ROOT / "research_lab/game_sim/official_score_overrides_2025.json"
    overrides = {int(k): value for k, value in json.loads(override_path.read_text()).items()}
    games, excluded = extract_historical_games(history, 2025, score_overrides=overrides)
    games = sorted(games, key=lambda game: (game.game_date, game.game_pk))
    if args.max_games:
        games = games[:args.max_games]
    del history
    gc.collect()

    shards = [
        list(chunk)
        for chunk in np.array_split(
            np.asarray(games, dtype=object), min(args.workers, len(games))
        )
        if len(chunk)
    ]
    ctx = mp.get_context("fork")
    processes: list[mp.Process] = []
    started = time.time()
    for shard_id, shard_games in enumerate(shards):
        process = ctx.Process(
            target=run_shard,
            args=(
                shard_id, shard_games, str(ROOT), str(output), args.sims,
                args.checkpoint_every, args.resume, args.variant,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise SystemExit(f"worker failed with exit code {process.exitcode}")

    shard_paths = sorted(output.glob(f"{args.variant}_shard_*.csv.gz"))
    frame = pd.concat([pd.read_csv(path, low_memory=False) for path in shard_paths], ignore_index=True)
    frame = frame.drop_duplicates("game_pk", keep="last").sort_values(
        ["game_date", "game_pk"], kind="mergesort"
    )
    final_path = output / f"{args.variant}_predictions.csv.gz"
    frame.to_csv(final_path, index=False, compression="gzip")
    receipt = {
        "schema": "baseball_research_lab.2025_variant_replay.v2",
        "variant": args.variant,
        "games": int(len(frame)),
        "simulations_per_game": int(args.sims),
        "workers": int(args.workers),
        "runtime_seconds": float(time.time() - started),
        "excluded_game_count": int(len(excluded)),
        "failures": [
            failure
            for path in sorted(output.glob(f"{args.variant}_shard_*_progress.json"))
            for failure in json.loads(path.read_text()).get("failures", [])
        ],
        "common_random_seed": "game_pk; identical path-seed generator across variants",
        "oracle_boundary": (
            "actual reliever identities/order from completed games; diagnostic only; never forecast"
            if args.variant == "oracle" else None
        ),
        "predictions": str(final_path.relative_to(ROOT)),
    }
    (output / f"{args.variant}_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    main()
