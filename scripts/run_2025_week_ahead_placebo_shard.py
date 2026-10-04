"""Run one 2025 week-ahead bullpen-placebo replay shard.

This diagnostic gives each target team the reliever order from a single game
5-9 days later, preferring exactly 7 days. It is postgame information and never
a forecast. Its purpose is to separate generic bullpen-roster/role value from
the same-game oracle's outcome-linked usage information.
"""
from __future__ import annotations

import argparse
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
from research_lab.game_sim.oracle import OracleRelieverOrderPolicy, apply_actual_reliever_order
from research_lab.game_sim.placebo import week_ahead_placebo_orders
from research_lab.game_sim.replay import (
    build_matchup,
    extract_historical_games,
    pitcher_appearances,
    replay_game,
    starter_expectations,
)
from research_lab.game_sim.sequential_history_compat import install_writeability_guard
from scripts.run_2025_variant_parallel_v2 import load_history


class WeekAheadPlaceboPolicy(OracleRelieverOrderPolicy):
    name = "placebo-week-ahead-reliever-order-diagnostic"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_environment() -> dict:
    names = [
        "Flask", "pandas", "numpy", "requests", "joblib",
        "scikit-learn", "scipy",
    ]
    return {
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in names},
        "github_sha": os.getenv("GITHUB_SHA"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.sims <= 1:
        raise SystemExit("--sims must exceed one")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard configuration")

    install_writeability_guard()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    history, history_path, model_path, hazard_path = load_history(ROOT)
    overrides = {
        int(key): value
        for key, value in json.loads(
            (ROOT / "research_lab/game_sim/official_score_overrides_2025.json").read_text()
        ).items()
    }
    games, excluded = extract_historical_games(history, 2025, score_overrides=overrides)
    games = sorted(games, key=lambda game: (game.game_date, game.game_pk))
    orders, source_meta = week_ahead_placebo_orders(history, games, season=2025)
    chunks = [
        list(chunk)
        for chunk in np.array_split(np.asarray(games, dtype=object), args.num_shards)
    ]
    shard_games = chunks[args.shard_index]

    apps = pitcher_appearances(history)
    hazard = joblib.load(hazard_path)
    first = shard_games[0]
    provider = LockedPAModelProvider(
        model_path,
        history_path,
        first.game_date,
        first.game_date,
        first.park,
        history_frame=history,
        sequential_history=True,
    )
    current_date = first.game_date
    rows: list[dict] = []
    failures: list[dict] = []
    started = time.time()

    fallback_sides = 0
    source_sides = 0
    future_only_pitchers = 0
    source_pitchers = 0
    days_ahead: list[int] = []

    for index, game in enumerate(shard_games, start=1):
        try:
            if game.game_date != current_date:
                current_date = game.game_date
                provider.advance_to(current_date, game_date=current_date, park=game.park)
            else:
                provider.set_park(game.park)

            matchup, notes = build_matchup(game, history, apps)
            pitcher_exp, team_exp = starter_expectations(
                apps, game.game_date, hazard["league_mean_bf"]
            )
            team_of = {
                int(matchup.away.starter.player_id): game.away_team,
                int(matchup.home.starter.player_id): game.home_team,
            }

            game_orders = orders[int(game.game_pk)]
            game_meta = source_meta[int(game.game_pk)]
            existing = {
                "away": {int(p.player_id) for p in matchup.away.bullpen},
                "home": {int(p.player_id) for p in matchup.home.bullpen},
            }
            game_future_only = {}
            for side in ("away", "home"):
                meta = game_meta[side]
                if meta.status == "week_ahead_source":
                    source_sides += 1
                    days_ahead.append(int(meta.days_ahead))
                else:
                    fallback_sides += 1
                ids = [int(pid) for pid, _ in game_orders[side]]
                source_pitchers += len(ids)
                missing_ids = [pid for pid in ids if pid not in existing[side]]
                future_only_pitchers += len(missing_ids)
                game_future_only[side] = missing_ids

            matchup = apply_actual_reliever_order(matchup, game_orders)
            manager = WeekAheadPlaceboPolicy(hazard, pitcher_exp, team_exp, team_of)
            notes["placebo_boundary"] = (
                "reliever identities/order from a different game 5-9 days later; "
                "postgame negative control; never a forecast"
            )
            notes["placebo_sources"] = {
                side: game_meta[side].to_dict() for side in ("away", "home")
            }
            notes["placebo_future_only_pitcher_ids"] = game_future_only

            calls_before = int(provider.calls)
            hits_before = int(provider.cache_hits)
            result = replay_game(
                game, provider, matchup, manager, args.sims, int(game.game_pk)
            )
            result["provider_calls"] = int(provider.calls) - calls_before
            result["provider_cache_hits"] = int(provider.cache_hits) - hits_before
            result["score_source"] = game.score_source
            result["common_random_seed"] = int(game.game_pk)
            result["fixture_notes"] = json.dumps(notes, sort_keys=True)
            result["variant"] = "placebo_week_ahead"
            result["placebo_away_source_game_pk"] = game_meta["away"].source_game_pk
            result["placebo_home_source_game_pk"] = game_meta["home"].source_game_pk
            result["placebo_away_days_ahead"] = game_meta["away"].days_ahead
            result["placebo_home_days_ahead"] = game_meta["home"].days_ahead
            result["placebo_away_reliever_count"] = game_meta["away"].reliever_count
            result["placebo_home_reliever_count"] = game_meta["home"].reliever_count
            result["placebo_away_future_only_count"] = len(game_future_only["away"])
            result["placebo_home_future_only_count"] = len(game_future_only["home"])
            rows.append(result)
        except Exception as exc:
            failures.append(
                {
                    "game_pk": int(game.game_pk),
                    "game_date": game.game_date,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if index % args.checkpoint_every == 0 or index == len(shard_games):
            frame = pd.DataFrame(rows).sort_values(
                ["game_date", "game_pk"], kind="mergesort"
            )
            prediction_path = output / f"placebo_week_shard_{args.shard_index:02d}.csv.gz"
            frame.to_csv(prediction_path, index=False, compression="gzip")
            elapsed = time.time() - started
            progress = {
                "schema": "baseball_research_lab.week_ahead_placebo_shard.v1",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "simulations_per_game": args.sims,
                "games_expected": len(shard_games),
                "games_scored": len(rows),
                "failures": failures,
                "elapsed_seconds": elapsed,
                "fallback_sides": fallback_sides,
                "source_sides": source_sides,
                "source_pitchers": source_pitchers,
                "future_only_pitchers": future_only_pitchers,
                "mean_days_ahead": float(np.mean(days_ahead)) if days_ahead else None,
                "environment": replay_environment(),
            }
            (output / f"placebo_week_shard_{args.shard_index:02d}_progress.json").write_text(
                json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
            )

    prediction_path = output / f"placebo_week_shard_{args.shard_index:02d}.csv.gz"
    receipt = {
        "schema": "baseball_research_lab.week_ahead_placebo_shard.v1",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "simulations_per_game": args.sims,
        "games_expected": len(shard_games),
        "games_scored": len(rows),
        "failure_count": len(failures),
        "excluded_source_games": len(excluded),
        "predictions_sha256": sha256(prediction_path),
        "environment": replay_environment(),
        "selection_window_days": [5, 9],
        "preferred_days_ahead": 7,
        "doubleheader_source_dates_excluded": True,
        "forecast_valid": False,
    }
    (output / f"placebo_week_shard_{args.shard_index:02d}_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )

    del history
    gc.collect()

    if failures:
        raise RuntimeError(
            f"placebo shard {args.shard_index} failed for {len(failures)} games: "
            f"{failures[:5]}"
        )
    if len(rows) != len(shard_games):
        raise RuntimeError(
            f"placebo shard {args.shard_index} incomplete: "
            f"expected {len(shard_games)}, found {len(rows)}"
        )


if __name__ == "__main__":
    main()
