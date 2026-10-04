"""Run one contiguous 2025 replay shard without child multiprocessing.

Designed for GitHub Actions matrix execution. Each job owns one provider state,
uses game_pk as the common seed, and writes a single variant/shard artifact.
Any game-level failure makes the shard job fail after the progress artifact is
written, so incomplete seasons can never look like successful benchmark runs.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_lab.game_sim.replay import extract_historical_games
from research_lab.game_sim.sequential_history_compat import install_writeability_guard
from scripts.run_2025_variant_parallel_v2 import load_history, run_shard


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("candidate", "flat", "oracle"), required=True)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.sims <= 1:
        raise SystemExit("--sims must exceed one")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be inside [0, num-shards)")

    install_writeability_guard()

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    history, _, _, _ = load_history(ROOT)
    override_path = ROOT / "research_lab/game_sim/official_score_overrides_2025.json"
    overrides = {int(k): value for k, value in json.loads(override_path.read_text()).items()}
    games, excluded = extract_historical_games(history, 2025, score_overrides=overrides)
    games = sorted(games, key=lambda game: (game.game_date, game.game_pk))
    chunks = [list(chunk) for chunk in np.array_split(np.asarray(games, dtype=object), args.num_shards)]
    shard_games = chunks[args.shard_index]
    del history
    gc.collect()

    started = time.time()
    run_shard(
        args.shard_index,
        shard_games,
        str(ROOT),
        str(output),
        args.sims,
        args.checkpoint_every,
        args.resume,
        args.variant,
    )

    progress_path = output / f"{args.variant}_shard_{args.shard_index:02d}_progress.json"
    if not progress_path.is_file():
        raise RuntimeError(f"missing shard progress receipt: {progress_path}")
    progress = json.loads(progress_path.read_text())
    failures = list(progress.get("failures", []))
    completed = int(progress.get("games_scored_total", 0))

    receipt = {
        "schema": "baseball_research_lab.2025_variant_shard.v3",
        "variant": args.variant,
        "simulations_per_game": args.sims,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "games_in_shard": len(shard_games),
        "games_scored": completed,
        "failure_count": len(failures),
        "first_game_pk": int(shard_games[0].game_pk) if shard_games else None,
        "last_game_pk": int(shard_games[-1].game_pk) if shard_games else None,
        "excluded_game_count_full_season": len(excluded),
        "runtime_seconds": time.time() - started,
        "common_seed": "game_pk",
        "environment": replay_environment(),
    }
    receipt_path = output / f"{args.variant}_shard_{args.shard_index:02d}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        raise RuntimeError(
            f"{args.variant} shard {args.shard_index} failed for {len(failures)} games: "
            f"{failures[:5]}"
        )
    if completed != len(shard_games):
        raise RuntimeError(
            f"{args.variant} shard {args.shard_index} incomplete: "
            f"expected {len(shard_games)} rows, found {completed}"
        )


if __name__ == "__main__":
    main()
