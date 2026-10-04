"""Run one contiguous 2025 replay shard without child multiprocessing.

Designed for GitHub Actions matrix execution. Each job owns one provider state,
uses game_pk as the common seed, and writes a single variant/shard artifact.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_2025_variant_parallel_v2 import load_history, run_shard
from research_lab.game_sim.replay import extract_historical_games


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
    receipt = {
        "schema": "baseball_research_lab.2025_variant_shard.v2",
        "variant": args.variant,
        "simulations_per_game": args.sims,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "games_in_shard": len(shard_games),
        "first_game_pk": int(shard_games[0].game_pk) if shard_games else None,
        "last_game_pk": int(shard_games[-1].game_pk) if shard_games else None,
        "excluded_game_count_full_season": len(excluded),
        "runtime_seconds": time.time() - started,
        "common_seed": "game_pk",
    }
    (output / f"{args.variant}_shard_{args.shard_index:02d}_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
