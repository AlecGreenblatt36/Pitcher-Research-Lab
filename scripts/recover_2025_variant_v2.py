"""Recover only games missing from the first 2025 1000-path replay run.

The original 72 shard jobs completed and preserved their outputs, but some
locked-provider games failed under pandas 3 because grouped NumPy rows were
read-only.  This script applies a narrow writeability guard, replays only the
missing games, then verifies and emits one complete file for the variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_lab.game_sim.locked_pa_provider import MODEL_LABELS, SequentialHistoryState
from research_lab.game_sim.replay import extract_historical_games
from scripts.run_2025_variant_parallel_v2 import load_history, run_shard


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_writeability_guard() -> None:
    """Make pandas-3 read-only grouped arrays mutable at first update."""

    def safe_increment(table: dict, key, class_index: int) -> None:
        counts = table.get(key)
        if counts is None:
            counts = np.zeros(len(MODEL_LABELS), dtype=float)
            table[key] = counts
        elif not counts.flags.writeable:
            counts = counts.copy()
            table[key] = counts
        counts[class_index] += 1.0

    def safe_increment_recent(
        cls, table: dict, queues: dict, key, class_index: int, window: int
    ) -> None:
        counts = table.get(key)
        if counts is None:
            counts = np.zeros(len(MODEL_LABELS), dtype=float)
            table[key] = counts
        elif not counts.flags.writeable:
            counts = counts.copy()
            table[key] = counts
        queue = queues.get(key)
        if queue is None:
            from collections import deque
            queue = deque()
            queues[key] = queue
        if len(queue) >= window:
            counts[int(queue.popleft())] -= 1.0
        queue.append(int(class_index))
        counts[class_index] += 1.0

    SequentialHistoryState._increment = staticmethod(safe_increment)
    SequentialHistoryState._increment_recent = classmethod(safe_increment_recent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("candidate", "flat", "oracle"), required=True)
    parser.add_argument("--old-shards", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--expected-games", type=int, default=2430)
    args = parser.parse_args()

    if args.sims <= 1:
        raise SystemExit("--sims must exceed one")
    apply_writeability_guard()

    old_dir = Path(args.old_shards)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    old_paths = sorted(old_dir.rglob(f"{args.variant}_shard_*.csv.gz"))
    if not old_paths:
        raise RuntimeError(f"no preserved {args.variant} shard outputs found")
    old = pd.concat([pd.read_csv(path, low_memory=False) for path in old_paths], ignore_index=True)
    old = old.drop_duplicates("game_pk", keep="last")

    history, _, _, _ = load_history(ROOT)
    overrides = {
        int(key): value
        for key, value in json.loads(
            (ROOT / "research_lab/game_sim/official_score_overrides_2025.json").read_text()
        ).items()
    }
    games, excluded = extract_historical_games(history, 2025, score_overrides=overrides)
    games = sorted(games, key=lambda game: (game.game_date, game.game_pk))
    expected_ids = {int(game.game_pk) for game in games}
    existing_ids = set(old.game_pk.astype(int))
    missing = [game for game in games if int(game.game_pk) not in existing_ids]

    recovery_dir = output / "recovery_shard"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        run_shard(
            99,
            missing,
            str(ROOT),
            str(recovery_dir),
            args.sims,
            10,
            False,
            args.variant,
        )
        recovery_path = recovery_dir / f"{args.variant}_shard_99.csv.gz"
        progress_path = recovery_dir / f"{args.variant}_shard_99_progress.json"
        if not recovery_path.is_file():
            raise RuntimeError("recovery replay did not produce a prediction file")
        progress = json.loads(progress_path.read_text())
        if progress.get("failures"):
            raise RuntimeError(
                f"recovery still failed for {len(progress['failures'])} games: "
                f"{progress['failures'][:5]}"
            )
        recovered = pd.read_csv(recovery_path, low_memory=False)
    else:
        recovered = pd.DataFrame(columns=old.columns)

    frame = pd.concat([old, recovered], ignore_index=True)
    duplicates = int(frame.game_pk.duplicated().sum())
    if duplicates:
        raise RuntimeError(f"{args.variant}: {duplicates} duplicate game rows after recovery")
    frame = frame.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    actual_ids = set(frame.game_pk.astype(int))
    still_missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if still_missing or unexpected or len(frame) != args.expected_games:
        raise RuntimeError(
            f"{args.variant}: expected {args.expected_games}, found {len(frame)}, "
            f"missing={still_missing[:10]}, unexpected={unexpected[:10]}"
        )
    if frame.home_win_probability.isna().any():
        raise RuntimeError(f"{args.variant}: missing win probabilities")
    if not frame.home_win_probability.between(0, 1).all():
        raise RuntimeError(f"{args.variant}: probability outside [0,1]")
    if not (frame.common_random_seed.astype(int) == frame.game_pk.astype(int)).all():
        raise RuntimeError(f"{args.variant}: common seed mismatch")

    final_path = output / f"{args.variant}_predictions.csv.gz"
    frame.to_csv(final_path, index=False, compression="gzip")
    receipt = {
        "schema": "baseball_research_lab.2025_variant_recovery.v1",
        "variant": args.variant,
        "simulations_per_game": int(args.sims),
        "old_shard_files": len(old_paths),
        "old_games_recovered": int(len(old)),
        "missing_games_replayed": int(len(missing)),
        "final_games": int(len(frame)),
        "excluded_source_games": len(excluded),
        "failure_cause": "pandas-3 read-only grouped NumPy arrays during sequential history updates",
        "fix": "copy count arrays on first mutation when flags.writeable is false",
        "predictions": str(final_path),
        "predictions_sha256": sha256(final_path),
        "source_shards": [
            {"path": str(path), "sha256": sha256(path)} for path in old_paths
        ],
    }
    (output / f"{args.variant}_recovery_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
