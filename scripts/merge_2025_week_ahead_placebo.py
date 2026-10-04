"""Merge and verify all 2025 week-ahead placebo replay shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-games", type=int, default=2430)
    args = parser.parse_args()

    source = Path(args.input_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(source.rglob("placebo_week_shard_*.csv.gz"))
    receipt_paths = sorted(source.rglob("placebo_week_shard_*_receipt.json"))
    progress_paths = sorted(source.rglob("placebo_week_shard_*_progress.json"))
    if not paths:
        raise RuntimeError(f"no placebo shard files found under {source}")

    receipts = [json.loads(path.read_text()) for path in receipt_paths]
    progress = [json.loads(path.read_text()) for path in progress_paths]
    failed = [row for item in progress for row in item.get("failures", [])]
    if failed:
        raise RuntimeError(f"placebo shards contain {len(failed)} failures: {failed[:5]}")

    frame = pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)
    duplicates = int(frame.game_pk.duplicated().sum())
    if duplicates:
        raise RuntimeError(f"placebo: {duplicates} duplicate game rows")
    frame = frame.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    if len(frame) != args.expected_games:
        raise RuntimeError(f"placebo: expected {args.expected_games}, found {len(frame)}")
    if frame.home_win_probability.isna().any():
        raise RuntimeError("placebo: missing win probabilities")
    if not frame.home_win_probability.between(0, 1).all():
        raise RuntimeError("placebo: probability outside [0,1]")
    if not (frame.common_random_seed.astype(int) == frame.game_pk.astype(int)).all():
        raise RuntimeError("placebo: common seed mismatch")

    target = output / "placebo_week_ahead_predictions.csv.gz"
    frame.to_csv(target, index=False, compression="gzip")

    sides = 2 * len(frame)
    fallback_sides = int(
        frame.placebo_away_source_game_pk.isna().sum()
        + frame.placebo_home_source_game_pk.isna().sum()
    )
    future_only = int(
        frame.placebo_away_future_only_count.fillna(0).sum()
        + frame.placebo_home_future_only_count.fillna(0).sum()
    )
    source_pitchers = int(
        frame.placebo_away_reliever_count.fillna(0).sum()
        + frame.placebo_home_reliever_count.fillna(0).sum()
    )
    days = pd.concat(
        [frame.placebo_away_days_ahead, frame.placebo_home_days_ahead],
        ignore_index=True,
    ).dropna()

    merge_receipt = {
        "schema": "baseball_research_lab.week_ahead_placebo_merge.v1",
        "games": int(len(frame)),
        "shards": int(len(paths)),
        "simulations_per_game": 1000,
        "common_seed": "game_pk",
        "selection_window_days": [5, 9],
        "preferred_days_ahead": 7,
        "fallback_sides": fallback_sides,
        "fallback_side_rate": fallback_sides / sides,
        "source_sides": sides - fallback_sides,
        "source_pitchers": source_pitchers,
        "future_only_pitchers": future_only,
        "future_only_pitcher_fraction": (
            future_only / source_pitchers if source_pitchers else None
        ),
        "mean_days_ahead": float(days.mean()) if len(days) else None,
        "median_days_ahead": float(days.median()) if len(days) else None,
        "forecast_valid": False,
        "predictions": str(target),
        "predictions_sha256": sha256(target),
        "source_shards": [
            {"path": str(path), "sha256": sha256(path)} for path in paths
        ],
        "shard_receipts": receipts,
    }
    (output / "WEEK_AHEAD_PLACEBO_MERGE_RECEIPT.json").write_text(
        json.dumps(merge_receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(merge_receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
