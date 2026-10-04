from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-games", type=int, default=2430)
    args = parser.parse_args()
    source = Path(args.input_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "baseball_research_lab.2025_shard_merge.v2", "variants": {}}
    for variant in ("candidate", "flat", "oracle"):
        paths = sorted(source.rglob(f"{variant}_shard_*.csv.gz"))
        if not paths:
            raise RuntimeError(f"no {variant} shard files found under {source}")
        frame = pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)
        duplicates = int(frame.game_pk.duplicated().sum())
        if duplicates:
            raise RuntimeError(f"{variant}: {duplicates} duplicate game rows")
        frame = frame.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
        if len(frame) != args.expected_games:
            raise RuntimeError(f"{variant}: expected {args.expected_games} games, found {len(frame)}")
        if frame.home_win_probability.isna().any():
            raise RuntimeError(f"{variant}: missing win probabilities")
        if not frame.home_win_probability.between(0, 1).all():
            raise RuntimeError(f"{variant}: probability outside [0,1]")
        if not (frame.common_random_seed.astype(int) == frame.game_pk.astype(int)).all():
            raise RuntimeError(f"{variant}: common seed mismatch")
        target = output / f"{variant}_predictions.csv.gz"
        frame.to_csv(target, index=False, compression="gzip")
        manifest["variants"][variant] = {
            "games": int(len(frame)),
            "shards": int(len(paths)),
            "path": str(target),
            "first_date": str(frame.game_date.min()),
            "last_date": str(frame.game_date.max()),
            "pa_cap_rate": float(frame.pa_cap_rate.mean()),
        }
    (output / "SHARD_MERGE_RECEIPT.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
