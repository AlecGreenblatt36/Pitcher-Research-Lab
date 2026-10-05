"""Merge and fail-closed verify frozen 2026 replay shards."""
from __future__ import annotations

import argparse
from collections import Counter
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
    parser.add_argument("--variants", nargs="+", default=["candidate", "flat", "oracle"])
    parser.add_argument("--expected-games", type=int, default=0)
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()

    source = Path(args.input_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema": "baseball_research_lab.2026_shard_merge.v1",
        "season": int(args.season),
        "variants": {},
    }
    reference_ids: list[int] | None = None
    inferred_expected: int | None = args.expected_games or None

    for variant in args.variants:
        paths = sorted(source.rglob(f"{variant}_shard_*.csv.gz"))
        receipt_paths = sorted(source.rglob(f"{variant}_shard_*_receipt.json"))
        progress_paths = sorted(source.rglob(f"{variant}_shard_*_progress.json"))
        if not paths:
            raise RuntimeError(f"no {variant} shard files found under {source}")
        if not receipt_paths:
            raise RuntimeError(f"no {variant} shard receipts found under {source}")

        receipts = [json.loads(path.read_text()) for path in receipt_paths]
        failures = [
            failure
            for path in progress_paths
            for failure in json.loads(path.read_text()).get("failures", [])
        ]
        if failures:
            raise RuntimeError(f"{variant}: shard failures present: {failures[:5]}")
        if any(int(receipt.get("season", -1)) != args.season for receipt in receipts):
            raise RuntimeError(f"{variant}: season mismatch in shard receipt")
        if any(receipt.get("variant") != variant for receipt in receipts):
            raise RuntimeError(f"{variant}: variant mismatch in shard receipt")

        frame = pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)
        duplicates = int(frame["game_pk"].duplicated().sum())
        if duplicates:
            raise RuntimeError(f"{variant}: {duplicates} duplicate game rows")
        frame = frame.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
        if inferred_expected is None:
            inferred_expected = len(frame)
        if len(frame) != inferred_expected:
            raise RuntimeError(
                f"{variant}: expected {inferred_expected} games, found {len(frame)}"
            )
        if frame["home_win_probability"].isna().any():
            raise RuntimeError(f"{variant}: missing win probabilities")
        if not frame["home_win_probability"].between(0, 1).all():
            raise RuntimeError(f"{variant}: probability outside [0,1]")
        if not (frame["common_random_seed"].astype(int) == frame["game_pk"].astype(int)).all():
            raise RuntimeError(f"{variant}: common seed mismatch")
        if not (frame["variant"] == variant).all():
            raise RuntimeError(f"{variant}: row variant mismatch")

        game_ids = frame["game_pk"].astype(int).tolist()
        if reference_ids is None:
            reference_ids = game_ids
        elif reference_ids != game_ids:
            raise RuntimeError(f"{variant}: game set/order differs from first variant")

        target = output / f"{variant}_predictions.csv.gz"
        frame.to_csv(target, index=False, compression="gzip")
        exclusion_counts = Counter()
        for receipt in receipts:
            exclusion_counts.update(receipt.get("excluded_reason_counts", {}))
        # The exclusion summary is repeated in every shard receipt. Preserve
        # one canonical copy rather than summing duplicates.
        exclusion_summary = receipts[0].get("excluded_reason_counts", {})
        manifest["variants"][variant] = {
            "games": int(len(frame)),
            "shards": int(len(paths)),
            "path": str(target),
            "sha256": sha256(target),
            "first_date": str(frame["game_date"].min()),
            "last_date": str(frame["game_date"].max()),
            "pa_cap_rate": float(frame["pa_cap_rate"].mean()),
            "simulations_per_game": int(receipts[0]["simulations_per_game"]),
            "games_available_from_history": int(receipts[0]["games_available_from_history"]),
            "excluded_game_count": int(receipts[0]["excluded_game_count"]),
            "excluded_reason_counts": exclusion_summary,
            "artifact_hashes": receipts[0].get("artifacts", {}),
        }

    manifest["expected_games"] = int(inferred_expected or 0)
    receipt_path = output / "SHARD_MERGE_RECEIPT.json"
    receipt_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
