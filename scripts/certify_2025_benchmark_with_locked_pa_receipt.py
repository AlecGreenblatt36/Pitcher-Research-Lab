"""Attach the pinned locked-PA reproduction receipt to a completed benchmark.

This is a post-processing certification step.  It never reruns simulations and
never changes benchmark metrics.  It fails closed unless the compatibility
receipt is PASS and the benchmark has the expected full-season protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--reproduction-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    receipt_path = Path(args.reproduction_receipt)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    if receipt.get("status") != "PASS":
        raise RuntimeError("locked PA pinned-environment reproduction is not PASS")
    protocol = benchmark.get("protocol", {})
    if int(protocol.get("games", -1)) != 2430:
        raise RuntimeError("benchmark is not a complete 2,430-game season")
    if int(protocol.get("simulations_per_game_per_variant", -1)) != 1000:
        raise RuntimeError("benchmark is not the frozen 1,000-path protocol")
    if bool(protocol.get("2026_rerun", True)):
        raise RuntimeError("benchmark receipt indicates an unauthorized 2026 rerun")

    certified = dict(benchmark)
    certified["certification"] = {
        "schema": "baseball_research_lab.2025_benchmark_certification.v1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": sha256(benchmark_path),
        },
        "locked_pa_pinned_reproduction": {
            "path": str(receipt_path),
            "sha256": sha256(receipt_path),
            "status": receipt["status"],
            "environment": receipt["environment"],
            "artifacts": receipt["artifacts"],
            "selected_dates": receipt["selected_dates"],
            "selected_rows": receipt["selected_rows"],
            "global_max_abs_differences": receipt[
                "global_max_abs_differences"
            ],
            "tolerance": receipt["tolerance"],
        },
        "boundary": (
            "Certification proves artifact/provider compatibility under the "
            "pinned replay environment. It does not promote the full-game model."
        ),
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(certified, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(certified["certification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
