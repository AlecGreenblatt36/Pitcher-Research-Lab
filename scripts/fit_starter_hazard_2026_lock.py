"""Fit the frozen starter-removal hazard used by the 2026 game replay.

Regularization C=1.0 was selected before this replay.  The model is fit on
2023-2025 only and emits a receipt proving that no 2026 decision row entered the
fit.  The 2026 game outcomes are never read by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_lab.game_sim.starter_hazard import (
    FEATURES,
    add_tendencies,
    build_decisions,
    fit_hazard,
)

USECOLS = [
    "game_pk", "at_bat_number", "pitcher", "home_team", "away_team",
    "inning", "inning_topbot", "outs_when_up", "bat_score", "fld_score",
    "runner_1b", "runner_2b", "runner_3b", "season", "date_key",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    train_seasons = sorted(set(args.train_seasons))
    if not train_seasons or max(train_seasons) >= args.test_season:
        raise SystemExit("train seasons must be non-empty and strictly precede test season")
    if args.C <= 0:
        raise SystemExit("C must be positive")

    history_path = Path(args.history)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(history_path, usecols=USECOLS, low_memory=False)
    history["date_key"] = history["date_key"].astype(str).str[:10]
    train_history = history[history["season"].isin(train_seasons)].copy()
    if train_history.empty:
        raise RuntimeError("no training rows found")
    if int(train_history["season"].max()) >= args.test_season:
        raise RuntimeError("test-season rows entered starter-hazard training")

    decisions, starts = build_decisions(train_history)
    league_mean = float(starts["bf_total"].mean())
    fitted = add_tendencies(decisions, starts, league_mean)
    model = fit_hazard(fitted, args.C)
    bundle = {
        "model": model,
        "league_mean_bf": league_mean,
        "selected_C": float(args.C),
        "features": list(FEATURES),
        "train_seasons": train_seasons,
        "test_season": int(args.test_season),
        "protocol": "fixed C from pre-2026 development; refit on all 2023-2025 rows",
    }
    model_path = output / "starter_hazard_2026_frozen.joblib"
    joblib.dump(bundle, model_path)

    receipt = {
        "schema": "baseball_research_lab.starter_hazard_2026_frozen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "train_seasons": train_seasons,
            "test_season": int(args.test_season),
            "selected_C": float(args.C),
            "regularization_source": "frozen pre-2026 development result",
            "test_outcomes_read": False,
            "fit_max_season": int(train_history["season"].max()),
        },
        "training": {
            "history_rows": int(len(train_history)),
            "decision_rows": int(len(fitted)),
            "starts": int(len(starts)),
            "removal_rate": float(fitted["y"].mean()),
            "league_mean_bf": league_mean,
            "date_min": str(train_history["date_key"].min()),
            "date_max": str(train_history["date_key"].max()),
            "features": list(FEATURES),
        },
        "artifacts": {
            "history_path": str(history_path),
            "history_sha256": sha256(history_path),
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
        },
    }
    receipt_path = output / "STARTER_HAZARD_2026_FROZEN_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
