"""Compare the forecast-valid engine, week-ahead placebo, and same-game oracle.

The week-ahead placebo estimates value from a more realistic bullpen identity
and role list without using the target game's realized reliever usage. The
same-game oracle remains a postgame ceiling with irreducible outcome leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_lab.game_sim.ablation_v2 import finite_path_log_loss_correction
from research_lab.game_sim.replay import bootstrap_metric_difference


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prefix(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    shared = [
        "game_pk", "game_date", "away_team", "home_team",
        "actual_away_runs", "actual_home_runs", "actual_home_win",
    ]
    return frame.rename(
        columns={column: f"{name}_{column}" for column in frame.columns if column not in shared}
    )


def per_game_scores(frame: pd.DataFrame, name: str, simulations: int) -> dict[str, np.ndarray]:
    y = frame.actual_home_win.to_numpy(float)
    p = np.clip(frame[f"{name}_home_win_probability"].to_numpy(float), 1e-8, 1 - 1e-8)
    brier = (p - y) ** 2 - p * (1 - p) / (simulations - 1)
    _, _, log_loss = finite_path_log_loss_correction(y, p, simulations)
    crps = (
        frame[f"{name}_away_crps"].to_numpy(float)
        + frame[f"{name}_home_crps"].to_numpy(float)
    ) / 2.0
    return {"brier": brier, "log_loss": log_loss, "crps": crps, "probability": p}


def comparison(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict:
    output = {"games": int(mask.sum())}
    for offset, metric in enumerate(("brier", "log_loss", "crps")):
        output[metric] = bootstrap_metric_difference(
            left[metric][mask], right[metric][mask], reps=reps, seed=seed + offset
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--placebo", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    paths = {
        "candidate": Path(args.candidate),
        "placebo": Path(args.placebo),
        "oracle": Path(args.oracle),
    }
    frames = {name: prefix(pd.read_csv(path, low_memory=False), name) for name, path in paths.items()}
    keys = [
        "game_pk", "game_date", "away_team", "home_team",
        "actual_away_runs", "actual_home_runs", "actual_home_win",
    ]
    frame = frames["candidate"].merge(frames["placebo"], on=keys, validate="one_to_one")
    frame = frame.merge(frames["oracle"], on=keys, validate="one_to_one")
    if len(frame) != 2430 or frame.game_pk.duplicated().any():
        raise RuntimeError(f"expected 2430 unique games, found {len(frame)}")

    scores = {
        name: per_game_scores(frame, name, args.simulations)
        for name in ("candidate", "placebo", "oracle")
    }
    margin = np.abs(frame.actual_home_runs.to_numpy(int) - frame.actual_away_runs.to_numpy(int))
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "margin_1_2": (margin >= 1) & (margin <= 2),
        "margin_3_4": (margin >= 3) & (margin <= 4),
        "margin_5_plus": margin >= 5,
    }

    comparisons = {}
    pair_specs = [
        ("placebo_minus_candidate", "placebo", "candidate", 100),
        ("oracle_minus_placebo", "oracle", "placebo", 200),
        ("oracle_minus_candidate", "oracle", "candidate", 300),
    ]
    for pair_name, left, right, seed in pair_specs:
        comparisons[pair_name] = {
            mask_name: comparison(
                scores[left], scores[right], mask,
                reps=args.bootstrap_replicates,
                seed=seed + 10 * index,
            )
            for index, (mask_name, mask) in enumerate(masks.items())
        }

    fallback_sides = int(
        frame.placebo_placebo_away_source_game_pk.isna().sum()
        + frame.placebo_placebo_home_source_game_pk.isna().sum()
    )
    total_sides = 2 * len(frame)
    future_only = int(
        frame.placebo_placebo_away_future_only_count.fillna(0).sum()
        + frame.placebo_placebo_home_future_only_count.fillna(0).sum()
    )
    source_pitchers = int(
        frame.placebo_placebo_away_reliever_count.fillna(0).sum()
        + frame.placebo_placebo_home_reliever_count.fillna(0).sum()
    )

    placebo_brier = comparisons["placebo_minus_candidate"]["all"]["brier"]
    decision = {
        "pregame_bullpen_pipeline_priority": bool(placebo_brier["ci_95_high"] < 0),
        "meaning": (
            "A clear placebo gain supports repairing the pregame active-roster, "
            "availability, and role pipeline. Oracle-minus-placebo remains descriptive "
            "and cannot authorize reliever models because it mixes usage value with leakage."
        ),
        "reliever_model_promotion_from_this_diagnostic": False,
    }

    result = {
        "schema": "baseball_research_lab.2025_bullpen_placebo_comparison.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "games": int(len(frame)),
            "simulations_per_game": int(args.simulations),
            "same_game_seed": "game_pk",
            "brier_correction": "p_hat*(1-p_hat)/(N-1), per game",
            "log_loss_correction": "second-order delta method, per game",
            "placebo": "actual reliever sequence from a different game 5-9 days later, preferring 7; never a forecast",
            "oracle": "actual reliever sequence from the target game; outcome-leaking ceiling; never a forecast",
        },
        "comparisons": comparisons,
        "placebo_coverage": {
            "fallback_sides": fallback_sides,
            "fallback_side_rate": fallback_sides / total_sides,
            "source_sides": total_sides - fallback_sides,
            "source_pitchers": source_pitchers,
            "future_only_pitchers": future_only,
            "future_only_pitcher_fraction": (
                future_only / source_pitchers if source_pitchers else None
            ),
        },
        "decision": decision,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
