"""Finalize the frozen 2026 full-game replay against forecast-valid baselines.

This script performs no model tuning.  It scores already-completed candidate,
flat-PA, and optional actual-reliever-order oracle simulations; constructs
strictly prior-date deterministic baselines from 2023-2025 history; applies
finite-path corrections; and computes paired game bootstrap intervals.

Because 2026 has already been inspected elsewhere in the project, this result
is explicitly a frozen diagnostic and cannot restore a pristine holdout claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_lab.game_sim.ablation_v2 import (
    build_nb_team_baseline_map,
    denoised_calibration,
    finite_path_log_loss_correction,
    fit_nb2_dispersion,
)
from research_lab.game_sim.replay import (
    bootstrap_metric_difference,
    calibration_parameters,
    extract_historical_games,
)
from research_lab.game_sim.replay_baselines import (
    build_league_nb_baseline_map,
    build_starter_adjusted_nb_baseline_map,
)

HISTORY_USECOLS = [
    "game_pk", "at_bat_number", "batter", "pitcher", "stand", "p_throws",
    "home_team", "away_team", "inning", "inning_topbot", "outs_when_up",
    "home_score", "away_score", "bat_score", "fld_score", "game_type",
    "terminal_event", "season", "date_key", "runner_1b", "runner_2b",
    "runner_3b", "park",
]
SHARED = [
    "game_pk", "game_date", "away_team", "home_team", "actual_away_runs",
    "actual_home_runs", "actual_home_win",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prefix_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(columns={column: f"{prefix}{column}" for column in frame.columns if column not in SHARED})


def score_variant(
    frame: pd.DataFrame,
    prefix: str,
    simulations: int | None,
) -> tuple[dict, dict[str, np.ndarray]]:
    y = frame["actual_home_win"].to_numpy(float)
    p = np.clip(frame[f"{prefix}home_win_probability"].to_numpy(float), 1e-8, 1 - 1e-8)
    brier_raw = (p - y) ** 2
    if simulations:
        brier_correction = p * (1.0 - p) / (simulations - 1)
        log_loss_raw, log_loss_bias, log_loss_corrected = finite_path_log_loss_correction(
            y, p, simulations
        )
        calibration = denoised_calibration(y, p, simulations)
    else:
        brier_correction = np.zeros_like(p)
        log_loss_raw = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        log_loss_bias = np.zeros_like(p)
        log_loss_corrected = log_loss_raw.copy()
        calibration = {"raw": calibration_parameters(y, p), "de_noised": None}

    away_error = frame[f"{prefix}away_mean_runs"].to_numpy(float) - frame["actual_away_runs"].to_numpy(float)
    home_error = frame[f"{prefix}home_mean_runs"].to_numpy(float) - frame["actual_home_runs"].to_numpy(float)
    run_errors = np.concatenate([away_error, home_error])
    game_crps = (
        frame[f"{prefix}away_crps"].to_numpy(float)
        + frame[f"{prefix}home_crps"].to_numpy(float)
    ) / 2.0
    game_mae = (np.abs(away_error) + np.abs(home_error)) / 2.0
    metrics = {
        "winner_brier_raw": float(brier_raw.mean()),
        "winner_brier_mc_correction": float(brier_correction.mean()),
        "winner_brier_unbiased": float((brier_raw - brier_correction).mean()),
        "winner_log_loss_raw": float(log_loss_raw.mean()),
        "winner_log_loss_mc_bias_estimate": float(log_loss_bias.mean()),
        "winner_log_loss_bias_corrected": float(log_loss_corrected.mean()),
        "winner_accuracy": float(np.mean((p >= 0.5) == (y >= 0.5))),
        "win_probability_mean": float(p.mean()),
        "win_probability_sd": float(p.std(ddof=1)),
        "calibration": calibration,
        "team_run_mae": float(np.mean(np.abs(run_errors))),
        "team_run_rmse": float(np.sqrt(np.mean(run_errors ** 2))),
        "team_run_crps": float(game_crps.mean()),
        "away_run_bias": float(away_error.mean()),
        "home_run_bias": float(home_error.mean()),
    }
    rows = {
        "winner_brier_unbiased": brier_raw - brier_correction,
        "winner_log_loss_bias_corrected": log_loss_corrected,
        "team_run_crps": game_crps,
        "team_run_mae": game_mae,
    }
    return metrics, rows


def add_map_columns(frame: pd.DataFrame, prefix: str, mapping: dict[int, dict]) -> None:
    required = [
        "home_win_probability", "away_mean_runs", "home_mean_runs",
        "away_crps", "home_crps", "prior_home_win_rate",
        "home_logit_offset", "neutral_home_win_probability",
    ]
    missing = [int(pk) for pk in frame["game_pk"] if int(pk) not in mapping]
    if missing:
        raise RuntimeError(f"{prefix}: baseline missing {len(missing)} games: {missing[:5]}")
    for column in required:
        frame[f"{prefix}{column}"] = frame["game_pk"].map(
            lambda value, c=column: mapping[int(value)][c]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--flat", required=True)
    parser.add_argument("--oracle")
    parser.add_argument("--history", required=True)
    parser.add_argument("--starter-hazard-receipt", required=True)
    parser.add_argument("--merge-receipt")
    parser.add_argument("--score-overrides-2026")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(args.candidate)
    flat_path = Path(args.flat)
    oracle_path = Path(args.oracle) if args.oracle else None
    history_path = Path(args.history)

    candidate = prefix_frame(pd.read_csv(candidate_path, low_memory=False), "candidate_")
    flat = prefix_frame(pd.read_csv(flat_path, low_memory=False), "flat_")
    frame = candidate.merge(flat, on=SHARED, validate="one_to_one")
    if oracle_path:
        oracle = prefix_frame(pd.read_csv(oracle_path, low_memory=False), "oracle_")
        frame = frame.merge(oracle, on=SHARED, validate="one_to_one")
    if frame["game_pk"].duplicated().any():
        raise RuntimeError("duplicate games after variant merge")
    if not np.array_equal(frame["game_pk"].to_numpy(int), candidate["game_pk"].to_numpy(int)):
        raise RuntimeError("variant merge changed candidate game order")
    frame = frame.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    history = pd.read_csv(history_path, usecols=HISTORY_USECOLS, low_memory=False)
    history["date_key"] = history["date_key"].astype(str).str[:10]
    override_2025_path = ROOT / "research_lab/game_sim/official_score_overrides_2025.json"
    overrides_2025 = {int(key): value for key, value in json.loads(override_2025_path.read_text()).items()}
    overrides_2026: dict[int, dict | list] = {}
    if args.score_overrides_2026:
        overrides_2026 = {
            int(key): value
            for key, value in json.loads(Path(args.score_overrides_2026).read_text()).items()
        }

    prior_games = []
    prior_excluded: dict[int, list[dict]] = {}
    for season in (2023, 2024, 2025):
        games, excluded = extract_historical_games(
            history,
            season,
            score_overrides=overrides_2025 if season == 2025 else None,
        )
        prior_games.extend(games)
        prior_excluded[season] = excluded
    target_games_all, target_excluded = extract_historical_games(
        history, args.season, score_overrides=overrides_2026
    )
    target_lookup = {int(game.game_pk): game for game in target_games_all}
    target_games = [target_lookup[int(game_pk)] for game_pk in frame["game_pk"]]

    nb_fit = fit_nb2_dispersion(prior_games)
    league_map = build_league_nb_baseline_map(target_games, prior_games, nb_fit)
    team_map = build_nb_team_baseline_map(target_games, prior_games, nb_fit)
    starter_map = build_starter_adjusted_nb_baseline_map(
        target_games, prior_games, history, nb_fit
    )
    add_map_columns(frame, "league_nb_", league_map)
    add_map_columns(frame, "team_nb_hfa_", team_map)
    add_map_columns(frame, "starter_nb_hfa_", starter_map)

    variant_specs: list[tuple[str, str, int | None]] = [
        ("candidate_locked_pa", "candidate_", args.simulations),
        ("ablated_flat_league_pa", "flat_", args.simulations),
        ("league_nb_baseline", "league_nb_", None),
        ("team_nb_hfa_baseline", "team_nb_hfa_", None),
        ("starter_adjusted_nb_hfa_baseline", "starter_nb_hfa_", None),
    ]
    if oracle_path:
        variant_specs.append(("oracle_actual_reliever_order", "oracle_", args.simulations))

    variants: dict[str, dict] = {}
    per_game: dict[str, dict[str, np.ndarray]] = {}
    for name, prefix, simulations in variant_specs:
        variants[name], per_game[name] = score_variant(frame, prefix, simulations)

    comparisons: dict[str, dict] = {}
    comparison_order = [name for name, _, _ in variant_specs if name != "candidate_locked_pa"]
    for index, other in enumerate(comparison_order):
        label = f"candidate_minus_{other}"
        comparisons[label] = {}
        for offset, metric in enumerate(
            (
                "winner_brier_unbiased",
                "winner_log_loss_bias_corrected",
                "team_run_crps",
                "team_run_mae",
            )
        ):
            comparisons[label][metric] = bootstrap_metric_difference(
                per_game["candidate_locked_pa"][metric],
                per_game[other][metric],
                reps=args.bootstrap_replicates,
                seed=202600 + index * 10 + offset,
            )

    if oracle_path:
        comparisons["oracle_minus_starter_adjusted_baseline"] = {}
        for offset, metric in enumerate(
            ("winner_brier_unbiased", "winner_log_loss_bias_corrected", "team_run_crps", "team_run_mae")
        ):
            comparisons["oracle_minus_starter_adjusted_baseline"][metric] = bootstrap_metric_difference(
                per_game["oracle_actual_reliever_order"][metric],
                per_game["starter_adjusted_nb_hfa_baseline"][metric],
                reps=args.bootstrap_replicates,
                seed=202690 + offset,
            )

    hazard_receipt_path = Path(args.starter_hazard_receipt)
    hazard_receipt = json.loads(hazard_receipt_path.read_text())
    merge_receipt = (
        json.loads(Path(args.merge_receipt).read_text()) if args.merge_receipt else None
    )
    candidate_vs_flat = comparisons["candidate_minus_ablated_flat_league_pa"]
    candidate_signal_survives = bool(
        candidate_vs_flat["winner_brier_unbiased"]["ci_95_high"] < 0
        or candidate_vs_flat["winner_log_loss_bias_corrected"]["ci_95_high"] < 0
        or candidate_vs_flat["team_run_crps"]["ci_95_high"] < 0
    )
    baseline_names = [
        "league_nb_baseline",
        "team_nb_hfa_baseline",
        "starter_adjusted_nb_hfa_baseline",
    ]
    candidate_beats_all_baselines = all(
        comparisons[f"candidate_minus_{name}"]["winner_brier_unbiased"]["ci_95_high"] < 0
        for name in baseline_names
    )

    result = {
        "schema": "baseball_research_lab.2026_frozen_full_game_replay.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": (
            "frozen 2026 historical replay diagnostic; 2026 was previously inspected; "
            "not a pristine holdout and not a live prospective forecast"
        ),
        "protocol": {
            "season": int(args.season),
            "games": int(len(frame)),
            "simulations_per_game_per_simulated_variant": int(args.simulations),
            "same_per_game_seed": True,
            "seed": "game_pk",
            "confirmed_lineup_boundary": (
                "actual first nine batters and first pitcher are treated as confirmed pregame inputs"
            ),
            "history_boundary": (
                "player, pitcher, bullpen, team and starter tendencies use dates strictly before target date; "
                "all games on a date are revealed together afterward"
            ),
            "brier_correction": "p_hat*(1-p_hat)/(N-1), per game",
            "log_loss_correction": "second-order delta-method finite-path bias, per game",
            "crps": (
                "unbiased empirical U-statistic for simulated variants; exact negative-binomial CDF for deterministic baselines"
            ),
            "bootstrap": "paired game bootstrap",
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "2026_previously_accessed": True,
            "development_frozen_before_run": True,
            "source_commit": os.getenv("GITHUB_SHA"),
        },
        "coverage": {
            "first_date": str(frame["game_date"].min()),
            "last_date": str(frame["game_date"].max()),
            "games": int(len(frame)),
            "target_games_available_from_history": int(len(target_games_all)),
            "target_games_excluded": int(len(target_excluded)),
            "target_exclusions": target_excluded,
            "prior_excluded_counts": {str(year): len(rows) for year, rows in prior_excluded.items()},
        },
        "negative_binomial_fit_2023_2025": {
            "alpha": float(nb_fit.alpha),
            "away_mean_prior": float(nb_fit.away_mean_2023_2024),
            "home_mean_prior": float(nb_fit.home_mean_2023_2024),
            "games": int(nb_fit.games),
            "observations": int(nb_fit.observations),
            "fit_seasons": [2023, 2024, 2025],
            "method": nb_fit.method,
        },
        "starter_hazard": hazard_receipt,
        "variants": variants,
        "paired_game_bootstrap": comparisons,
        "oracle_boundary": (
            "postgame actual reliever identities/order; diagnostic ceiling only; never a forecast"
            if oracle_path else None
        ),
        "decision": {
            "candidate_pa_signal_survives_downstream_engine": candidate_signal_survives,
            "candidate_beats_all_deterministic_baselines_on_winner_brier_with_clear_ci": candidate_beats_all_baselines,
            "promotion_authorized": False,
            "promotion_blockers": [
                "2026 was previously inspected and cannot be represented as a pristine untouched holdout",
                "runner transitions remain a development kernel",
                "forecast-valid reliever candidate, exit and selection layers are not yet validated",
                "historical replay is not a live pregame forecast record",
            ],
        },
        "artifacts": {
            "candidate": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
            "flat": {"path": str(flat_path), "sha256": sha256(flat_path)},
            "oracle": ({"path": str(oracle_path), "sha256": sha256(oracle_path)} if oracle_path else None),
            "history": {"path": str(history_path), "sha256": sha256(history_path)},
            "starter_hazard_receipt": {
                "path": str(hazard_receipt_path),
                "sha256": sha256(hazard_receipt_path),
            },
            "merge_receipt": (
                {"path": str(Path(args.merge_receipt)), "sha256": sha256(Path(args.merge_receipt))}
                if args.merge_receipt else None
            ),
        },
        "merge_receipt": merge_receipt,
    }
    result_path = output / "2026_FROZEN_FULL_GAME_REPLAY_RESULT.json"
    predictions_path = output / "2026_FROZEN_FULL_GAME_REPLAY_PREDICTIONS.csv.gz"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    frame.to_csv(predictions_path, index=False, compression="gzip")
    print(json.dumps({
        "result": str(result_path),
        "predictions": str(predictions_path),
        "games": len(frame),
        "decision": result["decision"],
        "variants": variants,
    }, indent=2)[:40000])


if __name__ == "__main__":
    main()
