"""Render the frozen 2026 full-game replay receipt as a compact Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def ci(comparison: dict) -> str:
    return (
        f"{float(comparison['difference']):+.6f} "
        f"[{float(comparison['ci_95_low']):+.6f}, {float(comparison['ci_95_high']):+.6f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text())
    variants = result["variants"]
    comparisons = result["paired_game_bootstrap"]
    protocol = result["protocol"]
    coverage = result["coverage"]
    decision = result["decision"]

    order = [
        ("candidate_locked_pa", "Candidate: locked PA + fitted starter hazard"),
        ("ablated_flat_league_pa", "Ablation: flat prior-date league PA"),
        ("league_nb_baseline", "League NB baseline"),
        ("team_nb_hfa_baseline", "Prior-date team NB + HFA baseline"),
        ("starter_adjusted_nb_hfa_baseline", "Starter-adjusted team NB + HFA baseline"),
        ("oracle_actual_reliever_order", "Postgame actual-reliever-order oracle"),
    ]
    lines = [
        "# Frozen 2026 full-game replay",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        "## Claim boundary",
        "",
        result["claim_status"],
        "",
        f"- Games scored: **{coverage['games']:,}**",
        f"- Dates: **{coverage['first_date']} through {coverage['last_date']}**",
        f"- Simulation paths per simulated variant/game: **{protocol['simulations_per_game_per_simulated_variant']:,}**",
        f"- Paired game bootstrap replicates: **{protocol['bootstrap_replicates']:,}**",
        f"- Games available from the history extractor: **{coverage['target_games_available_from_history']:,}**",
        f"- Games excluded by fail-closed extraction: **{coverage['target_games_excluded']:,}**",
        "",
        "Confirmed historical starting lineups and starters are treated as known pregame inputs. "
        "All player, team, bullpen, and starter tendencies use only dates strictly before the target date.",
        "",
        "## Proper-score results",
        "",
        "| Variant | Winner Brier* | Winner log loss* | Accuracy | Run CRPS | Run MAE | Win-probability SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in order:
        if key not in variants:
            continue
        row = variants[key]
        lines.append(
            f"| {label} | {fmt(row['winner_brier_unbiased'])} | "
            f"{fmt(row['winner_log_loss_bias_corrected'])} | "
            f"{fmt(row['winner_accuracy'], 4)} | {fmt(row['team_run_crps'])} | "
            f"{fmt(row['team_run_mae'])} | {fmt(row['win_probability_sd'])} |"
        )
    lines.extend([
        "",
        "*Finite-path Brier and log-loss corrections are applied to simulated variants. Lower is better for Brier, log loss, CRPS, and MAE.*",
        "",
        "## Candidate paired differences",
        "",
        "Negative differences favor the candidate.",
        "",
        "| Comparator | Brier difference [95% CI] | Log-loss difference [95% CI] | Run-CRPS difference [95% CI] | Run-MAE difference [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ])
    compare_order = [
        ("ablated_flat_league_pa", "Flat-PA ablation"),
        ("league_nb_baseline", "League NB"),
        ("team_nb_hfa_baseline", "Prior-date team NB + HFA"),
        ("starter_adjusted_nb_hfa_baseline", "Starter-adjusted team NB + HFA"),
        ("oracle_actual_reliever_order", "Actual-reliever-order oracle"),
    ]
    for key, label in compare_order:
        name = f"candidate_minus_{key}"
        if name not in comparisons:
            continue
        row = comparisons[name]
        lines.append(
            f"| {label} | {ci(row['winner_brier_unbiased'])} | "
            f"{ci(row['winner_log_loss_bias_corrected'])} | "
            f"{ci(row['team_run_crps'])} | {ci(row['team_run_mae'])} |"
        )

    lines.extend([
        "",
        "## Frozen decision",
        "",
        f"- PA signal survives the downstream engine: **{decision['candidate_pa_signal_survives_downstream_engine']}**",
        f"- Candidate clearly beats every deterministic winner-Brier baseline: **{decision['candidate_beats_all_deterministic_baselines_on_winner_brier_with_clear_ci']}**",
        f"- Promotion authorized: **{decision['promotion_authorized']}**",
        "",
        "Promotion remains blocked because:",
        "",
    ])
    for blocker in decision["promotion_blockers"]:
        lines.append(f"- {blocker}")

    lines.extend([
        "",
        "## Current engineering interpretation",
        "",
        "This replay answers whether the locked PA signal and fitted starter-removal layer survive the present full-game engine. "
        "It does not establish live pregame dependability. The actual-reliever-order variant, when present, is a postgame ceiling diagnostic only. "
        "Negative or null results are retained unchanged and must not be tuned away on the same 2026 games.",
        "",
        "## Reproducibility",
        "",
        f"- Source commit: `{protocol.get('source_commit')}`",
        f"- Candidate SHA-256: `{result['artifacts']['candidate']['sha256']}`",
        f"- Flat SHA-256: `{result['artifacts']['flat']['sha256']}`",
        f"- History SHA-256: `{result['artifacts']['history']['sha256']}`",
        f"- Starter-hazard receipt SHA-256: `{result['artifacts']['starter_hazard_receipt']['sha256']}`",
    ])
    if result["artifacts"].get("oracle"):
        lines.append(f"- Oracle SHA-256: `{result['artifacts']['oracle']['sha256']}`")

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
