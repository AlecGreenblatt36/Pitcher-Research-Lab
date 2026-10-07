from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Any

import numpy as np

from .engine import GameResult, GameSimulator
from .models import GameMatchup


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    spread = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return (max(0.0, center - spread), min(1.0, center + spread))


def _distribution_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _starter_outs(result: GameResult, starter_id: str) -> int:
    line = result.pitcher_lines.get(starter_id)
    return int(line["outs_recorded"]) if line else 0


def simulate_many(
    simulator: GameSimulator,
    matchup: GameMatchup,
    simulations: int = 1000,
    seed: int = 20261003,
) -> dict[str, Any]:
    if simulations < 50:
        raise ValueError("at least 50 simulations are required")
    if simulations > 20000:
        raise ValueError("simulations cannot exceed 20,000 in one request")

    seed_generator = np.random.default_rng(seed)
    simulation_seeds = seed_generator.integers(
        0,
        np.iinfo(np.int32).max,
        size=simulations,
        dtype=np.int64,
    )
    results: list[GameResult] = []
    scorelines: Counter[tuple[int, int]] = Counter()
    appearances: dict[str, Counter[str]] = {
        "away": Counter(),
        "home": Counter(),
    }
    inning_runs: dict[str, defaultdict[int, list[int]]] = {
        "away": defaultdict(list),
        "home": defaultdict(list),
    }
    outcome_totals: Counter[str] = Counter()

    for simulation_seed in simulation_seeds:
        result = simulator.simulate(
            matchup,
            int(simulation_seed),
            record_events=False,
        )
        results.append(result)
        scorelines[(result.away_score, result.home_score)] += 1
        outcome_totals.update(result.outcome_counts)
        for side in ("away", "home"):
            appearances[side].update(result.pitcher_appearances.get(side, []))

    max_inning = max(result.innings_played for result in results)
    for result in results:
        for side in ("away", "home"):
            runs_by_inning = result.inning_runs.get(side, {})
            for inning in range(1, max_inning + 1):
                inning_runs[side][inning].append(int(runs_by_inning.get(inning, 0)))

    away_wins = sum(result.winner == "away" for result in results)
    home_wins = sum(result.winner == "home" for result in results)
    ties = simulations - away_wins - home_wins
    away_scores = [result.away_score for result in results]
    home_scores = [result.home_score for result in results]
    totals = [result.away_score + result.home_score for result in results]
    margins = [result.home_score - result.away_score for result in results]
    away_starter_outs = [
        _starter_outs(result, matchup.away.starter.player_id)
        for result in results
    ]
    home_starter_outs = [
        _starter_outs(result, matchup.home.starter.player_id)
        for result in results
    ]

    away_interval = _wilson_interval(away_wins, simulations)
    home_interval = _wilson_interval(home_wins, simulations)

    top_scorelines = [
        {
            "away_runs": away_runs,
            "home_runs": home_runs,
            "count": count,
            "probability": count / simulations,
            "label": f"{matchup.away.name} {away_runs} – {matchup.home.name} {home_runs}",
        }
        for (away_runs, home_runs), count in scorelines.most_common(12)
    ]

    pitcher_appearance_probabilities: dict[str, list[dict[str, Any]]] = {}
    profile_lookup = {
        pitcher.player_id: pitcher
        for team in (matchup.away, matchup.home)
        for pitcher in (team.starter, *team.bullpen)
    }
    for side in ("away", "home"):
        pitcher_appearance_probabilities[side] = [
            {
                "pitcher_id": pitcher_id,
                "name": profile_lookup[pitcher_id].name,
                "role": profile_lookup[pitcher_id].role,
                "probability": count / simulations,
            }
            for pitcher_id, count in appearances[side].most_common()
        ]

    median_away = float(np.median(away_scores))
    median_home = float(np.median(home_scores))
    median_away_outs = float(np.median(away_starter_outs))
    median_home_outs = float(np.median(home_starter_outs))

    representative_result = min(
        results,
        key=lambda result: (
            (result.away_score - median_away) ** 2
            + (result.home_score - median_home) ** 2
            + 0.05
            * (
                (_starter_outs(result, matchup.away.starter.player_id) - median_away_outs) ** 2
                + (_starter_outs(result, matchup.home.starter.player_id) - median_home_outs) ** 2
            )
        ),
    )
    representative = simulator.simulate(
        matchup,
        representative_result.seed,
        record_events=True,
    )

    expected_inning_runs = {
        side: [
            {
                "inning": inning,
                "mean_runs": float(np.mean(values)),
                "scoreless_probability": float(np.mean(np.asarray(values) == 0)),
                "two_plus_probability": float(np.mean(np.asarray(values) >= 2)),
            }
            for inning, values in sorted(inning_runs[side].items())
        ]
        for side in ("away", "home")
    }

    return {
        "simulation_count": simulations,
        "seed": seed,
        "matchup": {
            "away": matchup.away.name,
            "home": matchup.home.name,
            "venue": matchup.venue,
            "park_factor": matchup.park_factor,
            "weather_run_factor": matchup.weather_run_factor,
        },
        "win_probabilities": {
            "away": away_wins / simulations,
            "home": home_wins / simulations,
            "tie": ties / simulations,
            "away_monte_carlo_95_interval": list(away_interval),
            "home_monte_carlo_95_interval": list(home_interval),
        },
        "runs": {
            "away": _distribution_summary([float(value) for value in away_scores]),
            "home": _distribution_summary([float(value) for value in home_scores]),
            "total": _distribution_summary([float(value) for value in totals]),
            "home_margin": _distribution_summary([float(value) for value in margins]),
        },
        "top_scorelines": top_scorelines,
        "inning_run_outlook": expected_inning_runs,
        "starters": {
            "away": {
                "pitcher_id": matchup.away.starter.player_id,
                "name": matchup.away.starter.name,
                "innings": _distribution_summary(
                    [value / 3.0 for value in away_starter_outs]
                ),
            },
            "home": {
                "pitcher_id": matchup.home.starter.player_id,
                "name": matchup.home.starter.name,
                "innings": _distribution_summary(
                    [value / 3.0 for value in home_starter_outs]
                ),
            },
        },
        "pitcher_appearance_probabilities": pitcher_appearance_probabilities,
        "outcome_mix": {
            label: outcome_totals[label] / max(1, sum(outcome_totals.values()))
            for label in sorted(outcome_totals)
        },
        "representative_game": representative.to_dict(),
        "model_status": {
            "pa_provider": getattr(simulator.provider, "name", type(simulator.provider).__name__),
            "pa_provider_validation": getattr(
                simulator.provider,
                "validation_status",
                "unknown",
            ),
            "game_engine_validation": "not-yet-game-level-validated",
            "interpretation": (
                "Win and score percentages are frequencies from this simulation engine. "
                "They are not calibrated real-world game probabilities until the frozen "
                "historical game replay is completed."
            ),
        },
    }
