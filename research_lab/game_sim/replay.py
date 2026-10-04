"""Leakage-controlled full-game historical replay utilities.

This module builds a pregame fixture from the PA history using only rows dated
strictly before the target game for player/pitcher histories, bullpen
availability, and starter tendencies. The actual starting lineup and starter
are taken from that game's first appearances, which is an allowed pregame input
for a confirmed-lineup replay.

The game layer is still experimental: terminal PA runner transitions and
reliever selection remain development components. Results from this module must
be reported as a frozen historical replay, not live prospective validation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp, lgamma, log
from pathlib import Path
import json
from typing import Iterable, Mapping, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .engine import GameSimulator
from .locked_pa_provider import (
    FIXED_LEAGUE_PRIOR, MODEL_LABELS, SIM_FROM_MODEL, LockedPAModelProvider, _posterior,
)
from .models import GameMatchup, PAContext, PitcherProfile, PlayerProfile, TeamProfile
from .starter_hazard import FittedStarterPolicy, build_decisions

OUT_EVENTS = {
    "strikeout", "field_out", "force_out", "grounded_into_double_play",
    "fielders_choice_out", "double_play", "triple_play", "strikeout_double_play",
}


@dataclass(frozen=True)
class HistoricalGame:
    game_pk: int
    game_date: str
    away_team: str
    home_team: str
    park: str
    away_lineup: tuple[tuple[int, str], ...]  # (MLBAM, bats)
    home_lineup: tuple[tuple[int, str], ...]
    away_starter: tuple[int, str]
    home_starter: tuple[int, str]
    actual_away_runs: int
    actual_home_runs: int
    score_source: str

    @property
    def actual_home_win(self) -> float:
        if self.actual_home_runs > self.actual_away_runs:
            return 1.0
        if self.actual_home_runs < self.actual_away_runs:
            return 0.0
        return 0.5


def add_post_pa_scores(history: pd.DataFrame) -> pd.DataFrame:
    """Add exact score deltas where the next PA exists in the same game."""
    h = history.sort_values(["game_pk", "at_bat_number"], kind="mergesort").copy()
    g = h.groupby("game_pk", sort=False)
    h["next_home_score"] = g["home_score"].shift(-1)
    h["next_away_score"] = g["away_score"].shift(-1)
    h["home_runs_on_pa"] = h["next_home_score"] - h["home_score"]
    h["away_runs_on_pa"] = h["next_away_score"] - h["away_score"]
    return h


def infer_final_score(last: pd.Series) -> tuple[int, int, str] | None:
    """Infer the final score from the final PA.

    The source PA table stores the score before each PA. The next PA supplies
    the post-PA score except for the game's final PA. Most games end on an out,
    and non-HR bottom-half endings are walk-offs where only the winning run is
    counted. Two rare top-half hit endings in the 2026 regular-season data are
    deliberately excluded rather than guessed.
    """
    home = int(last.home_score)
    away = int(last.away_score)
    event = str(last.terminal_event)
    half = str(last.inning_topbot).lower()
    runners = int(last.runner_1b) + int(last.runner_2b) + int(last.runner_3b)

    if event in OUT_EVENTS:
        return away, home, "last_pa_out"
    if event == "home_run":
        runs = 1 + runners
        if half.startswith("bot"):
            home += runs
        else:
            away += runs
        return away, home, "terminal_home_run"
    if half.startswith("bot"):
        # A final bottom-half non-HR reach is a walk-off; scoring stops once the
        # winning run crosses. Sac flies/bunts are included under the same rule.
        if home <= away:
            home = away + 1
            return away, home, "walkoff_non_hr"
        return None
    # Top-half walks/hits can include a run and an out on the bases. Without
    # post-score columns they are ambiguous, so fail closed.
    if event in {"walk", "intent_walk", "hit_by_pitch"} and runners == 3:
        return away + 1, home, "forced_run_top"
    if event == "sac_fly" and int(last.runner_3b):
        return away + 1, home, "sac_fly_top"
    return None


def extract_historical_games(
    history: pd.DataFrame,
    season: int | None = 2026,
    score_overrides: Mapping[int, tuple[int, int] | Mapping[str, Any]] | None = None,
) -> tuple[list[HistoricalGame], list[dict]]:
    if season is None:
        h = history[history["game_type"] == "R"]
    else:
        h = history[(history["season"] == season) & (history["game_type"] == "R")]
    games: list[HistoricalGame] = []
    excluded: list[dict] = []
    for game_pk, frame in h.groupby("game_pk", sort=True):
        frame = frame.sort_values("at_bat_number", kind="mergesort")
        first = frame.iloc[0]
        away_team, home_team = str(first.away_team), str(first.home_team)

        def lineup(top: bool) -> tuple[tuple[int, str], ...]:
            side = frame[frame["inning_topbot"].astype(str).str.lower().str.startswith("top" if top else "bot")]
            seen: set[int] = set(); rows: list[tuple[int, str]] = []
            for r in side.itertuples():
                pid = int(r.batter)
                if pid not in seen:
                    seen.add(pid); rows.append((pid, str(r.stand)))
                if len(rows) == 9:
                    break
            return tuple(rows)

        away_lineup, home_lineup = lineup(True), lineup(False)
        if len(away_lineup) != 9 or len(home_lineup) != 9:
            excluded.append({"game_pk": int(game_pk), "reason": "starting_lineup_not_reconstructable"})
            continue
        top = frame[frame["inning_topbot"].astype(str).str.lower().str.startswith("top")]
        bot = frame[frame["inning_topbot"].astype(str).str.lower().str.startswith("bot")]
        if top.empty or bot.empty:
            excluded.append({"game_pk": int(game_pk), "reason": "missing_half_inning"})
            continue
        home_starter = (int(top.iloc[0].pitcher), str(top.iloc[0].p_throws))
        away_starter = (int(bot.iloc[0].pitcher), str(bot.iloc[0].p_throws))
        score = infer_final_score(frame.iloc[-1])
        if score is None and score_overrides and int(game_pk) in score_overrides:
            override = score_overrides[int(game_pk)]
            if isinstance(override, Mapping):
                away_runs = int(override["away_runs"])
                home_runs = int(override["home_runs"])
            else:
                away_runs, home_runs = map(int, override)
            score = (away_runs, home_runs, "official_score_override")
        if score is None:
            excluded.append({"game_pk": int(game_pk), "reason": "ambiguous_final_score"})
            continue
        away_runs, home_runs, score_source = score
        games.append(HistoricalGame(
            game_pk=int(game_pk), game_date=str(first.date_key), away_team=away_team,
            home_team=home_team, park=str(first.park), away_lineup=away_lineup,
            home_lineup=home_lineup, away_starter=away_starter,
            home_starter=home_starter, actual_away_runs=away_runs,
            actual_home_runs=home_runs, score_source=score_source,
        ))
    return games, excluded


def pitcher_appearances(history: pd.DataFrame) -> pd.DataFrame:
    h = history.copy()
    top = h["inning_topbot"].astype(str).str.lower().str.startswith("top")
    h["fielding_team"] = np.where(top, h["home_team"], h["away_team"])
    first_pitcher = h.groupby(["game_pk", "fielding_team"], sort=False)["pitcher"].transform("first")
    h["is_start"] = h["pitcher"] == first_pitcher
    return (h.groupby(["game_pk", "pitcher"], sort=False)
              .agg(date=("date_key", "first"), team=("fielding_team", "first"),
                   bf=("batter", "size"), entry_inning=("inning", "min"),
                   throws=("p_throws", "first"), start=("is_start", "first"))
              .reset_index())


def starter_expectations(starts: pd.DataFrame, cutoff: str, league_mean: float = 21.94) -> tuple[dict[int, float], dict[str, float]]:
    prior = starts[(starts["date"] < cutoff) & starts["start"]]
    def shrunk(group: pd.Series) -> float:
        return float((group.sum() + 5.0 * league_mean) / (len(group) + 5.0))
    return (prior.groupby("pitcher")["bf"].apply(shrunk).to_dict(),
            prior.groupby("team")["bf"].apply(shrunk).to_dict())


def _rest_score(days: int) -> float:
    if days <= 1: return 0.35
    if days == 2: return 0.70
    return 1.0


def build_matchup(game: HistoricalGame, history: pd.DataFrame, apps: pd.DataFrame,
                  names: Mapping[int, str] | None = None) -> tuple[GameMatchup, dict]:
    names = names or {}
    prior_apps = apps[apps["date"] < game.game_date]
    cutoff = pd.Timestamp(game.game_date)
    window_start = (cutoff - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    fallback_start = (cutoff - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    p_exp, _ = starter_expectations(apps, game.game_date)
    notes: dict = {"bullpen_window": [window_start, game.game_date], "fallback_window": fallback_start}

    def pitcher_name(pid: int) -> str:
        return names.get(pid, f"MLBAM {pid}")

    def starter(team: str, spec: tuple[int, str]) -> PitcherProfile:
        pid, throws = spec
        exp_bf = int(round(p_exp.get(pid, 22.0)))
        return PitcherProfile(str(pid), pitcher_name(pid), throws, role="starter",
                              stamina=0.7, expected_batters=max(12, exp_bf),
                              max_batters=max(18, exp_bf + 6))

    def bullpen(team: str, starter_id: int) -> tuple[PitcherProfile, ...]:
        recent = prior_apps[(prior_apps["team"] == team) & (~prior_apps["start"]) &
                            (prior_apps["date"] >= window_start) & (prior_apps["pitcher"] != starter_id)]
        if recent["pitcher"].nunique() < 5:
            recent = prior_apps[(prior_apps["team"] == team) & (~prior_apps["start"]) &
                                (prior_apps["date"] >= fallback_start) & (prior_apps["pitcher"] != starter_id)]
            notes[f"{team}_bullpen_fallback"] = True
        season_prior = prior_apps[(prior_apps["team"] == team) & (~prior_apps["start"])]
        candidates = (recent.sort_values("date").groupby("pitcher").tail(1)
                      .sort_values("date", ascending=False).head(10)["pitcher"].tolist())
        out: list[PitcherProfile] = []
        for pid in candidates:
            g = season_prior[season_prior["pitcher"] == pid]
            if g.empty: continue
            late = float((g["entry_inning"] >= 8).mean())
            ninth = float((g["entry_inning"] >= 9).mean())
            exp_bf = int(max(3, round(g["bf"].median())))
            role = "closer" if ninth >= 0.55 else "setup" if late >= 0.45 else "long" if exp_bf >= 7 else "reliever"
            last_date = pd.Timestamp(g["date"].max())
            rest = _rest_score(max(1, (cutoff - last_date).days))
            throws = str(g.sort_values("date").iloc[-1].throws)
            out.append(PitcherProfile(str(int(pid)), pitcher_name(int(pid)), throws, role=role,
                                      leverage=min(1.0, 0.25 + late), rest=rest,
                                      expected_batters=exp_bf, max_batters=max(6, exp_bf + 3)))
        # One closer maximum.
        closers = [p for p in out if p.role == "closer"]
        if len(closers) > 1:
            keep = max(closers, key=lambda p: p.leverage)
            out = [p if p.role != "closer" or p.player_id == keep.player_id else PitcherProfile(
                p.player_id, p.name, p.throws, role="setup", leverage=p.leverage,
                rest=p.rest, expected_batters=p.expected_batters, max_batters=p.max_batters)
                for p in out]
        return tuple(out)

    def team_profile(team: str, lineup_spec: tuple[tuple[int, str], ...], starter_spec: tuple[int, str]) -> TeamProfile:
        lineup = tuple(PlayerProfile(str(pid), pitcher_name(pid), bats) for pid, bats in lineup_spec)
        sp = starter(team, starter_spec)
        return TeamProfile(team, team, lineup, sp, bullpen(team, int(sp.player_id)))

    away = team_profile(game.away_team, game.away_lineup, game.away_starter)
    home = team_profile(game.home_team, game.home_lineup, game.home_starter)
    notes["away_bullpen_count"] = len(away.bullpen)
    notes["home_bullpen_count"] = len(home.bullpen)
    return GameMatchup(away=away, home=home, venue=game.park, game_type="R"), notes


def crps_sample(samples: np.ndarray, actual: float) -> float:
    """Unbiased empirical CRPS from Monte Carlo samples.

    The familiar empirical CDF expression with an ``n**2`` pair denominator
    is upward biased at finite path counts.  For model comparisons at 1000
    paths we use the U-statistic denominator ``n * (n - 1)``.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n < 2:
        raise ValueError("CRPS requires at least two simulation paths")
    first = np.mean(np.abs(x - actual))
    # Sum_{i,j}|x_i-x_j| = 2*dot((2i-n-1), x_i).  Exclude the
    # diagonal and divide by n(n-1) for the unbiased U-statistic.
    weights = 2 * np.arange(1, n + 1) - n - 1
    pair = 2.0 * float(np.dot(weights, x)) / (n * (n - 1))
    return float(first - 0.5 * pair)


def poisson_pmf(k: int, lam: float) -> float:
    return exp(k * log(max(lam, 1e-12)) - lam - lgamma(k + 1))


def poisson_win_probability(away_mean: float, home_mean: float, cap: int = 20) -> float:
    away = np.array([poisson_pmf(k, away_mean) for k in range(cap + 1)])
    home = np.array([poisson_pmf(k, home_mean) for k in range(cap + 1)])
    away[-1] += max(0.0, 1.0 - away.sum()); home[-1] += max(0.0, 1.0 - home.sum())
    return float(sum(home[h] * away[:h].sum() for h in range(cap + 1)) +
                 0.5 * np.dot(home, away))


def prior_team_baseline(results: pd.DataFrame, game: HistoricalGame, shrink_games: float = 20.0) -> dict:
    prior = results[results["game_date"] < game.game_date]
    if prior.empty:
        league = 4.4
        return {"away_mean": league, "home_mean": league, "home_win": 0.5}
    team_rows = pd.concat([
        prior[["game_date", "away_team", "away_runs", "home_runs"]].rename(
            columns={"away_team": "team", "away_runs": "runs_for", "home_runs": "runs_against"}),
        prior[["game_date", "home_team", "home_runs", "away_runs"]].rename(
            columns={"home_team": "team", "home_runs": "runs_for", "away_runs": "runs_against"}),
    ], ignore_index=True)
    league = float(team_rows["runs_for"].mean())
    home_factor = float(prior["home_runs"].mean() / max(1e-9, league))
    away_factor = float(prior["away_runs"].mean() / max(1e-9, league))

    def rates(team: str) -> tuple[float, float]:
        g = team_rows[team_rows["team"] == team]
        off = (g["runs_for"].sum() + shrink_games * league) / (len(g) + shrink_games)
        deff = (g["runs_against"].sum() + shrink_games * league) / (len(g) + shrink_games)
        return float(off), float(deff)

    a_off, a_def = rates(game.away_team); h_off, h_def = rates(game.home_team)
    away_mean = league * np.sqrt((a_off / league) * (h_def / league)) * away_factor
    home_mean = league * np.sqrt((h_off / league) * (a_def / league)) * home_factor
    return {"away_mean": float(away_mean), "home_mean": float(home_mean),
            "home_win": poisson_win_probability(float(away_mean), float(home_mean))}


def replay_game(game: HistoricalGame, provider: Any, matchup: GameMatchup,
                manager: FittedStarterPolicy, simulations: int, seed: int) -> dict:
    simulator = GameSimulator(provider, manager_policy=manager)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=simulations)
    away_scores: list[int] = []; home_scores: list[int] = []; home_wins = 0.0
    away_sp_ip: list[float] = []; home_sp_ip: list[float] = []
    caps = 0
    for s in seeds:
        result = simulator.simulate(matchup, int(s), record_events=False)
        away_scores.append(result.away_score); home_scores.append(result.home_score)
        home_wins += 1.0 if result.home_score > result.away_score else 0.5 if result.home_score == result.away_score else 0.0
        away_sp_ip.append(result.pitcher_lines.get(matchup.away.starter.player_id, {}).get("innings_pitched", 0.0))
        home_sp_ip.append(result.pitcher_lines.get(matchup.home.starter.player_id, {}).get("innings_pitched", 0.0))
        caps += int(result.ended_by_plate_appearance_cap)
    a = np.asarray(away_scores); h = np.asarray(home_scores)
    score_counts = Counter(zip(a.tolist(), h.tolist()))
    modal, modal_n = score_counts.most_common(1)[0]
    return {
        "game_pk": game.game_pk, "game_date": game.game_date,
        "away_team": game.away_team, "home_team": game.home_team,
        "actual_away_runs": game.actual_away_runs, "actual_home_runs": game.actual_home_runs,
        "actual_home_win": game.actual_home_win,
        "home_win_probability": float(home_wins / simulations),
        "away_mean_runs": float(a.mean()), "home_mean_runs": float(h.mean()),
        "away_p10": float(np.quantile(a, .10)), "away_p90": float(np.quantile(a, .90)),
        "home_p10": float(np.quantile(h, .10)), "home_p90": float(np.quantile(h, .90)),
        "away_crps": crps_sample(a, game.actual_away_runs),
        "home_crps": crps_sample(h, game.actual_home_runs),
        "modal_away_runs": int(modal[0]), "modal_home_runs": int(modal[1]),
        "modal_probability": float(modal_n / simulations),
        "away_starter_ip_mean": float(np.mean(away_sp_ip)),
        "home_starter_ip_mean": float(np.mean(home_sp_ip)),
        "pa_cap_rate": float(caps / simulations),
        "provider_calls": int(getattr(provider, "calls", 0)),
        "provider_cache_hits": int(getattr(provider, "cache_hits", 0)),
        "home_win_mc_se": float(np.sqrt(max(0.0, (home_wins / simulations) * (1.0 - home_wins / simulations) / simulations))),
    }



def bootstrap_metric_difference(a: np.ndarray, b: np.ndarray, reps: int = 2000, seed: int = 36) -> dict:
    """Paired game bootstrap for candidate-minus-baseline metric rows."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("paired metric arrays must be non-empty and equal length")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    boot = d[idx].mean(axis=1)
    return {
        "difference": float(d.mean()),
        "ci_95_low": float(np.quantile(boot, 0.025)),
        "ci_95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_replicates": int(reps),
        "probability_candidate_better": float(np.mean(boot < 0)),
    }

def score_rows(rows: pd.DataFrame, simulations_per_game: int, baseline_cols: Iterable[str] = ("league", "team")) -> dict:
    y = rows["actual_home_win"].to_numpy(float)
    out: dict = {}
    for name, pcol, acol, hcol in [
        ("model", "home_win_probability", "away_mean_runs", "home_mean_runs"),
        ("league", "league_home_win", "league_away_mean", "league_home_mean"),
        ("team", "team_home_win", "team_away_mean", "team_home_mean"),
    ]:
        p = np.clip(rows[pcol].to_numpy(float), 1e-6, 1 - 1e-6)
        ae = rows[acol].to_numpy(float) - rows["actual_away_runs"].to_numpy(float)
        he = rows[hcol].to_numpy(float) - rows["actual_home_runs"].to_numpy(float)
        out[name] = {
            "winner_brier": float(np.mean((p - y) ** 2)),
            "winner_log_loss": float(np.mean(-(y * np.log(p) + (1-y) * np.log(1-p)))),
            "winner_accuracy": float(np.mean((p >= .5) == (y >= .5))),
            "team_run_mae": float(np.mean(np.abs(np.concatenate([ae, he])))),
            "team_run_rmse": float(np.sqrt(np.mean(np.concatenate([ae, he]) ** 2))),
        }
    out["model"]["team_run_crps"] = float(np.mean(
        np.concatenate([rows["away_crps"].to_numpy(), rows["home_crps"].to_numpy()])))
    out["model"]["middle_80_coverage"] = float(np.mean(np.concatenate([
        ((rows.actual_away_runs >= rows.away_p10) & (rows.actual_away_runs <= rows.away_p90)).to_numpy(),
        ((rows.actual_home_runs >= rows.home_p10) & (rows.actual_home_runs <= rows.home_p90)).to_numpy(),
    ])))
    out["model"]["exact_modal_score_rate"] = float(np.mean(
        (rows.modal_away_runs == rows.actual_away_runs) & (rows.modal_home_runs == rows.actual_home_runs)))
    # A finite Monte Carlo estimate adds p(1-p)/N variance to Brier score.
    # Report the raw operational score and a diagnostic noise-corrected estimate;
    # promotion decisions still require a sufficiently large simulation count.
    n_sims = int(simulations_per_game)
    mc_variance = rows["home_win_probability"].to_numpy(float) * (1.0 - rows["home_win_probability"].to_numpy(float)) / max(1, n_sims - 1)
    out["model"]["winner_brier_mc_noise_estimate"] = float(np.mean(mc_variance))
    out["model"]["winner_brier_mc_corrected_diagnostic"] = float(
        out["model"]["winner_brier"] - np.mean(mc_variance)
    )
    out["model"]["simulation_precision_status"] = (
        "promotion_eligible" if n_sims >= 500 else "development_only_below_500_paths"
    )

    model_p = np.clip(rows["home_win_probability"].to_numpy(float), 1e-6, 1 - 1e-6)
    model_brier = (model_p - y) ** 2
    model_ll = -(y * np.log(model_p) + (1-y) * np.log(1-model_p))
    model_mae = (np.abs(rows["away_mean_runs"] - rows["actual_away_runs"]) +
                 np.abs(rows["home_mean_runs"] - rows["actual_home_runs"])) / 2
    out["paired_bootstrap"] = {}
    for i, (name, pcol, acol, hcol) in enumerate([
        ("league", "league_home_win", "league_away_mean", "league_home_mean"),
        ("team", "team_home_win", "team_away_mean", "team_home_mean"),
    ]):
        bp = np.clip(rows[pcol].to_numpy(float), 1e-6, 1 - 1e-6)
        bb = (bp - y) ** 2
        bll = -(y * np.log(bp) + (1-y) * np.log(1-bp))
        bmae = (np.abs(rows[acol] - rows["actual_away_runs"]) +
                 np.abs(rows[hcol] - rows["actual_home_runs"])) / 2
        out["paired_bootstrap"][f"model_minus_{name}"] = {
            "winner_brier": bootstrap_metric_difference(model_brier, bb, seed=36+i),
            "winner_log_loss": bootstrap_metric_difference(model_ll, bll, seed=46+i),
            "team_run_mae": bootstrap_metric_difference(model_mae.to_numpy(float), bmae.to_numpy(float), seed=56+i),
        }
    out["diagnostics"] = {
        "actual_runs_per_team": float(rows[["actual_away_runs", "actual_home_runs"]].to_numpy().mean()),
        "model_runs_per_team": float(rows[["away_mean_runs", "home_mean_runs"]].to_numpy().mean()),
        "away_run_bias": float((rows["away_mean_runs"] - rows["actual_away_runs"]).mean()),
        "home_run_bias": float((rows["home_mean_runs"] - rows["actual_home_runs"]).mean()),
        "pa_cap_rate": float(rows["pa_cap_rate"].mean()),
    }
    return out


class FlatLeagueAverageProvider:
    """Time-valid flat PA probabilities for signal-survival ablation.

    The provider shares the locked provider's chronological history state, but
    discards batter, pitcher, platoon, park, recent-form, and game-context
    differences. Every PA on a date receives the same prior-date league
    outcome distribution. This isolates whether player-level PA information
    survives the downstream game engine.
    """

    name = "prior-date-flat-league-pa-v1"
    validation_status = "development ablation comparator; prior-date league rates only"

    def __init__(self, locked_provider: LockedPAModelProvider):
        self.locked_provider = locked_provider
        self.calls = 0
        self.cache_hits = 0
        self._date_cache: tuple[str, dict[str, float]] | None = None

    def probabilities(self, context: PAContext) -> Mapping[str, float]:
        self.calls += 1
        state = self.locked_provider.state
        cache_key = str(self.locked_provider.cutoff_date)
        if self._date_cache is not None and self._date_cache[0] == cache_key:
            self.cache_hits += 1
            return self._date_cache[1]
        eps = float(self.locked_provider._config.get("min_probability", 1e-7))
        league = _posterior(
            state.league_counts,
            float(state.league_counts.sum()),
            FIXED_LEAGUE_PRIOR,
            500.0,
            eps,
        )
        model_index = {label: i for i, label in enumerate(MODEL_LABELS)}
        result = {
            sim_label: float(league[model_index[model_label]])
            for sim_label, model_label in SIM_FROM_MODEL.items()
        }
        self._date_cache = (cache_key, result)
        return result


def poisson_crps(mean: float, actual: int) -> float:
    """Discrete CRPS for a Poisson forecast, evaluated by its CDF."""
    mean = float(max(mean, 1e-9))
    actual = int(actual)
    cap = int(max(40, actual + 25, np.ceil(mean + 12 * np.sqrt(mean + 1))))
    pmf = np.empty(cap + 1, dtype=float)
    pmf[0] = np.exp(-mean)
    for k in range(1, cap + 1):
        pmf[k] = pmf[k - 1] * mean / k
    cdf = np.cumsum(pmf)
    # Capture any tiny numerical tail at the final support point.
    cdf[-1] = 1.0
    observation_cdf = (np.arange(cap + 1) >= actual).astype(float)
    return float(np.sum((cdf - observation_cdf) ** 2))


def calibration_parameters(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | bool | str | None]:
    """Fit logit(P(Y=1)) = intercept + slope * logit(forecast).

    Uses a two-parameter Newton/IRLS solve with backtracking.  This avoids the
    precision-loss warnings that generic finite-difference BFGS can emit when a
    baseline's probabilities occupy a narrow range.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1.0 - p))
    if len(y) == 0:
        return {"slope": None, "intercept": None, "success": False, "message": "no rows"}
    if np.std(x) < 1e-12:
        observed = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return {
            "slope": 0.0,
            "intercept": float(np.log(observed / (1.0 - observed))),
            "success": False,
            "message": "forecast logits are constant; slope not identifiable",
        }

    design = np.column_stack([np.ones(len(x), dtype=float), x])
    observed = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    theta = np.array([np.log(observed / (1.0 - observed)), 1.0], dtype=float)

    def nll(value: np.ndarray) -> float:
        z = np.clip(design @ value, -50.0, 50.0)
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    success = False
    message = "maximum iterations reached"
    iterations = 0
    for iterations in range(1, 101):
        z = np.clip(design @ theta, -35.0, 35.0)
        q = 1.0 / (1.0 + np.exp(-z))
        gradient = design.T @ (q - y)
        weights = np.maximum(q * (1.0 - q), 1e-10)
        hessian = design.T @ (design * weights[:, None])
        # Tiny numerical ridge only; it is not a modeling penalty.
        hessian += np.eye(2) * 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            message = "singular calibration Hessian"
            break
        if np.max(np.abs(step)) < 1e-10:
            success = True
            message = "converged"
            break
        current = nll(theta)
        scale = 1.0
        accepted = False
        while scale >= 1e-8:
            candidate = theta - scale * step
            if np.isfinite(candidate).all() and nll(candidate) <= current + 1e-10:
                theta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            if np.linalg.norm(gradient, ord=np.inf) < 1e-7:
                success = True
                message = "converged (small gradient)"
            else:
                message = "line search failed"
            break
        if np.linalg.norm(gradient, ord=np.inf) < 1e-8:
            success = True
            message = "converged"
            break
    return {
        "slope": float(theta[1]),
        "intercept": float(theta[0]),
        "success": bool(success),
        "message": message,
        "iterations": int(iterations),
    }

def _variant_metrics(
    rows: pd.DataFrame,
    *,
    prefix: str,
    simulations_per_game: int | None,
    monte_carlo_probability: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    y = rows["actual_home_win"].to_numpy(float)
    p = np.clip(rows[f"{prefix}home_win_probability"].to_numpy(float), 1e-6, 1 - 1e-6)
    raw_brier_rows = (p - y) ** 2
    if monte_carlo_probability:
        if simulations_per_game is None or simulations_per_game <= 1:
            raise ValueError("simulations_per_game must exceed one for unbiased Brier correction")
        mc_correction_rows = p * (1.0 - p) / (int(simulations_per_game) - 1)
    else:
        mc_correction_rows = np.zeros_like(p)
    unbiased_brier_rows = raw_brier_rows - mc_correction_rows
    log_loss_rows = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    away_error = rows[f"{prefix}away_mean_runs"].to_numpy(float) - rows["actual_away_runs"].to_numpy(float)
    home_error = rows[f"{prefix}home_mean_runs"].to_numpy(float) - rows["actual_home_runs"].to_numpy(float)
    run_errors = np.concatenate([away_error, home_error])
    away_crps = rows[f"{prefix}away_crps"].to_numpy(float)
    home_crps = rows[f"{prefix}home_crps"].to_numpy(float)
    game_crps_rows = (away_crps + home_crps) / 2.0

    metrics = {
        "winner_brier_raw": float(raw_brier_rows.mean()),
        "winner_brier_mc_correction": float(mc_correction_rows.mean()),
        "winner_brier_unbiased": float(unbiased_brier_rows.mean()),
        "winner_log_loss": float(log_loss_rows.mean()),
        "winner_accuracy": float(np.mean((p >= 0.5) == (y >= 0.5))),
        "win_probability_sd": float(np.std(p, ddof=1)),
        "win_probability_mean": float(np.mean(p)),
        "mean_per_game_mc_se": float(np.mean(np.sqrt(p * (1.0 - p) / max(1, int(simulations_per_game or 1))))) if monte_carlo_probability else 0.0,
        "calibration": calibration_parameters(y, p),
        "team_run_mae": float(np.mean(np.abs(run_errors))),
        "team_run_rmse": float(np.sqrt(np.mean(run_errors ** 2))),
        "team_run_crps": float(game_crps_rows.mean()),
    }
    rows_out = {
        "winner_brier_unbiased": unbiased_brier_rows,
        "winner_brier_raw": raw_brier_rows,
        "winner_log_loss": log_loss_rows,
        "team_run_crps": game_crps_rows,
        "team_run_mae": (np.abs(away_error) + np.abs(home_error)) / 2.0,
    }
    return metrics, rows_out


def score_pa_ablation(rows: pd.DataFrame, simulations_per_game: int, bootstrap_replicates: int = 2000) -> dict:
    """Score locked-PA vs flat-league PA ablation and deterministic baselines."""
    variants: dict[str, dict] = {}
    per_game: dict[str, dict[str, np.ndarray]] = {}
    specs = {
        "candidate_locked_pa": ("candidate_", True),
        "ablated_flat_league_pa": ("flat_", True),
        "league_baseline": ("league_", False),
        "team_strength_baseline": ("team_", False),
    }
    for name, (prefix, mc) in specs.items():
        variants[name], per_game[name] = _variant_metrics(
            rows,
            prefix=prefix,
            simulations_per_game=simulations_per_game if mc else None,
            monte_carlo_probability=mc,
        )

    comparisons: dict[str, dict] = {}
    for i, other in enumerate(("ablated_flat_league_pa", "league_baseline", "team_strength_baseline")):
        label = f"candidate_minus_{other}"
        comparisons[label] = {}
        for j, metric in enumerate(("winner_brier_unbiased", "winner_log_loss", "team_run_crps", "team_run_mae")):
            comparisons[label][metric] = bootstrap_metric_difference(
                per_game["candidate_locked_pa"][metric],
                per_game[other][metric],
                reps=bootstrap_replicates,
                seed=360 + 10 * i + j,
            )
    return {
        "variants": variants,
        "paired_game_bootstrap": comparisons,
        "diagnostics": {
            "games": int(len(rows)),
            "actual_home_win_rate": float(rows["actual_home_win"].mean()),
            "actual_runs_per_team": float(rows[["actual_away_runs", "actual_home_runs"]].to_numpy().mean()),
            "candidate_pa_cap_rate": float(rows["candidate_pa_cap_rate"].mean()),
            "flat_pa_cap_rate": float(rows["flat_pa_cap_rate"].mean()),
        },
    }
