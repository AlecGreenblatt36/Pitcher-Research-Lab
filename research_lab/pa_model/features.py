from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import PAConfig


@dataclass
class _CounterState:
    counts: np.ndarray
    total: int = 0


class _OnlineCounter:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self._states: dict[Hashable, _CounterState] = {}

    def get(self, key: Hashable) -> _CounterState:
        return self._states.get(key, _CounterState(np.zeros(self.n_classes, dtype=float), 0))

    def update(self, key: Hashable, class_index: int) -> None:
        state = self._states.get(key)
        if state is None:
            state = _CounterState(np.zeros(self.n_classes, dtype=float), 0)
            self._states[key] = state
        state.counts[class_index] += 1.0
        state.total += 1


class _RollingCounter:
    def __init__(self, n_classes: int, window: int):
        self.n_classes = n_classes
        self.window = window
        self._queues: dict[Hashable, deque[int]] = defaultdict(deque)
        self._counts: dict[Hashable, np.ndarray] = {}

    def get(self, key: Hashable) -> _CounterState:
        queue = self._queues.get(key)
        if not queue:
            return _CounterState(np.zeros(self.n_classes, dtype=float), 0)
        return _CounterState(self._counts[key].copy(), len(queue))

    def update(self, key: Hashable, class_index: int) -> None:
        queue = self._queues[key]
        counts = self._counts.setdefault(key, np.zeros(self.n_classes, dtype=float))
        queue.append(class_index)
        counts[class_index] += 1.0
        if len(queue) > self.window:
            counts[queue.popleft()] -= 1.0


def _posterior(state: _CounterState, prior: np.ndarray, strength: float, eps: float) -> np.ndarray:
    result = (state.counts + strength * prior) / max(state.total + strength, eps)
    result = np.clip(result, eps, 1.0)
    return result / result.sum()


def _log_ratio(prob: np.ndarray, league: np.ndarray, eps: float) -> np.ndarray:
    return np.log(np.clip(prob, eps, 1.0)) - np.log(np.clip(league, eps, 1.0))


def build_time_valid_features(pa: pd.DataFrame, config: PAConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    config = config or PAConfig()
    required = {"date_key", "game_pk", "at_bat_number", "batter", "pitcher", "outcome", "stand", "p_throws", "park"}
    missing = required - set(pa.columns)
    if missing:
        raise ValueError(f"PA data missing feature columns: {sorted(missing)}")
    labels = list(config.outcome_labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    unknown = sorted(set(pa["outcome"]) - set(labels))
    if unknown:
        raise ValueError(f"unknown outcomes: {unknown}")
    ordered = pa.sort_values(["date_key", "game_pk", "at_bat_number"], kind="mergesort").reset_index(drop=True)
    n_classes = len(labels)
    eps = config.min_probability
    league = _OnlineCounter(n_classes)
    batter = _OnlineCounter(n_classes)
    pitcher = _OnlineCounter(n_classes)
    batter_split = _OnlineCounter(n_classes)
    pitcher_split = _OnlineCounter(n_classes)
    park = _OnlineCounter(n_classes)
    batter_recent = _RollingCounter(n_classes, config.batter_recent_window)
    pitcher_recent = _RollingCounter(n_classes, config.pitcher_recent_window)
    fixed_prior = np.array([0.49, 0.22, 0.09, 0.10, 0.035, 0.03, 0.035])
    fixed_prior /= fixed_prior.sum()
    records = []
    feature_columns = []
    for current_date, date_rows in ordered.groupby("date_key", sort=True):
        league_prob = _posterior(league.get("league"), fixed_prior, 500.0, eps)
        for row in date_rows.itertuples(index=False):
            stand, throws = str(getattr(row, "stand", "U")), str(getattr(row, "p_throws", "U"))
            batter_id, pitcher_id = int(row.batter), int(row.pitcher)
            b_split_key, p_split_key = (batter_id, throws), (pitcher_id, stand)
            b_prob = _posterior(batter.get(batter_id), league_prob, config.player_prior_pa, eps)
            p_prob = _posterior(pitcher.get(pitcher_id), league_prob, config.player_prior_pa, eps)
            groups = {
                "league": league_prob,
                "batter": b_prob,
                "pitcher": p_prob,
                "batter_split": _posterior(batter_split.get(b_split_key), b_prob, config.split_prior_pa, eps),
                "pitcher_split": _posterior(pitcher_split.get(p_split_key), p_prob, config.split_prior_pa, eps),
                "park": _posterior(park.get(str(getattr(row, "park", "UNK"))), league_prob, config.park_prior_pa, eps),
                "batter_recent": _posterior(batter_recent.get(batter_id), b_prob, config.recent_prior_pa, eps),
                "pitcher_recent": _posterior(pitcher_recent.get(pitcher_id), p_prob, config.recent_prior_pa, eps),
            }
            record = {"date_key": current_date, "season": int(row.season), "game_pk": int(row.game_pk), "at_bat_number": int(row.at_bat_number), "batter": batter_id, "pitcher": pitcher_id, "outcome": str(row.outcome)}
            for prefix, probs in groups.items():
                for idx, label in enumerate(labels):
                    col = f"p_{prefix}_{label}"
                    record[col] = float(probs[idx])
                    if col not in feature_columns:
                        feature_columns.append(col)
            for prefix, probs in groups.items():
                if prefix == "league":
                    continue
                for idx, label in enumerate(labels):
                    col = f"lr_{prefix}_{label}"
                    record[col] = float(_log_ratio(probs, league_prob, eps)[idx])
                    if col not in feature_columns:
                        feature_columns.append(col)
            context = {
                "platoon": float(getattr(row, "platoon", 0)), "is_home_batter": float(getattr(row, "is_home_batter", 0)),
                "inning": float(getattr(row, "inning", 1)), "outs_when_up": float(getattr(row, "outs_when_up", 0)),
                "runner_1b": float(getattr(row, "runner_1b", 0)), "runner_2b": float(getattr(row, "runner_2b", 0)),
                "runner_3b": float(getattr(row, "runner_3b", 0)), "bat_score_diff": float(getattr(row, "bat_score_diff", 0)),
                "n_thruorder_pitcher": float(getattr(row, "n_thruorder_pitcher", 1)),
                "batter_days_since_prev_game": float(getattr(row, "batter_days_since_prev_game", np.nan)),
                "pitcher_days_since_prev_game": float(getattr(row, "pitcher_days_since_prev_game", np.nan)),
                "age_bat": float(getattr(row, "age_bat", np.nan)), "age_pit": float(getattr(row, "age_pit", np.nan)),
                "batter_history_pa": float(batter.get(batter_id).total), "pitcher_history_pa": float(pitcher.get(pitcher_id).total),
                "batter_split_history_pa": float(batter_split.get(b_split_key).total), "pitcher_split_history_pa": float(pitcher_split.get(p_split_key).total),
            }
            for col, value in context.items():
                record[col] = value
                if col not in feature_columns:
                    feature_columns.append(col)
            records.append(record)
        for row in date_rows.itertuples(index=False):
            idx, batter_id, pitcher_id = label_to_index[str(row.outcome)], int(row.batter), int(row.pitcher)
            stand, throws = str(getattr(row, "stand", "U")), str(getattr(row, "p_throws", "U"))
            league.update("league", idx); batter.update(batter_id, idx); pitcher.update(pitcher_id, idx)
            batter_split.update((batter_id, throws), idx); pitcher_split.update((pitcher_id, stand), idx)
            park.update(str(getattr(row, "park", "UNK")), idx); batter_recent.update(batter_id, idx); pitcher_recent.update(pitcher_id, idx)
    return pd.DataFrame.from_records(records), feature_columns


def empirical_bayes_matchup_probabilities(features: pd.DataFrame, labels: Iterable[str], eps: float = 1e-7, parameters: dict[str, float] | None = None) -> np.ndarray:
    labels = list(labels)
    parameters = parameters or {"batter_weight": 0.5, "pitcher_weight": 0.5, "park_weight": 0.1, "batter_recent_weight": 0.1, "pitcher_recent_weight": 0.1, "temperature": 1.0}
    grab = lambda prefix: features[[f"p_{prefix}_{label}" for label in labels]].to_numpy(float)
    league, batter, pitcher, park, recent_b, recent_p = grab("league"), grab("batter_split"), grab("pitcher_split"), grab("park"), grab("batter_recent"), grab("pitcher_recent")
    lr = lambda values, reference: np.log(np.clip(values, eps, 1)) - np.log(np.clip(reference, eps, 1))
    log_score = np.log(np.clip(league, eps, 1))
    log_score += parameters["batter_weight"] * lr(batter, league)
    log_score += parameters["pitcher_weight"] * lr(pitcher, league)
    log_score += parameters["park_weight"] * lr(park, league)
    log_score += parameters["batter_recent_weight"] * lr(recent_b, batter)
    log_score += parameters["pitcher_recent_weight"] * lr(recent_p, pitcher)
    log_score /= max(float(parameters["temperature"]), eps)
    log_score -= log_score.max(axis=1, keepdims=True)
    score = np.exp(log_score)
    return score / score.sum(axis=1, keepdims=True)


def fit_empirical_bayes_baseline(validation: pd.DataFrame, y_index: np.ndarray, labels: Iterable[str], eps: float = 1e-7) -> tuple[dict[str, float], dict]:
    labels = list(labels)
    names = ("batter_weight", "pitcher_weight", "park_weight", "batter_recent_weight", "pitcher_recent_weight")
    def unpack(theta):
        values = {name: float(theta[i]) for i, name in enumerate(names)}
        values["temperature"] = float(np.exp(theta[-1]))
        return values
    row = np.arange(len(y_index))
    def objective(theta):
        probabilities = empirical_bayes_matchup_probabilities(validation, labels, eps, unpack(theta))
        return float(-np.log(np.clip(probabilities[row, y_index], eps, 1)).mean())
    initial = np.array([0.5, 0.5, 0.1, 0.1, 0.1, 0.0], dtype=float)
    bounds = [(0, 2), (0, 2), (0, 1), (0, 1), (0, 1), (float(np.log(0.5)), float(np.log(3.0)))]
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100, "ftol": 1e-11})
    selected = result.x if result.success else initial
    parameters = unpack(selected)
    return parameters, {"success": bool(result.success), "message": str(result.message), "iterations": int(getattr(result, "nit", 0)), "validation_log_loss": float(objective(selected)), "parameters": parameters}
