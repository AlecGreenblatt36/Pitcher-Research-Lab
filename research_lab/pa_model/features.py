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
    """Construct pre-PA features with strict date blocking.

    The implementation preallocates a float32 matrix rather than retaining one
    Python dictionary per PA. A full three-season MLB build is therefore
    practical on a standard CI runner while preserving the exact same temporal
    information boundary.
    """

    config = config or PAConfig()
    required = {
        "date_key", "game_pk", "at_bat_number", "batter", "pitcher",
        "outcome", "stand", "p_throws", "park",
    }
    missing = required - set(pa.columns)
    if missing:
        raise ValueError(f"PA data missing feature columns: {sorted(missing)}")

    labels = list(config.outcome_labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    unknown = sorted(set(pa["outcome"]) - set(labels))
    if unknown:
        raise ValueError(f"unknown outcomes: {unknown}")

    ordered = pa.sort_values(
        ["date_key", "game_pk", "at_bat_number"], kind="mergesort"
    ).reset_index(drop=True)
    n_classes = len(labels)
    eps = config.min_probability

    group_prefixes = (
        "league", "batter", "pitcher", "batter_split", "pitcher_split",
        "park", "batter_recent", "pitcher_recent",
    )
    ratio_prefixes = tuple(prefix for prefix in group_prefixes if prefix != "league")
    context_columns = (
        "platoon", "is_home_batter", "inning", "outs_when_up",
        "runner_1b", "runner_2b", "runner_3b", "bat_score_diff",
        "n_thruorder_pitcher", "batter_days_since_prev_game",
        "pitcher_days_since_prev_game", "age_bat", "age_pit",
        "batter_history_pa", "pitcher_history_pa",
        "batter_split_history_pa", "pitcher_split_history_pa",
    )

    feature_columns: list[str] = []
    probability_slices: dict[str, slice] = {}
    ratio_slices: dict[str, slice] = {}
    cursor = 0
    for prefix in group_prefixes:
        probability_slices[prefix] = slice(cursor, cursor + n_classes)
        feature_columns.extend(f"p_{prefix}_{label}" for label in labels)
        cursor += n_classes
    for prefix in ratio_prefixes:
        ratio_slices[prefix] = slice(cursor, cursor + n_classes)
        feature_columns.extend(f"lr_{prefix}_{label}" for label in labels)
        cursor += n_classes
    context_slice = slice(cursor, cursor + len(context_columns))
    feature_columns.extend(context_columns)
    matrix = np.empty((len(ordered), len(feature_columns)), dtype=np.float32)

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

    for _, date_rows in ordered.groupby("date_key", sort=True):
        # Freeze the league prior once for the whole date. No result from date D
        # can influence another PA on date D, including doubleheaders.
        league_prob = _posterior(league.get("league"), fixed_prior, 500.0, eps)

        for row in date_rows.itertuples(index=True):
            position = int(row.Index)
            stand = str(getattr(row, "stand", "U"))
            throws = str(getattr(row, "p_throws", "U"))
            batter_id = int(row.batter)
            pitcher_id = int(row.pitcher)
            b_split_key = (batter_id, throws)
            p_split_key = (pitcher_id, stand)

            b_state = batter.get(batter_id)
            p_state = pitcher.get(pitcher_id)
            bs_state = batter_split.get(b_split_key)
            ps_state = pitcher_split.get(p_split_key)
            park_key = str(getattr(row, "park", "UNK"))

            b_prob = _posterior(b_state, league_prob, config.player_prior_pa, eps)
            p_prob = _posterior(p_state, league_prob, config.player_prior_pa, eps)
            groups = {
                "league": league_prob,
                "batter": b_prob,
                "pitcher": p_prob,
                "batter_split": _posterior(bs_state, b_prob, config.split_prior_pa, eps),
                "pitcher_split": _posterior(ps_state, p_prob, config.split_prior_pa, eps),
                "park": _posterior(
                    park.get(park_key), league_prob, config.park_prior_pa, eps
                ),
                "batter_recent": _posterior(
                    batter_recent.get(batter_id), b_prob, config.recent_prior_pa, eps
                ),
                "pitcher_recent": _posterior(
                    pitcher_recent.get(pitcher_id), p_prob, config.recent_prior_pa, eps
                ),
            }

            for prefix, probabilities in groups.items():
                matrix[position, probability_slices[prefix]] = probabilities
            for prefix in ratio_prefixes:
                matrix[position, ratio_slices[prefix]] = _log_ratio(
                    groups[prefix], league_prob, eps
                )

            matrix[position, context_slice] = np.asarray(
                [
                    float(getattr(row, "platoon", 0)),
                    float(getattr(row, "is_home_batter", 0)),
                    float(getattr(row, "inning", 1)),
                    float(getattr(row, "outs_when_up", 0)),
                    float(getattr(row, "runner_1b", 0)),
                    float(getattr(row, "runner_2b", 0)),
                    float(getattr(row, "runner_3b", 0)),
                    float(getattr(row, "bat_score_diff", 0)),
                    float(getattr(row, "n_thruorder_pitcher", 1)),
                    float(getattr(row, "batter_days_since_prev_game", np.nan)),
                    float(getattr(row, "pitcher_days_since_prev_game", np.nan)),
                    float(getattr(row, "age_bat", np.nan)),
                    float(getattr(row, "age_pit", np.nan)),
                    float(b_state.total),
                    float(p_state.total),
                    float(bs_state.total),
                    float(ps_state.total),
                ],
                dtype=np.float32,
            )

        # Only after the full date is scored do its outcomes become historical
        # information for subsequent dates.
        for row in date_rows.itertuples(index=False):
            class_index = label_to_index[str(row.outcome)]
            batter_id = int(row.batter)
            pitcher_id = int(row.pitcher)
            stand = str(getattr(row, "stand", "U"))
            throws = str(getattr(row, "p_throws", "U"))
            park_key = str(getattr(row, "park", "UNK"))
            league.update("league", class_index)
            batter.update(batter_id, class_index)
            pitcher.update(pitcher_id, class_index)
            batter_split.update((batter_id, throws), class_index)
            pitcher_split.update((pitcher_id, stand), class_index)
            park.update(park_key, class_index)
            batter_recent.update(batter_id, class_index)
            pitcher_recent.update(pitcher_id, class_index)

    metadata_columns = [
        "date_key", "season", "game_pk", "at_bat_number",
        "batter", "pitcher", "outcome",
    ]
    metadata = ordered.loc[:, metadata_columns].reset_index(drop=True)
    numeric = pd.DataFrame(matrix, columns=feature_columns)
    return pd.concat([metadata, numeric], axis=1, copy=False), feature_columns


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
