"""BOOST-01: a gradient-boosted matchup model stacked into the at-bat model.

The at-bat model is a multinomial logistic regression: every feature moves the seven log-odds by a fixed amount
whatever the other features say. The pitch tracking, the chain and the history counters are different views of the
same two players, and how much one view should count can depend on the others (a chain strikeout rate means more
behind a pitcher whose whiff physics agree with it; a small history means the physics should carry more). Trees find
those interactions without being told where to look.

The boosted model (sklearn's histogram gradient boosting, seven classes) reads the matchup-level features only: the
talent counters, the handedness, ages, rest and history sizes, the physics and the chain. It never reads the in-game
state (inning, outs, runners, score, times through the order), so one evaluation per matchup per game serves every
plate appearance of that matchup, and the simulator's cost is unchanged. Its seven log probabilities are appended to
the logistic model's features (``gb_<label>``), the same way the chain's are, so the logistic model and its
calibration decide how much of the trees to believe.

Leakage control: the boosted model is fit on the training seasons only. Its predictions for the training rows come
out of fold (games split into K groups; each group is predicted by a model fit on the other groups), so the logistic
model never sees a tree's memory of a row it is trained on. The validation and test seasons are predicted by the model
fit on every training row, the one that goes in the bundle. The number of boosting rounds is picked on the tuning
block of the validation season (the block the logistic model's regularization is picked on) from the staged
predictions of one long fit.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .config import PAConfig
from .evaluation import probability_metrics

# In-game state the boosted model never sees (the benchmark's ablation set, pipeline._feature_families).
GAME_CONTEXT = ("is_home_batter", "inning", "outs_when_up", "runner_1b", "runner_2b", "runner_3b",
                "bat_score_diff", "n_thruorder_pitcher")
DEFAULT_PARAMS = {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 300,
                  "l2_regularization": 1.0, "folds": 4, "round_grid": (100, 200, 300, 400), "random_seed": 36}


def boost_feature_names(labels) -> list[str]:
    return [f"gb_{label}" for label in labels]


def boost_input_columns(all_columns) -> list[str]:
    """The matchup-level inputs: every feature column that is not in-game state (and not a boosted feature)."""
    return [str(c) for c in all_columns if str(c) not in GAME_CONTEXT and not str(c).startswith("gb_")]


def _estimator(params: dict, max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=float(params["learning_rate"]), max_iter=int(max_iter),
        max_leaf_nodes=int(params["max_leaf_nodes"]), min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]), early_stopping=False,
        random_state=int(params["random_seed"]),
    )


def _log_probs(model, x: np.ndarray, n_classes: int, eps: float) -> np.ndarray:
    """Log probabilities in the configured label order (the estimator may not have seen every class)."""
    raw = model.predict_proba(x)
    out = np.full((len(x), n_classes), eps, dtype=float)
    out[:, np.asarray(model.classes_, int)] = raw
    out = np.clip(out, eps, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return np.log(out)


def _staged_log_loss(model, x: np.ndarray, y: np.ndarray, n_classes: int, eps: float, rounds) -> dict[int, float]:
    """Log loss on (x, y) after each number of rounds in ``rounds`` from one fit's staged predictions."""
    wanted = sorted(set(int(r) for r in rounds if int(r) <= model.n_iter_))
    out: dict[int, float] = {}
    for i, probs in enumerate(model.staged_predict_proba(x), start=1):
        if i in wanted:
            full = np.full((len(x), n_classes), eps, dtype=float)
            full[:, np.asarray(model.classes_, int)] = probs
            out[i] = float(probability_metrics(y, full / full.sum(axis=1, keepdims=True)).log_loss)
        if i >= wanted[-1]:
            break
    return out


def fit_boost(features: pd.DataFrame, columns: list[str], tune_mask: np.ndarray, config: PAConfig,
              params: dict | None = None) -> tuple[HistGradientBoostingClassifier, np.ndarray, dict]:
    """Fit the boosted matchup model on the training seasons and return (the model fit on every training row, the
    stacked log probabilities for every row of ``features`` (out of fold on the training rows), an audit)."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    labels = list(config.outcome_labels)
    n_classes = len(labels)
    eps = float(config.min_probability)
    t0 = time.time()
    y = features["outcome"].map({label: i for i, label in enumerate(labels)}).to_numpy(int)
    x = features[columns].to_numpy(dtype=np.float32)
    train = features["season"].isin(config.train_years).to_numpy()
    if not train.any():
        raise ValueError("boost: training period contains no rows")
    groups = features["game_pk"].to_numpy()

    # rounds: one long fit on every training row, scored on the tuning block after each candidate number of rounds
    longest = _estimator(p, int(p["max_iter"]))
    longest.fit(x[train], y[train])
    by_rounds = _staged_log_loss(longest, x[tune_mask], y[tune_mask], n_classes, eps, p["round_grid"]) if tune_mask.any() else {}
    rounds = min(by_rounds, key=by_rounds.get) if by_rounds else int(p["max_iter"])

    # out-of-fold predictions for the training rows at the chosen number of rounds
    stacked = np.full((len(features), n_classes), np.log(1.0 / n_classes), dtype=float)
    folds = int(p["folds"])
    train_idx = np.where(train)[0]
    for k, (fit_part, hold_part) in enumerate(GroupKFold(n_splits=folds).split(train_idx, groups=groups[train_idx])):
        model = _estimator(p, rounds)
        model.fit(x[train_idx[fit_part]], y[train_idx[fit_part]])
        stacked[train_idx[hold_part]] = _log_probs(model, x[train_idx[hold_part]], n_classes, eps)
    # every other row (validation, test, anything later) from the model fit on all training rows
    final = longest if rounds == longest.n_iter_ else _estimator(p, rounds).fit(x[train], y[train])
    rest = ~train
    if rest.any():
        stacked[rest] = _log_probs(final, x[rest], n_classes, eps)
    oof = float(probability_metrics(y[train], np.exp(stacked[train])).log_loss)
    audit = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}, "rounds": int(rounds),
             "tune_log_loss_by_rounds": {str(k): round(v, 6) for k, v in sorted(by_rounds.items())},
             "oof_train_log_loss": round(oof, 6), "input_columns": len(columns), "train_rows": int(train.sum()),
             "folds": folds, "seconds": round(time.time() - t0, 1)}
    return final, stacked, audit


class BoostPath:
    """The provider's side: the fitted model plus the indices of its inputs inside the feature row."""

    def __init__(self, model, input_columns: list[str], row_columns: list[str], n_classes: int, eps: float):
        index = {c: i for i, c in enumerate(row_columns)}
        missing = [c for c in input_columns if c not in index]
        if missing:
            raise ValueError(f"boost inputs missing from the feature row: {missing[:5]}")
        self.model = model
        self.idx = np.array([index[c] for c in input_columns], dtype=int)
        self.n_classes = n_classes
        self.eps = eps
        self._cache: dict = {}

    def features(self, row: np.ndarray) -> np.ndarray:
        """The seven stacked log probabilities for a feature row (without them); cached on the inputs' bytes."""
        x = np.asarray(row, dtype=np.float32)[self.idx]
        key = x.tobytes()
        out = self._cache.get(key)
        if out is None:
            if len(self._cache) > 4096:
                self._cache.clear()
            out = _log_probs(self.model, x[None, :], self.n_classes, self.eps)[0]
            self._cache[key] = out
        return out
