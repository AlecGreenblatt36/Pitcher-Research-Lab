"""Locked seven-outcome PA model as the game simulator's probability provider.

Replaces ``RatingsProbabilityProvider`` (the hand-tuned, ratings-based
development fallback) with the serialized model validated on the locked
chronological 2026 replay.

Rules this module enforces:

* Fail closed. Missing/mismatched artifact hash, history file, labels, or
  feature schema raises ``LockedModelError``. There is no silent fallback.
* Time valid. Player history is built only from PAs dated strictly before
  ``cutoff_date``. Nothing on or after the cutoff can enter a feature.
* Exact schema. The 122 features are rebuilt with the same shrinkage priors,
  rolling windows, clipping, and float32 storage as
  ``research_lab.pa_model.features.build_time_valid_features``.
* Same scoring path. Probabilities = 0.9 * calibrated multinomial model +
  0.1 * empirical-Bayes matchup baseline, using the saved parameters.

Validation boundary: this validates the PA probability layer only. Game-level
win/score outputs built on top of it are NOT validated until a frozen
chronological full-game replay is run.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from .models import OUTCOME_LABELS, PAContext

EXPECTED_MODEL_SHA256 = "3c87e4deedfb5253ac81252ad7fa2f117b16457f2c383c465670e2c3ee2fa095"
MODEL_LABELS = ("BIP_OUT", "K", "BB_HBP", "1B", "2B_3B", "HR", "OTHER_REACH")
# Simulator label <- model label (same order).
SIM_FROM_MODEL = dict(zip(OUTCOME_LABELS, MODEL_LABELS))

GROUPS = (
    "league", "batter", "pitcher", "batter_split", "pitcher_split",
    "park", "batter_recent", "pitcher_recent",
)
RATIO_GROUPS = GROUPS[1:]
CONTEXT_COLUMNS = (
    "platoon", "is_home_batter", "inning", "outs_when_up",
    "runner_1b", "runner_2b", "runner_3b", "bat_score_diff",
    "n_thruorder_pitcher", "batter_days_since_prev_game",
    "pitcher_days_since_prev_game", "age_bat", "age_pit",
    "batter_history_pa", "pitcher_history_pa",
    "batter_split_history_pa", "pitcher_split_history_pa",
)
FIXED_LEAGUE_PRIOR = np.array([0.49, 0.22, 0.09, 0.10, 0.035, 0.03, 0.035])
FIXED_LEAGUE_PRIOR = FIXED_LEAGUE_PRIOR / FIXED_LEAGUE_PRIOR.sum()


class LockedModelError(RuntimeError):
    """Raised instead of falling back to any other probability source."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_feature_columns() -> list[str]:
    columns = [f"p_{g}_{label}" for g in GROUPS for label in MODEL_LABELS]
    columns += [f"lr_{g}_{label}" for g in RATIO_GROUPS for label in MODEL_LABELS]
    columns += list(CONTEXT_COLUMNS)
    return columns


def _posterior(counts: np.ndarray, total: float, prior: np.ndarray, strength: float, eps: float) -> np.ndarray:
    result = (counts + strength * prior) / max(total + strength, eps)
    result = np.clip(result, eps, 1.0)
    return result / result.sum()


def _log_ratio(prob: np.ndarray, league: np.ndarray, eps: float) -> np.ndarray:
    return np.log(np.clip(prob, eps, 1.0)) - np.log(np.clip(league, eps, 1.0))


def _count_table(frame: pd.DataFrame, keys: list[str], n_classes: int) -> dict:
    grouped = frame.groupby(keys + ["_y"], sort=False).size().unstack("_y", fill_value=0)
    grouped = grouped.reindex(columns=range(n_classes), fill_value=0)
    values = grouped.to_numpy(float)
    index = grouped.index
    out = {}
    for i, key in enumerate(index):
        out[key] = values[i]
    return out


@dataclass
class HistoryState:
    """Counter state exactly as it stands at the start of ``cutoff_date``."""

    cutoff_date: str
    n_rows: int
    league_counts: np.ndarray
    batter: dict
    pitcher: dict
    batter_split: dict
    pitcher_split: dict
    park: dict
    batter_recent: dict
    pitcher_recent: dict
    batter_last_date: dict
    pitcher_last_date: dict
    batter_age: dict
    pitcher_age: dict
    last_history_date: str

    @classmethod
    def build(cls, history: pd.DataFrame, cutoff_date: str, config: Mapping) -> "HistoryState":
        label_index = {label: i for i, label in enumerate(MODEL_LABELS)}
        frame = history.loc[history["date_key"] < cutoff_date].copy()
        if frame.empty:
            raise LockedModelError(f"no PA history before cutoff {cutoff_date}")
        unknown = set(frame["outcome"]) - set(MODEL_LABELS)
        if unknown:
            raise LockedModelError(f"history contains unknown outcomes {sorted(unknown)}")
        frame = frame.sort_values(["date_key", "game_pk", "at_bat_number"], kind="mergesort")
        frame["_y"] = frame["outcome"].map(label_index).astype(int)
        frame["stand"] = frame["stand"].astype(str)
        frame["p_throws"] = frame["p_throws"].astype(str)
        frame["park"] = frame["park"].astype(str)
        n = len(MODEL_LABELS)

        def rolling(key: str, window: int) -> dict:
            tail = frame.groupby(key, sort=False).tail(window)
            return _count_table(tail, [key], n)

        last_b = frame.groupby("batter")["date_key"].max().to_dict()
        last_p = frame.groupby("pitcher")["date_key"].max().to_dict()
        age_b = frame.groupby("batter")["age_bat"].last().to_dict()
        age_p = frame.groupby("pitcher")["age_pit"].last().to_dict()
        return cls(
            cutoff_date=cutoff_date,
            n_rows=int(len(frame)),
            league_counts=np.bincount(frame["_y"].to_numpy(), minlength=n).astype(float),
            batter=_count_table(frame, ["batter"], n),
            pitcher=_count_table(frame, ["pitcher"], n),
            batter_split=_count_table(frame, ["batter", "p_throws"], n),
            pitcher_split=_count_table(frame, ["pitcher", "stand"], n),
            park=_count_table(frame, ["park"], n),
            batter_recent=rolling("batter", int(config["batter_recent_window"])),
            pitcher_recent=rolling("pitcher", int(config["pitcher_recent_window"])),
            batter_last_date=last_b,
            pitcher_last_date=last_p,
            batter_age=age_b,
            pitcher_age=age_p,
            last_history_date=str(frame["date_key"].max()),
        )




class SequentialHistoryState:
    """Mutable, strictly chronological history for large game replays.

    ``HistoryState.build`` is ideal for one forecast, but rebuilding all
    grouped counters from hundreds of thousands of rows for every game date is
    prohibitively slow.  This class builds the first cutoff once, then advances
    through later dates by applying only newly available PAs.  It preserves the
    same full-history and rolling-window counters as ``HistoryState``.
    """

    def __init__(self, history: pd.DataFrame, cutoff_date: str, config: Mapping):
        frame = history.copy()
        frame["date_key"] = frame["date_key"].astype(str).str[:10]
        frame = frame.sort_values(
            ["date_key", "game_pk", "at_bat_number"], kind="mergesort"
        ).reset_index(drop=True)
        label_index = {label: i for i, label in enumerate(MODEL_LABELS)}
        unknown = set(frame["outcome"]) - set(MODEL_LABELS)
        if unknown:
            raise LockedModelError(f"history contains unknown outcomes {sorted(unknown)}")
        frame["_y"] = frame["outcome"].map(label_index).astype(int)
        frame["stand"] = frame["stand"].astype(str)
        frame["p_throws"] = frame["p_throws"].astype(str)
        frame["park"] = frame["park"].astype(str)

        past = frame.loc[frame["date_key"] < cutoff_date].copy()
        if past.empty:
            raise LockedModelError(f"no PA history before cutoff {cutoff_date}")
        base = HistoryState.build(past, cutoff_date, config)
        self.__dict__.update(base.__dict__)
        self._config = dict(config)
        self._future = frame.loc[frame["date_key"] >= cutoff_date].reset_index(drop=True)
        self._future_dates = self._future["date_key"].to_numpy(dtype=str)
        self._position = 0
        self._batter_recent_queues = self._recent_queues(
            past, "batter", int(config["batter_recent_window"])
        )
        self._pitcher_recent_queues = self._recent_queues(
            past, "pitcher", int(config["pitcher_recent_window"])
        )

    @staticmethod
    def _recent_queues(frame: pd.DataFrame, key: str, window: int) -> dict:
        tail = frame.groupby(key, sort=False).tail(window)
        return {
            group_key: deque(group["_y"].astype(int).tolist())
            for group_key, group in tail.groupby(key, sort=False)
        }

    @staticmethod
    def _increment(table: dict, key, class_index: int) -> None:
        counts = table.get(key)
        if counts is None:
            counts = np.zeros(len(MODEL_LABELS), dtype=float)
            table[key] = counts
        counts[class_index] += 1.0

    @classmethod
    def _increment_recent(
        cls, table: dict, queues: dict, key, class_index: int, window: int
    ) -> None:
        counts = table.get(key)
        if counts is None:
            counts = np.zeros(len(MODEL_LABELS), dtype=float)
            table[key] = counts
        queue = queues.get(key)
        if queue is None:
            queue = deque()
            queues[key] = queue
        if len(queue) >= window:
            counts[int(queue.popleft())] -= 1.0
        queue.append(int(class_index))
        counts[class_index] += 1.0

    def advance_to(self, cutoff_date: str) -> None:
        cutoff_date = str(cutoff_date)[:10]
        if cutoff_date < self.cutoff_date:
            raise LockedModelError(
                f"sequential history cannot move backward: {self.cutoff_date} -> {cutoff_date}"
            )
        stop = int(np.searchsorted(self._future_dates, cutoff_date, side="left"))
        if stop < self._position:
            raise LockedModelError("sequential history position would move backward")
        rows = self._future.iloc[self._position:stop]
        b_window = int(self._config["batter_recent_window"])
        p_window = int(self._config["pitcher_recent_window"])
        update_columns = [
            "_y", "batter", "pitcher", "stand", "p_throws", "park",
            "date_key", "age_bat", "age_pit",
        ]
        for (y, batter_id, pitcher_id, stand, throws, park, date_key,
             age_bat, age_pit) in rows[update_columns].itertuples(index=False, name=None):
            y, batter_id, pitcher_id = int(y), int(batter_id), int(pitcher_id)
            stand, throws, park = str(stand), str(throws), str(park)
            self.league_counts[y] += 1.0
            self._increment(self.batter, batter_id, y)
            self._increment(self.pitcher, pitcher_id, y)
            self._increment(self.batter_split, (batter_id, throws), y)
            self._increment(self.pitcher_split, (pitcher_id, stand), y)
            self._increment(self.park, park, y)
            self._increment_recent(
                self.batter_recent, self._batter_recent_queues,
                batter_id, y, b_window,
            )
            self._increment_recent(
                self.pitcher_recent, self._pitcher_recent_queues,
                pitcher_id, y, p_window,
            )
            date_key = str(date_key)[:10]
            self.batter_last_date[batter_id] = date_key
            self.pitcher_last_date[pitcher_id] = date_key
            if not pd.isna(age_bat):
                self.batter_age[batter_id] = float(age_bat)
            if not pd.isna(age_pit):
                self.pitcher_age[pitcher_id] = float(age_pit)
            self.n_rows += 1
            self.last_history_date = date_key
        self._position = stop
        self.cutoff_date = cutoff_date

@dataclass
class _LinearPath:
    """Exact numpy replica of the fitted sklearn pipeline + logit calibration."""

    statistics: np.ndarray
    indicator_features: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    temperature: float
    class_biases: np.ndarray

    @classmethod
    def from_fitted(cls, fitted) -> "_LinearPath":
        numeric = fitted.estimator.named_steps["preprocess"].named_transformers_["numeric"]
        imputer = numeric.named_steps["imputer"]
        scaler = numeric.named_steps["scaler"]
        model = fitted.estimator.named_steps["model"]
        indicator = getattr(imputer, "indicator_", None)
        features = np.asarray(indicator.features_ if indicator is not None else [], dtype=int)
        return cls(
            statistics=np.asarray(imputer.statistics_, float),
            indicator_features=features,
            mean=np.asarray(scaler.mean_, float),
            scale=np.asarray(scaler.scale_, float),
            coef=np.asarray(model.coef_, float),
            intercept=np.asarray(model.intercept_, float),
            temperature=float(fitted.temperature),
            class_biases=np.asarray(fitted.class_biases, float),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        missing = np.isnan(x)
        filled = np.where(missing, self.statistics, x)
        if self.indicator_features.size:
            filled = np.hstack([filled, missing[:, self.indicator_features].astype(float)])
        z = (filled - self.mean) / self.scale
        logits = z @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        raw = np.exp(logits)
        raw /= raw.sum(axis=1, keepdims=True)
        cal = np.log(np.clip(raw, 1e-12, 1.0)) / max(self.temperature, 1e-6) + self.class_biases
        cal -= cal.max(axis=1, keepdims=True)
        out = np.exp(cal)
        return out / out.sum(axis=1, keepdims=True)


@dataclass
class LockedPAModelProvider:
    """Seven-outcome probabilities from the serialized locked PA model."""

    artifact_path: Path
    history_path: Path
    cutoff_date: str
    game_date: str
    park: str
    expected_sha256: str = EXPECTED_MODEL_SHA256
    name: str = "locked-pa-2026-v1"
    validation_status: str = (
        "PA layer: passed locked chronological 2026 replay; "
        "game-level outputs: NOT validated"
    )
    birthdates: Mapping[int, tuple[int, int, int]] | None = None
    history_frame: pd.DataFrame | None = None
    sequential_history: bool = False
    model_sha256: str = field(init=False, default="")
    history_sha256: str = field(init=False, default="")
    calls: int = field(init=False, default=0)
    cache_hits: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.artifact_path = Path(self.artifact_path)
        self.history_path = Path(self.history_path)
        if not self.artifact_path.is_file():
            raise LockedModelError(f"model artifact missing: {self.artifact_path}")
        if not self.history_path.is_file():
            raise LockedModelError(f"PA history missing: {self.history_path}")
        if self.cutoff_date > self.game_date:
            raise LockedModelError("cutoff_date cannot be after game_date")
        self.model_sha256 = sha256_file(self.artifact_path)
        if self.model_sha256 != self.expected_sha256:
            raise LockedModelError(
                f"model hash mismatch: got {self.model_sha256}, expected {self.expected_sha256}"
            )
        bundle = joblib.load(self.artifact_path)
        try:
            fitted = bundle["talent_plus_context"]
            self._eb = dict(bundle["empirical_bayes_parameters"])
            self._model_weight = float(bundle["model_weight"])
            self._config = dict(bundle["config"])
        except (KeyError, TypeError) as exc:
            raise LockedModelError(f"model bundle missing required key: {exc}") from exc
        if tuple(bundle.get("labels", ())) != MODEL_LABELS:
            raise LockedModelError(f"label order mismatch: {bundle.get('labels')}")
        if list(fitted.feature_columns) != expected_feature_columns():
            raise LockedModelError("feature schema mismatch between artifact and provider")
        self._fitted = fitted
        self._linear = _LinearPath.from_fitted(fitted)
        self._eps = float(self._config.get("min_probability", 1e-7))
        self.history_sha256 = sha256_file(self.history_path)
        needed = ["date_key", "game_pk", "at_bat_number", "batter", "pitcher",
                  "stand", "p_throws", "park", "outcome", "age_bat", "age_pit"]
        if self.history_frame is None:
            history = pd.read_csv(self.history_path, usecols=needed)
        else:
            missing = set(needed) - set(self.history_frame.columns)
            if missing:
                raise LockedModelError(f"preloaded history missing columns: {sorted(missing)}")
            history = self.history_frame.loc[:, needed]
        self.state = (
            SequentialHistoryState(history, self.cutoff_date, self._config)
            if self.sequential_history
            else HistoryState.build(history, self.cutoff_date, self._config)
        )
        self._talent_cache: dict = {}
        self._probability_cache: dict[tuple, dict[str, float]] = {}

    def advance_to(self, cutoff_date: str, *, game_date: str | None = None, park: str | None = None) -> None:
        """Advance a replay provider without rebuilding historical counters."""
        if not isinstance(self.state, SequentialHistoryState):
            raise LockedModelError("provider was not initialized with sequential_history=True")
        next_game_date = str(game_date or cutoff_date)[:10]
        cutoff_date = str(cutoff_date)[:10]
        if cutoff_date > next_game_date:
            raise LockedModelError("cutoff_date cannot be after game_date")
        self.state.advance_to(cutoff_date)
        self.cutoff_date = cutoff_date
        self.game_date = next_game_date
        if park is not None:
            self.park = str(park)
        # Replace the dictionaries instead of ``clear()`` so Python can release
        # the large hash tables accumulated during high-path game simulations.
        self._talent_cache = {}
        self._probability_cache = {}

    def set_park(self, park: str) -> None:
        if str(park) != self.park:
            self.park = str(park)
            self._talent_cache = {}
            self._probability_cache = {}

    # ----- feature construction -------------------------------------------------
    def _get(self, table: dict, key) -> tuple[np.ndarray, float]:
        counts = table.get(key)
        if counts is None:
            return np.zeros(len(MODEL_LABELS)), 0.0
        return counts, float(counts.sum())

    def talent_block(self, batter_id: int, pitcher_id: int, stand: str, throws: str) -> tuple[np.ndarray, dict]:
        key = (batter_id, pitcher_id, stand, throws)
        cached = self._talent_cache.get(key)
        if cached is not None:
            return cached
        c, s, eps = self._config, self.state, self._eps
        league = _posterior(s.league_counts, float(s.league_counts.sum()), FIXED_LEAGUE_PRIOR, 500.0, eps)
        b_counts, b_total = self._get(s.batter, batter_id)
        p_counts, p_total = self._get(s.pitcher, pitcher_id)
        bs_counts, bs_total = self._get(s.batter_split, (batter_id, throws))
        ps_counts, ps_total = self._get(s.pitcher_split, (pitcher_id, stand))
        pk_counts, pk_total = self._get(s.park, self.park)
        br_counts, br_total = self._get(s.batter_recent, batter_id)
        pr_counts, pr_total = self._get(s.pitcher_recent, pitcher_id)
        b = _posterior(b_counts, b_total, league, c["player_prior_pa"], eps)
        p = _posterior(p_counts, p_total, league, c["player_prior_pa"], eps)
        groups = {
            "league": league,
            "batter": b,
            "pitcher": p,
            "batter_split": _posterior(bs_counts, bs_total, b, c["split_prior_pa"], eps),
            "pitcher_split": _posterior(ps_counts, ps_total, p, c["split_prior_pa"], eps),
            "park": _posterior(pk_counts, pk_total, league, c["park_prior_pa"], eps),
            "batter_recent": _posterior(br_counts, br_total, b, c["recent_prior_pa"], eps),
            "pitcher_recent": _posterior(pr_counts, pr_total, p, c["recent_prior_pa"], eps),
        }
        parts = [groups[g] for g in GROUPS] + [_log_ratio(groups[g], league, eps) for g in RATIO_GROUPS]
        block = np.concatenate(parts)
        meta = {
            "batter_history_pa": b_total, "pitcher_history_pa": p_total,
            "batter_split_history_pa": bs_total, "pitcher_split_history_pa": ps_total,
            "groups": groups,
        }
        self._talent_cache[key] = (block, meta)
        return block, meta

    # Statcast convention in the training data: days-since-previous-game is
    # NaN on a player's first game of a season and never exceeds 30 (max in
    # 697,615 rows, values above 30 are capped). Cross-season gaps become NaN, not 300+ day values
    # the model never saw.
    MAX_DAYS_SINCE = 30

    def season_age(self, player_id: int, observed: Mapping) -> float:
        """Statcast age fields are season age as of June 30 (checked on 2026 rows)."""
        born = (self.birthdates or {}).get(player_id)
        if born:
            year = int(str(self.game_date)[:4])
            y, m, d = born
            return float(year - y - (1 if (m, d) > (6, 30) else 0))
        return float(observed.get(player_id, np.nan))

    def _days_since(self, last: str | None) -> float:
        if last is None:
            return float("nan")
        if str(last)[:4] != str(self.game_date)[:4]:
            return float("nan")
        days = float((pd.Timestamp(self.game_date) - pd.Timestamp(last)).days)
        return min(days, float(self.MAX_DAYS_SINCE))  # Statcast caps at 30

    def feature_row(
        self, batter_id: int, pitcher_id: int, stand: str, throws: str, *,
        is_home_batter: int, inning: int, outs: int, runners: tuple[int, int, int],
        bat_score_diff: int, times_through: int,
        batter_days: float | None = None, pitcher_days: float | None = None,
        age_bat: float | None = None, age_pit: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        block, meta = self.talent_block(batter_id, pitcher_id, stand, throws)
        s = self.state
        if batter_days is None:
            batter_days = self._days_since(s.batter_last_date.get(batter_id))
        if pitcher_days is None:
            pitcher_days = self._days_since(s.pitcher_last_date.get(pitcher_id))
        if age_bat is None:
            age_bat = self.season_age(batter_id, s.batter_age)
        if age_pit is None:
            age_pit = self.season_age(pitcher_id, s.pitcher_age)
        context = np.array([
            float(stand.upper() == throws.upper()),  # training definition: same-handed = 1
            float(is_home_batter),
            float(min(20, max(1, inning))),
            float(min(2, max(0, outs))),
            float(runners[0]), float(runners[1]), float(runners[2]),
            float(min(10, max(-10, bat_score_diff))),
            float(min(8, max(1, times_through))),
            float(batter_days), float(pitcher_days),
            float(age_bat), float(age_pit),
            meta["batter_history_pa"], meta["pitcher_history_pa"],
            meta["batter_split_history_pa"], meta["pitcher_split_history_pa"],
        ])
        # Training stored features as float32; replicate exactly.
        row = np.concatenate([block, context]).astype(np.float32).astype(float)
        return row, meta

    def _eb_probs(self, row: np.ndarray) -> np.ndarray:
        n, eps, w = len(MODEL_LABELS), self._eps, self._eb
        sl = {g: slice(i * n, (i + 1) * n) for i, g in enumerate(GROUPS)}
        league, batter, pitcher = row[sl["league"]], row[sl["batter_split"]], row[sl["pitcher_split"]]
        park, rb, rp = row[sl["park"]], row[sl["batter_recent"]], row[sl["pitcher_recent"]]
        lr = lambda a, b: np.log(np.clip(a, eps, 1)) - np.log(np.clip(b, eps, 1))
        score = np.log(np.clip(league, eps, 1))
        score = score + w["batter_weight"] * lr(batter, league) + w["pitcher_weight"] * lr(pitcher, league)
        score = score + w["park_weight"] * lr(park, league)
        score = score + w["batter_recent_weight"] * lr(rb, batter) + w["pitcher_recent_weight"] * lr(rp, pitcher)
        score = score / max(float(w["temperature"]), eps)
        score -= score.max()
        out = np.exp(score)
        return out / out.sum()

    def predict_row(self, row: np.ndarray) -> dict[str, np.ndarray]:
        model = self._linear.predict(row)[0]
        eb = self._eb_probs(row)
        ensemble = self._model_weight * model + (1.0 - self._model_weight) * eb
        return {"model": model, "eb": eb, "ensemble": ensemble}

    # ----- simulator interface --------------------------------------------------
    @staticmethod
    def _stand(bats: str, throws: str) -> str:
        bats = (bats or "R").upper()[0]
        if bats == "S":
            return "L" if throws.upper() == "R" else "R"
        return bats

    def probabilities(self, context: PAContext) -> Mapping[str, float]:
        try:
            batter_id = int(context.batter.player_id)
            pitcher_id = int(context.pitcher.player_id)
        except ValueError as exc:
            raise LockedModelError(
                f"locked model needs real MLB IDs; got {context.batter.player_id!r} / {context.pitcher.player_id!r}"
            ) from exc
        throws = context.pitcher.throws
        stand = self._stand(context.batter.bats, throws)
        runners = tuple(int(b is not None) for b in context.bases)
        cache_key = (
            batter_id, pitcher_id, stand, str(throws).upper(),
            int(context.batting_side == "home"),
            int(min(20, max(1, context.inning))),
            int(min(2, max(0, context.outs))),
            runners,
            int(min(10, max(-10, context.score_diff))),
            int(min(8, max(1, context.times_through_order))),
        )
        self.calls += 1
        cached = self._probability_cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        row, _ = self.feature_row(
            batter_id, pitcher_id, stand, throws,
            is_home_batter=int(context.batting_side == "home"),
            inning=context.inning, outs=context.outs,
            runners=runners,
            bat_score_diff=context.score_diff,
            times_through=context.times_through_order,
        )
        ensemble = self.predict_row(row)["ensemble"]
        model_index = {label: i for i, label in enumerate(MODEL_LABELS)}
        result = {sim: float(ensemble[model_index[m]]) for sim, m in SIM_FROM_MODEL.items()}
        self._probability_cache[cache_key] = result
        return result

    def matchup_answer(self, batter_id: int, pitcher_id: int, bats: str, throws: str, *,
                       is_home_batter: int, inning: int = 1, outs: int = 0) -> dict:
        stand = self._stand(bats, throws)
        row, meta = self.feature_row(
            batter_id, pitcher_id, stand, throws, is_home_batter=is_home_batter,
            inning=inning, outs=outs, runners=(0, 0, 0), bat_score_diff=0, times_through=1,
        )
        parts = self.predict_row(row)
        league = meta["groups"]["league"]
        return {
            "batter_id": batter_id, "pitcher_id": pitcher_id, "stand": stand, "throws": throws,
            "probabilities": dict(zip(MODEL_LABELS, map(float, parts["ensemble"]))),
            "league_reference": dict(zip(MODEL_LABELS, map(float, league))),
            "history_pa": {k: int(meta[k]) for k in (
                "batter_history_pa", "pitcher_history_pa",
                "batter_split_history_pa", "pitcher_split_history_pa")},
            "provenance": self.provenance(),
        }

    def provenance(self) -> dict:
        return {
            "provider": self.name,
            "validation_status": self.validation_status,
            "model_sha256": self.model_sha256,
            "history_sha256": self.history_sha256,
            "cutoff_date": self.cutoff_date,
            "last_history_date_used": self.state.last_history_date,
            "history_rows_used": self.state.n_rows,
            "game_date": self.game_date,
            "park": self.park,
            "blend": {"model": self._model_weight, "empirical_bayes": 1.0 - self._model_weight},
            "fallback_used": False,
        }
