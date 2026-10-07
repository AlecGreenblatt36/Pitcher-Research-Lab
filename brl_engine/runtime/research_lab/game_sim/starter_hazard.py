"""Fitted starter-removal hazard (replaces the hand-tuned starter rule).

Decision unit: after each PA a starting pitcher completes, does he face the
batting team's next batter (y=0) or is he replaced (y=1)? The state is the one
the simulator sees at that moment (engine._maybe_change_pitcher): inning just
pitched, outs (3 if the half-inning ended), runners, runs allowed, BF, lead.
Opponent score after the PA comes from the batting team's next PA; the fielding
team's own score is taken from before its next turn at bat, so nothing after
the decision leaks in.

Pitcher and team tendencies are expanding means of PRIOR starts only (strictly
earlier dates), shrunk toward the training-league mean.

Protocol: fit 2023-2024, choose regularization on 2025, report 2026 once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SHRINK_STARTS = 5.0
FEATURES = [
    "bf", "bf_sq", "bf_minus_exp", "bf_minus_exp_sq", "over_exp", "runs", "runs_sq",
    "inning_ended", "inning", "late", "outs", "runners", "lead", "abs_lead_capped",
    "tto3", "pitcher_exp_bf", "team_exp_bf", "ended_x_bfdelta", "runs_x_early",
]


def _fielding_team(df: pd.DataFrame) -> np.ndarray:
    top = df["inning_topbot"].astype(str).str.lower().str.startswith("top")
    return np.where(top, df["home_team"], df["away_team"])


def build_decisions(history: pd.DataFrame) -> pd.DataFrame:
    h = history.sort_values(["date_key", "game_pk", "at_bat_number"], kind="mergesort").copy()
    h["fielding_team"] = _fielding_team(h)
    h["batting_team"] = np.where(h["fielding_team"] == h["home_team"], h["away_team"], h["home_team"])
    first = h.groupby(["game_pk", "fielding_team"], sort=False)["pitcher"].transform("first")
    h["starter"] = first
    g = h.groupby(["game_pk", "batting_team"], sort=False)
    for col in ("pitcher", "inning", "outs_when_up", "bat_score", "runner_1b", "runner_2b", "runner_3b"):
        h[f"next_{col}"] = g[col].shift(-1)
    h["first_bat_score"] = g["bat_score"].transform("first")
    d = h[(h["pitcher"] == h["starter"]) & h["next_pitcher"].notna()].copy()
    d["bf"] = d.groupby(["game_pk", "fielding_team"]).cumcount() + 1
    d["y"] = (d["next_pitcher"] != d["starter"]).astype(int)
    d["inning_ended"] = (d["next_inning"] != d["inning"]).astype(int)
    d["outs"] = np.where(d["inning_ended"] == 1, 3, d["next_outs_when_up"])
    runners = d["next_runner_1b"] + d["next_runner_2b"] + d["next_runner_3b"]
    d["runners"] = np.where(d["inning_ended"] == 1, 0, runners)
    d["runs"] = (d["next_bat_score"] - d["first_bat_score"]).clip(lower=0)
    d["lead"] = (d["fld_score"] - d["next_bat_score"]).clip(-8, 8)
    starts = (d.groupby(["game_pk", "fielding_team"], sort=False)
                .agg(date_key=("date_key", "first"), season=("season", "first"), starter=("starter", "first"),
                     bf_total=("bf", "max"), removed=("y", "max")).reset_index())
    return d, starts


def add_tendencies(d: pd.DataFrame, starts: pd.DataFrame, league_mean: float) -> pd.DataFrame:
    s = starts.sort_values(["date_key", "game_pk"]).copy()

    def prior(key: str) -> pd.Series:
        # strictly earlier DATES: cumulative sums through previous date
        daily = s.groupby([key, "date_key"])["bf_total"].agg(["sum", "count"]).groupby(level=0).cumsum()
        daily = daily.groupby(level=0).shift(1).fillna(0.0)
        merged = s[[key, "date_key"]].merge(daily.reset_index(), on=[key, "date_key"], how="left")
        return ((merged["sum"] + SHRINK_STARTS * league_mean) / (merged["count"] + SHRINK_STARTS)).to_numpy()

    s["pitcher_exp_bf"] = prior("starter")
    s["team_exp_bf"] = prior("fielding_team")
    out = d.merge(s[["game_pk", "fielding_team", "pitcher_exp_bf", "team_exp_bf"]], on=["game_pk", "fielding_team"])
    return engineer(out)


def engineer(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["bf_sq"] = d["bf"] ** 2 / 30.0
    d["bf_minus_exp"] = d["bf"] - d["pitcher_exp_bf"]
    d["bf_minus_exp_sq"] = d["bf_minus_exp"] ** 2 / 10.0
    d["over_exp"] = (d["bf_minus_exp"] > 0).astype(float)
    d["runs_sq"] = d["runs"] ** 2
    d["late"] = (d["inning"] >= 6).astype(float)
    d["abs_lead_capped"] = d["lead"].abs().clip(upper=5)
    d["tto3"] = (d["bf"] >= 18).astype(float)
    d["ended_x_bfdelta"] = d["inning_ended"] * d["bf_minus_exp"]
    d["runs_x_early"] = d["runs"] * (d["inning"] <= 4)
    return d


def heuristic_probability(d: pd.DataFrame) -> np.ndarray:
    """Exact replica of ManagerPolicy.should_remove for starters (fixture profile)."""
    exp = np.maximum(1.0, np.round(d["pitcher_exp_bf"].to_numpy()))
    max_b = exp + 5
    bf, runs, inn = d["bf"].to_numpy(float), d["runs"].to_numpy(float), d["inning"].to_numpy(float)
    ended = d["inning_ended"].to_numpy(bool)
    fatigue = np.clip(bf / max_b - 0.40 * 0.7, 0.0, 1.5)
    inning_w = np.clip((inn - 3) / 6.0, 0.05, 1.0)
    closeness = np.maximum(0.0, 1.0 - np.minimum(np.abs(d["lead"].to_numpy(float)), 6) / 6.0)
    traffic = d["runners"].to_numpy(float) / 3.0
    out_p = (2 - np.minimum(d["outs"].to_numpy(float), 2)) / 2.0
    leverage = np.minimum(1.0, 0.15 + 0.48 * inning_w * closeness + 0.22 * traffic + 0.15 * out_p)
    delta = (bf - exp) / np.maximum(2.5, exp * 0.22)
    pressure = -1.35 + 1.05 * delta + 0.34 * runs + 0.72 * fatigue + 0.22 * np.maximum(0, inn - 5) + 0.25 * leverage + 0.42 * ended
    p = 1.0 / (1.0 + np.exp(-pressure))
    p = np.where((inn <= 3) & (bf < exp * 0.72) & (runs < 5), 0.0, p)
    p = np.where(~ended & (bf < 3), 0.0, p)
    p = np.where(bf >= max_b, 1.0, p)
    return np.clip(p, 1e-4, 1 - 1e-4)


@dataclass
class BFOnlyBaseline:
    table: dict

    @classmethod
    def fit(cls, d: pd.DataFrame) -> "BFOnlyBaseline":
        key = list(zip(d["bf"].clip(upper=35), d["inning_ended"]))
        frame = pd.DataFrame({"k": key, "y": d["y"].to_numpy()})
        agg = frame.groupby("k")["y"].agg(["sum", "count"])
        base = d["y"].mean()
        return cls({k: (r["sum"] + 5 * base) / (r["count"] + 5) for k, r in agg.iterrows()} | {"_base": base})

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        keys = zip(d["bf"].clip(upper=35), d["inning_ended"])
        return np.clip(np.array([self.table.get(k, self.table["_base"]) for k in keys]), 1e-4, 1 - 1e-4)


def fit_hazard(train: pd.DataFrame, c: float) -> Pipeline:
    model = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=c, max_iter=2000))])
    model.fit(train[FEATURES], train["y"])
    return model


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def clustered_diff_ci(y, p_a, p_b, clusters, reps=1000, seed=36):
    p_a, p_b = np.clip(p_a, 1e-6, 1 - 1e-6), np.clip(p_b, 1e-6, 1 - 1e-6)
    la = -(y * np.log(p_a) + (1 - y) * np.log(1 - p_a))
    lb = -(y * np.log(p_b) + (1 - y) * np.log(1 - p_b))
    frame = pd.DataFrame({"c": clusters, "d": la - lb, "n": 1}).groupby("c")[["d", "n"]].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(frame), size=(reps, len(frame)))
    sums, counts = frame["d"].to_numpy()[idx].sum(1), frame["n"].to_numpy()[idx].sum(1)
    boot = sums / counts
    return float(frame["d"].sum() / frame["n"].sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def tendencies_at(history: pd.DataFrame, cutoff: str, league_mean: float) -> tuple[dict, dict]:
    """Pitcher and team expected starter BF from starts strictly before cutoff."""
    _, starts = build_decisions(history.loc[history["date_key"] < cutoff])
    shrink = lambda g: (g.sum() + SHRINK_STARTS * league_mean) / (g.count() + SHRINK_STARTS)
    return (starts.groupby("starter")["bf_total"].apply(shrink).to_dict(),
            starts.groupby("fielding_team")["bf_total"].apply(shrink).to_dict())


def actual_exit_innings(history: pd.DataFrame, pitcher_id: int, season: int) -> list[float]:
    d, _ = build_decisions(history[history["season"] == season])
    exits = d[(d["starter"] == pitcher_id) & (d["y"] == 1)]
    return [float(r.inning if r.inning_ended else r.inning - 1 + r.outs / 3) for r in exits.itertuples()]


from .manager import ManagerPolicy  # noqa: E402


class FittedStarterPolicy(ManagerPolicy):
    """Starter removal from the fitted hazard; relievers keep the dev heuristic."""

    def __init__(self, bundle: dict, pitcher_exp: dict, team_exp: dict, team_of_pitcher: dict, **kw):
        super().__init__(**kw)
        self.model, self.league = bundle["model"], float(bundle["league_mean_bf"])
        self.pitcher_exp, self.team_exp, self.team_of = pitcher_exp, team_exp, team_of_pitcher
        self.name = "fitted-starter-hazard-v1 (relievers: heuristic)"
        scaler, lr = self.model.named_steps["s"], self.model.named_steps["m"]
        self._mu, self._sd = scaler.mean_, scaler.scale_
        self._w, self._b = lr.coef_[0], float(lr.intercept_[0])

    def _prob(self, bf, runs, inning, ended, outs, runners, lead, p_exp, t_exp) -> float:
        """Numpy replica of engineer() + scaler + logistic (same order as FEATURES)."""
        d = bf - p_exp
        x = np.array([bf, bf * bf / 30.0, d, d * d / 10.0, float(d > 0), runs, runs * runs,
                      ended, inning, float(inning >= 6), outs, runners, lead, min(abs(lead), 5),
                      float(bf >= 18), p_exp, t_exp, ended * d, runs * float(inning <= 4)])
        z = float(((x - self._mu) / self._sd) @ self._w + self._b)
        return 1.0 / (1.0 + np.exp(-z))

    def should_remove(self, pitcher, line, state, fielding_side, inning_ended, rng):
        if not line.is_starter:
            return super().should_remove(pitcher, line, state, fielding_side, inning_ended, rng)
        if not self.removal_is_legal(line, inning_ended):
            return False
        pid = int(pitcher.player_id)
        lead = state.score_for(fielding_side) - state.opponent_score_for(fielding_side)
        p = self._prob(float(line.batters_faced), float(line.runs_allowed), float(state.inning),
                       float(inning_ended), 3.0 if inning_ended else float(state.outs),
                       0.0 if inning_ended else float(sum(b is not None for b in state.bases)),
                       float(max(-8, min(8, lead))), float(self.pitcher_exp.get(pid, self.league)),
                       float(self.team_exp.get(self.team_of.get(pid), self.league)))
        return bool(rng.random() < p)
