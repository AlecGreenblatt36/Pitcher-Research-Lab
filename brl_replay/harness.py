"""Replay one season of games through the simulator with time-valid inputs.

Inputs for a game dated D come only from plate appearances dated before D: the provider's
history state (and physics state for a physics model), bullpen candidates (relievers the team
used in the last 14 days, roles from the prior 365 days), starter tendencies for the fitted
starter policy, and batter handedness. The production context offsets (brl_live/provider_adjust)
are applied the same way the live lane applies them, so a replay scores what the site shows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.locked_pa_provider import HistoryState, LockedPAModelProvider
from research_lab.game_sim.models import GameMatchup, PitcherProfile, PlayerProfile, TeamProfile
from research_lab.game_sim.starter_hazard import FittedStarterPolicy, tendencies_at

MAX_RUNS = 30
HISTORY_COLUMNS = ["date_key", "game_pk", "at_bat_number", "batter", "pitcher", "stand", "p_throws", "park", "outcome", "age_bat", "age_pit"]


class CachedProvider:
    """Memoizes the provider within one game (a pure function of these inputs, so outputs are identical)."""

    def __init__(self, base):
        self.base, self.cache, self.hits, self.misses = base, {}, 0, 0
        self.name, self.validation_status = base.name, base.validation_status

    def reset(self):
        self.cache.clear()

    def probabilities(self, c):
        key = (c.batter.player_id, c.pitcher.player_id, c.batting_side, min(20, max(1, c.inning)),
               min(2, max(0, c.outs)), tuple(b is not None for b in c.bases),
               max(-10, min(10, c.score_diff)), min(8, max(1, c.times_through_order)))
        out = self.cache.get(key)
        if out is None:
            out = self.base.probabilities(c)
            self.cache[key] = out
            self.misses += 1
        else:
            self.hits += 1
        return out

    def __getattr__(self, item):
        return getattr(self.base, item)


def appearances(h: pd.DataFrame) -> pd.DataFrame:
    top = h["inning_topbot"].str.lower().str.startswith("top")
    h = h.assign(fielding_team=np.where(top, h["home_team"], h["away_team"]))
    first_ab = h.groupby(["game_pk", "fielding_team"])["at_bat_number"].transform("min")
    h = h.assign(_first=h.groupby(["game_pk", "pitcher"])["at_bat_number"].transform("min") == first_ab)
    h = h.assign(_k=(h["outcome"] == "K").astype(int), _bb=(h["outcome"] == "BB_HBP").astype(int),
                 _hr=(h["outcome"] == "HR").astype(int), _lead=h["fld_score"] - h["bat_score"])
    last_ab = h.groupby(["game_pk", "fielding_team"])["at_bat_number"].transform("max")
    h = h.assign(_finish=h.groupby(["game_pk", "pitcher"])["at_bat_number"].transform("max") == last_ab)
    return h.groupby(["game_pk", "pitcher"]).agg(
        date=("date_key", "first"), team=("fielding_team", "first"), bf=("at_bat_number", "size"),
        entry_inning=("inning", "min"), throws=("p_throws", "first"), start=("_first", "first"),
        k=("_k", "sum"), bb=("_bb", "sum"), hr=("_hr", "sum"), entry_lead=("_lead", "first"),
        finished=("_finish", "first"),
    ).reset_index()


def bullpen(app: pd.DataFrame, team: str, cutoff: str, starter: int) -> tuple[PitcherProfile, ...]:
    """Relievers the team used in the last 14 days (its last 10 games early in a season); roles from the prior 365 days."""
    prior = app[app["date"] < cutoff]
    team_games = prior[prior["team"] == team].sort_values("date")
    window_start = (pd.Timestamp(cutoff) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    recent = team_games[(team_games["date"] >= window_start) & (~team_games["start"]) & (team_games["pitcher"] != starter)]
    if recent["pitcher"].nunique() < 6:
        last_games = team_games["game_pk"].drop_duplicates().tail(10)
        recent = team_games[team_games["game_pk"].isin(last_games) & (~team_games["start"]) & (team_games["pitcher"] != starter)]
    year_start = (pd.Timestamp(cutoff) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    hist = prior[(prior["date"] >= year_start) & (~prior["start"])]
    out = []
    for pid, g in recent.groupby("pitcher"):
        sg = hist[hist["pitcher"] == pid]
        late = float((sg["entry_inning"] >= 8).mean()) if len(sg) else 0.0
        ninth = float((sg["entry_inning"] >= 9).mean()) if len(sg) else 0.0
        exp_bf = int(max(3, round(sg["bf"].median()))) if len(sg) else 4
        role = "closer" if ninth >= 0.6 else "setup" if late >= 0.5 else "long" if exp_bf >= 7 else "reliever"
        out.append(PitcherProfile(str(pid), str(pid), str(g["throws"].iloc[-1]), role=role, leverage=min(1.0, 0.3 + late),
                                  rest=1.0, expected_batters=exp_bf, max_batters=max(exp_bf + 3, 6)))
    closers = [p for p in out if p.role == "closer"]
    if len(closers) > 1:
        keep = max(closers, key=lambda p: p.leverage)
        out = [p if p.role != "closer" or p is keep else PitcherProfile(
            p.player_id, p.name, p.throws, role="setup", leverage=p.leverage, rest=p.rest,
            expected_batters=p.expected_batters, max_batters=p.max_batters) for p in out]
    return tuple(out)


def starter_profile(app: pd.DataFrame, pid: int, throws: str, cutoff: str) -> PitcherProfile:
    year_start = (pd.Timestamp(cutoff) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    starts = app[(app["pitcher"] == pid) & app["start"] & (app["date"] < cutoff) & (app["date"] >= year_start)]["bf"]
    exp_bf = int(round(starts.median())) if len(starts) else 22
    max_bf = int(round(starts.quantile(0.95))) if len(starts) >= 5 else 27
    return PitcherProfile(str(pid), str(pid), throws, role="starter", stamina=0.7,
                          expected_batters=exp_bf, max_batters=max(max_bf, exp_bf + 2))


def bats_lookup(h: pd.DataFrame, cutoff: str) -> dict:
    prior = h[h["date_key"] < cutoff]
    return prior.groupby("batter")["stand"].agg(lambda s: "S" if s.nunique() > 1 else s.iloc[0]).to_dict()


def replay_dates(h: pd.DataFrame, app: pd.DataFrame, games: pd.DataFrame, dates: list, *, model_path: Path, model_sha256: str,
                 history_path: Path, hazard_path: Path, n_sims: int, physics_table=None, offsets=None, log=print) -> list[dict]:
    """Simulate every game on the given dates; one record per game (win counts, run histograms, starter outs)."""
    from brl_live.provider_adjust import ContextAdjust
    hcols = h[HISTORY_COLUMNS]
    first = dates[0]
    base = LockedPAModelProvider(model_path, history_path, cutoff_date=first, game_date=first, park="NYY",
                                 expected_sha256=model_sha256, physics_table=physics_table)
    cached = CachedProvider(base)
    provider = ContextAdjust(cached, offsets) if offsets else cached
    hazard = joblib.load(hazard_path)
    records = []
    t0 = time.time()
    for di, date in enumerate(dates):
        day = games[games["date"] == date]
        if day.empty:
            continue
        td = time.time()
        base.state = HistoryState.build(hcols, date, base._config)
        base.cutoff_date = base.game_date = date
        defense = None
        if base.physics is not None:
            from research_lab.pa_model.physics import PhysicsState, DefenseState
            params = dict(base.physics.sums.p)
            base.physics = PhysicsState.build(physics_table, date, params)
            base._physics_cache = {}
            if 'f_def' in base.physics_features:
                defense = DefenseState.build(h, date, params)
        pexp, texp = tendencies_at(h, date, float(hazard["league_mean_bf"]))
        hands = bats_lookup(h, date)
        first_stand = h[h["date_key"] == date].groupby("batter")["stand"].first().to_dict()
        for g in day.itertuples():
            base.park = g.park
            base._talent_cache = {}
            cached.reset()
            teams = {}
            for side in ("away", "home"):
                lineup = tuple(PlayerProfile(str(b), str(b), hands.get(b, first_stand.get(b, "R"))) for b in getattr(g, f"{side}_lineup"))
                sid = int(getattr(g, f"{side}_starter"))
                team = getattr(g, side)
                starter = starter_profile(app, sid, getattr(g, f"{side}_starter_throws"), date)
                pen = tuple(p for p in bullpen(app, team, date, sid) if p.player_id != str(sid))
                teams[side] = TeamProfile(team, team, lineup, starter, pen, defense=(float(defense.value(team)) if defense is not None else 0.0))
            matchup = GameMatchup(away=teams["away"], home=teams["home"], venue=g.park, game_type="R")
            policy = FittedStarterPolicy(hazard, pexp, texp, {int(g.away_starter): g.away, int(g.home_starter): g.home})
            sim = GameSimulator(provider, manager_policy=policy)
            rng = np.random.default_rng(int(g.game_pk))
            seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_sims, dtype=np.int64)
            hw = ties = 0
            ha = np.zeros(MAX_RUNS + 1, int); aa = np.zeros(MAX_RUNS + 1, int)
            s_outs = {"away": [], "home": []}
            for s in seeds:
                r = sim.simulate(matchup, int(s), record_events=False)
                ha[min(r.home_score, MAX_RUNS)] += 1
                aa[min(r.away_score, MAX_RUNS)] += 1
                if r.winner == "home":
                    hw += 1
                elif r.winner not in ("away", "home"):
                    ties += 1
                for side in ("away", "home"):
                    line = r.pitcher_lines.get(teams[side].starter.player_id)
                    s_outs[side].append(int(line["outs_recorded"]) if line else 0)
            records.append({"game_pk": int(g.game_pk), "date": date, "home": g.home, "away": g.away, "n": n_sims,
                            "home_wins": hw, "ties": ties, "home_hist": ha.tolist(), "away_hist": aa.tolist(),
                            "home_starter_outs": float(np.mean(s_outs["home"])), "away_starter_outs": float(np.mean(s_outs["away"])),
                            "home_runs": int(g.home_runs), "away_runs": int(g.away_runs)})
        log(f"{date} {len(day)} games {time.time() - td:.1f}s ({di + 1}/{len(dates)}) cache hit {cached.hits / max(1, cached.hits + cached.misses):.2f}")
    log(f"done {time.time() - t0:.0f}s")
    return records
