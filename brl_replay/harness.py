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
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.locked_pa_provider import HistoryState, LockedPAModelProvider
from research_lab.game_sim.models import GameMatchup, PitcherProfile, PlayerProfile, TeamProfile
from research_lab.game_sim.running_events import KINDS as RUNNING_KINDS
from research_lab.game_sim.starter_hazard import FittedStarterPolicy, tendencies_at

MAX_RUNS = 30
MAX_MARGIN = 15
NON_PA = frozenset({"stolen_base", "caught_stealing", *RUNNING_KINDS})     # events between plate appearances
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


def pitcher_outs(sub: pd.DataFrame, home_won=None) -> dict:
    """{(game_pk, pitcher): outs he recorded} from plate appearances (game_pk, at_bat_number, inning, inning_topbot,
    outs_when_up, pitcher, outcome): the outs before the next plate appearance of the same half-inning minus the outs before
    his (an out on the bases between plate appearances goes to the pitcher on the mound, as the official line counts it),
    and 3 at the end of a half-inning. The one half-inning that can end short of three is the home team's last when it wins
    the game there (a walk-off): its last plate appearance adds its own out, if it made one (a sacrifice fly). home_won is
    the set of game_pks the home team won; without it, any game's last bottom half is read that way."""
    s_ = sub.sort_values(["game_pk", "at_bat_number"], kind="mergesort")
    half = s_["inning_topbot"].astype(str).str.lower().str.startswith("top").to_numpy()
    key = pd.Series(list(zip(s_["game_pk"].to_numpy(), s_["inning"].to_numpy(), half)), index=s_.index)
    up = s_["outs_when_up"].astype(int)
    nxt = up.groupby(key.values).shift(-1).to_numpy()
    last_pa_game = (s_["game_pk"] != s_["game_pk"].shift(-1)).to_numpy()
    won = np.ones(len(s_), bool) if home_won is None else s_["game_pk"].isin(set(home_won)).to_numpy()
    walk_off = last_pa_game & ~half & won
    made_out = s_["outcome"].isin(["BIP_OUT", "K"]).to_numpy()
    after = np.where(~np.isnan(nxt), nxt, np.where(walk_off, up.to_numpy() + made_out, 3))
    rec = np.clip(after - up.to_numpy(), 0, 3)
    out = {}
    for g_, p_, r_ in zip(s_["game_pk"].to_numpy(), s_["pitcher"].to_numpy(), rec):
        k_ = (int(g_), int(p_)); out[k_] = out.get(k_, 0) + int(r_)
    return out


ENTRY_INN = (5, 6, 7, 8, 9)            # entry inning buckets: 5th or earlier, 6th, 7th, 8th, 9th or later
ENTRY_LEAD = (-3, -1, 0, 3)            # the fielding team's lead at the half-inning's start: <=-3, -2..-1, 0, 1..3, >=4


def entry_key(inning, lead, top2) -> str:
    """BULLPEN-05's cell for one relief entry: inning bucket, lead bucket and whether the arm is one of the team's top two
    by 14-day share."""
    i = min(max(int(inning), 5), 9)
    lead = int(lead)
    lb = 0 if lead <= -3 else 1 if lead <= -1 else 2 if lead == 0 else 3 if lead <= 3 else 4
    return f"{i}:{lb}:{int(bool(top2))}"


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


def bullpen(app: pd.DataFrame, team: str, cutoff: str, starter: int, rest: bool = False) -> tuple[PitcherProfile, ...]:
    """Relievers the team used in the last 14 days (its last 10 games early in a season); roles from the prior 365 days;
    availability from the three days before when rest is on (brl_live.live_feed.reliever_rest)."""
    from brl_live.live_feed import bullpen_usage, reliever_rest
    prior = app[app["date"] < cutoff]
    team_games = prior[prior["team"] == team].sort_values("date")
    window_start = (pd.Timestamp(cutoff) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    recent = team_games[(team_games["date"] >= window_start) & (~team_games["start"]) & (team_games["pitcher"] != starter)]
    if recent["pitcher"].nunique() < 6:
        last_games = team_games["game_pk"].drop_duplicates().tail(10)
        recent = team_games[team_games["game_pk"].isin(last_games) & (~team_games["start"]) & (team_games["pitcher"] != starter)]
    year_start = (pd.Timestamp(cutoff) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    hist = prior[(prior["date"] >= year_start) & (~prior["start"])]
    usage = bullpen_usage(app, team, cutoff, list(recent["pitcher"].unique()))
    out = []
    for pid, g in recent.groupby("pitcher"):
        sg = hist[hist["pitcher"] == pid]
        late = float((sg["entry_inning"] >= 8).mean()) if len(sg) else 0.0
        ninth = float((sg["entry_inning"] >= 9).mean()) if len(sg) else 0.0
        exp_bf = int(max(3, round(sg["bf"].median()))) if len(sg) else 4
        role = "closer" if ninth >= 0.6 else "setup" if late >= 0.5 else "long" if exp_bf >= 7 else "reliever"
        out.append(PitcherProfile(str(pid), str(pid), str(g["throws"].iloc[-1]), role=role, leverage=min(1.0, 0.3 + late),
                                  rest=reliever_rest(app, pid, cutoff) if rest else 1.0, expected_batters=exp_bf, max_batters=max(exp_bf + 3, 6),
                                  usage=usage.get(pid, {})))
    closers = [p for p in out if p.role == "closer"]
    if len(closers) > 1:
        keep = max(closers, key=lambda p: p.leverage)
        out = [p if p.role != "closer" or p is keep else replace(p, role="setup") for p in out]
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
                 history_path: Path, hazard_path: Path, n_sims: int, physics_table=None, offsets=None, rest=False, environment=None,
                 team_offsets=None, age_layer=None, steals=None, win_states=False, starter_lines=False, role_offsets=None, real_pa_check=False, transitions=None, hitter_lines=False, running_events=None, reliever_choice=None, leash=None, base_state=None, relief_exit=None, relief_hooks=None, day_form=0.0, paired_streams=False, matchup=None, level_offsets=None, hazard_correction=None, pitch_counts=None, state_history=None, pitch_table=None, log=print) -> list[dict]:
    """Simulate every game on the given dates; one record per game (win counts, run histograms, starter outs).

    environment: optional {game_pk: seven log-multipliers} (brl_live/environment.py); games without an entry are unadjusted.
    team_offsets: optional {date: {team: {'bat': [7], 'fld': [7]}}} (brl_live/team_offsets.by_date), offsets at the start of each date.
    age_layer: optional aging and recency layer (brl_live/age_layer.py); its AgingState advances date by date from the history.
    steals: optional {'per_pa': ...} base-running settings (brl_live/running.py); each date uses running statistics
    through the season before it, so nothing from the replayed season enters.
    win_states: also build the game's win table from the simulated plate appearances (brl_live/win_table.py, as the
    live forecast does) and record its win chance at the start of every real plate appearance of that game
    ([inning, half 0 top 1 bottom, outs, runners mask, home lead, chance]); game states only, no player data.
    win_states='split' adds the chances from the even and the odd simulated games separately and the number of
    simulated plate appearances in that state, so the table's own noise can be measured.
    starter_lines: also record each starter's simulated distributions (strikeouts, batters faced, hits, walks and hit
    batters, outs) with his actual line in that game (box-score facts) and a simple baseline from earlier dates only
    (strikeout share over the prior 365 days shrunk toward 22% by 150 batters; expected batters faced from the hazard).
    real_pa_check: also run the game's full probability stack (model, context, environment, team and role offsets) on
    every real plate appearance of the game (its batter, pitcher, inning, outs, runners, score and times through the
    order) and record the predicted and observed counts per outcome class for starters and relievers, by groups of side,
    inning, role and times through the order, and by bases (empty, first only, a runner in scoring position) and outs, so
    the simulator's own totals can be told apart from the probabilities it draws from.
    transitions: optional base-running kernel after each outcome (research_lab.game_sim.transitions.EmpiricalKernel).
    running_events: optional running plays between plate appearances other than steals (research_lab.game_sim.running_events).
    reliever_choice: optional fitted choice of the entering reliever (research_lab.game_sim.reliever_choice.RelieverChoice).
    relief_exit: optional fitted reliever exits (research_lab.game_sim.relief_exit.ReliefExit).
    relief_hooks: optional team hook offsets for the fitted exits (brl_replay.relief_decisions.HookOffsets), prior dates.
    day_form: sd of a per-team, per-world shock on reaching base (brl_live.provider_adjust.DayForm); 0 is off.
    matchup: optional {'table': {(batter, pitcher): (chase points, zone-swing points)}, 'coefs': {'K': (..), 'BB': (..)}}:
        decision-moment matchups (brl_live.provider_adjust.MatchupAdjust), from earlier seasons only.
    pitch_counts: optional research_lab.game_sim.pitch_counts.PitchCounts: each plate appearance's pitches drawn by outcome
        (LEASH-03), for the hazard correction's pitch terms.
    state_history: optional plate-appearance rows (HISTORY_COLUMNS) the model's history state is built from instead of h.
    pitch_table: the per-pitch table (tools/brl_pitch_table.py) a model with pitch-chain features needs (PITCH-02).
    hazard_correction: optional {'terms', 'coef'}: LEASH-02's correction on the starter-exit hazard's log-odds by inning, its end
        and the score (research_lab.game_sim.starter_hazard.CORRECTION_TERMS).
    paired_streams: tie every random draw to a team's plate appearance number (engine paired streams), so replays of two
        settings with the same seeds stay paired and their comparison carries far less simulation noise; both replays of a
        comparison must use the same choice.
    base_state: optional base-state offsets (brl_live.provider_adjust.load_base_state): the stack shaped by bases and outs,
        applied after the team offsets.
    leash: optional (research_lab.game_sim.starter_leash.Leash, AppearanceIndex of regular-season appearances): each
        starter's expected batters faced moved for short rest, a relief outing before, a return from a layoff and the month.
    hitter_lines: also record each lineup hitter's simulated chances (at least one hit, a home run, a strikeout, a walk), mean
    plate appearances, hits and home runs, the histograms of his hits, home runs, walks and strikeouts, his actual line in that game (box-score facts) and a baseline
    from earlier dates only (his hit and home-run rates per plate appearance over the prior 365 days, shrunk toward the league)."""
    from brl_live.provider_adjust import ContextAdjust, EnvironmentAdjust, TeamAdjust, RoleAdjust
    # state_history (LONG-01): the plate appearances the model's counters are built from, when they reach further back
    # than the replayed history h (the sealed 2015-2022 seasons in front of it); games and real states still come from h.
    hcols = (state_history if state_history is not None else h)[HISTORY_COLUMNS]
    first = dates[0]
    base = LockedPAModelProvider(model_path, history_path, cutoff_date=first, game_date=first, park="NYY",
                                 expected_sha256=model_sha256, physics_table=physics_table, pitch_table=pitch_table)
    chain_q = None
    if base.chain is not None:
        from research_lab.pa_model.pitch_chain import prepare
        chain_q = prepare(pitch_table, {int(g): str(k) for g, k in zip(hcols["game_pk"], hcols["park"].astype(str))})
    cached = CachedProvider(base)
    provider = ContextAdjust(cached, offsets) if offsets else cached
    chain = [('bare', cached)] + ([('context', provider)] if offsets else [])      # each layer, for the real plate appearance check
    env = None
    if environment is not None:
        env = provider = EnvironmentAdjust(provider); chain.append(('environment', env))
    tadj = None
    if team_offsets is not None:
        tadj = provider = TeamAdjust(provider); chain.append(('team', tadj))
    if base_state is not None:
        from brl_live.provider_adjust import BaseStateAdjust
        provider = BaseStateAdjust(provider, base_state); chain.append(('base_state', provider))
    if matchup is not None:
        from brl_live.provider_adjust import MatchupAdjust
        provider = MatchupAdjust(provider, matchup['table'], matchup['coefs']); chain.append(('matchup', provider))
    dform = None
    if day_form:
        from brl_live.provider_adjust import DayForm
        if isinstance(day_form, dict):
            dform = provider = DayForm(provider, float(day_form.get('sigma', 0.0)), float(day_form.get('center', 0.0)))
        else:
            dform = provider = DayForm(provider, float(day_form))
        chain.append(('day_form', provider))
    radj = None
    if role_offsets is not None:
        radj = provider = RoleAdjust(provider); chain.append(('role', radj))
    aadj = None
    if age_layer is not None:
        from research_lab.pa_model.physics import AgingState
        from brl_live.age_layer import AgeAdjust
        a_state = AgingState(age_layer.get('params') or {'aging': True, 'decay_days': 365, 'k_dec': 60.0})
        hs = hcols.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort')
        a_dates = hs['date_key'].astype(str).str[:10].to_numpy()
        a_b, a_p, a_o = hs['batter'].to_numpy(int), hs['pitcher'].to_numpy(int), hs['outcome'].astype(str).to_numpy()
        a_pos = 0
        aadj = provider = AgeAdjust(provider, age_layer, a_state, 0); chain.append(('age', aadj))
    ladj = None
    if level_offsets is not None:
        # LEVEL-02: the season's league level re-anchored in-season, each date's offsets from earlier dates only (LEVEL-01)
        from brl_live.provider_adjust import LevelAdjust
        ladj = provider = LevelAdjust(provider); chain.append(('level', ladj))
    hazard = joblib.load(hazard_path)
    real_states = {}
    if win_states:
        from brl_live.win_table import WinTable, state_index
        want = set(int(x) for x in games[games["date"].isin(dates)]["game_pk"])
        cols = ["game_pk", "at_bat_number", "inning", "inning_topbot", "outs_when_up", "runner_1b", "runner_2b", "runner_3b", "home_score", "away_score"]
        rs = h.loc[h["game_pk"].isin(want), cols].sort_values(["game_pk", "at_bat_number"], kind="mergesort")
        for gpk, grp in rs.groupby("game_pk", sort=False):
            real_states[int(gpk)] = [(int(r.inning), "top" if str(r.inning_topbot).lower().startswith("top") else "bottom", int(r.outs_when_up),
                                      (bool(r.runner_1b), bool(r.runner_2b), bool(r.runner_3b)), int(r.home_score) - int(r.away_score))
                                     for r in grp.itertuples(index=False)]
    act_lines, per_k, act_relievers, act_entries = {}, {}, {}, {}
    act_innings = {}
    if starter_lines:
        # the batting side's score before its first plate appearance of each inning (runs by inning follow with the final)
        want_i = set(int(x) for x in games[games["date"].isin(dates)]["game_pk"])
        si_ = h.loc[h["game_pk"].isin(want_i), ["game_pk", "at_bat_number", "inning", "inning_topbot", "bat_score"]].sort_values(["game_pk", "at_bat_number"])
        side_i = np.where(si_["inning_topbot"].astype(str).str.lower().str.startswith("top"), "away", "home")
        for (gpk_, side_), gg in si_.groupby([si_["game_pk"].to_numpy(), side_i]):
            firsts = gg.groupby("inning")["bat_score"].first()
            act_innings.setdefault(int(gpk_), {})[str(side_)] = [[int(i_), int(v_)] for i_, v_ in firsts.items()]
    if starter_lines:
        want_g = set(int(x) for x in games[games["date"].isin(dates)]["game_pk"])
        sub = h.loc[h["game_pk"].isin(want_g), ["game_pk", "pitcher", "outcome", "inning_topbot"]]
        gw_ = games[games["game_pk"].isin(want_g)]
        home_won_ = set(int(x) for x in gw_.loc[gw_["home_runs"] > gw_["away_runs"], "game_pk"]) if {"home_runs", "away_runs"} <= set(gw_.columns) else None
        outs_by = pitcher_outs(h.loc[h["game_pk"].isin(want_g), ["game_pk", "at_bat_number", "inning", "inning_topbot", "outs_when_up", "pitcher", "outcome"]],
                               home_won_) if {"at_bat_number", "inning", "outs_when_up"} <= set(h.columns) else {}
        sub_side = np.where(sub["inning_topbot"].astype(str).str.lower().str.startswith("top"), "home", "away")
        npit_act = sub.groupby([sub["game_pk"].to_numpy(), sub_side])["pitcher"].nunique().to_dict()
        if "at_bat_number" in h.columns:
            # the relievers each side actually used, in order (BULLPEN-03: the chance-to-pitch table against them)
            sub2 = h.loc[h["game_pk"].isin(want_g), ["game_pk", "at_bat_number", "pitcher", "inning_topbot"]].sort_values(["game_pk", "at_bat_number"], kind="mergesort")
            side2 = np.where(sub2["inning_topbot"].astype(str).str.lower().str.startswith("top"), "home", "away")
            for (gpk_, side_), pids_ in sub2.groupby([sub2["game_pk"].to_numpy(), side2], sort=False)["pitcher"]:
                act_relievers[(int(gpk_), str(side_))] = [int(x) for x in pd.unique(pids_.to_numpy())][1:]
            if {"fld_score", "bat_score", "inning"} <= set(h.columns):
                # BULLPEN-05: each real relief entry's inning and the fielding team's lead at that half-inning's start
                s3 = h.loc[h["game_pk"].isin(want_g), ["game_pk", "at_bat_number", "pitcher", "inning", "inning_topbot", "fld_score", "bat_score"]].sort_values(["game_pk", "at_bat_number"], kind="mergesort")
                s3 = s3.assign(fside=np.where(s3["inning_topbot"].astype(str).str.lower().str.startswith("top"), "home", "away"))
                half_first = s3.groupby(["game_pk", "inning", "inning_topbot"], sort=False)[["fld_score", "bat_score"]].transform("first")
                s3 = s3.assign(lead0=(half_first["fld_score"] - half_first["bat_score"]).to_numpy())
                firsts = s3.drop_duplicates(["game_pk", "fside", "pitcher"], keep="first")
                for (gpk_, side_), gg in firsts.groupby(["game_pk", "fside"], sort=False):
                    act_entries[(int(gpk_), str(side_))] = [(int(r_.pitcher), int(r_.inning), int(r_.lead0)) for r_ in gg.iloc[1:].itertuples()]
        for (gpk_, side_), oc in sub.groupby([sub["game_pk"].to_numpy(), sub_side])["outcome"]:
            vc = oc.value_counts()
            act_lines[(int(gpk_), str(side_))] = {"bf": int(len(oc)), "k": int(vc.get("K", 0)), "bb": int(vc.get("BB_HBP", 0)),
                                                  "h": int(vc.get("1B", 0) + vc.get("2B_3B", 0) + vc.get("HR", 0)), "hr": int(vc.get("HR", 0)),
                                                  "pitchers": int(npit_act.get((gpk_, side_), 0))}
        for (gpk_, pid_), oc in sub.groupby(["game_pk", "pitcher"])["outcome"]:
            vc = oc.value_counts()
            act_lines[(int(gpk_), int(pid_))] = {"bf": int(len(oc)), "k": int(vc.get("K", 0)), "bb": int(vc.get("BB_HBP", 0)),
                                                 "h": int(vc.get("1B", 0) + vc.get("2B_3B", 0) + vc.get("HR", 0)), "hr": int(vc.get("HR", 0)),
                                                 "outs": outs_by.get((int(gpk_), int(pid_)))}
        hk = pd.DataFrame({"pitcher": h["pitcher"].to_numpy(), "date_key": h["date_key"].astype(str).str[:10].to_numpy(), "k": (h["outcome"] == "K").to_numpy().astype(int)})
        daily = hk.groupby(["pitcher", "date_key"]).agg(bf=("k", "size"), k=("k", "sum")).reset_index()
        daily["ord"] = [pd.Timestamp(x).toordinal() for x in daily["date_key"]]
        for pid_, gq in daily.groupby("pitcher"):
            per_k[int(pid_)] = (gq["ord"].to_numpy(), np.cumsum(gq["bf"].to_numpy()), np.cumsum(gq["k"].to_numpy()))

    act_bat, per_h = {}, {}
    if hitter_lines:
        want_h = set(int(x) for x in games[games["date"].isin(dates)]["game_pk"])
        sub_b = h.loc[h["game_pk"].isin(want_h), ["game_pk", "batter", "outcome"]]
        for (gpk_, bid_), oc in sub_b.groupby(["game_pk", "batter"])["outcome"]:
            vc = oc.value_counts()
            act_bat[(int(gpk_), int(bid_))] = {"pa": int(len(oc)), "h": int(vc.get("1B", 0) + vc.get("2B_3B", 0) + vc.get("HR", 0)), "hr": int(vc.get("HR", 0)),
                                               "xbh": int(vc.get("2B_3B", 0) + vc.get("HR", 0)),
                                               "k": int(vc.get("K", 0)), "bb": int(vc.get("BB_HBP", 0))}
        hb = pd.DataFrame({"batter": h["batter"].to_numpy(), "date_key": h["date_key"].astype(str).str[:10].to_numpy(),
                           "h": h["outcome"].isin(["1B", "2B_3B", "HR"]).to_numpy().astype(int), "hr": (h["outcome"] == "HR").to_numpy().astype(int)})
        daily_b = hb.groupby(["batter", "date_key"]).agg(pa=("h", "size"), h=("h", "sum"), hr=("hr", "sum")).reset_index()
        daily_b["ord"] = [pd.Timestamp(x).toordinal() for x in daily_b["date_key"]]
        for bid_, gq in daily_b.groupby("batter"):
            per_h[int(bid_)] = (gq["ord"].to_numpy(), np.cumsum(gq["pa"].to_numpy()), np.cumsum(gq["h"].to_numpy()), np.cumsum(gq["hr"].to_numpy()))

    def prior_hit_rates(bid_, date_, lg_h=0.218, lg_hr=0.031, k_=150.0):
        v = per_h.get(int(bid_))
        if v is None:
            return lg_h, lg_hr, 0
        ords, cpa, ch, chr_ = v; d_ = pd.Timestamp(date_).toordinal()
        lo, hi = np.searchsorted(ords, d_ - 365, "left"), np.searchsorted(ords, d_, "left")
        if hi <= lo:
            return lg_h, lg_hr, 0
        sub_ = lambda c: int(c[hi - 1] - (c[lo - 1] if lo > 0 else 0))
        pa_, h_, hr_ = sub_(cpa), sub_(ch), sub_(chr_)
        return (h_ + k_ * lg_h) / (pa_ + k_), (hr_ + k_ * lg_hr) / (pa_ + k_), pa_

    def prior_k_rate(pid_, date_):
        v = per_k.get(int(pid_))
        if v is None:
            return 0.22, 0
        ords, cbf, ck = v; d_ = pd.Timestamp(date_).toordinal()
        lo, hi = np.searchsorted(ords, d_ - 365, "left"), np.searchsorted(ords, d_, "left")
        if hi <= lo:
            return 0.22, 0
        bf_ = int(cbf[hi - 1] - (cbf[lo - 1] if lo > 0 else 0)); k_ = int(ck[hi - 1] - (ck[lo - 1] if lo > 0 else 0))
        return (k_ + 0.22 * 150.0) / (bf_ + 150.0), bf_
    real_rows = {}
    if real_pa_check:
        want_r = set(int(x) for x in games[games["date"].isin(dates)]["game_pk"])
        rcols = ["game_pk", "at_bat_number", "batter", "pitcher", "stand", "p_throws", "is_home_batter", "inning", "outs_when_up",
                 "runner_1b", "runner_2b", "runner_3b", "bat_score", "fld_score", "n_thruorder_pitcher", "outcome"]
        rsub = h.loc[h["game_pk"].isin(want_r), rcols].sort_values(["game_pk", "at_bat_number"], kind="mergesort")
        for gpk_, gq in rsub.groupby("game_pk", sort=False):
            real_rows[int(gpk_)] = list(gq.itertuples(index=False))
    MODEL_ORDER = ("BIP_OUT", "K", "BB_HBP", "1B", "2B_3B", "HR", "OTHER_REACH")
    records = []
    t0 = time.time()
    for di, date in enumerate(dates):
        day = games[games["date"] == date]
        if day.empty:
            continue
        td = time.time()
        base.state = HistoryState.build(hcols, date, base._config)
        base.cutoff_date = base.game_date = date
        if aadj is not None:
            while a_pos < len(a_dates) and a_dates[a_pos] < date:
                a_end = a_pos
                while a_end < len(a_dates) and a_dates[a_end] == a_dates[a_pos]:
                    a_end += 1
                a_state.absorb_date(pd.Timestamp(a_dates[a_pos]).toordinal(), a_b[a_pos:a_end], a_p[a_pos:a_end], a_o[a_pos:a_end])
                a_pos = a_end
            aadj.set_day(a_state, pd.Timestamp(date).toordinal())
        if ladj is not None:
            ladj.set_level(level_offsets.get(str(date)[:10]))
        defense = None
        if base.physics is not None:
            from research_lab.pa_model.physics import PhysicsState, DefenseState
            params = dict(base.physics.sums.p)
            base.physics = PhysicsState.build(physics_table, date, params).with_history(hcols)
            base._physics_cache = {}
            if 'f_def' in base.physics_features:
                defense = DefenseState.build(h, date, params)
        if base.chain is not None:
            # PITCH-02: the chain's tables advance to the start of this date (pitches of earlier dates only)
            from research_lab.pa_model.pitch_chain import day_ordinal
            base.chain.absorb_through(chain_q, int(day_ordinal([date])[0]))
            base._chain_cache = {}
        pexp, texp = tendencies_at(h, date, float(hazard["league_mean_bf"]))
        hands = bats_lookup(h, date)
        first_stand = h[h["date_key"] == date].groupby("batter")["stand"].first().to_dict()
        for g in day.itertuples():
            base.park = g.park
            base._talent_cache = {}
            cached.reset()
            if env is not None:
                env.set_environment(environment.get(int(g.game_pk)))
            if aadj is not None:
                aadj.reset()
            if radj is not None:
                radj.set_roles(role_offsets.get(date) or {})
            if tadj is not None:
                tday = team_offsets.get(date) or {}
                get = lambda team, side: np.asarray(((tday.get(str(team)) or {}).get(side)) or [0.0] * 7, float)
                tadj.set_teams({'away': (get(g.away, 'bat') + get(g.home, 'fld')).tolist(), 'home': (get(g.home, 'bat') + get(g.away, 'fld')).tolist()})
            teams = {}
            for side in ("away", "home"):
                lineup = tuple(PlayerProfile(str(b), str(b), hands.get(b, first_stand.get(b, "R"))) for b in getattr(g, f"{side}_lineup"))
                sid = int(getattr(g, f"{side}_starter"))
                team = getattr(g, side)
                starter = starter_profile(app, sid, getattr(g, f"{side}_starter_throws"), date)
                pen = tuple(p for p in bullpen(app, team, date, sid, rest=rest) if p.player_id != str(sid))
                teams[side] = TeamProfile(team, team, lineup, starter, pen, defense=(float(defense.value(team)) if defense is not None else 0.0))
            matchup = GameMatchup(away=teams["away"], home=teams["home"], venue=g.park, game_type="R")
            pexp_g = pexp
            if leash is not None:
                pexp_g = dict(pexp)
                for sid_l in (int(g.away_starter), int(g.home_starter)):
                    pexp_g[sid_l] = leash[0].adjust(float(pexp.get(sid_l, hazard["league_mean_bf"])), leash[1].facts(sid_l, date))
            policy = FittedStarterPolicy(hazard, pexp_g, texp, {int(g.away_starter): g.away, int(g.home_starter): g.home},
                                         correction=hazard_correction, reliever_choice=reliever_choice, relief_exit=relief_exit,
                                         relief_offsets=({'away': relief_hooks.at(g.away, date), 'home': relief_hooks.at(g.home, date)}
                                                         if relief_hooks is not None else None))
            steal = None
            if steals is not None:
                from brl_live.running import steal_model
                steal = steal_model(int(str(date)[:4]) - 1, **{k: float(v) for k, v in steals.items() if k in ('per_pa', 'third')})
            sim = GameSimulator(provider, manager_policy=policy, steals=steal, transitions=transitions, running_events=running_events, pitch_counts=pitch_counts,
                                paired_streams=bool(paired_streams))
            rng = np.random.default_rng(int(g.game_pk))
            seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_sims, dtype=np.int64)
            hw = ties = 0
            ha = np.zeros(MAX_RUNS + 1, int); aa = np.zeros(MAX_RUNS + 1, int)
            mh = np.zeros(2 * MAX_MARGIN + 1, int)                  # home minus away runs, -MAX_MARGIN..+MAX_MARGIN (the run line needs the joint)
            f5m = np.zeros(2 * MAX_MARGIN + 1, int); f5t = np.zeros(MAX_RUNS + 1, int)   # after five innings: margin and total
            first = np.zeros(3, int)                                # worlds with no run in the first inning, the road team scoring in it, the home team
            s_outs = {"away": [], "home": []}
            sl = {side: {m_: np.zeros(n_, int) for m_, n_ in (("k", 21), ("bf", 46), ("h", 21), ("bb", 16), ("outs", 28))} for side in ("away", "home")} if starter_lines else None
            tot = {side: np.zeros(4) for side in ("away", "home")} if starter_lines else None   # team pitching: K, BF, BB+HBP, hits
            npit = {side: 0 for side in ("away", "home")}                                         # pitchers used, summed over worlds
            rel_n = {side: {} for side in ("away", "home")}                                       # worlds each reliever pitched in
            ent_n = {side: {} for side in ("away", "home")}                                       # relief entries by BULLPEN-05's cells, summed
            top2_ = {side: set(str(b_.player_id) for b_ in sorted(teams[side].bullpen, key=lambda b_: -float(dict(getattr(b_, "usage", None) or ()).get("share14", 0.0)))[:2])
                     for side in ("away", "home")} if starter_lines else None
            nhr = {side: 0 for side in ("away", "home")}                                          # home runs allowed, summed over worlds
            keys_n = {side: np.zeros(5, int) for side in ("away", "home")}                       # KEYS-02: scored first, and won; won; starter 18+ outs, and won
            inn_sum = {side: np.zeros(10) for side in ("away", "home")}                           # runs by inning 1-9 and extras, summed
            wt = WinTable() if win_states else None
            halves = (WinTable(), WinTable()) if win_states == "split" else None
            hl = None
            if hitter_lines:
                # per lineup hitter: worlds with >=1 hit, >=1 home run, >=1 strikeout, >=1 walk; sums of PA, hits, home runs; hits histogram
                hl = {(side, str(b.player_id)): {"any": np.zeros(4), "sum": np.zeros(3), "h_hist": np.zeros(6, int), "hr_hist": np.zeros(4, int),
                                                 "bb_hist": np.zeros(5, int), "k_hist": np.zeros(5, int),
                                                 "hs_hist": np.zeros((6, 6), int), "kw": np.zeros(2, int)}
                      for side in ("away", "home") for b in teams[side].lineup}
            for si, s in enumerate(seeds):
                if dform is not None:
                    dform.new_world(int(s))
                r = sim.simulate(matchup, int(s), record_events=bool(win_states) or bool(hitter_lines))
                if wt is not None:
                    box = {"score": {"home": r.home_score, "away": r.away_score},
                           "plays": [e for e in r.events if e.get("outcome") not in NON_PA]}
                    wt.add(box)
                    if halves is not None:
                        halves[si % 2].add(box)
                ha[min(r.home_score, MAX_RUNS)] += 1
                aa[min(r.away_score, MAX_RUNS)] += 1
                mh[min(max(r.home_score - r.away_score, -MAX_MARGIN), MAX_MARGIN) + MAX_MARGIN] += 1
                ir_ = {sd_: {int(i_): int(v_) for i_, v_ in ((r.inning_runs or {}).get(sd_) or {}).items()} for sd_ in ("away", "home")}
                f5h = sum(v_ for i_, v_ in ir_["home"].items() if i_ <= 5); f5a = sum(v_ for i_, v_ in ir_["away"].items() if i_ <= 5)
                f5m[min(max(f5h - f5a, -MAX_MARGIN), MAX_MARGIN) + MAX_MARGIN] += 1; f5t[min(f5h + f5a, MAX_RUNS)] += 1
                a1, h1 = ir_["away"].get(1, 0), ir_["home"].get(1, 0)
                first += (a1 + h1 == 0, a1 > 0, h1 > 0)
                if r.winner == "home":
                    hw += 1
                elif r.winner not in ("away", "home"):
                    ties += 1
                ia_ = min((i_ for i_, v_ in ir_["away"].items() if v_ > 0), default=99); ih_ = min((i_ for i_, v_ in ir_["home"].items() if v_ > 0), default=99)
                fs_ = "away" if ia_ < 99 and ia_ <= ih_ else ("home" if ih_ < 99 else None)    # the road team bats first in each inning
                for side in ("away", "home"):
                    line = r.pitcher_lines.get(teams[side].starter.player_id)
                    s_outs[side].append(int(line["outs_recorded"]) if line else 0)
                    won_, sp6_ = r.winner == side, bool(line and int(line["outs_recorded"]) >= 18)
                    keys_n[side] += (fs_ == side, fs_ == side and won_, won_, sp6_, sp6_ and won_)
                    if sl is not None and line:
                        for m_, f_ in (("k", "strikeouts"), ("bf", "batters_faced"), ("h", "hits_allowed"), ("bb", "walks_hbp"), ("outs", "outs_recorded")):
                            arr = sl[side][m_]; arr[min(int(line[f_]), len(arr) - 1)] += 1
                if tot is not None:
                    for side_r, by_inn in (r.inning_runs or {}).items():
                        if side_r in inn_sum:
                            for inn_r, runs_r in by_inn.items():
                                inn_sum[side_r][min(int(inn_r), 10) - 1] += runs_r
                    for ln in r.pitcher_lines.values():
                        sd_ = ln.get("team_side")
                        if sd_ in tot:
                            tot[sd_] += (ln["strikeouts"], ln["batters_faced"], ln["walks_hbp"], ln["hits_allowed"])
                            npit[sd_] += 1
                            nhr[sd_] += int(ln.get("home_runs") or 0)
                            if not ln.get("is_starter"):
                                rel_n[sd_][str(ln.get("pitcher_id"))] = rel_n[sd_].get(str(ln.get("pitcher_id")), 0) + 1
                                ei_ = int(ln.get("entry_inning") or 1); ir2_ = r.inning_runs or {}
                                own_ = sum(v_ for i_, v_ in (ir2_.get(sd_) or {}).items() if int(i_) < ei_ + (1 if sd_ == "away" else 0))
                                opp_ = sum(v_ for i_, v_ in (ir2_.get("home" if sd_ == "away" else "away") or {}).items() if int(i_) < ei_)
                                k_ = entry_key(ei_, own_ - opp_, str(ln.get("pitcher_id")) in top2_[sd_])
                                ent_n[sd_][k_] = ent_n[sd_].get(k_, 0) + 1
                if hl is not None:
                    cnt = {}
                    for e in r.events:
                        o_ = e.get("outcome"); bid_ = (str(e.get("batting_side")), str(e.get("batter_id")))
                        if o_ in NON_PA or bid_ not in hl:
                            continue
                        c_ = cnt.setdefault(bid_, [0, 0, 0, 0, 0, 0])         # PA, hits, home runs, strikeouts, walks, singles
                        c_[0] += 1
                        c_[1] += o_ in ("single", "double_triple", "home_run"); c_[2] += o_ == "home_run"
                        c_[3] += o_ == "strikeout"; c_[4] += o_ == "bb_hbp"; c_[5] += o_ == "single"
                    for bid_, c_ in cnt.items():
                        a_ = hl[bid_]
                        a_["any"] += (c_[1] > 0, c_[2] > 0, c_[3] > 0, c_[4] > 0)
                        a_["sum"] += (c_[0], c_[1], c_[2])
                        a_["h_hist"][min(c_[1], 5)] += 1
                        a_["hr_hist"][min(c_[2], 3)] += 1; a_["bb_hist"][min(c_[4], 4)] += 1; a_["k_hist"][min(c_[3], 4)] += 1
                        a_["hs_hist"][min(c_[1], 5), min(c_[5], 5)] += 1          # hits by singles among them (total bases 2+ needs both)
                        won_h = r.winner == bid_[0]
                        a_["kw"] += (c_[2] > 0 and won_h, c_[1] >= 2 and won_h)  # KEYS-02: worlds his team won with his home run, with his 2+ hits
                    for bid_ in hl:
                        if bid_ not in cnt:
                            hl[bid_]["h_hist"][0] += 1; hl[bid_]["hr_hist"][0] += 1; hl[bid_]["bb_hist"][0] += 1; hl[bid_]["k_hist"][0] += 1
                            hl[bid_]["hs_hist"][0, 0] += 1
            records.append({"game_pk": int(g.game_pk), "date": date, "home": g.home, "away": g.away, "n": n_sims,
                            "home_wins": hw, "ties": ties, "home_hist": ha.tolist(), "away_hist": aa.tolist(), "margin_hist": mh.tolist(),
                            "f5_margin_hist": f5m.tolist(), "f5_total_hist": f5t.tolist(), "first_inning": first.tolist(),
                            "home_starter_outs": float(np.mean(s_outs["home"])) if s_outs["home"] else 0.0, "away_starter_outs": float(np.mean(s_outs["away"])) if s_outs["away"] else 0.0,
                            "home_runs": int(g.home_runs), "away_runs": int(g.away_runs)})
            if hl is not None:
                out_h = []
                for side in ("away", "home"):
                    for spot_, b in enumerate(teams[side].lineup, start=1):
                        a_ = hl[(side, str(b.player_id))]
                        ph_, phr_, prior_pa_ = prior_hit_rates(int(b.player_id), date)
                        out_h.append({"batter": int(b.player_id), "side": side, "spot": spot_,
                                      "p_hit": round(float(a_["any"][0] / n_sims), 4), "p_hr": round(float(a_["any"][1] / n_sims), 4),
                                      "p_k": round(float(a_["any"][2] / n_sims), 4), "p_bb": round(float(a_["any"][3] / n_sims), 4),
                                      "mean_pa": round(float(a_["sum"][0] / n_sims), 3), "mean_h": round(float(a_["sum"][1] / n_sims), 4),
                                      "mean_hr": round(float(a_["sum"][2] / n_sims), 4), "h_hist": a_["h_hist"].tolist(),
                                      "hr_hist": a_["hr_hist"].tolist(), "bb_hist": a_["bb_hist"].tolist(), "k_hist": a_["k_hist"].tolist(),
                                      "hs_hist": a_["hs_hist"].tolist(), "kw": a_["kw"].tolist(),
                                      "prior_h_rate": round(float(ph_), 4), "prior_hr_rate": round(float(phr_), 4), "prior_pa_365": prior_pa_,
                                      "actual": act_bat.get((int(g.game_pk), int(b.player_id)))})
                records[-1]["hitters"] = out_h
            if real_pa_check:
                from types import SimpleNamespace
                from research_lab.game_sim.locked_pa_provider import SIM_FROM_MODEL
                model_of_sim = {v: k for k, v in SIM_FROM_MODEL.items()}
                agg = {r_: {"pred": np.zeros(7), "obs": np.zeros(7), "n": 0, "layers": {nm_: np.zeros(7) for nm_, _ in chain[:-1]},
                            "ll": {nm_: 0.0 for nm_, _ in chain[:-1] + [("full", None)]}, "s2": np.zeros((7, 7))} for r_ in ("starter", "reliever")}
                starters_ = {int(g.home_starter), int(g.away_starter)}
                grp_ = {}      # (batting side 0 away 1 home, inning bucket, role, times through the order capped at 3): n, observed, bare, full
                bgrp_ = {}     # (bases 0 empty, 1 first only, 2 a runner in scoring position; outs): n, observed, full
                for row in real_rows.get(int(g.game_pk), []):
                    try:
                        side_b = "home" if int(row.is_home_batter) == 1 else "away"
                        fld_team = teams["away" if side_b == "home" else "home"]
                        ctx = SimpleNamespace(
                            batter=PlayerProfile(str(int(row.batter)), str(int(row.batter)), str(row.stand or "R")),
                            pitcher=PitcherProfile(str(int(row.pitcher)), str(int(row.pitcher)), str(row.p_throws or "R"), role="starter" if int(row.pitcher) in starters_ else "reliever"),
                            batting_side=side_b, inning=int(row.inning), half="bottom" if side_b == "home" else "top", outs=int(row.outs_when_up),
                            bases=tuple(True if int(x or 0) else None for x in (row.runner_1b, row.runner_2b, row.runner_3b)),
                            batting_score=int(row.bat_score), fielding_score=int(row.fld_score), score_diff=int(row.bat_score) - int(row.fld_score),
                            times_through_order=int(row.n_thruorder_pitcher or 1), fielding_team_defense=float(getattr(fld_team, "defense", 0.0) or 0.0))
                        probs = provider.probabilities(ctx)
                        inner_ = [(nm_, ob_.probabilities(ctx)) for nm_, ob_ in chain[:-1]]
                    except Exception as exc_:
                        agg.setdefault("errors", []).append(type(exc_).__name__ + ": " + str(exc_)[:80])
                        continue
                    r_ = "starter" if int(row.pitcher) in starters_ else "reliever"
                    pv_ = np.array([probs[model_of_sim[lab]] for lab in MODEL_ORDER])
                    agg[r_]["pred"] += pv_
                    agg[r_]["s2"] += np.outer(pv_, pv_)                      # second moments, for a level offset's log-likelihood (LEVEL-01)
                    for nm_, pr_ in inner_:
                        agg[r_]["layers"][nm_] += np.array([pr_[model_of_sim[lab]] for lab in MODEL_ORDER])
                    if str(row.outcome) in MODEL_ORDER:
                        agg[r_]["obs"][MODEL_ORDER.index(str(row.outcome))] += 1
                        lab_ = model_of_sim[str(row.outcome)]          # the simulator's label of the observed outcome
                        agg[r_]["ll"]["full"] += float(np.log(max(probs[lab_], 1e-12)))
                        for nm_, pr_ in inner_:
                            agg[r_]["ll"][nm_] += float(np.log(max(pr_[lab_], 1e-12)))
                    agg[r_]["n"] += 1
                    inn_ = int(row.inning)
                    gk_ = (int(side_b == "home"), "1st" if inn_ == 1 else ("mid" if inn_ <= 8 else "late"), r_[0], min(max(int(row.n_thruorder_pitcher or 1), 1), 3))
                    ge_ = grp_.setdefault(gk_, [0, np.zeros(7), np.zeros(7), np.zeros(7)])
                    ge_[0] += 1
                    if str(row.outcome) in MODEL_ORDER:
                        ge_[1][MODEL_ORDER.index(str(row.outcome))] += 1
                    ge_[2] += np.array([inner_[0][1][model_of_sim[lab]] for lab in MODEL_ORDER]) if inner_ else np.array([probs[model_of_sim[lab]] for lab in MODEL_ORDER])
                    ge_[3] += np.array([probs[model_of_sim[lab]] for lab in MODEL_ORDER])
                    on_ = tuple(int(x or 0) for x in (row.runner_1b, row.runner_2b, row.runner_3b))
                    bk_ = (0 if not any(on_) else 1 if not (on_[1] or on_[2]) else 2, min(max(int(row.outs_when_up), 0), 2))
                    be_ = bgrp_.setdefault(bk_, [0, np.zeros(7), np.zeros(7)])
                    be_[0] += 1
                    if str(row.outcome) in MODEL_ORDER:
                        be_[1][MODEL_ORDER.index(str(row.outcome))] += 1
                    be_[2] += np.array([probs[model_of_sim[lab]] for lab in MODEL_ORDER])
                errs = agg.pop("errors", [])
                records[-1]["real_pa"] = {r_: {"n": v_["n"], "pred": [round(float(x), 3) for x in v_["pred"]], "obs": [int(x) for x in v_["obs"]],
                                               "layers": {nm_: [round(float(x), 3) for x in lv_] for nm_, lv_ in v_["layers"].items()},
                                               "ll": {nm_: round(float(x), 4) for nm_, x in v_["ll"].items()},
                                               "s2": [round(float(x), 5) for x in v_["s2"].ravel()]} for r_, v_ in agg.items()}
                records[-1]["real_pa"]["groups"] = [[k_[0], k_[1], k_[2], k_[3], v_[0], [int(x) for x in v_[1]], [round(float(x), 4) for x in v_[2]], [round(float(x), 4) for x in v_[3]]]
                                                    for k_, v_ in sorted(grp_.items())]
                records[-1]["real_pa"]["bases"] = [[k_[0], k_[1], v_[0], [int(x) for x in v_[1]], [round(float(x), 4) for x in v_[2]]]
                                                   for k_, v_ in sorted(bgrp_.items())]
                if errs:
                    records[-1]["real_pa"]["errors"] = {"count": len(errs), "first": errs[0]}
            if sl is not None:
                st_out = {}
                for side in ("away", "home"):
                    sid_ = int(getattr(g, f"{side}_starter"))
                    rate_, prior_bf = prior_k_rate(sid_, date)
                    st_out[side] = {"pitcher": sid_, **{m_: v_.tolist() for m_, v_ in sl[side].items()}, "actual": act_lines.get((int(g.game_pk), sid_)),
                                    "prior_k_rate": round(float(rate_), 4), "prior_bf_365": prior_bf, "expected_bf": round(float(pexp_g.get(sid_, hazard["league_mean_bf"])), 2),
                                    "expected_bf_record": round(float(pexp.get(sid_, hazard["league_mean_bf"])), 2),
                                    "team_sim_mean": [round(float(v_ / n_sims), 3) for v_ in tot[side]], "team_actual": act_lines.get((int(g.game_pk), side)),
                                    "team_pitchers_sim_mean": round(npit[side] / n_sims, 3), "team_hr_sim_mean": round(nhr[side] / n_sims, 3)}
                    # BULLPEN-03: each bullpen candidate's share of worlds pitched in and whether he pitched (relievers who
                    # pitched without being a candidate get a share of 0)
                    used_ = act_relievers.get((int(g.game_pk), side))
                    if used_ is not None:
                        cand_ = [str(b.player_id) for b in teams[side].bullpen]
                        ids_ = cand_ + [str(x) for x in used_ if str(x) not in cand_]
                        st_out[side]["relievers"] = [[int(x) if str(x).isdigit() else x, round(rel_n[side].get(x, 0) / n_sims, 4), int(int(x) in used_ if str(x).isdigit() else 0)] for x in ids_]
                    ents_ = act_entries.get((int(g.game_pk), side))
                    if ents_ is not None:
                        real_ = {}
                        for pid_, inn_, lead_ in ents_:
                            k_ = entry_key(inn_, lead_, str(pid_) in top2_[side]); real_[k_] = real_.get(k_, 0) + 1
                        st_out[side]["entries"] = {"sim": {k_: round(v_ / n_sims, 4) for k_, v_ in sorted(ent_n[side].items())}, "real": real_}
                for side in ("away", "home"):
                    st_out[side]["keys"] = [int(x) for x in keys_n[side]]       # KEYS-02, over n worlds: first, first and won, won, sp 18+ outs, that and won
                records[-1]["starters"] = st_out
                records[-1]["innings"] = {"sim_mean": {sd_: [round(float(v_) / n_sims, 4) for v_ in inn_sum[sd_]] for sd_ in ("away", "home")},
                                          "actual_first_scores": act_innings.get(int(g.game_pk))}
            if wt is not None:
                tb = wt.table()
                tabs = [h_.table() for h_ in halves] if halves is not None else []
                st = []
                for inn, half, outs, bases, lead in real_states.get(int(g.game_pk), []):
                    key = state_index(inn, half, outs, bases, lead)
                    st.append([inn, 0 if half == "top" else 1, outs, int(sum(1 << k for k, b in enumerate(bases) if b)), lead, round(float(tb[key]), 4)]
                              + [round(float(t_[key]), 4) for t_ in tabs] + ([int(wt.n[key])] if tabs else []))
                k0 = state_index(1, "top", 0, (), 0)
                records[-1]["p_table_start"] = round(float(tb[k0]), 4)
                if tabs:
                    records[-1]["p_table_start_halves"] = [round(float(t_[k0]), 4) for t_ in tabs]
                records[-1]["states"] = st
        log(f"{date} {len(day)} games {time.time() - td:.1f}s ({di + 1}/{len(dates)}) cache hit {cached.hits / max(1, cached.hits + cached.misses):.2f}")
    log(f"done {time.time() - t0:.0f}s")
    return records
