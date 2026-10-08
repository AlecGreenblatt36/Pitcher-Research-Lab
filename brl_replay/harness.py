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


def bullpen(app: pd.DataFrame, team: str, cutoff: str, starter: int, rest: bool = False) -> tuple[PitcherProfile, ...]:
    """Relievers the team used in the last 14 days (its last 10 games early in a season); roles from the prior 365 days;
    availability from the three days before when rest is on (brl_live.live_feed.reliever_rest)."""
    from brl_live.live_feed import reliever_rest
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
                                  rest=reliever_rest(app, pid, cutoff) if rest else 1.0, expected_batters=exp_bf, max_batters=max(exp_bf + 3, 6)))
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
                 history_path: Path, hazard_path: Path, n_sims: int, physics_table=None, offsets=None, rest=False, environment=None,
                 team_offsets=None, age_layer=None, steals=None, win_states=False, starter_lines=False, role_offsets=None, real_pa_check=False, transitions=None, log=print) -> list[dict]:
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
    order) and record the predicted and observed counts per outcome class for starters and relievers, so the
    simulator's own totals can be told apart from the probabilities it draws from.
    transitions: optional base-running kernel after each outcome (research_lab.game_sim.transitions.EmpiricalKernel)."""
    from brl_live.provider_adjust import ContextAdjust, EnvironmentAdjust, TeamAdjust, RoleAdjust
    hcols = h[HISTORY_COLUMNS]
    first = dates[0]
    base = LockedPAModelProvider(model_path, history_path, cutoff_date=first, game_date=first, park="NYY",
                                 expected_sha256=model_sha256, physics_table=physics_table)
    cached = CachedProvider(base)
    provider = ContextAdjust(cached, offsets) if offsets else cached
    chain = [('bare', cached)] + ([('context', provider)] if offsets else [])      # each layer, for the real plate appearance check
    env = None
    if environment is not None:
        env = provider = EnvironmentAdjust(provider); chain.append(('environment', env))
    tadj = None
    if team_offsets is not None:
        tadj = provider = TeamAdjust(provider); chain.append(('team', tadj))
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
    act_lines, per_k = {}, {}
    if starter_lines:
        want_g = set(int(x) for x in games[games["date"].isin(dates)]["game_pk"])
        sub = h.loc[h["game_pk"].isin(want_g), ["game_pk", "pitcher", "outcome", "inning_topbot"]]
        sub_side = np.where(sub["inning_topbot"].astype(str).str.lower().str.startswith("top"), "home", "away")
        for (gpk_, side_), oc in sub.groupby([sub["game_pk"].to_numpy(), sub_side])["outcome"]:
            vc = oc.value_counts()
            act_lines[(int(gpk_), str(side_))] = {"bf": int(len(oc)), "k": int(vc.get("K", 0)), "bb": int(vc.get("BB_HBP", 0)),
                                                  "h": int(vc.get("1B", 0) + vc.get("2B_3B", 0) + vc.get("HR", 0)), "hr": int(vc.get("HR", 0))}
        for (gpk_, pid_), oc in sub.groupby(["game_pk", "pitcher"])["outcome"]:
            vc = oc.value_counts()
            act_lines[(int(gpk_), int(pid_))] = {"bf": int(len(oc)), "k": int(vc.get("K", 0)), "bb": int(vc.get("BB_HBP", 0)),
                                                 "h": int(vc.get("1B", 0) + vc.get("2B_3B", 0) + vc.get("HR", 0)), "hr": int(vc.get("HR", 0))}
        hk = pd.DataFrame({"pitcher": h["pitcher"].to_numpy(), "date_key": h["date_key"].astype(str).str[:10].to_numpy(), "k": (h["outcome"] == "K").to_numpy().astype(int)})
        daily = hk.groupby(["pitcher", "date_key"]).agg(bf=("k", "size"), k=("k", "sum")).reset_index()
        daily["ord"] = [pd.Timestamp(x).toordinal() for x in daily["date_key"]]
        for pid_, gq in daily.groupby("pitcher"):
            per_k[int(pid_)] = (gq["ord"].to_numpy(), np.cumsum(gq["bf"].to_numpy()), np.cumsum(gq["k"].to_numpy()))

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
        defense = None
        if base.physics is not None:
            from research_lab.pa_model.physics import PhysicsState, DefenseState
            params = dict(base.physics.sums.p)
            base.physics = PhysicsState.build(physics_table, date, params).with_history(hcols)
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
            policy = FittedStarterPolicy(hazard, pexp, texp, {int(g.away_starter): g.away, int(g.home_starter): g.home})
            steal = None
            if steals is not None:
                from brl_live.running import steal_model
                steal = steal_model(int(str(date)[:4]) - 1, **{k: float(v) for k, v in steals.items() if k in ('per_pa', 'third')})
            sim = GameSimulator(provider, manager_policy=policy, steals=steal, transitions=transitions)
            rng = np.random.default_rng(int(g.game_pk))
            seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_sims, dtype=np.int64)
            hw = ties = 0
            ha = np.zeros(MAX_RUNS + 1, int); aa = np.zeros(MAX_RUNS + 1, int)
            s_outs = {"away": [], "home": []}
            sl = {side: {m_: np.zeros(n_, int) for m_, n_ in (("k", 21), ("bf", 46), ("h", 21), ("bb", 16), ("outs", 28))} for side in ("away", "home")} if starter_lines else None
            tot = {side: np.zeros(4) for side in ("away", "home")} if starter_lines else None   # team pitching: K, BF, BB+HBP, hits
            wt = WinTable() if win_states else None
            halves = (WinTable(), WinTable()) if win_states == "split" else None
            for si, s in enumerate(seeds):
                r = sim.simulate(matchup, int(s), record_events=bool(win_states))
                if wt is not None:
                    box = {"score": {"home": r.home_score, "away": r.away_score},
                           "plays": [e for e in r.events if e.get("outcome") not in ("stolen_base", "caught_stealing")]}
                    wt.add(box)
                    if halves is not None:
                        halves[si % 2].add(box)
                ha[min(r.home_score, MAX_RUNS)] += 1
                aa[min(r.away_score, MAX_RUNS)] += 1
                if r.winner == "home":
                    hw += 1
                elif r.winner not in ("away", "home"):
                    ties += 1
                for side in ("away", "home"):
                    line = r.pitcher_lines.get(teams[side].starter.player_id)
                    s_outs[side].append(int(line["outs_recorded"]) if line else 0)
                    if sl is not None and line:
                        for m_, f_ in (("k", "strikeouts"), ("bf", "batters_faced"), ("h", "hits_allowed"), ("bb", "walks_hbp"), ("outs", "outs_recorded")):
                            arr = sl[side][m_]; arr[min(int(line[f_]), len(arr) - 1)] += 1
                if tot is not None:
                    for ln in r.pitcher_lines.values():
                        sd_ = ln.get("team_side")
                        if sd_ in tot:
                            tot[sd_] += (ln["strikeouts"], ln["batters_faced"], ln["walks_hbp"], ln["hits_allowed"])
            records.append({"game_pk": int(g.game_pk), "date": date, "home": g.home, "away": g.away, "n": n_sims,
                            "home_wins": hw, "ties": ties, "home_hist": ha.tolist(), "away_hist": aa.tolist(),
                            "home_starter_outs": float(np.mean(s_outs["home"])) if s_outs["home"] else 0.0, "away_starter_outs": float(np.mean(s_outs["away"])) if s_outs["away"] else 0.0,
                            "home_runs": int(g.home_runs), "away_runs": int(g.away_runs)})
            if real_pa_check:
                from types import SimpleNamespace
                from research_lab.game_sim.locked_pa_provider import SIM_FROM_MODEL
                model_of_sim = {v: k for k, v in SIM_FROM_MODEL.items()}
                agg = {r_: {"pred": np.zeros(7), "obs": np.zeros(7), "n": 0, "layers": {nm_: np.zeros(7) for nm_, _ in chain[:-1]}} for r_ in ("starter", "reliever")}
                starters_ = {int(g.home_starter), int(g.away_starter)}
                grp_ = {}      # (batting side 0 away 1 home, inning bucket, role, times through the order capped at 3): n, observed, bare, full
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
                    agg[r_]["pred"] += np.array([probs[model_of_sim[lab]] for lab in MODEL_ORDER])
                    for nm_, pr_ in inner_:
                        agg[r_]["layers"][nm_] += np.array([pr_[model_of_sim[lab]] for lab in MODEL_ORDER])
                    if str(row.outcome) in MODEL_ORDER:
                        agg[r_]["obs"][MODEL_ORDER.index(str(row.outcome))] += 1
                    agg[r_]["n"] += 1
                    inn_ = int(row.inning)
                    gk_ = (int(side_b == "home"), "1st" if inn_ == 1 else ("mid" if inn_ <= 8 else "late"), r_[0], min(max(int(row.n_thruorder_pitcher or 1), 1), 3))
                    ge_ = grp_.setdefault(gk_, [0, np.zeros(7), np.zeros(7), np.zeros(7)])
                    ge_[0] += 1
                    if str(row.outcome) in MODEL_ORDER:
                        ge_[1][MODEL_ORDER.index(str(row.outcome))] += 1
                    ge_[2] += np.array([inner_[0][1][model_of_sim[lab]] for lab in MODEL_ORDER]) if inner_ else np.array([probs[model_of_sim[lab]] for lab in MODEL_ORDER])
                    ge_[3] += np.array([probs[model_of_sim[lab]] for lab in MODEL_ORDER])
                errs = agg.pop("errors", [])
                records[-1]["real_pa"] = {r_: {"n": v_["n"], "pred": [round(float(x), 3) for x in v_["pred"]], "obs": [int(x) for x in v_["obs"]],
                                               "layers": {nm_: [round(float(x), 3) for x in lv_] for nm_, lv_ in v_["layers"].items()}} for r_, v_ in agg.items()}
                records[-1]["real_pa"]["groups"] = [[k_[0], k_[1], k_[2], k_[3], v_[0], [int(x) for x in v_[1]], [round(float(x), 4) for x in v_[2]], [round(float(x), 4) for x in v_[3]]]
                                                    for k_, v_ in sorted(grp_.items())]
                if errs:
                    records[-1]["real_pa"]["errors"] = {"count": len(errs), "first": errs[0]}
            if sl is not None:
                st_out = {}
                for side in ("away", "home"):
                    sid_ = int(getattr(g, f"{side}_starter"))
                    rate_, prior_bf = prior_k_rate(sid_, date)
                    st_out[side] = {"pitcher": sid_, **{m_: v_.tolist() for m_, v_ in sl[side].items()}, "actual": act_lines.get((int(g.game_pk), sid_)),
                                    "prior_k_rate": round(float(rate_), 4), "prior_bf_365": prior_bf, "expected_bf": round(float(pexp.get(sid_, hazard["league_mean_bf"])), 2),
                                    "team_sim_mean": [round(float(v_ / n_sims), 3) for v_ in tot[side]], "team_actual": act_lines.get((int(g.game_pk), side))}
                records[-1]["starters"] = st_out
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
