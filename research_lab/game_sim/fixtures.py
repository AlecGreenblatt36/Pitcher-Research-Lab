"""Official-game fixtures built only from real IDs and pre-cutoff history.

Padres at Brewers, gamePk 849830, 2026-10-03, American Family Field.
Lineups/starters are as recorded from the official MLB feed in the
ChatGPT handoff (CLAUDE_MASTER_HANDOFF section 13); this session had no
network access to statsapi.mlb.com to re-verify them.

Bullpen candidates are NOT an official roster. They are pitchers who
relieved for each club in the final 14 days of history before the cutoff,
with workload taken from that history. Postseason games after the last
history date (2026-09-27) are missing, so rest is unknown and assumed full.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import GameMatchup, PitcherProfile, PlayerProfile, TeamProfile

GAME_PK = 849830
GAME_DATE = "2026-10-03"
VENUE = "American Family Field"
PARK = "MIL"
LINEUPS = {
    "SD": [("Fernando Tatis Jr.", "R", "RF", 665487), ("Dustin Harris", "L", "LF", 687957),
           ("Manny Machado", "R", "3B", 592518), ("Ty France", "R", "1B", 664034),
           ("Jackson Merrill", "L", "CF", 701538), ("Xander Bogaerts", "R", "SS", 593428),
           ("Ethan Salas", "L", "C", 806956), ("Luis Campusano", "R", "DH", 669134),
           ("Jake Cronenworth", "L", "2B", 630105)],
    "MIL": [("Jackson Chourio", "R", "LF", 694192), ("Brice Turang", "L", "2B", 668930),
            ("William Contreras", "R", "C", 661388), ("Jake Bauers", "L", "1B", 641343),
            ("Christian Yelich", "L", "DH", 592885), ("Cooper Pratt", "R", "SS", 806198),
            ("Luis Lara", "S", "RF", 800325), ("Garrett Mitchell", "L", "CF", 669003),
            ("Joey Ortiz", "R", "3B", 687401)],
}
STARTERS = {"SD": ("Robbie Ray", "L", 592662), "MIL": ("Jacob Misiorowski", "R", 694819)}
TEAM_NAMES = {"SD": "San Diego Padres", "MIL": "Milwaukee Brewers"}


def load_names(path: Path) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    reg = pd.read_csv(path)
    names = {int(r.key_mlbam): f"{r.name_first} {r.name_last}" for r in reg.itertuples()}
    births = {
        int(r.key_mlbam): (int(r.birth_year), int(r.birth_month), int(r.birth_day))
        for r in reg.dropna(subset=["birth_year", "birth_month", "birth_day"]).itertuples()
    }
    return names, births


def _appearances(history: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    h = history.loc[history["date_key"] < cutoff].copy()
    h["fielding_team"] = np.where(h["inning_topbot"].str.lower().str.startswith("top"), h["home_team"], h["away_team"])
    first_inning = h.groupby(["game_pk", "pitcher"])["inning"].transform("min")
    game_first = h.groupby(["game_pk", "fielding_team"])["at_bat_number"].transform("min")
    h["is_start"] = h.groupby(["game_pk", "pitcher"])["at_bat_number"].transform("min") == game_first
    app = h.groupby(["game_pk", "pitcher"]).agg(
        date=("date_key", "first"), team=("fielding_team", "first"), bf=("at_bat_number", "size"),
        entry_inning=("inning", "min"), throws=("p_throws", "first"), start=("is_start", "first"),
    ).reset_index()
    return app


def build_matchup(history: pd.DataFrame, cutoff: str, names: dict[int, str]) -> tuple[GameMatchup, dict]:
    app = _appearances(history, cutoff)
    season = app[app["date"] >= cutoff[:4]]
    window_start = (pd.Timestamp(cutoff) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    notes: dict = {"bullpen_source": f"relievers used {window_start}..{cutoff} (pre-cutoff history), not official roster"}

    def starter_profile(team: str) -> PitcherProfile:
        name, throws, pid = STARTERS[team]
        starts = season[(season["pitcher"] == pid) & season["start"]]["bf"]
        exp_bf = int(round(starts.median())) if len(starts) else 22
        max_bf = int(round(starts.quantile(0.95))) if len(starts) >= 5 else 27
        notes[f"{team}_starter_bf"] = {"starts_2026": int(len(starts)), "median_bf": exp_bf, "p95_bf": max_bf}
        return PitcherProfile(str(pid), name, throws, role="starter", stamina=0.7,
                              expected_batters=exp_bf, max_batters=max(max_bf, exp_bf + 2))

    def bullpen(team: str, starter_id: int) -> tuple[PitcherProfile, ...]:
        recent = app[(app["team"] == team) & (app["date"] >= window_start) & (~app["start"])]
        recent = recent[recent["pitcher"] != starter_id]
        out = []
        for pid, g in recent.groupby("pitcher"):
            season_g = season[(season["pitcher"] == pid) & (~season["start"])]
            late = float((season_g["entry_inning"] >= 8).mean()) if len(season_g) else 0.0
            ninth = float((season_g["entry_inning"] >= 9).mean()) if len(season_g) else 0.0
            exp_bf = int(max(3, round(season_g["bf"].median()))) if len(season_g) else 4
            role = "closer" if ninth >= 0.6 else "setup" if late >= 0.5 else "long" if exp_bf >= 7 else "reliever"
            out.append(PitcherProfile(str(pid), names.get(int(pid), f"MLBAM {pid}"), str(g["throws"].iloc[0]),
                                      role=role, leverage=min(1.0, 0.3 + late), rest=1.0,
                                      expected_batters=exp_bf, max_batters=max(exp_bf + 3, 6)))
        # one closer max: keep the most 9th-inning-heavy as closer, demote others to setup
        closers = [p for p in out if p.role == "closer"]
        if len(closers) > 1:
            keep = max(closers, key=lambda p: p.leverage)
            out = [p if p.role != "closer" or p is keep else PitcherProfile(
                p.player_id, p.name, p.throws, role="setup", leverage=p.leverage, rest=p.rest,
                expected_batters=p.expected_batters, max_batters=p.max_batters) for p in out]
        notes[f"{team}_bullpen"] = [{"id": p.player_id, "name": p.name, "throws": p.throws, "role": p.role,
                                     "exp_bf": p.expected_batters} for p in out]
        return tuple(out)

    teams = {}
    for team in ("SD", "MIL"):
        lineup = tuple(PlayerProfile(str(pid), name, bats) for name, bats, _, pid in LINEUPS[team])
        starter = starter_profile(team)
        teams[team] = TeamProfile(team, TEAM_NAMES[team], lineup, starter, bullpen(team, int(starter.player_id)))
    return GameMatchup(away=teams["SD"], home=teams["MIL"], venue=VENUE, game_type="P"), notes
