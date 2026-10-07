"""Reconstruct every game in the PA table: teams, park, lineups, starters, final score.

Final scores come from the last plate appearance of each game:
  * game ends on a third out  -> final = pre-PA score
  * bottom-half walk-off HR   -> home = pre_home + 1 + runners on base
  * other bottom-half ending  -> walk-off, home = pre_away + 1
Anything that does not fit those patterns is flagged ambiguous and excluded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_LIKE = {"BIP_OUT", "K"}


def load_history(path: str | Path) -> pd.DataFrame:
    h = pd.read_csv(path, low_memory=False)
    h["date_key"] = h["date_key"].astype(str)
    h["season"] = h["date_key"].str[:4].astype(int)
    return h.sort_values(["date_key", "game_pk", "at_bat_number"], kind="mergesort").reset_index(drop=True)


def final_scores(h: pd.DataFrame) -> pd.DataFrame:
    last = h.groupby("game_pk", sort=False).tail(1).set_index("game_pk")
    rows = []
    for pk, r in last.iterrows():
        ph, pa = int(r.home_score), int(r.away_score)
        top = str(r.inning_topbot).lower().startswith("top")
        out_like = r.outcome in OUT_LIKE
        runners = int(r.runner_1b) + int(r.runner_2b) + int(r.runner_3b)
        inning = int(r.inning)
        source, ok, home, away = "", True, ph, pa
        if inning < 9:
            # called game: only trust an out with no possible run scoring on the play
            source, ok = "called_game", out_like and int(r.outs_when_up) == 2
        elif top:
            if r.outcome == "HR":
                away, source = pa + 1 + runners, "top_hr_then_out"
                ok = ph > away
            elif out_like or runners == 0:
                source, ok = "top_final_out", ph > pa
            else:
                source, ok = "top_unknown_runs", False
        else:
            if ph > pa:
                source, ok = "bottom_home_leading", False
            elif r.outcome == "HR":
                home, source = ph + 1 + runners, "walkoff_hr"
                ok = home > pa
            elif ph == pa:
                # tied: any non-third-out ending is a walk-off by one run
                if out_like and int(r.outs_when_up) == 2:
                    source, ok = "tied_third_out", False
                else:
                    home, source = pa + 1, "walkoff"
                    ok = runners >= 1 or not out_like
            else:
                if out_like:
                    source = "bottom_final_out"
                elif runners >= (pa - ph) + 1:
                    home, source = pa + 1, "walkoff"
                elif runners == 0:
                    source = "bottom_final_out_after_reach"
                else:
                    source, ok = "bottom_unknown_runs", False
        rows.append({"game_pk": int(pk), "home_runs": home, "away_runs": away, "score_source": source,
                     "ambiguous": not ok, "last_inning": int(r.inning)})
    return pd.DataFrame(rows)


def batter_hands(h: pd.DataFrame) -> pd.DataFrame:
    """Per batter and date: handedness known strictly before that date (S if both sides seen)."""
    g = h.groupby(["batter", "date_key"])["stand"].agg(lambda s: "".join(sorted(set(s))))
    return g.reset_index()


def reconstruct(h: pd.DataFrame) -> pd.DataFrame:
    fs = final_scores(h).set_index("game_pk")
    games = []
    for pk, g in h.groupby("game_pk", sort=False):
        first = g.iloc[0]
        top = g[g["inning_topbot"].str.lower().str.startswith("top")]
        bot = g[~g["inning_topbot"].str.lower().str.startswith("top")]
        if top.empty or bot.empty:
            continue

        def lineup(side: pd.DataFrame) -> list[int]:
            seen: list[int] = []
            for b in side["batter"]:
                if b not in seen:
                    seen.append(int(b))
                if len(seen) == 9:
                    break
            return seen

        la, lh = lineup(top), lineup(bot)
        f = fs.loc[pk]
        games.append({
            "game_pk": int(pk), "date": first.date_key, "season": int(first.season),
            "home": first.home_team, "away": first.away_team, "park": first.park,
            "away_lineup": la, "home_lineup": lh,
            "away_starter": int(bot["pitcher"].iloc[0]), "home_starter": int(top["pitcher"].iloc[0]),
            "away_starter_throws": str(bot["p_throws"].iloc[0]), "home_starter_throws": str(top["p_throws"].iloc[0]),
            "home_runs": int(f.home_runs), "away_runs": int(f.away_runs),
            "score_source": f.score_source, "ambiguous": bool(f.ambiguous),
            "valid_lineups": len(la) == 9 and len(lh) == 9,
        })
    return pd.DataFrame(games)

