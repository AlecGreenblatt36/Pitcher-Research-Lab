"""Which reliever comes in: a conditional logit fitted on every real relief entry (BULLPEN-01).

At each pitching change the candidates are the bullpen's arms not yet used in the game. Each candidate's utility is a
linear function of his recent use (share of the team's relief batters over 14 and 30 days, appearances yesterday and
over the last two and three days, days since his last outing), his role over the last year (share of entries in the
ninth and in the eighth or later, median batters per outing), his strikeout record relative to the other candidates,
the platoon matchup with the next batter, and the game situation (save spot, tied late, close late, early, blowout,
extra innings, entering with runners on or outs made). The reliever is drawn from the softmax of the utilities.

doc (brl_live/reliever_choice.json, from the fit): {'name': ..., 'names': [feature, ...], 'beta': [...]}.
Candidate facts come from PitcherProfile.usage (brl_live.live_feed.bullpen_usage); a candidate without them gets the
values of an unknown arm (no recent use, league strikeout rate).
"""
from __future__ import annotations

import math

import numpy as np

LEAGUE_K = 0.235
K_PRIOR_BF = 100.0
DEFAULT = {"share14": 0.0, "share30": 0.0, "pitched_d1": 0.0, "apps_d2": 0.0, "apps_d3": 0.0, "days_since": 99.0,
           "ninth": 0.0, "late": 0.0, "med_bf": 4.0, "k365": 0.0, "bf365": 0.0, "apps21": 0.0, "ninth21": 0.0, "late21": 0.0}


def situation(inning: int, lead: int, outs: int, runners: int) -> dict:
    """Game-situation indicators at the reliever's entry; lead is the fielding team's."""
    return {
        "save": float(inning >= 9 and 1 <= lead <= 3),
        "tie_late": float(inning >= 9 and lead == 0),
        "late_close": float(inning >= 7 and abs(lead) <= 2),
        "early": float(inning <= 5),
        "blowout": float(abs(lead) >= 5),
        "before8": float(inning < 8),
        "mid_inning": float(outs > 0 or runners > 0),
        "extra": float(inning >= 10),
    }


def features(facts: list[dict], same_hand: list[bool], sit: dict, same3=None) -> dict[str, np.ndarray]:
    """Feature columns for a choice set (one row per candidate), as in the fit. same_hand: the candidate throws with the
    hand the next hitter bats with; same3: how many of the next three hitters bat with the candidate's hand (switch
    hitters never count)."""
    f = {k: np.asarray([float(x.get(k, DEFAULT[k])) for x in facts], float) for k in DEFAULT}
    s3 = np.asarray(same3 if same3 is not None else [float(x) for x in same_hand], float)
    ls14, ls30 = np.log(f["share14"] + 0.01), np.log(f["share30"] + 0.01)
    medbf = np.minimum(f["med_bf"], 12.0)
    same = np.asarray(same_hand, float)
    q = (f["k365"] + K_PRIOR_BF * LEAGUE_K) / (f["bf365"] + K_PRIOR_BF)
    q_rel = (q - q.mean()) / 0.03
    s = sit
    return {
        "log_share14": ls14, "log_share30": ls30,
        "pitched_d1": (f["pitched_d1"] > 0).astype(float), "back_to_back": (f["apps_d2"] >= 2).astype(float),
        "apps_d3": f["apps_d3"], "days_since": np.minimum(f["days_since"], 14.0) / 14.0,
        "same_hand": same,
        "ninth_x_save": f["ninth"] * s["save"], "ninth_x_tie_late": f["ninth"] * s["tie_late"], "ninth_x_before8": f["ninth"] * s["before8"],
        "late_x_late_close": f["late"] * s["late_close"], "late_x_early": f["late"] * s["early"], "late_x_blowout": f["late"] * s["blowout"],
        "share_x_late_close": ls14 * s["late_close"], "share_x_blowout": ls14 * s["blowout"],
        "medbf_x_early": medbf * s["early"], "medbf_x_blowout": medbf * s["blowout"], "medbf_x_mid": medbf * s["mid_inning"],
        "same_hand_x_mid": same * s["mid_inning"],
        "quality": q_rel, "quality_x_late_close": q_rel * s["late_close"], "quality_x_save": q_rel * s["save"],
        "quality_x_blowout": q_rel * s["blowout"], "ninth_x_extra": f["ninth"] * s["extra"],
        "same3": s3, "same3_x_late": s3 * float(s["before8"] == 0.0), "same3_x_mid": s3 * s["mid_inning"],
        "ninth21_x_save": f["ninth21"] * s["save"], "late21_x_late_close": f["late21"] * s["late_close"],
    }


class RelieverChoice:
    def __init__(self, doc: dict):
        self.name = str(doc.get("name") or "reliever choice")
        self.names = [str(n) for n in doc["names"]]
        self.beta = np.asarray(doc["beta"], float)
        if len(self.names) != len(self.beta):
            raise ValueError("reliever choice: names and coefficients differ in length")

    def probabilities(self, candidates, inning: int, lead: int, outs: int, runners: int, batter_hand: str,
                      upcoming_hands=None) -> np.ndarray:
        """Choice probabilities over `candidates`; upcoming_hands lists how the next three hitters bat ('L', 'R', 'S'),
        the first being the next batter."""
        facts = [dict(getattr(p, "usage", ()) or ()) for p in candidates]
        hand = (batter_hand or "R").upper()[:1]
        same = [hand != "S" and p.throws == hand for p in candidates]
        hands = [str(h or "R").upper()[:1] for h in (upcoming_hands or [hand])]
        same3 = [float(sum(h != "S" and p.throws == h for h in hands)) for p in candidates]
        cols = features(facts, same, situation(inning, lead, outs, runners), same3)
        u = np.zeros(len(candidates))
        for name, b in zip(self.names, self.beta):
            if name in cols:
                u += b * cols[name]
        u -= u.max()
        e = np.exp(u)
        return e / e.sum()

    def select(self, candidates, state, fielding_side, next_batter, rng, upcoming=None):
        """The entering reliever. At an inning's end (three outs on the board) he takes over at the start of the next
        inning; otherwise he enters with the current outs and runners. upcoming: the next three hitters (next first)."""
        lead = int(state.score_for(fielding_side) - state.opponent_score_for(fielding_side))
        if state.outs >= 3:
            inning, outs, runners = state.inning + 1, 0, int(state.inning + 1 >= 10)
        else:
            inning, outs, runners = state.inning, state.outs, sum(r is not None for r in state.bases)
        hands = [getattr(b, "bats", "R") for b in upcoming] if upcoming else None
        p = self.probabilities(candidates, inning, lead, outs, runners, getattr(next_batter, "bats", "R"), hands)
        return candidates[int(rng.choice(len(candidates), p=p))]
