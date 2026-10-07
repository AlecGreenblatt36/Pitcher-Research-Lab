"""Projected-game selection across all simulated worlds (replaces "modal score, first match").

v1 took the most common final score and then the FIRST world with that score, so the
"most typical" game could be a 15-inning marathon with a 136-pitch starter. v2 scores every
world on how close it sits to the middle of all worlds and picks:

  projected  favorite wins, nine innings, smallest distance to the medians
  high       nine innings, total runs at or above the 85th percentile, then most central
  low        nine innings, total runs at or below the 15th percentile, then most central
  upset      underdog wins, nine innings, most central

Distance = sum over features of |x - median| / IQR-based scale, using team runs, team hits,
each starter's outs, strikeouts and pitches. Deterministic (ties broken by world index).

Used by brl_live.boxscore.BoxAccumulator; `runners_up` lists the next most central worlds so
the saved sample count can always reach five.
"""
from __future__ import annotations

import numpy as np

SIDE = ("away", "home")
SELECTION_NOTE = ("projected: favorite wins, nine innings, closest to the median of team runs, team hits and "
                  "both starters' outs, strikeouts and pitches; then most central high-scoring, low-scoring and upset worlds")


def world_features(box: dict) -> dict:
    innings = max([int(k) for s in SIDE for k in box["innings"][s].keys()] or [9])
    f = {"innings": innings, "away": box["score"]["away"], "home": box["score"]["home"]}
    for s in SIDE:
        f[f"{s}_hits"] = sum(int(v.get("H", 0)) for v in box["innings"][s].values())
        st = (box["pitching"][s] or [{}])[0]
        f[f"{s}_outs"] = int(st.get("outs", 0))
        f[f"{s}_K"] = int(st.get("K", 0))
        f[f"{s}_PC"] = int(st.get("PC", 0))
    return f


KEYS = ("away", "home", "away_hits", "home_hits", "away_outs", "home_outs", "away_K", "home_K", "away_PC", "home_PC")


def select_worlds(features: list[dict], score_pairs: list[tuple[int, int]] | None = None) -> dict:
    X = np.array([[f[k] for k in KEYS] for f in features], dtype=float)
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    scale = np.where(iqr > 0, iqr / 1.349, 1.0)
    dist = (np.abs(X - med) / scale).sum(axis=1)
    away, home = X[:, 0], X[:, 1]
    nine = np.array([f["innings"] == 9 for f in features])
    home_wins = home > away
    fav_home = home_wins.mean() >= 0.5
    fav_wins = home_wins if fav_home else ~home_wins
    total = away + home
    hi_cut, lo_cut = np.percentile(total, 85), np.percentile(total, 15)

    def best(mask):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return None
        return int(idx[np.lexsort((idx, dist[idx]))[0]])

    picks = {
        "projected": best(nine & fav_wins) if best(nine & fav_wins) is not None else best(fav_wins),
        "high": best(nine & (total >= hi_cut)),
        "low": best(nine & (total <= lo_cut)),
        "upset": best(nine & ~fav_wins),
    }
    seen, out = set(), {}
    for k, v in picks.items():
        out[k] = v if v is not None and v not in seen else None
        if v is not None:
            seen.add(v)
    order = np.lexsort((np.arange(len(dist)), dist))
    out["runners_up"] = [int(i) for i in order if int(i) not in seen][:8]
    return out


if __name__ == "__main__":
    # self-test with random worlds
    rng = np.random.default_rng(1)
    feats = []
    for i in range(10000):
        inn = 9 if rng.random() > 0.09 else int(rng.integers(10, 16))
        a, h = int(rng.poisson(4.1)), int(rng.poisson(4.5))
        if a == h:
            h += 1
        feats.append({"innings": inn, "away": a, "home": h, "away_hits": a * 2, "home_hits": h * 2,
                      "away_outs": int(rng.normal(16, 3)), "home_outs": int(rng.normal(16, 3)),
                      "away_K": int(rng.poisson(5)), "home_K": int(rng.poisson(6)),
                      "away_PC": int(rng.normal(88, 12)), "home_PC": int(rng.normal(90, 12))})
    picks = select_worlds(feats)
    for k, v in picks.items():
        f = feats[v]
        print(k, v, f"innings {f['innings']}  score {f['away']}-{f['home']}  starter outs {f['away_outs']}/{f['home_outs']}  PC {f['away_PC']}/{f['home_PC']}")
        assert f["innings"] == 9
    assert feats[picks["projected"]]["home"] > feats[picks["projected"]]["away"]
    assert feats[picks["upset"]]["home"] < feats[picks["upset"]]["away"]
    print("self-test passed")
