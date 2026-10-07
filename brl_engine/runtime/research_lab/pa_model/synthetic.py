from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PAConfig


def make_synthetic_pa(rows_per_year: int = 5000, years: tuple[int, ...] = (2023, 2024, 2025), seed: int = 36) -> pd.DataFrame:
    labels = list(PAConfig().outcome_labels)
    rng = np.random.default_rng(seed)
    batter_count, pitcher_count = 180, 150
    base = np.array([0.49, 0.22, 0.09, 0.10, 0.035, 0.03, 0.035])
    batter_effect = rng.normal(0, 0.22, size=(batter_count, len(labels))); batter_effect -= batter_effect.mean(axis=1, keepdims=True)
    pitcher_effect = rng.normal(0, 0.20, size=(pitcher_count, len(labels))); pitcher_effect -= pitcher_effect.mean(axis=1, keepdims=True)
    records, game_pk = [], 100000
    for year in years:
        dates = pd.date_range(f"{year}-03-28", f"{year}-09-29", freq="D")
        for i in range(rows_per_year):
            batter, pitcher = int(rng.integers(batter_count)), int(rng.integers(pitcher_count))
            stand, throws = ("L" if rng.random() < 0.35 else "R"), ("L" if rng.random() < 0.28 else "R")
            platoon = int(stand == throws)
            runners = rng.random(3) < np.array([0.28, 0.18, 0.10])
            logits = np.log(base) + batter_effect[batter] + pitcher_effect[pitcher]
            logits[1] -= 0.12 * platoon; logits[2] += 0.08 * platoon; logits[5] += 0.08 * int(runners.any())
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
            outcome = labels[int(rng.choice(len(labels), p=probs))]
            if i % 75 == 0: game_pk += 1
            date = dates[i % len(dates)]
            records.append({"game_date": date, "date_key": date.normalize(), "season": year, "game_pk": game_pk, "at_bat_number": i % 75 + 1, "pitch_number": 1, "batter": 500000 + batter, "pitcher": 600000 + pitcher, "terminal_event": outcome, "outcome": outcome, "stand": stand, "p_throws": throws, "home_team": f"T{int(rng.integers(1,31)):02d}", "away_team": f"T{int(rng.integers(1,31)):02d}", "park": f"T{int(rng.integers(1,31)):02d}", "inning": int(rng.integers(1,10)), "inning_topbot": "Bot" if rng.random() < .5 else "Top", "outs_when_up": int(rng.integers(0,3)), "runner_1b": int(runners[0]), "runner_2b": int(runners[1]), "runner_3b": int(runners[2]), "on_1b": 1 if runners[0] else np.nan, "on_2b": 1 if runners[1] else np.nan, "on_3b": 1 if runners[2] else np.nan, "bat_score_diff": int(rng.integers(-5,6)), "n_thruorder_pitcher": int(rng.integers(1,4)), "batter_days_since_prev_game": int(rng.integers(0,8)), "pitcher_days_since_prev_game": int(rng.integers(0,8)), "age_bat": float(rng.normal(28,3)), "age_pit": float(rng.normal(28,3)), "platoon": platoon, "is_home_batter": int(rng.random() < .5), "matchup_key": f"{batter}_{pitcher}"})
    return pd.DataFrame(records).sort_values(["date_key", "game_pk", "at_bat_number"], kind="mergesort").reset_index(drop=True)
