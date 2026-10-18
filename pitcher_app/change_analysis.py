"""Outing-level comparisons shared by change screening and release measurements."""

import sqlite3
import pandas as pd

METRICS = [
    ("Velocity", "release_speed", 1.0, "mph"),
    ("Spin Rate", "release_spin_rate", 1.0, "rpm"),
    ("Extension", "release_extension", 1.0, "ft"),
    ("Horizontal Release", "release_pos_x", 1.0, "ft"),
    ("Vertical Release", "release_pos_z", 1.0, "ft"),
    ("Horizontal Movement", "pfx_x", 12.0, "in"),
    ("Vertical Movement", "pfx_z", 12.0, "in"),
    ("Arm Angle", "arm_angle", 1.0, "deg"),
]
RELEASE_KEYS = {"release_extension", "release_pos_x", "release_pos_z", "arm_angle"}


def sustained_change(current, baseline_mean, baseline_std):
    current = current.sort_values(["game_date", "game_pk"]).copy()
    rolling_z = (current["value"].rolling(3).mean() - baseline_mean) / baseline_std
    for index in range(2, len(current)):
        window = rolling_z.iloc[index - 2 : index + 1]
        if window.notna().all():
            if (window <= -2).all():
                return (
                    current.iloc[index - 2]["game_date"].strftime("%Y-%m-%d"),
                    "Below baseline",
                )
            if (window >= 2).all():
                return (
                    current.iloc[index - 2]["game_date"].strftime("%Y-%m-%d"),
                    "Above baseline",
                )
    return None, None


def compare_metrics(
    database, pitcher_id, target_season, comparison, pitch=None, release=False
):
    columns = ", ".join(spec[1] for spec in METRICS)
    with sqlite3.connect(database) as connection:
        data = pd.read_sql_query(
            f"SELECT season, game_date, game_pk, pitch_type, {columns} FROM pitches "
            "WHERE game_type = 'R' AND CAST(pitcher AS INTEGER) = ? AND pitch_type IS NOT NULL",
            connection,
            params=(pitcher_id,),
        )
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    if comparison.scope == "season":
        data = data[
            pd.to_numeric(data["season"], errors="coerce").eq(target_season)
        ].copy()
    data["period"] = data["game_date"].apply(comparison.classify)
    data = data[data["period"].isin(["early", "post"])].copy()
    if pitch:
        data = data[data["pitch_type"].eq(pitch)].copy()
    results = []
    for pitch_type, pitch_data in data.groupby("pitch_type"):
        if not release and (pitch_data["period"] == "post").sum() < 20:
            continue
        for label, column, multiplier, unit in METRICS:
            if release and column not in RELEASE_KEYS:
                continue
            measured = pitch_data.assign(
                value=pd.to_numeric(pitch_data[column], errors="coerce") * multiplier
            )
            measured = measured.dropna(subset=["value", "game_date", "game_pk"])
            outings = (
                measured.groupby(["period", "game_date", "game_pk"])
                .agg(pitches=("value", "count"), value=("value", "mean"))
                .reset_index()
            )
            screened = outings[outings["pitches"] >= 5]
            baseline_screen = screened[screened["period"] == "early"]
            current_screen = screened[screened["period"] == "post"]
            std = baseline_screen["value"].std()
            eligible = (
                len(baseline_screen) >= 3
                and len(current_screen) >= 2
                and pd.notna(std)
                and std > 1e-9
            )
            if not release and not eligible:
                continue
            sample = outings if release else screened
            baseline = sample[sample["period"] == "early"]
            current = sample[sample["period"] == "post"]
            baseline_mean, current_mean = (
                baseline["value"].mean(),
                current["value"].mean(),
            )
            change = current_mean - baseline_mean
            first, direction = (
                sustained_change(current_screen, baseline_screen["value"].mean(), std)
                if eligible
                else (None, None)
            )

            def number(value):
                return round(float(value), 2) if pd.notna(value) else None

            results.append(
                {
                    "metric": f"{pitch_type} {label}",
                    "pitch_type": pitch_type,
                    "metric_key": column,
                    "unit": unit,
                    "target_season": int(target_season),
                    "baseline_seasons": sorted(
                        baseline["game_date"].dt.year.unique().astype(int).tolist()
                    ),
                    "baseline_outings": len(baseline),
                    "current_outings": len(current),
                    "baseline_pitches": int(baseline["pitches"].sum()),
                    "current_pitches": int(current["pitches"].sum()),
                    "baseline_mean": number(baseline_mean),
                    "current_mean": number(current_mean),
                    "change": number(change),
                    "z_score": (
                        number(
                            (
                                current_screen["value"].mean()
                                - baseline_screen["value"].mean()
                            )
                            / std
                        )
                        if eligible
                        else None
                    ),
                    "screen_eligible": bool(eligible),
                    "first_sustained_change": first,
                    "direction": direction,
                    "comparison_periods": comparison.payload(),
                }
            )
    results.sort(key=lambda row: abs(row["z_score"] or 0), reverse=True)
    return results
