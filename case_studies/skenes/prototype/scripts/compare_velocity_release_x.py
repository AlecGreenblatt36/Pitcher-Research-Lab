import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2026_raw.csv"

data = pd.read_csv(data_file)

regular_season = data[data["game_type"] == "R"].copy()

fastballs = regular_season[
    regular_season["pitch_type"] == "FF"
].copy()

fastballs["game_date"] = pd.to_datetime(fastballs["game_date"])

outing_data = (
    fastballs.groupby("game_date")
    .agg(
        avg_velocity=("release_speed", "mean"),
        avg_release_x=("release_pos_x", "mean")
    )
    .reset_index()
    .sort_values("game_date")
)

outing_data = outing_data.round(2)

print(outing_data.to_string(index=False))

correlation = outing_data["avg_velocity"].corr(
    outing_data["avg_release_x"]
)

print(
    "\nVelocity/Horizontal Release Correlation:",
    round(correlation, 3)
)