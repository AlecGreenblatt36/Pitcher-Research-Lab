import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
regular_season = data[data["game_type"] == "R"].copy()

# Four-seam fastballs only
fastballs = regular_season[regular_season["pitch_type"] == "FF"].copy()

# Treat game_date as an actual date
fastballs["game_date"] = pd.to_datetime(fastballs["game_date"])

# Calculate velocity and vertical release position for each outing
outing_data = (
    fastballs.groupby("game_date")
    .agg(
        avg_velocity=("release_speed", "mean"),
        avg_release_z=("release_pos_z", "mean")
    )
    .reset_index()
    .sort_values("game_date")
)

outing_data = outing_data.round(2)

print(outing_data.to_string(index=False))

# Calculate correlation
correlation = outing_data["avg_velocity"].corr(
    outing_data["avg_release_z"]
)

print(
    "\nVelocity/Vertical Release Correlation:",
    round(correlation, 3)
)