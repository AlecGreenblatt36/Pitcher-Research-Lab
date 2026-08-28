import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()

# A batted ball has an exit velocity measurement
data["batted_ball"] = data["launch_speed"].notna()

# Statcast considers 95+ mph hard contact
data["hard_hit"] = data["launch_speed"] >= 95

contact_quality = (
    data.groupby(["season", "pitch_type"])
    .agg(
        pitches=("pitch_type", "count"),
        batted_balls=("batted_ball", "sum"),
        avg_exit_velocity=("launch_speed", "mean"),
        hard_hit_balls=("hard_hit", "sum"),
        avg_launch_angle=("launch_angle", "mean")
    )
    .reset_index()
)

# Hard-hit percentage
contact_quality["hard_hit_percent"] = (
    contact_quality["hard_hit_balls"]
    / contact_quality["batted_balls"]
    * 100
)

contact_quality = contact_quality.round(2)

print(contact_quality.to_string(index=False))