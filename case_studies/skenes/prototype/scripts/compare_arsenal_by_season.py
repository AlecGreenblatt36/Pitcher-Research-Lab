import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()

# Count pitches and average velocity by season and pitch type
arsenal = (
    data.groupby(["season", "pitch_type"])
    .agg(
        pitches=("pitch_type", "count"),
        avg_velocity=("release_speed", "mean")
    )
    .reset_index()
)

# Calculate total pitches thrown in each season
season_totals = (
    data.groupby("season")
    .size()
    .reset_index(name="total_pitches")
)

# Add season totals to our arsenal table
arsenal = arsenal.merge(
    season_totals,
    on="season"
)

# Calculate pitch usage percentage
arsenal["usage_percent"] = (
    arsenal["pitches"] / arsenal["total_pitches"] * 100
)

arsenal = arsenal.round(2)

print(
    arsenal[
        [
            "season",
            "pitch_type",
            "pitches",
            "usage_percent",
            "avg_velocity"
        ]
    ].to_string(index=False)
)