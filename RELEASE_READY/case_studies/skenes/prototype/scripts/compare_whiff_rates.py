import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()

# Descriptions that count as a swing
swing_descriptions = [
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play"
]

# Descriptions that count as a whiff
whiff_descriptions = [
    "swinging_strike",
    "swinging_strike_blocked"
]

data["swing"] = data["description"].isin(swing_descriptions)

data["whiff"] = data["description"].isin(whiff_descriptions)

results = (
    data.groupby(["season", "pitch_type"])
    .agg(
        pitches=("pitch_type", "count"),
        swings=("swing", "sum"),
        whiffs=("whiff", "sum")
    )
    .reset_index()
)

results["whiff_percent"] = (
    results["whiffs"] / results["swings"] * 100
)

results = results.round(2)

print(results.to_string(index=False))