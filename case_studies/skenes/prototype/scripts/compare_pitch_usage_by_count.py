import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()

# Create the count as text: 0-0, 1-2, 3-1, etc.
data["count"] = (
    data["balls"].astype(str)
    + "-"
    + data["strikes"].astype(str)
)

# Count ALL pitches thrown in each count
count_totals = (
    data.groupby(["season", "count"])
    .size()
    .reset_index(name="total_pitches")
)

# Focus on the four pitches we are investigating
focus_pitches = data[
    data["pitch_type"].isin(["FF", "CH", "FS", "SI"])
].copy()

# Count how many of each pitch he threw in each count
count_usage = (
    focus_pitches.groupby(["season", "count", "pitch_type"])
    .size()
    .reset_index(name="pitches")
)

# Add the total number of ALL pitches thrown in that count
count_usage = count_usage.merge(
    count_totals,
    on=["season", "count"]
)

# Calculate true usage percentage
count_usage["usage_percent"] = (
    count_usage["pitches"]
    / count_usage["total_pitches"]
    * 100
)

count_usage = count_usage.round(1)

print(
    count_usage[
        [
            "season",
            "count",
            "pitch_type",
            "pitches",
            "usage_percent"
        ]
    ].to_string(index=False)
)