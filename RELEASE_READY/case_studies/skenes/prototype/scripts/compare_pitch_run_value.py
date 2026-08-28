import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()

# Make sure run expectancy is numeric
data["delta_run_exp"] = pd.to_numeric(
    data["delta_run_exp"],
    errors="coerce"
)

# Remove pitches without run expectancy information
data = data.dropna(subset=["delta_run_exp"])

# Savant's delta_run_exp is from the offense's perspective.
# Flip the sign so positive = good for the pitcher.
data["pitcher_run_value"] = -data["delta_run_exp"]

run_value = (
    data.groupby(["season", "pitch_type"])
    .agg(
        pitches=("pitch_type", "count"),
        total_run_value=("pitcher_run_value", "sum")
    )
    .reset_index()
)

# Put everything on the same 100-pitch scale
run_value["run_value_per_100"] = (
    run_value["total_run_value"]
    / run_value["pitches"]
    * 100
)

run_value = run_value.round(2)

print(run_value.to_string(index=False))