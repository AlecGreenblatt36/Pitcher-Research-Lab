import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()

# Compare the physical characteristics of each pitch
pitch_shape = (
    data.groupby(["season", "pitch_type"])
    .agg(
        pitches=("pitch_type", "count"),
        avg_velocity=("release_speed", "mean"),
        avg_horizontal_movement=("pfx_x", "mean"),
        avg_vertical_movement=("pfx_z", "mean"),
        avg_spin_rate=("release_spin_rate", "mean"),
        avg_release_x=("release_pos_x", "mean"),
        avg_release_z=("release_pos_z", "mean"),
        avg_extension=("release_extension", "mean")
    )
    .reset_index()
)

# Convert movement from feet to inches
pitch_shape["avg_horizontal_movement"] *= 12
pitch_shape["avg_vertical_movement"] *= 12

pitch_shape = pitch_shape.round(2)

print(pitch_shape.to_string(index=False))