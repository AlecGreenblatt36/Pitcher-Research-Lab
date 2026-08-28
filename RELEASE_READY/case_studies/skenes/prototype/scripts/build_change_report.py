import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()


# -----------------------------
# 2024-2025 historical baseline
# -----------------------------

baseline_data = data[data["season"].isin([2024, 2025])].copy()

baseline_total_pitches = len(baseline_data)

baseline = (
    baseline_data.groupby("pitch_type")
    .agg(
        baseline_pitches=("pitch_type", "count"),
        baseline_velocity=("release_speed", "mean"),
        baseline_horizontal_movement=("pfx_x", "mean"),
        baseline_vertical_movement=("pfx_z", "mean"),
        baseline_spin=("release_spin_rate", "mean"),
        baseline_release_x=("release_pos_x", "mean"),
        baseline_release_z=("release_pos_z", "mean"),
        baseline_extension=("release_extension", "mean")
    )
    .reset_index()
)

baseline["baseline_usage"] = (
    baseline["baseline_pitches"] / baseline_total_pitches * 100
)


# -----------------------------
# 2026 current version
# -----------------------------

current_data = data[data["season"] == 2026].copy()

current_total_pitches = len(current_data)

current = (
    current_data.groupby("pitch_type")
    .agg(
        current_pitches=("pitch_type", "count"),
        current_velocity=("release_speed", "mean"),
        current_horizontal_movement=("pfx_x", "mean"),
        current_vertical_movement=("pfx_z", "mean"),
        current_spin=("release_spin_rate", "mean"),
        current_release_x=("release_pos_x", "mean"),
        current_release_z=("release_pos_z", "mean"),
        current_extension=("release_extension", "mean")
    )
    .reset_index()
)

current["current_usage"] = (
    current["current_pitches"] / current_total_pitches * 100
)


# -----------------------------
# Combine baseline and 2026
# -----------------------------

report = baseline.merge(
    current,
    on="pitch_type",
    how="inner"
)


# -----------------------------
# Calculate changes
# -----------------------------

report["velocity_change"] = (
    report["current_velocity"] - report["baseline_velocity"]
)

report["horizontal_movement_change"] = (
    report["current_horizontal_movement"]
    - report["baseline_horizontal_movement"]
) * 12

report["vertical_movement_change"] = (
    report["current_vertical_movement"]
    - report["baseline_vertical_movement"]
) * 12

report["spin_change"] = (
    report["current_spin"] - report["baseline_spin"]
)

report["release_x_change"] = (
    report["current_release_x"] - report["baseline_release_x"]
) * 12

report["release_z_change"] = (
    report["current_release_z"] - report["baseline_release_z"]
) * 12

report["extension_change"] = (
    report["current_extension"] - report["baseline_extension"]
) * 12

report["usage_change"] = (
    report["current_usage"] - report["baseline_usage"]
)


# Round numbers
report = report.round(2)


# Keep the change columns we care about
change_report = report[
    [
        "pitch_type",
        "baseline_pitches",
        "current_pitches",
        "velocity_change",
        "horizontal_movement_change",
        "vertical_movement_change",
        "spin_change",
        "release_x_change",
        "release_z_change",
        "extension_change",
        "usage_change"
    ]
]

print(change_report.to_string(index=False))