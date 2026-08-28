import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season four-seam fastballs only
fastballs = data[
    (data["game_type"] == "R") &
    (data["pitch_type"] == "FF")
].copy()

# Average four-seam velocity for each outing
outing_velocity = (
    fastballs.groupby(["season", "game_date"])
    .agg(
        avg_velocity=("release_speed", "mean"),
        fastballs_thrown=("release_speed", "count")
    )
    .reset_index()
)

# Historical baseline = 2024 and 2025 outings
baseline = outing_velocity[
    outing_velocity["season"].isin([2024, 2025])
]

# Current = 2026 outings
current = outing_velocity[
    outing_velocity["season"] == 2026
]

# Historical average outing velocity
baseline_mean = baseline["avg_velocity"].mean()

# How much Skenes normally varies from outing to outing
baseline_std = baseline["avg_velocity"].std()

# Average outing velocity in 2026
current_mean = current["avg_velocity"].mean()

# Difference from historical baseline
velocity_change = current_mean - baseline_mean

# Standardized deviation
z_score = (
    current_mean - baseline_mean
) / baseline_std

print("2024-25 baseline velocity:", round(baseline_mean, 2), "mph")
print("Historical outing variation:", round(baseline_std, 2), "mph")
print("2026 average velocity:", round(current_mean, 2), "mph")
print("Velocity change:", round(velocity_change, 2), "mph")
print("Deviation score:", round(z_score, 2))