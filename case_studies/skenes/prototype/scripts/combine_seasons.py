import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

# Load each season
data_2024 = pd.read_csv(data_folder / "skenes_2024_raw.csv")
data_2025 = pd.read_csv(data_folder / "skenes_2025_raw.csv")
data_2026 = pd.read_csv(data_folder / "skenes_2026_raw.csv")

# Add a season column so we always know which year each pitch came from
data_2024["season"] = 2024
data_2025["season"] = 2025
data_2026["season"] = 2026

# Stack all three seasons together
all_seasons = pd.concat(
    [data_2024, data_2025, data_2026],
    ignore_index=True
)

print("Combined dataset size:", all_seasons.shape)

print("\nPitches by season:")
print(all_seasons["season"].value_counts().sort_index())

# Save the combined dataset
all_seasons.to_csv(
    data_folder / "skenes_2024_2026_raw.csv",
    index=False
)

print("\nSaved master dataset.")