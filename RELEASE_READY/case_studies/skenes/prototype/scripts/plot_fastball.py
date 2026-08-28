import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2026_raw.csv"

data = pd.read_csv(data_file)

# Keep only four-seam fastballs
regular_season = data[data["game_type"] == "R"].copy()

fastballs = regular_season[regular_season["pitch_type"] == "FF"].copy()

# Make sure game_date is treated as an actual date
fastballs["game_date"] = pd.to_datetime(fastballs["game_date"])

# Find average fastball velocity for each outing
velocity_by_outing = (
    fastballs.groupby("game_date")["release_speed"]
    .mean()
    .reset_index()
    .sort_values("game_date")
)

print(velocity_by_outing)

print("\nGame types:")
print(data["game_type"].value_counts())
# Create the graph
plt.figure(figsize=(10, 5))

plt.plot(
    velocity_by_outing["game_date"],
    velocity_by_outing["release_speed"],
    marker="o"
)

plt.title("Paul Skenes Four-Seam Fastball Velocity by Outing")
plt.xlabel("Game Date")
plt.ylabel("Average Velocity (mph)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
