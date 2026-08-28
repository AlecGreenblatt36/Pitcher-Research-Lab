import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2026_raw.csv"

data = pd.read_csv(data_file)

# Keep only regular-season pitches
regular_season = data[data["game_type"] == "R"].copy()

# Keep only four-seam fastballs
fastballs = regular_season[regular_season["pitch_type"] == "FF"].copy()

# Treat game_date as a real date
fastballs["game_date"] = pd.to_datetime(fastballs["game_date"])

# Find average extension for each outing
extension_by_outing = (
    fastballs.groupby("game_date")["release_extension"]
    .mean()
    .reset_index()
    .sort_values("game_date")
)

print(extension_by_outing)

# Create the graph
plt.figure(figsize=(10, 5))

plt.plot(
    extension_by_outing["game_date"],
    extension_by_outing["release_extension"],
    marker="o"
)

plt.title("Paul Skenes Four-Seam Fastball Extension by Outing")
plt.xlabel("Game Date")
plt.ylabel("Average Extension (ft)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()