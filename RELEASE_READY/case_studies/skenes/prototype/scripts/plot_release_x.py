import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
regular_season = data[data["game_type"] == "R"].copy()

# Four-seam fastballs only
fastballs = regular_season[regular_season["pitch_type"] == "FF"].copy()

# Treat game_date as an actual date
fastballs["game_date"] = pd.to_datetime(fastballs["game_date"])

# Find average horizontal release position for each outing
release_x_by_outing = (
    fastballs.groupby("game_date")["release_pos_x"]
    .mean()
    .reset_index()
    .sort_values("game_date")
)

release_x_by_outing = release_x_by_outing.round(2)

print(release_x_by_outing.to_string(index=False))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
regular_season = data[data["game_type"] == "R"].copy()

# Four-seam fastballs only
fastballs = regular_season[regular_season["pitch_type"] == "FF"].copy()

# Treat game_date as an actual date
fastballs["game_date"] = pd.to_datetime(fastballs["game_date"])

# Find average horizontal release position for each outing
release_x_by_outing = (
    fastballs.groupby("game_date")["release_pos_x"]
    .mean()
    .reset_index()
    .sort_values("game_date")
)

release_x_by_outing = release_x_by_outing.round(2)

print(release_x_by_outing.to_string(index=False))


# Create a cleaner graph
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    release_x_by_outing["game_date"],
    release_x_by_outing["release_pos_x"],
    marker="o",
    linewidth=2
)

ax.set_title(
    "Paul Skenes Four-Seam Horizontal Release Position",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Game Date")

ax.set_ylabel(
    "Horizontal Release Position (ft)\nCloser to 0 = closer to field centerline"
)

# Zoom in so the horizontal changes are easier to see
ax.set_ylim(-3.0, -1.8)

# Add horizontal grid lines
ax.grid(axis="y", alpha=0.3)

# Make dates easier to read
fig.autofmt_xdate()

plt.tight_layout()

# Create an outputs folder if it doesn't already exist
output_folder = project_folder / "outputs"
output_folder.mkdir(exist_ok=True)

# Save a high-quality image of the graph
plt.savefig(
    output_folder / "skenes_release_x.png",
    dpi=300,
    bbox_inches="tight"
)

# Show the graph
plt.show()