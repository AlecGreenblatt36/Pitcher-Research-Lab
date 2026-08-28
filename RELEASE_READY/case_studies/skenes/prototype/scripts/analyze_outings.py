import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2026_raw.csv"

data = pd.read_csv(data_file)

# Keep only four-seam fastballs
fastballs = data[data["pitch_type"] == "FF"]

# Calculate averages for each outing
fastball_by_outing = (
    fastballs.groupby("game_date")[
        [
            "release_speed",
            "release_extension",
            "release_pos_x",
            "release_pos_z",
            "pfx_x",
            "pfx_z"
        ]
    ]
    .mean()
    .reset_index()
)

# Round numbers so they are easier to read
fastball_by_outing = fastball_by_outing.round(2)

print(fastball_by_outing.to_string(index=False))