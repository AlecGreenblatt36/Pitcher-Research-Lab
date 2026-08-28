import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent

raw_file = project_folder / "data" / "skenes_2026_raw.csv"

data = pd.read_csv(raw_file)

columns_we_want = [
    "game_date",
    "pitch_type",
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z"
]

skenes = data[columns_we_want]

print(skenes.head())
print("\nDataset size:", skenes.shape)

print("\nPitch types:")
print(skenes["pitch_type"].value_counts())