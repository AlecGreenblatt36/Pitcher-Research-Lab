import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

data_2024 = pd.read_csv(data_folder / "skenes_2024_raw.csv")
data_2025 = pd.read_csv(data_folder / "skenes_2025_raw.csv")
data_2026 = pd.read_csv(data_folder / "skenes_2026_raw.csv")

seasons = {
    2024: data_2024,
    2025: data_2025,
    2026: data_2026
}

for year, data in seasons.items():

    print(f"\n--- {year} ---")

    print("Dataset size:", data.shape)

    print("\nGame types:")
    print(data["game_type"].value_counts())

    print("Number of columns:", len(data.columns))