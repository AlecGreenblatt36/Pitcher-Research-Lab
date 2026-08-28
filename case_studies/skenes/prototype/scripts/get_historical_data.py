from pybaseball import statcast_pitcher
from pathlib import Path
skenes_id = 694973
project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"
# Pull 2024 data
data_2024 = statcast_pitcher(
    "2024-03-01",
    "2024-10-01",
    skenes_id
)
data_2024.to_csv(
    data_folder / "skenes_2024_raw.csv",
    index=False
)
print("2024 complete:", data_2024.shape)
# Pull 2025 data
data_2025 = statcast_pitcher(
    "2025-03-01",
    "2025-10-01",
    skenes_id
)
data_2025.to_csv(
    data_folder / "skenes_2025_raw.csv",
    index=False
)
print("2025 complete:", data_2025.shape)
