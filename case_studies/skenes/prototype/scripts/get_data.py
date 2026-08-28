from pathlib import Path
from pybaseball import statcast_pitcher
skenes_id = 694973
data = statcast_pitcher(
    "2026-03-01",
    "2026-08-16",
    skenes_id
)
print(data.head())
print(data.shape)
project_folder = Path(__file__).resolve().parent
data_folder = project_folder /"data"
data_folder.mkdir(exist_ok=True)
data.to_csv("data/skenes_2026_raw.csv", index=False)