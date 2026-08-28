import pandas as pd
from pathlib import Path
project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2026_raw.csv"
data = pd.read_csv(data_file)
print("Dataset size:", data.shape)

print("\nColumns:")
for number, column in enumerate(data.columns, start=1):
    print(number,column)