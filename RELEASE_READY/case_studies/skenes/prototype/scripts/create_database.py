import pandas as pd
import sqlite3
from pathlib import Path

project_folder = Path(__file__).resolve().parent

csv_file = (
    project_folder
    / "data"
    / "skenes_2024_2026_raw.csv"
)

database_file = (
    project_folder
    / "data"
    / "pitcher_research.db"
)

# Load our master Statcast dataset
data = pd.read_csv(csv_file)

# Connect to a SQLite database
connection = sqlite3.connect(database_file)

# Store the DataFrame as a SQL table called "pitches"
data.to_sql(
    "pitches",
    connection,
    if_exists="replace",
    index=False
)

print("Database created successfully.")

# Run our first SQL query
query = """
SELECT
    season,
    COUNT(*) AS pitches
FROM pitches
GROUP BY season
ORDER BY season;
"""

results = pd.read_sql_query(
    query,
    connection
)

print("\nPitches stored in database:")
print(results.to_string(index=False))

connection.close()