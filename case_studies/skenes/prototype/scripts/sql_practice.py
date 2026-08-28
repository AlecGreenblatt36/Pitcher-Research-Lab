import sqlite3
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
database_file = project_folder / "data" / "pitcher_research.db"

connection = sqlite3.connect(database_file)

query = """
SELECT
    season,
    COUNT(*) AS pitches,
    ROUND(AVG(release_speed), 2) AS avg_velocity
FROM pitches
WHERE
    game_type = 'R'
    AND pitch_type = 'FF'
GROUP BY season
ORDER BY season;
"""

results = pd.read_sql_query(
    query,
    connection
)

print(results.to_string(index=False))

connection.close()