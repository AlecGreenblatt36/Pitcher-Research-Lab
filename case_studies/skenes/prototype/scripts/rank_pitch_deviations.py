import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()

data["game_date"] = pd.to_datetime(data["game_date"])


# Metrics we want to investigate
metrics = {
    "release_speed": "Velocity",
    "release_extension": "Extension",
    "release_pos_x": "Release X",
    "release_pos_z": "Release Z",
    "pfx_x": "Horizontal Movement",
    "pfx_z": "Vertical Movement",
    "release_spin_rate": "Spin Rate"
}


# Create outing-level averages for every pitch type
outing_data = (
    data.groupby(["season", "game_date", "pitch_type"])
    .agg(
        pitches_in_outing=("pitch_type", "count"),
        release_speed=("release_speed", "mean"),
        release_extension=("release_extension", "mean"),
        release_pos_x=("release_pos_x", "mean"),
        release_pos_z=("release_pos_z", "mean"),
        pfx_x=("pfx_x", "mean"),
        pfx_z=("pfx_z", "mean"),
        release_spin_rate=("release_spin_rate", "mean")
    )
    .reset_index()
)


# Require at least 5 of that pitch in an outing
# so a one-pitch outing does not distort the baseline
outing_data = outing_data[
    outing_data["pitches_in_outing"] >= 5
].copy()


results = []


for pitch_type in outing_data["pitch_type"].unique():

    pitch_data = outing_data[
        outing_data["pitch_type"] == pitch_type
    ]

    baseline = pitch_data[
        pitch_data["season"].isin([2024, 2025])
    ]

    current = pitch_data[
        pitch_data["season"] == 2026
    ]

    for column, metric_name in metrics.items():

        baseline_values = baseline[column].dropna()
        current_values = current[column].dropna()

        # Skip comparisons where we don't have enough data
        if len(baseline_values) < 5 or len(current_values) < 3:
            continue

        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std()
        current_mean = current_values.mean()

        if baseline_std == 0:
            continue

        change = current_mean - baseline_mean
        z_score = change / baseline_std

        # Convert movement/release changes from feet to inches
        if column in [
            "release_extension",
            "release_pos_x",
            "release_pos_z",
            "pfx_x",
            "pfx_z"
        ]:
            display_change = change * 12
            unit = "in"

        elif column == "release_speed":
            display_change = change
            unit = "mph"

        elif column == "release_spin_rate":
            display_change = change
            unit = "rpm"

        results.append({
            "pitch_type": pitch_type,
            "metric": metric_name,
            "baseline_outings": len(baseline_values),
            "2026_outings": len(current_values),
            "change": round(display_change, 2),
            "unit": unit,
            "z_score": round(z_score, 2),
            "absolute_z_score": round(abs(z_score), 2)
        })


report = pd.DataFrame(results)

# Biggest deviations first
report = report.sort_values(
    "absolute_z_score",
    ascending=False
)

print(report.to_string(index=False))


# Save the report
output_folder = project_folder / "outputs"
output_folder.mkdir(exist_ok=True)

report.to_csv(
    output_folder / "skenes_2026_deviation_report.csv",
    index=False
)

print("\nSaved deviation report.")