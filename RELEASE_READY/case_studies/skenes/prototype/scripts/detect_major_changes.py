import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season only
data = data[data["game_type"] == "R"].copy()

data["game_date"] = pd.to_datetime(data["game_date"])


# --------------------------------------------------
# Things we already identified as interesting
# --------------------------------------------------

tests = [
    {
        "pitch_type": "FF",
        "column": "release_speed",
        "name": "Four-Seam Velocity"
    },

    {
        "pitch_type": "FF",
        "column": "pfx_z",
        "name": "Four-Seam Vertical Movement"
    },

    {
        "pitch_type": "SL",
        "column": "pfx_z",
        "name": "Slider Vertical Movement"
    },

    {
        "pitch_type": "SI",
        "column": "release_spin_rate",
        "name": "Sinker Spin Rate"
    },

    {
        "pitch_type": "SI",
        "column": "release_speed",
        "name": "Sinker Velocity"
    }
]


results = []


# --------------------------------------------------
# Run the same analysis for every metric
# --------------------------------------------------

for test in tests:

    pitch_type = test["pitch_type"]
    column = test["column"]
    name = test["name"]

    pitch_data = data[
        data["pitch_type"] == pitch_type
    ].copy()

    # Average the metric for each outing
    outing_data = (
        pitch_data.groupby(["season", "game_date"])
        .agg(
            pitches=("pitch_type", "count"),
            metric_value=(column, "mean")
        )
        .reset_index()
        .sort_values("game_date")
    )

    # Require at least 5 of that pitch in the outing
    outing_data = outing_data[
        outing_data["pitches"] >= 5
    ].copy()

    # Historical baseline
    baseline = outing_data[
        outing_data["season"].isin([2024, 2025])
    ].copy()

    # 2026
    current = outing_data[
        outing_data["season"] == 2026
    ].copy()

    current = current.reset_index(drop=True)

    # Make sure we have enough data
    if len(baseline) < 5 or len(current) < 3:
        continue

    baseline_mean = baseline["metric_value"].mean()
    baseline_std = baseline["metric_value"].std()

    if baseline_std == 0:
        continue

    # 3-outing rolling average
    current["rolling_value"] = (
        current["metric_value"]
        .rolling(window=3)
        .mean()
    )

    # Z-score relative to historical baseline
    current["rolling_z"] = (
        current["rolling_value"] - baseline_mean
    ) / baseline_std

    # Detect 2 SD below historical baseline
    current["below_2sd"] = (
        current["rolling_z"] <= -2
    )

    # Detect 2 SD above historical baseline
    current["above_2sd"] = (
        current["rolling_z"] >= 2
    )

    # Require 3 consecutive outings in the same direction
    current["sustained_below"] = (
        current["below_2sd"]
        .rolling(window=3)
        .sum()
        == 3
    )

    current["sustained_above"] = (
        current["above_2sd"]
        .rolling(window=3)
        .sum()
        == 3
    )

    change_date = None
    direction = None

    # First sustained low change
    below_flags = current[
        current["sustained_below"]
    ]

    if not below_flags.empty:

        third_position = below_flags.index[0]

        change_date = current.loc[
            third_position - 2,
            "game_date"
        ]

        direction = "Below baseline"

    # First sustained high change
    above_flags = current[
        current["sustained_above"]
    ]

    if not above_flags.empty:

        third_position = above_flags.index[0]

        above_date = current.loc[
            third_position - 2,
            "game_date"
        ]

        # If both happen, keep whichever happened first
        if (
            change_date is None
            or above_date < change_date
        ):
            change_date = above_date
            direction = "Above baseline"

    results.append({
        "metric": name,
        "baseline_outings": len(baseline),
        "2026_outings": len(current),
        "baseline_mean": round(baseline_mean, 2),
        "baseline_std": round(baseline_std, 2),
        "first_sustained_change": (
            change_date.date()
            if change_date is not None
            else "None"
        ),
        "direction": (
            direction
            if direction is not None
            else "No sustained change"
        )
    })


# --------------------------------------------------
# Print final change timeline
# --------------------------------------------------

report = pd.DataFrame(results)

print("\nPAUL SKENES 2026 CHANGE DETECTION")
print("--------------------------------")

print(report.to_string(index=False))


# Save it
output_folder = project_folder / "outputs"
output_folder.mkdir(exist_ok=True)

report.to_csv(
    output_folder / "skenes_change_timeline.csv",
    index=False
)

print("\nSaved change timeline.")