import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular season four-seam fastballs only
fastballs = data[
    (data["game_type"] == "R") &
    (data["pitch_type"] == "FF")
].copy()

# Treat game_date as an actual date
fastballs["game_date"] = pd.to_datetime(fastballs["game_date"])

# Calculate average four-seam velocity for each outing
outing_data = (
    fastballs.groupby(["season", "game_date"])
    .agg(
        avg_velocity=("release_speed", "mean"),
        fastballs=("release_speed", "count")
    )
    .reset_index()
    .sort_values("game_date")
)

# -----------------------------------
# Build 2024-2025 historical baseline
# -----------------------------------

baseline = outing_data[
    outing_data["season"].isin([2024, 2025])
].copy()

baseline_mean = baseline["avg_velocity"].mean()
baseline_std = baseline["avg_velocity"].std()

print("2024-25 baseline velocity:",
      round(baseline_mean, 2), "mph")

print("Historical outing variation:",
      round(baseline_std, 2), "mph")


# -----------------------------------
# Look at 2026
# -----------------------------------

current = outing_data[
    outing_data["season"] == 2026
].copy()

current = current.reset_index(drop=True)

# Calculate a 3-outing rolling velocity average
current["rolling_velocity"] = (
    current["avg_velocity"]
    .rolling(window=3)
    .mean()
)

# Calculate how far each rolling average is
# from Skenes' historical normal
current["rolling_z"] = (
    current["rolling_velocity"] - baseline_mean
) / baseline_std


# -----------------------------------
# Detect sustained velocity change
# -----------------------------------

# True when rolling velocity is at least
# 2 standard deviations below baseline
current["below_2sd"] = (
    current["rolling_z"] <= -2
)

# Look for 3 consecutive outings below -2 SD
current["sustained_change"] = (
    current["below_2sd"]
    .rolling(window=3)
    .sum()
    == 3
)

flagged = current[
    current["sustained_change"]
]

if not flagged.empty:

    # This row is the third outing in the streak
    third_position = flagged.index[0]

    # Go back two outings to find where streak began
    start_position = third_position - 2

    sustained_change_date = current.loc[
        start_position,
        "game_date"
    ]

    print(
        "\nFirst sustained velocity departure:",
        sustained_change_date.date()
    )

else:

    sustained_change_date = None

    print(
        "\nNo sustained velocity departure detected."
    )


# -----------------------------------
# Print March through June
# -----------------------------------

early_season = current[
    current["game_date"] <= "2026-06-30"
].copy()

display_table = early_season[
    [
        "game_date",
        "avg_velocity",
        "rolling_velocity",
        "rolling_z"
    ]
].copy()

display_table["avg_velocity"] = (
    display_table["avg_velocity"].round(2)
)

display_table["rolling_velocity"] = (
    display_table["rolling_velocity"].round(2)
)

display_table["rolling_z"] = (
    display_table["rolling_z"].round(2)
)

print("\n2026 Early-Season Velocity Progression:")

print(
    display_table.to_string(index=False)
)


# -----------------------------------
# Create graph
# -----------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

# Individual outing averages
ax.plot(
    current["game_date"],
    current["avg_velocity"],
    marker="o",
    alpha=0.45,
    label="Outing Average"
)

# 3-outing rolling average
ax.plot(
    current["game_date"],
    current["rolling_velocity"],
    linewidth=3,
    label="3-Outing Rolling Average"
)

# Historical baseline
ax.axhline(
    baseline_mean,
    linestyle="--",
    linewidth=2,
    label="2024-25 Baseline"
)

# One standard deviation below baseline
ax.axhline(
    baseline_mean - baseline_std,
    linestyle=":",
    linewidth=1.5,
    label="1 SD Below Baseline"
)

# Two standard deviations below baseline
ax.axhline(
    baseline_mean - (2 * baseline_std),
    linestyle=":",
    linewidth=2,
    label="2 SD Below Baseline"
)

# Mark sustained change date
if sustained_change_date is not None:

    ax.axvline(
        sustained_change_date,
        linestyle="--",
        linewidth=2
    )

    ax.text(
        sustained_change_date,
        current["avg_velocity"].min(),
        "  Sustained Change Begins",
        rotation=90,
        verticalalignment="bottom"
    )


# -----------------------------------
# Graph formatting
# -----------------------------------

ax.set_title(
    "Paul Skenes 2026 Four-Seam Velocity vs Historical Baseline",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Game Date")

ax.set_ylabel(
    "Average Four-Seam Velocity (mph)"
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend()

fig.autofmt_xdate()

plt.tight_layout()


# -----------------------------------
# Save graph
# -----------------------------------

output_folder = project_folder / "outputs"

output_folder.mkdir(
    exist_ok=True
)

plt.savefig(
    output_folder /
    "skenes_fastball_change_detection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()