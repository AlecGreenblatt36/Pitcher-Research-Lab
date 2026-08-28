import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
data_file = project_folder / "data" / "skenes_2024_2026_raw.csv"

data = pd.read_csv(data_file)

# Regular-season four-seam fastballs only
fastballs = data[
    (data["season"] == 2026) &
    (data["game_type"] == "R") &
    (data["pitch_type"] == "FF")
].copy()

fastballs["game_date"] = pd.to_datetime(fastballs["game_date"])


# --------------------------------------------------
# Dates we want to investigate on video
# --------------------------------------------------

review_dates = {
    "Before Change": [
        "2026-04-18",
        "2026-04-24"
    ],

    "Transition": [
        "2026-05-06",
        "2026-05-12"
    ],

    "After Change": [
        "2026-05-28",
        "2026-06-09"
    ]
}


review_pitches = []


# --------------------------------------------------
# Find representative pitches from each outing
# --------------------------------------------------

for phase, dates in review_dates.items():

    for date in dates:

        game_pitches = fastballs[
            fastballs["game_date"] == date
        ].copy()

        if game_pitches.empty:
            continue

        # Average four-seam velocity for this outing
        outing_avg_velocity = (
            game_pitches["release_speed"].mean()
        )

        # Find how far each pitch was from that outing average
        game_pitches["distance_from_avg"] = abs(
            game_pitches["release_speed"]
            - outing_avg_velocity
        )

        # Choose the 3 pitches closest to the outing's average velocity
        representative = (
            game_pitches
            .sort_values("distance_from_avg")
            .head(3)
            .copy()
        )

        representative["phase"] = phase
        representative["outing_avg_velocity"] = (
            outing_avg_velocity
        )

        review_pitches.append(representative)


# --------------------------------------------------
# Combine selected pitches
# --------------------------------------------------

video_queue = pd.concat(
    review_pitches,
    ignore_index=True
)


# Create readable count
video_queue["count"] = (
    video_queue["balls"].astype(str)
    + "-"
    + video_queue["strikes"].astype(str)
)


# Convert movement to inches
video_queue["horizontal_movement_in"] = (
    video_queue["pfx_x"] * 12
)

video_queue["vertical_movement_in"] = (
    video_queue["pfx_z"] * 12
)


# --------------------------------------------------
# Keep useful video-review information
# --------------------------------------------------

video_queue = video_queue[
    [
        "phase",
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "count",
        "stand",
        "release_speed",
        "outing_avg_velocity",
        "release_extension",
        "release_pos_x",
        "release_pos_z",
        "horizontal_movement_in",
        "vertical_movement_in",
        "release_spin_rate",
        "plate_x",
        "plate_z",
        "description"
    ]
].copy()


# Round numbers
numeric_columns = [
    "release_speed",
    "outing_avg_velocity",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "horizontal_movement_in",
    "vertical_movement_in",
    "release_spin_rate",
    "plate_x",
    "plate_z"
]

video_queue[numeric_columns] = (
    video_queue[numeric_columns].round(2)
)


# --------------------------------------------------
# Print and save
# --------------------------------------------------

print("\nPAUL SKENES VIDEO REVIEW QUEUE")
print("------------------------------")

print(
    video_queue.to_string(index=False)
)

output_folder = project_folder / "outputs"
output_folder.mkdir(exist_ok=True)

video_queue.to_csv(
    output_folder / "skenes_video_review_queue.csv",
    index=False
)

print("\nSaved video review queue.")