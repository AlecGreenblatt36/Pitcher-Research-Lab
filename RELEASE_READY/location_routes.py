from flask import Blueprint, jsonify, request
import sqlite3
import pandas as pd
from pitcher_core import DATABASE_FILE, current_research_season
from comparison import ComparisonError, resolve_comparison


# ==================================================
# Blueprint
# ==================================================

location_bp = Blueprint(
    "location",
    __name__
)


database_file = DATABASE_FILE


# ==================================================
# Helpers
# ==================================================

def clean_number(
    value,
    decimals=2
):

    if pd.isna(value):
        return None

    return round(
        float(value),
        decimals
    )


def safe_percentage(
    numerator,
    denominator
):

    if denominator == 0:
        return None

    return (
        numerator
        /
        denominator
        *
        100
    )


# ==================================================
# Location API
# ==================================================

@location_bp.route("/api/pitchers/<int:pitcher_id>/location")
def pitcher_location(pitcher_id):

    pitch_type = (
        request.args.get(
            "pitch",
            "FF"
        )
        .upper()
    )

    batter_hand = (
        request.args.get(
            "hand",
            "ALL"
        )
        .upper()
    )

    if batter_hand not in {"ALL", "L", "R"}:
        return jsonify({"error": "Batter hand must be ALL, L or R."}), 400

    target_season = request.args.get("season", type=int)
    if target_season is None:
        target_season = current_research_season(pitcher_id)
    if target_season is None:
        return jsonify({"error": "No regular-season Statcast data found."}), 404

    try:
        comparison = resolve_comparison(request.args, pitcher_id, target_season)
    except ComparisonError as exc:
        return jsonify({"error": str(exc), "code": "invalid_comparison_periods"}), 400
    scope = comparison.scope


    # ==================================================
    # Load pitches
    # ==================================================

    connection = sqlite3.connect(
        database_file
    )


    query = """
    SELECT
        game_date,
        game_pk,
        pitch_type,
        stand,

        plate_x,
        plate_z,
        sz_top,
        sz_bot,

        description,

        launch_speed,

        delta_run_exp

    FROM pitches

    WHERE
        game_type = 'R'
        AND CAST(pitcher AS INTEGER) = ?
        AND pitch_type IS NOT NULL;
    """

    if scope == "season":
        query = query.replace(
            "AND pitch_type IS NOT NULL",
            "AND CAST(season AS INTEGER) = ? AND pitch_type IS NOT NULL",
        )
        query_params = (int(pitcher_id), int(target_season))
    else:
        query_params = (int(pitcher_id),)


    data = pd.read_sql_query(
        query,
        connection,
        params=query_params
    )


    connection.close()


    # ==================================================
    # Clean
    # ==================================================

    data["game_date"] = (
        pd.to_datetime(
            data["game_date"]
        )
    )


    numeric_columns = [

        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "launch_speed",
        "delta_run_exp"

    ]


    for column in numeric_columns:

        data[column] = (
            pd.to_numeric(
                data[column],
                errors="coerce"
            )
        )


    # ==================================================
    # Pitch filter
    # ==================================================

    data = data[
        data["pitch_type"]
        ==
        pitch_type
    ].copy()


    # ==================================================
    # Handedness
    # ==================================================

    if batter_hand in [
        "L",
        "R"
    ]:

        data = data[
            data["stand"]
            ==
            batter_hand
        ].copy()


    if data.empty:

        return jsonify(
            {
                "error":
                    "No matching pitches found."
            }
        ), 404


    # ==================================================
    # Period
    # ==================================================

    data["period"] = (
        data["game_date"]
        .apply(comparison.classify)
    )
    data = data[data["period"].notna()].copy()


    # ==================================================
    # Normalize strike zone
    # ==================================================

    valid_location = (

        data["plate_x"].notna()

        &

        data["plate_z"].notna()

        &

        data["sz_top"].notna()

        &

        data["sz_bot"].notna()

        &

        (
            data["sz_top"]
            >
            data["sz_bot"]
        )

    )


    data = data[
        valid_location
    ].copy()


    if data.empty:

        return jsonify(
            {
                "error":
                    "No usable location data."
            }
        ), 404


    horizontal_half_width = 0.83


    data["zone_center_z"] = (

        data["sz_top"]
        +
        data["sz_bot"]

    ) / 2


    data["zone_half_height"] = (

        data["sz_top"]
        -
        data["sz_bot"]

    ) / 2


    data["normalized_x"] = (

        data["plate_x"]
        /
        horizontal_half_width

    )


    data["normalized_z"] = (

        (
            data["plate_z"]
            -
            data["zone_center_z"]
        )

        /

        data["zone_half_height"]

    )


    # ==================================================
    # Zone regions
    # ==================================================

    data["in_zone"] = (

        (
            data["normalized_x"]
            .abs()
            <=
            1
        )

        &

        (
            data["normalized_z"]
            .abs()
            <=
            1
        )

    )


    data["in_heart"] = (

        (
            data["normalized_x"]
            .abs()
            <=
            0.50
        )

        &

        (
            data["normalized_z"]
            .abs()
            <=
            0.50
        )

    )


    data["on_edge"] = (

        data["in_zone"]

        &

        (

            (
                data["normalized_x"]
                .abs()
                >=
                0.67
            )

            |

            (
                data["normalized_z"]
                .abs()
                >=
                0.67
            )

        )

    )


    # ==================================================
    # Swing / whiff
    # ==================================================

    swing_descriptions = [

        "swinging_strike",
        "swinging_strike_blocked",

        "foul",
        "foul_tip",

        "hit_into_play",
        "foul_bunt",
        "missed_bunt",
        "swinging_pitchout"

    ]


    whiff_descriptions = [

        "swinging_strike",
        "swinging_strike_blocked",
        "missed_bunt",
        "swinging_pitchout"

    ]


    data["is_swing"] = (

        data["description"]
        .isin(
            swing_descriptions
        )

    )


    data["is_whiff"] = (

        data["description"]
        .isin(
            whiff_descriptions
        )

    )


    data["is_chase"] = (

        (~data["in_zone"])

        &

        data["is_swing"]

    )


    # ==================================================
    # Contact
    # ==================================================

    data["is_batted_ball"] = (

        data["launch_speed"]
        .notna()

        &

        data["description"]
        .eq("hit_into_play")

    )


    data["is_hard_hit"] = (

        data["launch_speed"]
        >=
        95

    )


    # ==================================================
    # Pitcher run value
    #
    # Positive = good for pitcher
    # ==================================================

    data["pitcher_run_value"] = (

        -data["delta_run_exp"]

    )


    # ==================================================
    # Period summary
    # ==================================================

    def summarize_period(
        period_data,
        period_name
    ):

        pitch_count = len(
            period_data
        )


        zone_pitches = int(
            period_data[
                "in_zone"
            ].sum()
        )


        heart_pitches = int(
            period_data[
                "in_heart"
            ].sum()
        )


        edge_pitches = int(
            period_data[
                "on_edge"
            ].sum()
        )


        outside_zone = (
            ~period_data[
                "in_zone"
            ]
        )


        out_of_zone_count = int(
            outside_zone.sum()
        )


        chase_count = int(
            period_data[
                "is_chase"
            ].sum()
        )


        swings = int(
            period_data[
                "is_swing"
            ].sum()
        )


        whiffs = int(
            period_data[
                "is_whiff"
            ].sum()
        )


        zone_swings = int(

            (
                period_data[
                    "in_zone"
                ]

                &

                period_data[
                    "is_swing"
                ]

            ).sum()

        )


        zone_whiffs = int(

            (
                period_data[
                    "in_zone"
                ]

                &

                period_data[
                    "is_whiff"
                ]

            ).sum()

        )


        contact = period_data[
            period_data[
                "is_batted_ball"
            ]
        ]


        hard_hits = int(
            contact[
                "is_hard_hit"
            ].sum()
        )


        run_value = (
            period_data[
                "pitcher_run_value"
            ].sum(min_count=1)
        )


        run_value_pitches = int(
            period_data[
                "pitcher_run_value"
            ].notna().sum()
        )


        return {

            "period":
                period_name,

            "pitches":
                int(
                    pitch_count
                ),

            "swings":
                int(
                    swings
                ),

            "batted_balls":
                int(
                    len(contact)
                ),

            "zone_pct":
                clean_number(
                    safe_percentage(
                        zone_pitches,
                        pitch_count
                    )
                ),

            "heart_pct":
                clean_number(
                    safe_percentage(
                        heart_pitches,
                        pitch_count
                    )
                ),

            "edge_pct":
                clean_number(
                    safe_percentage(
                        edge_pitches,
                        pitch_count
                    )
                ),

            "chase_pct":
                clean_number(
                    safe_percentage(
                        chase_count,
                        out_of_zone_count
                    )
                ),

            "whiff_pct":
                clean_number(
                    safe_percentage(
                        whiffs,
                        swings
                    )
                ),

            "zone_whiff_pct":
                clean_number(
                    safe_percentage(
                        zone_whiffs,
                        zone_swings
                    )
                ),

            "avg_ev":
                clean_number(
                    contact[
                        "launch_speed"
                    ]
                    .mean()
                ),

            "hard_hit_pct":
                clean_number(
                    safe_percentage(
                        hard_hits,
                        len(contact)
                    )
                ),

            "run_value":
                clean_number(
                    run_value
                ),

            "run_value_pitches":
                run_value_pitches,

            "run_value_per_100":

                clean_number(

                    (
                        run_value
                        /
                        run_value_pitches
                        *
                        100
                    )

                    if run_value_pitches and pd.notna(run_value)
                    else None

                ),

            "avg_x":
                clean_number(
                    period_data[
                        "normalized_x"
                    ]
                    .mean()
                ),

            "avg_z":
                clean_number(
                    period_data[
                        "normalized_z"
                    ]
                    .mean()
                )

        }


    # ==================================================
    # 7 x 7 heatmap
    # ==================================================

    def create_heatmap(
        period_data
    ):

        minimum = -1.75

        maximum = 1.75

        grid_size = 7


        bin_width = (

            maximum
            -
            minimum

        ) / grid_size


        visible = period_data[

            (
                period_data[
                    "normalized_x"
                ]
                >=
                minimum
            )

            &

            (
                period_data[
                    "normalized_x"
                ]
                <=
                maximum
            )

            &

            (
                period_data[
                    "normalized_z"
                ]
                >=
                minimum
            )

            &

            (
                period_data[
                    "normalized_z"
                ]
                <=
                maximum
            )

        ].copy()


        visible["x_bin"] = (

            (
                (
                    visible[
                        "normalized_x"
                    ]
                    -
                    minimum
                )

                /
                bin_width
            )

            .astype(int)

            .clip(
                0,
                grid_size - 1
            )

        )


        visible["z_bin"] = (

            (
                (
                    visible[
                        "normalized_z"
                    ]
                    -
                    minimum
                )

                /
                bin_width
            )

            .astype(int)

            .clip(
                0,
                grid_size - 1
            )

        )


        bins = []


        for x_bin in range(
            grid_size
        ):

            for z_bin in range(
                grid_size
            ):

                cell = visible[

                    (
                        visible[
                            "x_bin"
                        ]
                        ==
                        x_bin
                    )

                    &

                    (
                        visible[
                            "z_bin"
                        ]
                        ==
                        z_bin
                    )

                ]


                pitch_count = len(
                    cell
                )


                if pitch_count == 0:

                    bins.append(
                        {

                            "x_bin":
                                x_bin,

                            "z_bin":
                                z_bin,

                            "count":
                                0,

                            "swings":
                                0,

                            "whiffs":
                                0,

                            "batted_balls":
                                0,

                            "run_value_per_100":
                                None,

                            "avg_ev":
                                None,

                            "hard_hit_pct":
                                None

                        }
                    )

                    continue


                swings = int(
                    cell[
                        "is_swing"
                    ].sum()
                )


                whiffs = int(
                    cell[
                        "is_whiff"
                    ].sum()
                )


                run_value = (
                    cell[
                        "pitcher_run_value"
                    ].sum(min_count=1)
                )


                run_value_pitches = int(
                    cell[
                        "pitcher_run_value"
                    ].notna().sum()
                )


                contact = cell[
                    cell[
                        "is_batted_ball"
                    ]
                ]


                hard_hits = int(
                    contact[
                        "is_hard_hit"
                    ].sum()
                )


                bins.append(
                    {

                        "x_bin":
                            x_bin,

                        "z_bin":
                            z_bin,

                        "count":
                            int(
                                pitch_count
                            ),

                        "swings":
                            swings,

                        "whiffs":
                            whiffs,

                        "batted_balls":
                            int(
                                len(contact)
                            ),

                        "run_value_per_100":

                            clean_number(

                                run_value
                                /
                                run_value_pitches
                                *
                                100

                            ) if run_value_pitches and pd.notna(run_value) else None,

                        "avg_ev":
                            clean_number(
                                contact[
                                    "launch_speed"
                                ]
                                .mean()
                            ),

                        "hard_hit_pct":

                            clean_number(

                                safe_percentage(
                                    hard_hits,
                                    len(contact)
                                )

                            )

                    }
                )


        return bins


    # ==================================================
    # Return periods
    # ==================================================

    period_results = {}


    for period_name in [

        "early",
        "transition",
        "post"

    ]:

        period_data = data[
            data["period"]
            ==
            period_name
        ].copy()


        if period_data.empty:

            continue


        period_results[
            period_name
        ] = {

            "summary":
                summarize_period(
                    period_data,
                    period_name
                ),

            "heatmap":
                create_heatmap(
                    period_data
                )

        }


    return jsonify(
        {

            "pitch_type":
                pitch_type,

            "batter_hand":
                batter_hand,

            "comparison_periods": comparison.payload(),
            "transition_window": comparison.legacy_payload(),

            "periods":
                period_results,

            "definitions":
                {

                    "zone":
                        "Pitch inside the normalized batter-specific strike zone.",

                    "heart":
                        "Pitch in the central 50% of the normalized zone.",

                    "edge":
                        "Pitch in the outer third of the normalized strike zone.",

                    "chase":
                        "Swing at a pitch outside the normalized strike zone.",

                    "whiff":
                        "Swing that results in a miss.",

                    "hard_hit":
                        "Batted ball with exit velocity of at least 95 mph.",

                    "run_value":
                        "Pitcher-perspective run value. Positive is better for the pitcher."

                }

        }
    )
