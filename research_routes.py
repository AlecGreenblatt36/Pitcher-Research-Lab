from flask import Blueprint, jsonify, request
import sqlite3
import pandas as pd
from pitcher_core import DATABASE_FILE, current_research_season
from comparison import ComparisonError, resolve_comparison


# ==================================================
# Blueprint setup
# ==================================================

research_bp = Blueprint(
    "research",
    __name__
)


database_file = DATABASE_FILE


# ==================================================
# Helper
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


# ==================================================
# Research Command Center API
# ==================================================

@research_bp.route("/api/pitchers/<int:pitcher_id>/research")
def pitcher_research(pitcher_id):

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


    connection = sqlite3.connect(
        database_file
    )


    query = """
    SELECT
        game_date,
        game_pk,
        pitch_type,

        release_speed,
        release_spin_rate,
        release_extension,

        pfx_x,
        pfx_z,

        description,
        delta_run_exp,

        launch_speed,
        launch_angle

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
    # Clean data
    # ==================================================

    data["game_date"] = (
        pd.to_datetime(
            data["game_date"]
        )
    )


    numeric_columns = [

        "release_speed",
        "release_spin_rate",
        "release_extension",

        "pfx_x",
        "pfx_z",

        "delta_run_exp",

        "launch_speed",
        "launch_angle"

    ]


    for column in numeric_columns:

        data[column] = (
            pd.to_numeric(
                data[column],
                errors="coerce"
            )
        )


    # ==================================================
    # Swing / whiff definitions
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
        .astype(int)
    )


    data["is_whiff"] = (
        data["description"]
        .isin(
            whiff_descriptions
        )
        .astype(int)
    )


    # ==================================================
    # Pitcher run value
    #
    # Statcast delta_run_exp is offensive
    # perspective, so flip sign.
    # ==================================================

    data["pitcher_run_value"] = -data["delta_run_exp"]


    # ==================================================
    # Contact quality
    # ==================================================

    data["is_batted_ball"] = (
        data["launch_speed"]
        .notna()

        &

        data["description"]
        .eq("hit_into_play")
        .astype(int)
    )


    data["is_hard_hit"] = (
        (
            data["launch_speed"]
            >= 95
        )
        .astype(int)
    )


    # ==================================================
    # Divide the selected season into research periods
    # ==================================================

    data["period"] = (
        data["game_date"]
        .apply(comparison.classify)
    )
    data = data[data["period"].notna()].copy()


    # ==================================================
    # Total pitches per period
    #
    # Needed for arsenal usage %
    # ==================================================

    period_totals = (
        data
        .groupby(
            "period"
        )
        .size()
        .to_dict()
    )


    # ==================================================
    # Aggregate by period and pitch type
    # ==================================================

    grouped = (

        data

        .groupby(
            [
                "period",
                "pitch_type"
            ]
        )

        .agg(

            pitch_count=(
                "pitch_type",
                "size"
            ),

            avg_velocity=(
                "release_speed",
                "mean"
            ),

            avg_spin=(
                "release_spin_rate",
                "mean"
            ),

            avg_extension=(
                "release_extension",
                "mean"
            ),

            avg_hmov=(
                "pfx_x",
                "mean"
            ),

            avg_vmov=(
                "pfx_z",
                "mean"
            ),

            swings=(
                "is_swing",
                "sum"
            ),

            whiffs=(
                "is_whiff",
                "sum"
            ),

            run_value=(
                "pitcher_run_value",
                lambda values: values.sum(min_count=1)
            ),

            run_value_pitches=(
                "pitcher_run_value",
                "count"
            ),

            batted_balls=(
                "is_batted_ball",
                "sum"
            ),

            hard_hits=(
                "is_hard_hit",
                "sum"
            ),

            avg_ev=(
                "launch_speed",
                "mean"
            ),

            avg_launch_angle=(
                "launch_angle",
                "mean"
            )

        )

        .reset_index()

    )


    # ==================================================
    # Derived metrics
    # ==================================================

    grouped["usage_pct"] = (

        grouped.apply(

            lambda row:

            (
                row["pitch_count"]
                /
                period_totals[
                    row["period"]
                ]
                *
                100
            ),

            axis=1
        )

    )


    grouped["whiff_pct"] = float("nan")


    has_swings = (
        grouped["swings"]
        >
        0
    )


    grouped.loc[
        has_swings,
        "whiff_pct"
    ] = (

        grouped.loc[
            has_swings,
            "whiffs"
        ]

        /

        grouped.loc[
            has_swings,
            "swings"
        ]

        *

        100

    )


    grouped[
        "run_value_per_100"
    ] = (

        grouped["run_value"]

        /

        grouped["run_value_pitches"].replace(0, pd.NA)

        *

        100

    )


    grouped[
        "hard_hit_pct"
    ] = float("nan")


    has_batted_balls = (
        grouped["batted_balls"]
        >
        0
    )


    grouped.loc[
        has_batted_balls,
        "hard_hit_pct"
    ] = (

        grouped.loc[
            has_batted_balls,
            "hard_hits"
        ]

        /

        grouped.loc[
            has_batted_balls,
            "batted_balls"
        ]

        *

        100

    )


    # Statcast movement:
    # feet -> inches

    grouped["avg_hmov"] = (
        grouped["avg_hmov"]
        *
        12
    )


    grouped["avg_vmov"] = (
        grouped["avg_vmov"]
        *
        12
    )


    # ==================================================
    # Convert to clean JSON
    # ==================================================

    pitch_results = []


    for _, row in grouped.iterrows():

        pitch_results.append(
            {

                "period":
                    row["period"],

                "pitch_type":
                    row["pitch_type"],

                "pitch_count":
                    int(
                        row["pitch_count"]
                    ),

                "usage_pct":
                    clean_number(
                        row["usage_pct"]
                    ),

                # --------------------------
                # Stuff
                # --------------------------

                "avg_velocity":
                    clean_number(
                        row["avg_velocity"]
                    ),

                "avg_spin":
                    clean_number(
                        row["avg_spin"],
                        0
                    ),

                "avg_extension":
                    clean_number(
                        row["avg_extension"]
                    ),

                "avg_hmov":
                    clean_number(
                        row["avg_hmov"]
                    ),

                "avg_vmov":
                    clean_number(
                        row["avg_vmov"]
                    ),

                # --------------------------
                # Swing results
                # --------------------------

                "swings":
                    int(
                        row["swings"]
                    ),

                "whiffs":
                    int(
                        row["whiffs"]
                    ),

                "whiff_pct":
                    clean_number(
                        row["whiff_pct"]
                    ),

                # --------------------------
                # Run value
                # --------------------------

                "run_value":
                    clean_number(
                        row["run_value"]
                    ),

                "run_value_pitches":
                    int(row["run_value_pitches"]),

                "run_value_per_100":
                    clean_number(
                        row[
                            "run_value_per_100"
                        ]
                    ),

                # --------------------------
                # Contact quality
                # --------------------------

                "batted_balls":
                    int(
                        row["batted_balls"]
                    ),

                "avg_ev":
                    clean_number(
                        row["avg_ev"]
                    ),

                "hard_hit_pct":
                    clean_number(
                        row["hard_hit_pct"]
                    ),

                "avg_launch_angle":
                    clean_number(
                        row[
                            "avg_launch_angle"
                        ]
                    )

            }
        )


    # ==================================================
    # Overall period summaries
    # ==================================================

    overall_results = []


    for period_name in [

        "early",
        "transition",
        "post"

    ]:

        period_data = data[
            data["period"]
            ==
            period_name
        ]


        if period_data.empty:
            continue


        total_pitches = len(
            period_data
        )


        total_swings = int(
            period_data[
                "is_swing"
            ].sum()
        )


        total_whiffs = int(
            period_data[
                "is_whiff"
            ].sum()
        )


        total_run_value = (
            period_data[
                "pitcher_run_value"
            ].sum(min_count=1)
        )

        run_value_pitches = int(period_data["pitcher_run_value"].count())


        contact_data = (
            period_data[
                period_data[
                    "launch_speed"
                ]
                .notna()
            ]
        )


        hard_hits = (
            contact_data[
                "launch_speed"
            ]
            .ge(95)
            .sum()
        )


        overall_results.append(
            {

                "period":
                    period_name,

                "outings":
                    int(
                        period_data[
                            "game_date"
                        ]
                        .nunique()
                    ),

                "pitches":
                    int(
                        total_pitches
                    ),

                "whiff_pct":

                    clean_number(

                        (
                            total_whiffs
                            /
                            total_swings
                            *
                            100
                        )

                        if total_swings
                        else None

                    ),

                "run_value":
                    clean_number(
                        total_run_value
                    ),

                "run_value_pitches": run_value_pitches,

                "run_value_per_100":

                    clean_number(

                        total_run_value / run_value_pitches * 100
                        if run_value_pitches else None

                    ),

                "avg_ev":
                    clean_number(
                        contact_data[
                            "launch_speed"
                        ]
                        .mean()
                    ),

                "hard_hit_pct":

                    clean_number(

                        (
                            hard_hits
                            /
                            len(
                                contact_data
                            )
                            *
                            100
                        )

                        if len(
                            contact_data
                        )
                        else None

                    )

            }
        )


    return jsonify(
        {

            "comparison_periods": comparison.payload(),
            "transition_window": comparison.legacy_payload(),

            "periods":
                {

                    "early": "Baseline",
                    "transition": "Between periods",
                    "post": "Comparison"

                },

            "overall":
                overall_results,

            "pitches":
                pitch_results

        }
    )
