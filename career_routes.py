from __future__ import annotations

import sqlite3
from pitcher_core import DATABASE_FILE, current_research_season, default_transition_window, get_pitcher_profile

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request


career_bp = Blueprint(
    "career",
    __name__
)






SWING_DESCRIPTIONS = {

    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "foul_bunt",
    "missed_bunt",
    "swinging_pitchout"

}


WHIFF_DESCRIPTIONS = {

    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "swinging_pitchout"

}


def connect_database():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


def table_exists(
    connection,
    table_name
):

    row = connection.execute(
        """
        SELECT
            1

        FROM
            sqlite_master

        WHERE
            type = 'table'

            AND

            name = ?

        LIMIT 1;
        """,
        (
            table_name,
        )
    ).fetchone()

    return (
        row is not None
    )


def safe_number(
    value,
    decimals=None
):

    if value is None:
        return None

    try:

        number = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return None


    if (
        pd.isna(
            number
        )

        or

        np.isinf(
            number
        )
    ):

        return None


    if decimals is None:
        return number


    return round(
        number,
        decimals
    )


def pct(
    numerator,
    denominator,
    decimals=1
):

    if (
        denominator is None
        or
        denominator == 0
    ):

        return None


    return round(
        (
            float(
                numerator
            )
            /
            float(
                denominator
            )
        )
        *
        100.0,
        decimals
    )


def load_official_outings(
    connection,
    pitcher_id
):

    if not table_exists(
        connection,
        "official_outings"
    ):

        return {}


    rows = connection.execute(
        """
        SELECT
            *

        FROM
            official_outings

        WHERE
            pitcher_id = ?

        ORDER BY
            game_date;
        """,
        (
            int(pitcher_id),
        )
    ).fetchall()


    return {

        int(
            row[
                "game_pk"
            ]
        ):
            dict(
                row
            )

        for row
        in rows

    }


def add_flags(
    data
):

    data = data.copy()


    numeric_columns = [

        "release_speed",
        "release_spin_rate",
        "release_extension",

        "release_pos_x",
        "release_pos_z",

        "pfx_x",
        "pfx_z",

        "plate_x",
        "plate_z",

        "sz_top",
        "sz_bot",

        "launch_speed",

        "estimated_woba_using_speedangle",
        "woba_value",
        "woba_denom",

        "delta_run_exp"

    ]


    for column in numeric_columns:

        data[
            column
        ] = pd.to_numeric(
            data[
                column
            ],
            errors="coerce"
        )


    data[
        "is_swing"
    ] = (

        data[
            "description"
        ]
        .isin(
            SWING_DESCRIPTIONS
        )

    )


    data[
        "is_whiff"
    ] = (

        data[
            "description"
        ]
        .isin(
            WHIFF_DESCRIPTIONS
        )

    )


    valid_zone = (

        data[
            "plate_x"
        ].notna()

        &

        data[
            "plate_z"
        ].notna()

        &

        data[
            "sz_top"
        ].notna()

        &

        data[
            "sz_bot"
        ].notna()

        &

        (
            data[
                "sz_top"
            ]
            >
            data[
                "sz_bot"
            ]
        )

    )


    zone_center = (

        data[
            "sz_top"
        ]

        +

        data[
            "sz_bot"
        ]

    ) / 2.0


    zone_half_height = (

        data[
            "sz_top"
        ]

        -

        data[
            "sz_bot"
        ]

    ) / 2.0


    data[
        "normalized_x"
    ] = (

        data[
            "plate_x"
        ]

        /

        0.83

    )


    data[
        "normalized_z"
    ] = (

        (
            data[
                "plate_z"
            ]

            -

            zone_center

        )

        /

        zone_half_height

    )


    data[
        "zone_eligible"
    ] = (
        valid_zone
    )


    data[
        "in_zone"
    ] = (

        valid_zone

        &

        (
            data[
                "normalized_x"
            ].abs()
            <=
            1.0
        )

        &

        (
            data[
                "normalized_z"
            ].abs()
            <=
            1.0
        )

    )


    data[
        "out_of_zone"
    ] = (

        valid_zone

        &

        ~data[
            "in_zone"
        ]

    )


    data[
        "is_chase"
    ] = (

        data[
            "out_of_zone"
        ]

        &

        data[
            "is_swing"
        ]

    )


    data[
        "tracked_contact"
    ] = (

        data[
            "launch_speed"
        ].notna()

        &

        data[
            "description"
        ].eq(
            "hit_into_play"
        )

    )


    data[
        "hard_hit"
    ] = (

        data[
            "tracked_contact"
        ]

        &

        (
            data[
                "launch_speed"
            ]
            >=
            95.0
        )

    )


    data[
        "pitcher_run_value"
    ] = (

        -data[
            "delta_run_exp"
        ]

    )


    data[
        "is_terminal_pa"
    ] = (

        data[
            "events"
        ].notna()

    )


    data[
        "xwoba_component"
    ] = (

        data[
            "estimated_woba_using_speedangle"
        ]

    )


    missing_estimate = (

        data[
            "xwoba_component"
        ].isna()

    )


    data.loc[
        missing_estimate,
        "xwoba_component"
    ] = (

        data.loc[
            missing_estimate,
            "woba_value"
        ]

    )


    return data


def overall_process_summary(
    group
):

    pitches = len(
        group
    )


    swings = int(
        group[
            "is_swing"
        ].sum()
    )


    whiffs = int(
        group[
            "is_whiff"
        ].sum()
    )


    zone_eligible = int(
        group[
            "zone_eligible"
        ].sum()
    )


    in_zone = int(
        group[
            "in_zone"
        ].sum()
    )


    out_of_zone = int(
        group[
            "out_of_zone"
        ].sum()
    )


    chases = int(
        group[
            "is_chase"
        ].sum()
    )


    tracked_contact = int(
        group[
            "tracked_contact"
        ].sum()
    )


    hard_hits = int(
        group[
            "hard_hit"
        ].sum()
    )


    terminal_pas = (

        group[
            group[
                "is_terminal_pa"
            ]
        ]
        .copy()

    )


    xwoba_rows = terminal_pas[

        terminal_pas[
            "xwoba_component"
        ].notna()

        &

        terminal_pas[
            "woba_denom"
        ].notna()

        &

        (
            terminal_pas[
                "woba_denom"
            ]
            >
            0
        )

    ]


    xwoba_allowed = None


    xwoba_denominator = (

        xwoba_rows[
            "woba_denom"
        ].sum()

    )


    if (
        xwoba_denominator
        >
        0
    ):

        xwoba_allowed = (

            (

                xwoba_rows[
                    "xwoba_component"
                ]

                *

                xwoba_rows[
                    "woba_denom"
                ]

            ).sum()

            /

            xwoba_denominator

        )


    run_value = (

        group[
            "pitcher_run_value"
        ]
        .sum(
            min_count=1
        )

    )


    run_value_per_100 = None


    if (
        pitches > 0

        and

        pd.notna(
            run_value
        )
    ):

        run_value_per_100 = (

            run_value

            /

            pitches

            *

            100.0

        )


    avg_ev = (

        group.loc[

            group[
                "tracked_contact"
            ],

            "launch_speed"

        ]
        .mean()

    )


    return {

        "pitches":
            int(
                pitches
            ),

        "swings":
            swings,

        "whiffs":
            whiffs,

        "whiff_pct":
            pct(
                whiffs,
                swings
            ),

        "zone_eligible_pitches":
            zone_eligible,

        "zone_pct":
            pct(
                in_zone,
                zone_eligible
            ),

        "out_of_zone_pitches":
            out_of_zone,

        "chase_pct":
            pct(
                chases,
                out_of_zone
            ),

        "tracked_batted_balls":
            tracked_contact,

        "hard_hit_pct":
            pct(
                hard_hits,
                tracked_contact
            ),

        "avg_exit_velocity":
            safe_number(
                avg_ev,
                1
            ),

        "xwoba_allowed":
            safe_number(
                xwoba_allowed,
                3
            ),

        "pitch_value_per_100":
            safe_number(
                run_value_per_100,
                2
            )

    }


def official_payload(
    row
):

    if not row:
        return None


    batters_faced = safe_number(
        row.get(
            "batters_faced"
        )
    )


    strikeouts = safe_number(
        row.get(
            "strikeouts"
        )
    )


    walks = safe_number(
        row.get(
            "walks"
        )
    )


    k_minus_bb_pct = None


    if (
        batters_faced

        and

        strikeouts is not None

        and

        walks is not None
    ):

        k_minus_bb_pct = (

            (
                strikeouts
                -
                walks
            )

            /

            batters_faced

            *

            100.0

        )


    return {

        "innings_pitched":
            row.get(
                "innings_pitched"
            ),

        "earned_runs":

            int(
                row[
                    "earned_runs"
                ]
            )

            if row.get(
                "earned_runs"
            )
            is not None

            else None,


        "runs":

            int(
                row[
                    "runs"
                ]
            )

            if row.get(
                "runs"
            )
            is not None

            else None,


        "hits":

            int(
                row[
                    "hits"
                ]
            )

            if row.get(
                "hits"
            )
            is not None

            else None,


        "walks":

            int(
                row[
                    "walks"
                ]
            )

            if row.get(
                "walks"
            )
            is not None

            else None,


        "strikeouts":

            int(
                row[
                    "strikeouts"
                ]
            )

            if row.get(
                "strikeouts"
            )
            is not None

            else None,


        "home_runs":

            int(
                row[
                    "home_runs"
                ]
            )

            if row.get(
                "home_runs"
            )
            is not None

            else None,


        "batters_faced":

            int(
                row[
                    "batters_faced"
                ]
            )

            if row.get(
                "batters_faced"
            )
            is not None

            else None,


        "k_minus_bb_pct":
            safe_number(
                k_minus_bb_pct,
                1
            ),

        "opponent":
            row.get(
                "opponent"
            ),

        "home_away":
            row.get(
                "home_away"
            )

    }


def pitch_outing_summary(
    group,
    outing_total
):

    pitches = len(
        group
    )


    swings = int(
        group[
            "is_swing"
        ].sum()
    )


    whiffs = int(
        group[
            "is_whiff"
        ].sum()
    )


    run_value = (

        group[
            "pitcher_run_value"
        ]
        .sum(
            min_count=1
        )

    )


    run_value_per_100 = None


    if (
        pitches > 0

        and

        pd.notna(
            run_value
        )
    ):

        run_value_per_100 = (

            run_value

            /

            pitches

            *

            100.0

        )


    return {

        "pitch_count":
            int(
                pitches
            ),

        "usage_pct":
            pct(
                pitches,
                outing_total
            ),

        "avg_velocity":
            safe_number(
                group[
                    "release_speed"
                ].mean(),
                2
            ),

        "avg_spin":
            safe_number(
                group[
                    "release_spin_rate"
                ].mean(),
                1
            ),

        "avg_extension":
            safe_number(
                group[
                    "release_extension"
                ].mean(),
                2
            ),

        "avg_release_x":
            safe_number(
                group[
                    "release_pos_x"
                ].mean(),
                2
            ),

        "avg_release_z":
            safe_number(
                group[
                    "release_pos_z"
                ].mean(),
                2
            ),

        "avg_arm_angle":
            safe_number(
                group[
                    "arm_angle"
                ].mean(),
                2
            ),

        "avg_horizontal_movement":
            safe_number(
                group[
                    "pfx_x"
                ].mean()
                *
                12.0,
                2
            ),

        "avg_vertical_movement":
            safe_number(
                group[
                    "pfx_z"
                ].mean()
                *
                12.0,
                2
            ),

        "swings":
            swings,

        "whiff_pct":
            pct(
                whiffs,
                swings
            ),

        "pitch_value_per_100":
            safe_number(
                run_value_per_100,
                2
            )

    }


@career_bp.route("/api/pitchers/<int:pitcher_id>/career")
def pitcher_career(pitcher_id):

    connection = (
        connect_database()
    )


    query = """
    SELECT

        season,

        game_date,

        game_pk,

        pitch_type,

        release_speed,

        release_spin_rate,

        release_extension,

        release_pos_x,

        release_pos_z,

        arm_angle,

        pfx_x,

        pfx_z,

        description,

        events,

        plate_x,

        plate_z,

        sz_top,

        sz_bot,

        launch_speed,

        estimated_woba_using_speedangle,

        woba_value,

        woba_denom,

        delta_run_exp

    FROM pitches

    WHERE

        game_type = 'R'

        AND

        CAST(
            pitcher
            AS INTEGER
        )
        = ?


    ORDER BY

        game_date,

        game_pk,

        at_bat_number,

        pitch_number;
    """


    data = pd.read_sql_query(
        query,
        connection,
        params=(
            int(pitcher_id),
        )
    )


    official_by_game = (
        load_official_outings(
            connection,
            pitcher_id
        )
    )


    connection.close()


    if data.empty:

        return jsonify(
            {
                "error":
                    "No regular-season Statcast data found."
            }
        ), 404


    data[
        "game_date"
    ] = pd.to_datetime(
        data[
            "game_date"
        ],
        errors="coerce"
    )


    data[
        "season"
    ] = pd.to_numeric(
        data[
            "season"
        ],
        errors="coerce"
    )


    data = (

        data[
            data[
                "game_date"
            ].notna()

            &

            data[
                "season"
            ].notna()
        ]
        .copy()

    )


    data = add_flags(
        data
    )


    overall_outings = []

    pitch_outings = []


    grouped_games = data.groupby(
        [
            "season",
            "game_date",
            "game_pk"
        ],
        sort=True
    )


    for (
        season,
        game_date,
        game_pk
    ), outing in grouped_games:

        game_pk = int(
            game_pk
        )


        outing_total = len(
            outing
        )


        overall_outings.append(
            {

                "season":
                    int(
                        season
                    ),

                "game_date":
                    game_date.strftime(
                        "%Y-%m-%d"
                    ),

                "game_pk":
                    game_pk,

                "official":
                    official_payload(
                        official_by_game.get(
                            game_pk
                        )
                    ),

                "process":
                    overall_process_summary(
                        outing
                    )

            }
        )


        pitch_data = (

            outing[
                outing[
                    "pitch_type"
                ].notna()
            ]

        )


        for (
            pitch_type,
            pitch_group
        ) in pitch_data.groupby(
            "pitch_type"
        ):

            pitch_outings.append(
                {

                    "season":
                        int(
                            season
                        ),

                    "game_date":
                        game_date.strftime(
                            "%Y-%m-%d"
                        ),

                    "game_pk":
                        game_pk,

                    "pitch_type":
                        pitch_type,

                    **pitch_outing_summary(
                        pitch_group,
                        outing_total
                    )

                }
            )


    overall_outings.sort(
        key=lambda row:
        (
            row[
                "game_date"
            ],
            row[
                "game_pk"
            ]
        )
    )


    pitch_outings.sort(
        key=lambda row:
        (
            row[
                "game_date"
            ],
            row[
                "game_pk"
            ],
            row[
                "pitch_type"
            ]
        )
    )


    target_season = request.args.get("season", type=int)
    if target_season is None:
        target_season = current_research_season(pitcher_id)
    screen_start, screen_end = default_transition_window(pitcher_id, target_season)
    try:
        player_profile = get_pitcher_profile(pitcher_id)
    except Exception:
        player_profile = {"name": f"MLB Pitcher {pitcher_id}"}

    return jsonify(
        {

            "player": {

                "name":
                    player_profile.get("name"),

                "mlbam_id":
                    int(pitcher_id)

            },


            "career_start":

                overall_outings[
                    0
                ][
                    "game_date"
                ]

                if overall_outings

                else None,


            "career_end":

                overall_outings[
                    -1
                ][
                    "game_date"
                ]

                if overall_outings

                else None,


            "current_screen_window": {

                "start":
                    screen_start,

                "end":
                    screen_end,

                "label":
                    f"{target_season} research screening window" if target_season else "Research screening window"

            },


            "overall_outings":
                overall_outings,


            "pitch_outings":
                pitch_outings,


            "definitions": {

                "career_timeline":

                    (
                        "Every regular-season MLB outing currently "
                        "stored for the selected pitcher. The timeline does not "
                        "assume that a meaningful change began in any specific season."
                    ),


                "rolling_average":

                    (
                        "The chart can overlay a three-outing "
                        "rolling average for readability. It is "
                        "descriptive and is not itself a formal "
                        "change-point result."
                    ),


                "screen_window":

                    (
                        "The shaded interval is the current research screening window, shown "
                        "for comparison rather than treated as "
                        "a known starting point."
                    ),


                "season_audit":

                    (
                        "Season cards compare the selected "
                        "metric's full-season average with the "
                        "first five and last five usable outings. "
                        "This is a descriptive drift check, not "
                        "a causal or formal structural-break test."
                    )

            }

        }
    )
