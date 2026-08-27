from __future__ import annotations

import sqlite3

from pitcher_core import (
    DATABASE_FILE,
    current_research_season,
    get_pitcher_profile,
)
from comparison import ComparisonError, resolve_comparison

import numpy as np
import pandas as pd

from flask import (
    Blueprint,
    jsonify,
    request
)


performance_bp = Blueprint(
    "performance",
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


STRIKEOUT_EVENTS = {

    "strikeout",

    "strikeout_double_play"

}


WALK_EVENTS = {

    "walk",

    "intent_walk"

}


HIT_EVENTS = {

    "single",

    "double",

    "triple",

    "home_run"

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

    row = (
        connection
        .execute(

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

        )
        .fetchone()
    )


    return (
        row
        is not None
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


def safe_int(
    value
):

    number = (
        safe_number(
            value
        )
    )


    if number is None:

        return None


    return int(
        round(
            number
        )
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

        float(
            numerator
        )

        /

        float(
            denominator
        )

        *

        100.0,

        decimals

    )


def outs_to_innings(
    outs
):

    if outs is None:

        return None


    outs = int(
        outs
    )


    return (

        f"{outs // 3}."
        f"{outs % 3}"

    )


def classify_period(
    game_date,
    start_date,
    end_date
):

    if (
        game_date
        <
        start_date
    ):

        return "early"


    if (
        game_date
        <=
        end_date
    ):

        return "transition"


    return "post"


def add_pitch_flags(
    data
):

    data = (
        data.copy()
    )


    numeric_columns = [

        "plate_x",

        "plate_z",

        "sz_top",

        "sz_bot",

        "launch_speed",

        "launch_angle",

        "estimated_woba_using_speedangle",

        "woba_value",

        "woba_denom",

        "delta_run_exp",

        "bat_score",

        "post_bat_score"

    ]


    for column in numeric_columns:

        data[
            column
        ] = (

            pd.to_numeric(
                data[
                    column
                ],
                errors="coerce"
            )

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


    data[
        "zone_eligible"
    ] = (
        valid_zone
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
        "in_zone"
    ] = (

        valid_zone

        &

        (
            data[
                "normalized_x"
            ]
            .abs()
            <=
            1.0
        )

        &

        (
            data[
                "normalized_z"
            ]
            .abs()
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


    # Hard-hit rate should use tracked balls in play,
    # not every pitch that happens to carry an EV field.

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


    # Existing project convention:
    # positive pitcher value = good.

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
        "is_strikeout"
    ] = (

        data[
            "events"
        ]
        .isin(
            STRIKEOUT_EVENTS
        )

    )


    data[
        "is_walk"
    ] = (

        data[
            "events"
        ]
        .isin(
            WALK_EVENTS
        )

    )


    data[
        "is_hbp"
    ] = (

        data[
            "events"
        ]
        .eq(
            "hit_by_pitch"
        )

    )


    data[
        "is_hit"
    ] = (

        data[
            "events"
        ]
        .isin(
            HIT_EVENTS
        )

    )


    data[
        "is_home_run"
    ] = (

        data[
            "events"
        ]
        .eq(
            "home_run"
        )

    )


    score_change = (

        data[
            "post_bat_score"
        ]

        -

        data[
            "bat_score"
        ]

    )


    data[
        "runs_scored_on_pitch"
    ] = (

        score_change
        .where(
            score_change > 0,
            0
        )
        .fillna(
            0
        )

    )


    # Expected wOBA:
    #
    # Use Statcast's estimated value on batted-ball
    # contact when it exists.
    #
    # On BB/HBP/K/non-contact outcomes, retain
    # the actual wOBA value.

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


def process_summary(
    data
):

    pitch_count = len(
        data
    )


    swings = int(

        data[
            "is_swing"
        ].sum()

    )


    whiffs = int(

        data[
            "is_whiff"
        ].sum()

    )


    zone_eligible = int(

        data[
            "zone_eligible"
        ].sum()

    )


    in_zone = int(

        data[
            "in_zone"
        ].sum()

    )


    out_of_zone = int(

        data[
            "out_of_zone"
        ].sum()

    )


    chases = int(

        data[
            "is_chase"
        ].sum()

    )


    tracked_contact = int(

        data[
            "tracked_contact"
        ].sum()

    )


    hard_hits = int(

        data[
            "hard_hit"
        ].sum()

    )


    terminal_pas = (

        data[
            data[
                "is_terminal_pa"
            ]
        ]
        .copy()

    )


    batters_faced = len(
        terminal_pas
    )


    woba_rows = terminal_pas[

        terminal_pas[
            "woba_value"
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


    woba_denominator = (

        woba_rows[
            "woba_denom"
        ].sum()

    )


    woba_allowed = None


    if (
        woba_denominator
        >
        0
    ):

        woba_allowed = (

            (

                woba_rows[
                    "woba_value"
                ]

                *

                woba_rows[
                    "woba_denom"
                ]

            ).sum()

            /

            woba_denominator

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


    xwoba_denominator = (

        xwoba_rows[
            "woba_denom"
        ].sum()

    )


    xwoba_allowed = None


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

        data[
            "pitcher_run_value"
        ]
        .sum(
            min_count=1
        )

    )


    run_value_per_100 = None


    if (
        pitch_count > 0

        and

        pd.notna(
            run_value
        )
    ):

        run_value_per_100 = (

            run_value

            /

            pitch_count

            *

            100.0

        )


    avg_ev = (

        data.loc[

            data[
                "tracked_contact"
            ],

            "launch_speed"

        ]
        .mean()

    )


    strikeouts = int(

        terminal_pas[
            "is_strikeout"
        ].sum()

    )


    walks = int(

        terminal_pas[
            "is_walk"
        ].sum()

    )


    hbp = int(

        terminal_pas[
            "is_hbp"
        ].sum()

    )


    hits = int(

        terminal_pas[
            "is_hit"
        ].sum()

    )


    home_runs = int(

        terminal_pas[
            "is_home_run"
        ].sum()

    )


    runs_while_pitching = int(

        round(

            data[
                "runs_scored_on_pitch"
            ].sum()

        )

    )


    return {

        "pitches":
            int(
                pitch_count
            ),

        "batters_faced_statcast":
            int(
                batters_faced
            ),

        "strikeouts_statcast":
            strikeouts,

        "walks_statcast":
            walks,

        "hit_by_pitch_statcast":
            hbp,

        "hits_statcast":
            hits,

        "home_runs_statcast":
            home_runs,

        "runs_while_pitching_statcast":
            runs_while_pitching,

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

        "in_zone_pitches":
            in_zone,

        "zone_pct":
            pct(
                in_zone,
                zone_eligible
            ),

        "out_of_zone_pitches":
            out_of_zone,

        "chases":
            chases,

        "chase_pct":
            pct(
                chases,
                out_of_zone
            ),

        "tracked_batted_balls":
            tracked_contact,

        "hard_hits":
            hard_hits,

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

        "woba_allowed":
            safe_number(
                woba_allowed,
                3
            ),

        "xwoba_allowed":
            safe_number(
                xwoba_allowed,
                3
            ),

        "pitcher_run_value":
            safe_number(
                run_value,
                2
            ),

        "pitch_value_per_100":
            safe_number(
                run_value_per_100,
                2
            )

    }


def official_summary(
    rows
):

    usable = [

        row

        for row
        in rows

        if row

    ]


    if not usable:

        return None


    def total(
        key
    ):

        values = [

            row.get(
                key
            )

            for row
            in usable

            if row.get(
                key
            )
            is not None

        ]


        if not values:

            return None


        return int(
            sum(
                values
            )
        )


    outs = (
        total(
            "outs_recorded"
        )
    )


    earned_runs = (
        total(
            "earned_runs"
        )
    )


    strikeouts = (
        total(
            "strikeouts"
        )
    )


    walks = (
        total(
            "walks"
        )
    )


    batters_faced = (
        total(
            "batters_faced"
        )
    )


    era = None


    if (
        outs

        and

        earned_runs
        is not None
    ):

        era = (

            earned_runs

            *

            27.0

            /

            outs

        )


    k_bb_pct = None


    if (
        batters_faced

        and

        strikeouts
        is not None

        and

        walks
        is not None
    ):

        k_bb_pct = (

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

        "outings":
            len(
                usable
            ),

        "innings_pitched":
            outs_to_innings(
                outs
            ),

        "outs_recorded":
            outs,

        "hits":
            total(
                "hits"
            ),

        "runs":
            total(
                "runs"
            ),

        "earned_runs":
            earned_runs,

        "walks":
            walks,

        "intentional_walks":
            total(
                "intentional_walks"
            ),

        "hit_by_pitch":
            total(
                "hit_by_pitch"
            ),

        "strikeouts":
            strikeouts,

        "home_runs":
            total(
                "home_runs"
            ),

        "batters_faced":
            batters_faced,

        "pitches":
            total(
                "pitches"
            ),

        "strikes":
            total(
                "strikes"
            ),

        "balls":
            total(
                "balls"
            ),

        "era":
            safe_number(
                era,
                2
            ),

        "k_minus_bb_pct":
            safe_number(
                k_bb_pct,
                1
            )

    }


def load_official_outings(
    connection,
    pitcher_id
):

    if not table_exists(
        connection,
        "official_outings"
    ):

        return {}


    rows = (
        connection
        .execute(

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

        )
        .fetchall()
    )


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


def build_outing_identity(
    group
):

    first = (
        group.iloc[0]
    )


    inning_topbot = str(

        first.get(
            "inning_topbot"
        )
        or
        ""

    )


    home_team = (
        first.get(
            "home_team"
        )
    )


    away_team = (
        first.get(
            "away_team"
        )
    )


    if (
        inning_topbot.lower()
        ==
        "top"
    ):

        team = (
            home_team
        )

        opponent = (
            away_team
        )

        home_away = (
            "Home"
        )

    else:

        team = (
            away_team
        )

        opponent = (
            home_team
        )

        home_away = (
            "Away"
        )


    return {

        "team":
            team,

        "opponent":
            opponent,

        "home_away":
            home_away

    }


def build_pitch_usage(
    target_data
):

    target_data = (

        target_data[
            target_data[
                "pitch_type"
            ].notna()
        ]
        .copy()

    )


    grouped = (

        target_data
        .groupby(
            [
                "game_date",
                "game_pk",
                "pitch_type"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="pitch_count"
        )

    )


    totals = (

        grouped
        .groupby(
            [
                "game_date",
                "game_pk"
            ]
        )[
            "pitch_count"
        ]
        .sum()
        .reset_index(
            name="outing_pitches"
        )

    )


    grouped = (
        grouped.merge(

            totals,

            on=[
                "game_date",
                "game_pk"
            ],

            how="left"

        )
    )


    grouped[
        "usage_pct"
    ] = (

        grouped[
            "pitch_count"
        ]

        /

        grouped[
            "outing_pitches"
        ]

        *

        100.0

    )


    results = []


    for row in (

        grouped
        .sort_values(
            [
                "game_date",
                "pitch_type"
            ]
        )
        .itertuples(
            index=False
        )

    ):

        results.append(
            {

                "game_date":
                    row.game_date.strftime(
                        "%Y-%m-%d"
                    ),

                "game_pk":
                    int(
                        row.game_pk
                    ),

                "pitch_type":
                    row.pitch_type,

                "pitch_count":
                    int(
                        row.pitch_count
                    ),

                "usage_pct":
                    round(
                        float(
                            row.usage_pct
                        ),
                        1
                    )

            }
        )


    return results


def build_pitch_periods(
    target_data
):

    results = []


    target_data = (

        target_data[
            target_data[
                "pitch_type"
            ].notna()
        ]
        .copy()

    )


    for (
        period,
        pitch_type
    ), group in (

        target_data
        .groupby(
            [
                "period",
                "pitch_type"
            ]
        )

    ):

        period_total = len(

            target_data[
                target_data[
                    "period"
                ]
                ==
                period
            ]

        )


        summary = (
            process_summary(
                group
            )
        )


        summary.update(
            {

                "period":
                    period,

                "pitch_type":
                    pitch_type,

                "usage_pct":

                    round(

                        len(
                            group
                        )

                        /

                        period_total

                        *

                        100.0,

                        1

                    )

                    if period_total

                    else None

            }
        )


        results.append(
            summary
        )


    order = {

        "early":
            0,

        "transition":
            1,

        "post":
            2

    }


    results.sort(

        key=lambda row:

            (
                order.get(
                    row[
                        "period"
                    ],
                    99
                ),

                row[
                    "pitch_type"
                ]
                or
                ""
            )

    )


    return results


@performance_bp.route("/api/pitchers/<int:pitcher_id>/performance")
def pitcher_performance(pitcher_id):

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


    connection = (
        connect_database()
    )


    query = """
    SELECT

        season,

        game_date,

        game_pk,

        pitch_type,

        home_team,

        away_team,

        inning_topbot,

        events,

        description,

        plate_x,

        plate_z,

        sz_top,

        sz_bot,

        launch_speed,

        launch_angle,

        estimated_woba_using_speedangle,

        woba_value,

        woba_denom,

        delta_run_exp,

        bat_score,

        post_bat_score,

        at_bat_number,

        pitch_number,

        stand

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


    data = (
        pd.read_sql_query(

            query,

            connection,

            params=(
                int(pitcher_id),
            )

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
                    "No regular-season Statcast data found for this pitcher."
            }
        ), 404


    data[
        "game_date"
    ] = (

        pd.to_datetime(
            data[
                "game_date"
            ],
            errors="coerce"
        )

    )


    data[
        "season"
    ] = (

        pd.to_numeric(
            data[
                "season"
            ],
            errors="coerce"
        )

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


    data = (
        add_pitch_flags(
            data
        )
    )


    target_data = (
        data.copy()
        if scope == "career"
        else data[data["season"] == target_season].copy()
    )


    if target_data.empty:

        return jsonify(
            {
                "error":
                    f"No {target_season} Statcast data found for this pitcher."
            }
        ), 404


    target_data[
        "period"
    ] = (

        target_data[
            "game_date"
        ]
        .apply(comparison.classify)

    )
    target_data = target_data[target_data["period"].notna()].copy()


    outings = []


    for (
        game_date,
        game_pk
    ), group in (

        target_data
        .groupby(
            [
                "game_date",
                "game_pk"
            ]
        )

    ):

        game_pk = int(
            game_pk
        )


        identity = (
            build_outing_identity(
                group
            )
        )


        process = (
            process_summary(
                group
            )
        )


        official = (
            official_by_game.get(
                game_pk
            )
        )


        period = comparison.classify(game_date)


        official_payload = None


        if official:

            outs = (
                safe_int(
                    official.get(
                        "outs_recorded"
                    )
                )
            )


            earned_runs = (
                safe_int(
                    official.get(
                        "earned_runs"
                    )
                )
            )


            strikeouts = (
                safe_int(
                    official.get(
                        "strikeouts"
                    )
                )
            )


            walks = (
                safe_int(
                    official.get(
                        "walks"
                    )
                )
            )


            batters_faced = (
                safe_int(
                    official.get(
                        "batters_faced"
                    )
                )
            )


            outing_era = None


            if (
                outs

                and

                earned_runs
                is not None
            ):

                outing_era = (

                    earned_runs

                    *

                    27.0

                    /

                    outs

                )


            k_bb_pct = None


            if (
                batters_faced

                and

                strikeouts
                is not None

                and

                walks
                is not None
            ):

                k_bb_pct = (

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


            official_payload = {

                "innings_pitched":
                    official.get(
                        "innings_pitched"
                    ),

                "outs_recorded":
                    outs,

                "hits":
                    safe_int(
                        official.get(
                            "hits"
                        )
                    ),

                "runs":
                    safe_int(
                        official.get(
                            "runs"
                        )
                    ),

                "earned_runs":
                    earned_runs,

                "walks":
                    walks,

                "intentional_walks":
                    safe_int(
                        official.get(
                            "intentional_walks"
                        )
                    ),

                "hit_by_pitch":
                    safe_int(
                        official.get(
                            "hit_by_pitch"
                        )
                    ),

                "strikeouts":
                    strikeouts,

                "home_runs":
                    safe_int(
                        official.get(
                            "home_runs"
                        )
                    ),

                "batters_faced":
                    batters_faced,

                "pitches":
                    safe_int(
                        official.get(
                            "pitches"
                        )
                    ),

                "strikes":
                    safe_int(
                        official.get(
                            "strikes"
                        )
                    ),

                "balls":
                    safe_int(
                        official.get(
                            "balls"
                        )
                    ),

                "era":
                    safe_number(
                        outing_era,
                        2
                    ),

                "k_minus_bb_pct":
                    safe_number(
                        k_bb_pct,
                        1
                    )

            }


            identity[
                "team"
            ] = (

                official.get(
                    "team"
                )
                or
                identity[
                    "team"
                ]

            )


            identity[
                "opponent"
            ] = (

                official.get(
                    "opponent"
                )
                or
                identity[
                    "opponent"
                ]

            )


            identity[
                "home_away"
            ] = (

                official.get(
                    "home_away"
                )
                or
                identity[
                    "home_away"
                ]

            )


        outings.append(
            {

                "game_date":
                    game_date.strftime(
                        "%Y-%m-%d"
                    ),

                "game_pk":
                    game_pk,

                "period":
                    period,

                **identity,

                "official":
                    official_payload,

                "process":
                    process

            }
        )


    outings.sort(
        key=lambda row:
            row[
                "game_date"
            ]
    )


    period_summaries = []


    for period in (
        "early",
        "transition",
        "post"
    ):

        period_pitches = (

            target_data[
                target_data[
                    "period"
                ]
                ==
                period
            ]
            .copy()

        )


        period_outings = [

            row

            for row
            in outings

            if row[
                "period"
            ]
            ==
            period

        ]


        official_rows = [

            official_by_game.get(
                row[
                    "game_pk"
                ]
            )

            for row
            in period_outings

        ]


        period_summaries.append(
            {

                "period":
                    period,

                "outing_count":
                    len(
                        period_outings
                    ),

                "official":
                    official_summary(
                        official_rows
                    ),

                "process":

                    process_summary(
                        period_pitches
                    )

                    if not period_pitches.empty

                    else None

            }
        )


    season_summaries = []


    for season in sorted(
        int(value) for value in data["season"].dropna().unique().tolist()
    ):

        season_pitches = (

            data[
                data[
                    "season"
                ]
                ==
                season
            ]
            .copy()

        )


        if season_pitches.empty:

            continue


        game_pks = [

            int(
                value
            )

            for value
            in season_pitches[
                "game_pk"
            ]
            .dropna()
            .unique()
            .tolist()

        ]


        official_rows = [

            official_by_game.get(
                game_pk
            )

            for game_pk
            in game_pks

        ]


        season_summaries.append(
            {

                "season":
                    season,

                "outing_count":
                    len(
                        game_pks
                    ),

                "official":
                    official_summary(
                        official_rows
                    ),

                "process":
                    process_summary(
                        season_pitches
                    )

            }
        )



    try:
        player_profile = get_pitcher_profile(pitcher_id)
    except Exception:
        player_profile = {"name": f"MLB Pitcher {pitcher_id}", "mlbam_id": pitcher_id}

    response = {

        "player": {

            "name":
                player_profile.get("name"),

            "mlbam_id":
                int(pitcher_id),

            "target_season":
                int(target_season)

        },


        "comparison_periods": comparison.payload(),
        "transition_window": comparison.legacy_payload(),


        "data_status": {

            "official_outings_available":
                len(
                    official_by_game
                )
                >
                0,

            "official_outings_cached":
                len(
                    official_by_game
                ),

            "latest_game_date":

                outings[-1][
                    "game_date"
                ]

                if outings

                else None,

            "latest_game_pk":

                outings[-1][
                    "game_pk"
                ]

                if outings

                else None

        },


        "definitions": {

            "official_outcomes":

                (
                    "IP, H, R, ER, BB, K, HR "
                    "and official pitch counts "
                    "come from cached MLB boxscores."
                ),


            "whiff_pct":

                (
                    "Whiffs divided by swings."
                ),


            "chase_pct":

                (
                    "Swings at pitches outside "
                    "the normalized strike zone "
                    "divided by located pitches "
                    "outside the zone."
                ),


            "zone_pct":

                (
                    "Located pitches inside the "
                    "normalized strike zone divided "
                    "by all pitches with usable "
                    "location and zone data."
                ),


            "hard_hit_pct":

                (
                    "Tracked batted balls at "
                    "95+ mph divided by tracked "
                    "balls in play."
                ),


            "pitch_value_per_100":

                (
                    "Negative Statcast delta run "
                    "expectancy, scaled to 100 pitches "
                    "so positive is better for "
                    "the pitcher."
                ),


            "woba_allowed":

                (
                    "Actual wOBA allowed across "
                    "plate appearances with a "
                    "Statcast wOBA denominator."
                ),


            "xwoba_allowed":

                (
                    "Expected wOBA uses Statcast "
                    "estimated wOBA on tracked contact "
                    "and actual wOBA values for "
                    "non-contact outcomes such as "
                    "walks, hit-by-pitches and "
                    "strikeouts."
                ),


            "statcast_runs_note":

                (
                    "runs_while_pitching_statcast "
                    "counts scoring changes on pitches "
                    "the selected pitcher threw. It is not used as "
                    "a substitute for official "
                    "earned runs."
                )

        },


        "outings":
            outings,


        "periods":
            period_summaries,


        "seasons":
            season_summaries,


        "pitch_usage":
            build_pitch_usage(
                target_data
            ),


        "pitch_periods":
            build_pitch_periods(
                target_data
            )

    }


    return jsonify(
        response
    )
