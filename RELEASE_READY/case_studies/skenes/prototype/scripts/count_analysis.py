from pathlib import Path
import sqlite3
import json

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "pitcher_research.db"
OUT_PATH = ROOT / "data" / "skenes_count_analysis.json"

PITCHER_ID = 694973
PERFORMANCE_BREAK = pd.Timestamp("2026-05-17")

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "foul_bunt",
    "missed_bunt",
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def pct(num, den):
    if den is None or den == 0:
        return None
    return round(float(num) / float(den) * 100.0, 2)


def clean_number(value):
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return round(float(value), 4)

    if isinstance(value, float):
        if np.isnan(value):
            return None
        return round(value, 4)

    return value


def clean_records(records):
    cleaned = []

    for record in records:
        row = {}
        for key, value in record.items():
            if pd.isna(value):
                row[key] = None
            else:
                row[key] = clean_number(value)
        cleaned.append(row)

    return cleaned


def summarize(group):
    if len(group) == 0:
        return {}

    valid_loc = group["location_valid"]
    zone = group["zone"]
    edge = group["edge"]
    heart = group["heart"]
    outside = group["outside_zone"]

    swings = group["swing"]
    whiffs = group["whiff"]

    outside_valid = valid_loc & outside
    chase_swings = outside_valid & swings

    rv = group["pitcher_run_value"].dropna()

    return {
        "pitches": int(len(group)),
        "location_pitches": int(valid_loc.sum()),

        "zone_pct": pct(
            (valid_loc & zone).sum(),
            valid_loc.sum()
        ),

        "outside_zone_pct": pct(
            outside_valid.sum(),
            valid_loc.sum()
        ),

        "edge_pct": pct(
            (valid_loc & edge).sum(),
            valid_loc.sum()
        ),

        "heart_pct": pct(
            (valid_loc & heart).sum(),
            valid_loc.sum()
        ),

        "chase_pct": pct(
            chase_swings.sum(),
            outside_valid.sum()
        ),

        "swings": int(swings.sum()),

        "whiff_pct": pct(
            whiffs.sum(),
            swings.sum()
        ),

        "run_value": round(float(rv.sum()), 3) if len(rv) else None,

        "run_value_per_100": (
            round(float(rv.sum()) / len(group) * 100.0, 3)
            if len(rv)
            else None
        ),
    }


def grouped_summary(df, group_columns):
    rows = []

    for keys, group in df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            column: key
            for column, key in zip(group_columns, keys)
        }

        row.update(summarize(group))
        rows.append(row)

    return clean_records(rows)


def pitch_usage(df, group_columns):
    rows = []

    for keys, group in df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = {
            column: key
            for column, key in zip(group_columns, keys)
        }

        total = len(group)

        counts = (
            group["pitch_type"]
            .fillna("UNKNOWN")
            .value_counts()
        )

        for pitch_type, count in counts.items():
            row = dict(base)
            row.update({
                "pitch_type": pitch_type,
                "pitches": int(count),
                "usage_pct": pct(count, total),
            })
            rows.append(row)

    return clean_records(rows)


# ------------------------------------------------------------
# LOAD 2026 PITCH DATA
# ------------------------------------------------------------

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Database not found:\n{DB_PATH}"
    )

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    game_date,
    game_pk,
    at_bat_number,
    pitch_number,
    pitcher,
    batter,
    stand,
    pitch_type,
    pitch_name,
    balls,
    strikes,
    plate_x,
    plate_z,
    sz_top,
    sz_bot,
    description,
    events,
    type,
    delta_run_exp,
    delta_pitcher_run_exp,
    game_type,
    game_year
FROM pitches
WHERE pitcher = ?
  AND game_type = 'R'
  AND game_year = 2026
ORDER BY
    game_date,
    game_pk,
    at_bat_number,
    pitch_number
"""

df = pd.read_sql_query(
    query,
    conn,
    params=[PITCHER_ID]
)

conn.close()

if df.empty:
    raise RuntimeError(
        "No 2026 regular-season Skenes pitches were found."
    )

df["game_date"] = pd.to_datetime(df["game_date"])

numeric_columns = [
    "balls",
    "strikes",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "delta_run_exp",
    "delta_pitcher_run_exp",
    "pitch_number",
    "at_bat_number",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# PERIOD
# ------------------------------------------------------------

df["period"] = np.where(
    df["game_date"] < PERFORMANCE_BREAK,
    "Pre May 17",
    "May 17 onward"
)


# ------------------------------------------------------------
# NORMALIZED STRIKE ZONE
#
# Same basic concept used by Pitcher Research Lab:
# plate_x normalized to half-width of zone
# plate_z normalized using batter-specific sz_top / sz_bot
# ------------------------------------------------------------

zone_height = df["sz_top"] - df["sz_bot"]

df["location_valid"] = (
    df["plate_x"].notna()
    & df["plate_z"].notna()
    & df["sz_top"].notna()
    & df["sz_bot"].notna()
    & (zone_height > 0)
)

zone_mid = (df["sz_top"] + df["sz_bot"]) / 2.0
zone_half_height = zone_height / 2.0

df["x_norm"] = np.where(
    df["location_valid"],
    df["plate_x"] / 0.83,
    np.nan
)

df["z_norm"] = np.where(
    df["location_valid"],
    (df["plate_z"] - zone_mid) / zone_half_height,
    np.nan
)

df["zone"] = (
    df["location_valid"]
    & (df["x_norm"].abs() <= 1.0)
    & (df["z_norm"].abs() <= 1.0)
)

df["outside_zone"] = (
    df["location_valid"]
    & ~df["zone"]
)

# Central 50% of normalized zone
df["heart"] = (
    df["zone"]
    & (df["x_norm"].abs() <= 0.50)
    & (df["z_norm"].abs() <= 0.50)
)

# Outer third of normalized strike zone
df["edge"] = (
    df["zone"]
    & (
        (df["x_norm"].abs() >= (2.0 / 3.0))
        | (df["z_norm"].abs() >= (2.0 / 3.0))
    )
)


# ------------------------------------------------------------
# SWINGS / WHIFFS / RUN VALUE
# ------------------------------------------------------------

df["swing"] = (
    df["description"]
    .fillna("")
    .isin(SWING_DESCRIPTIONS)
)

df["whiff"] = (
    df["description"]
    .fillna("")
    .isin(WHIFF_DESCRIPTIONS)
)

# Match the research project's pitcher-perspective convention.
# Positive = good for pitcher.
df["pitcher_run_value"] = -pd.to_numeric(
    df["delta_run_exp"],
    errors="coerce"
)


# ------------------------------------------------------------
# COUNT LABELS
# ------------------------------------------------------------

df["exact_count"] = (
    df["balls"].fillna(-1).astype(int).astype(str)
    + "-"
    + df["strikes"].fillna(-1).astype(int).astype(str)
)


def classify_count(row):
    balls = row["balls"]
    strikes = row["strikes"]

    if pd.isna(balls) or pd.isna(strikes):
        return "Unknown"

    balls = int(balls)
    strikes = int(strikes)

    if balls == 0 and strikes == 0:
        return "0-0"

    if balls > strikes:
        return "Hitter ahead"

    if balls < strikes:
        return "Pitcher ahead"

    return "Even"


df["count_state"] = df.apply(
    classify_count,
    axis=1
)

df["two_strike"] = df["strikes"] == 2


# ------------------------------------------------------------
# 1. OVERALL COUNT BEHAVIOR
# ------------------------------------------------------------

overall_by_count_state = grouped_summary(
    df,
    ["period", "count_state"]
)

two_strike_summary = grouped_summary(
    df[df["two_strike"]],
    ["period"]
)

exact_count_summary = grouped_summary(
    df,
    ["period", "exact_count"]
)


# ------------------------------------------------------------
# 2. RHH HARD-STUFF COMMAND
#
# This directly tests the hypothesis:
# Is he attacking RHH less with FF + SI after the performance break?
# ------------------------------------------------------------

rhh_hard = df[
    (df["stand"] == "R")
    & (df["pitch_type"].isin(["FF", "SI"]))
].copy()

rhh_hard_overall = grouped_summary(
    rhh_hard,
    ["period"]
)

rhh_hard_by_count_state = grouped_summary(
    rhh_hard,
    ["period", "count_state"]
)

rhh_hard_two_strike = grouped_summary(
    rhh_hard[rhh_hard["two_strike"]],
    ["period"]
)

rhh_hard_exact_counts = grouped_summary(
    rhh_hard,
    ["period", "exact_count"]
)


# ------------------------------------------------------------
# 3. PITCH USAGE BY COUNT VS RHH
#
# Tests whether his arsenal deployment changed in specific counts.
# ------------------------------------------------------------

rhh = df[df["stand"] == "R"].copy()

rhh_usage_by_count_state = pitch_usage(
    rhh,
    ["period", "count_state"]
)

rhh_exact_count_usage = pitch_usage(
    rhh,
    ["period", "exact_count"]
)


# ------------------------------------------------------------
# 4. PLATE-APPEARANCE CONTROL BEHAVIOR
#
# Tests whether he is reaching disadvantage counts more often.
# ------------------------------------------------------------

pa_rows = []

for (period, hand, game_pk, at_bat_number), group in df.groupby(
    ["period", "stand", "game_pk", "at_bat_number"],
    dropna=False
):
    group = group.sort_values("pitch_number")

    first_pitch = group.iloc[0]

    events = set(
        group["events"]
        .dropna()
        .astype(str)
    )

    pa_rows.append({
        "period": period,
        "hand": hand,
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,

        "pitches": len(group),

        "first_pitch_ball": (
            str(first_pitch["type"]) == "B"
        ),

        "first_pitch_strike": (
            str(first_pitch["type"]) == "S"
        ),

        "reached_1_0": bool(
            ((group["balls"] == 1) & (group["strikes"] == 0)).any()
        ),

        "reached_2_0": bool(
            ((group["balls"] == 2) & (group["strikes"] == 0)).any()
        ),

        "reached_3_ball_count": bool(
            (group["balls"] == 3).any()
        ),

        "reached_3_1": bool(
            ((group["balls"] == 3) & (group["strikes"] == 1)).any()
        ),

        "reached_full_count": bool(
            ((group["balls"] == 3) & (group["strikes"] == 2)).any()
        ),

        "walk": "walk" in events,
    })

pa = pd.DataFrame(pa_rows)


def pa_summary(group):
    n = len(group)

    return {
        "plate_appearances": int(n),

        "avg_pitches_per_pa": round(
            float(group["pitches"].mean()),
            2
        ),

        "first_pitch_ball_pct": pct(
            group["first_pitch_ball"].sum(),
            n
        ),

        "first_pitch_strike_pct": pct(
            group["first_pitch_strike"].sum(),
            n
        ),

        "reached_1_0_pct": pct(
            group["reached_1_0"].sum(),
            n
        ),

        "reached_2_0_pct": pct(
            group["reached_2_0"].sum(),
            n
        ),

        "reached_3_ball_count_pct": pct(
            group["reached_3_ball_count"].sum(),
            n
        ),

        "reached_3_1_pct": pct(
            group["reached_3_1"].sum(),
            n
        ),

        "reached_full_count_pct": pct(
            group["reached_full_count"].sum(),
            n
        ),

        "walks": int(group["walk"].sum()),

        "walk_pct": pct(
            group["walk"].sum(),
            n
        ),
    }


pa_overall = []

for period, group in pa.groupby("period"):
    row = {"period": period}
    row.update(pa_summary(group))
    pa_overall.append(row)


pa_by_hand = []

for (period, hand), group in pa.groupby(
    ["period", "hand"]
):
    row = {
        "period": period,
        "hand": hand,
    }

    row.update(pa_summary(group))
    pa_by_hand.append(row)


# ------------------------------------------------------------
# 5. THREE-BALL COUNTS
#
# When a pitcher really needs a strike, where is he throwing
# and what pitches is he using?
# ------------------------------------------------------------

three_ball = df[df["balls"] == 3].copy()

three_ball_summary = grouped_summary(
    three_ball,
    ["period", "stand", "exact_count"]
)

three_ball_usage = pitch_usage(
    three_ball,
    ["period", "stand", "exact_count"]
)


# ------------------------------------------------------------
# 6. WALK TERMINAL PITCHES
# ------------------------------------------------------------

walk_pitches = df[
    df["events"].fillna("") == "walk"
].copy()

walk_terminal = []

for _, row in walk_pitches.iterrows():
    walk_terminal.append({
        "period": row["period"],
        "game_date": row["game_date"].strftime("%Y-%m-%d"),
        "game_pk": clean_number(row["game_pk"]),
        "hand": row["stand"],
        "count": row["exact_count"],
        "pitch_type": row["pitch_type"],
        "description": row["description"],
        "plate_x": clean_number(row["plate_x"]),
        "plate_z": clean_number(row["plate_z"]),
        "x_norm": clean_number(row["x_norm"]),
        "z_norm": clean_number(row["z_norm"]),
        "zone": bool(row["zone"]),
        "edge": bool(row["edge"]),
        "heart": bool(row["heart"]),
    })


# ------------------------------------------------------------
# SAVE EVERYTHING
# ------------------------------------------------------------

output = {
    "player": "Paul Skenes",
    "pitcher_id": PITCHER_ID,

    "analysis_scope": {
        "season": 2026,
        "game_type": "Regular Season",
        "performance_break": "2026-05-17",
        "pre_definition": "Games before May 17, 2026",
        "post_definition": "May 17, 2026 onward",
    },

    "definitions": {
        "zone": (
            "Pitch inside normalized batter-specific strike zone."
        ),
        "heart": (
            "Pitch in central 50% of normalized strike zone."
        ),
        "edge": (
            "Pitch in outer third of normalized strike zone."
        ),
        "outside_zone": (
            "Located pitch outside normalized strike zone."
        ),
        "chase": (
            "Swing at a pitch outside normalized strike zone."
        ),
        "whiff": (
            "Swing resulting in a miss."
        ),
        "count_state": {
            "0-0": "First pitch of plate appearance.",
            "Pitcher ahead": "Strikes greater than balls.",
            "Even": "Balls equal strikes, excluding 0-0.",
            "Hitter ahead": "Balls greater than strikes.",
        },
    },

    "overall_by_count_state": overall_by_count_state,
    "overall_two_strike": two_strike_summary,
    "overall_exact_counts": exact_count_summary,

    "rhh_fastballs": {
        "pitch_types": ["FF", "SI"],
        "overall": rhh_hard_overall,
        "by_count_state": rhh_hard_by_count_state,
        "two_strike": rhh_hard_two_strike,
        "exact_counts": rhh_hard_exact_counts,
    },

    "rhh_pitch_usage": {
        "by_count_state": rhh_usage_by_count_state,
        "by_exact_count": rhh_exact_count_usage,
    },

    "plate_appearance_control": {
        "overall": clean_records(pa_overall),
        "by_batter_hand": clean_records(pa_by_hand),
    },

    "three_ball_counts": {
        "location_and_results": three_ball_summary,
        "pitch_usage": three_ball_usage,
    },

    "walk_terminal_pitches": walk_terminal,
}

with open(
    OUT_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        output,
        file,
        indent=2,
        allow_nan=False
    )


# ------------------------------------------------------------
# SHORT TERMINAL REPORT
# ------------------------------------------------------------

print()
print("=" * 72)
print("PAUL SKENES — 2026 COUNT / COMMAND ANALYSIS")
print("=" * 72)

print()
print("OUTPUT FILE:")
print(OUT_PATH)

print()
print("-" * 72)
print("PLATE APPEARANCE CONTROL")
print("-" * 72)

for row in pa_overall:
    print()
    print(row["period"])
    print(
        f"  PA: {row['plate_appearances']}"
    )
    print(
        f"  First-pitch ball%: "
        f"{row['first_pitch_ball_pct']}"
    )
    print(
        f"  Reach 2-0%: "
        f"{row['reached_2_0_pct']}"
    )
    print(
        f"  Reach 3-ball count%: "
        f"{row['reached_3_ball_count_pct']}"
    )
    print(
        f"  Walk%: "
        f"{row['walk_pct']}"
    )

print()
print("-" * 72)
print("RHH — FOUR-SEAM + SINKER")
print("-" * 72)

for row in rhh_hard_overall:
    print()
    print(row["period"])
    print(
        f"  Pitches: {row['pitches']}"
    )
    print(
        f"  Zone%: {row['zone_pct']}"
    )
    print(
        f"  Outside Zone%: {row['outside_zone_pct']}"
    )
    print(
        f"  Edge%: {row['edge_pct']}"
    )
    print(
        f"  Heart%: {row['heart_pct']}"
    )
    print(
        f"  Chase%: {row['chase_pct']}"
    )
    print(
        f"  Whiff%: {row['whiff_pct']}"
    )

print()
print("=" * 72)
print("DONE")
print("=" * 72)
print()
print(
)
print(OUT_PATH)
print()