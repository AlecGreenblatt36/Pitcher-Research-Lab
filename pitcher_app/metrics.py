"""Shared eligibility rules for Statcast summaries."""

import pandas as pd

NON_CONTACT_EVENTS = {
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "strikeout",
    "strikeout_double_play",
}


def xwoba_components(data):
    # Unestimated contact stays missing: actual contact results are not estimates.
    contact = data["description"].eq("hit_into_play")
    components = data["estimated_woba_using_speedangle"].where(contact)
    non_contact = data["events"].isin(NON_CONTACT_EVENTS)
    return components.where(~non_contact, data["woba_value"])


def xwoba_coverage(data):
    eligible = data["events"].notna() & data["woba_denom"].gt(0)
    covered = eligible & data["xwoba_component"].notna()
    contact = eligible & data["description"].eq("hit_into_play")
    count = int(eligible.sum())
    return {
        "xwoba_eligible_pa": count,
        "xwoba_covered_pa": int(covered.sum()),
        "xwoba_coverage_pct": round(covered.sum() / count * 100, 1) if count else None,
        "xwoba_missing_contact": int((contact & ~covered).sum()),
    }


def run_value_summary(data):
    values = data["pitcher_run_value"]
    count = int(values.count())
    total = values.sum(min_count=1)
    return total, count, total / count * 100 if count else None
