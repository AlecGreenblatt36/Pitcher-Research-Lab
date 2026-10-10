from __future__ import annotations

import pandas as pd

OUTCOME_ORDER = (
    "BIP_OUT",
    "K",
    "BB_HBP",
    "1B",
    "2B_3B",
    "HR",
    "OTHER_REACH",
)

_EVENT_TO_OUTCOME = {
    "strikeout": "K",
    "strikeout_double_play": "K",
    "walk": "BB_HBP",
    "intent_walk": "BB_HBP",
    "hit_by_pitch": "BB_HBP",
    "single": "1B",
    "double": "2B_3B",
    "triple": "2B_3B",
    "home_run": "HR",
    "field_out": "BIP_OUT",
    "force_out": "BIP_OUT",
    "grounded_into_double_play": "BIP_OUT",
    "fielders_choice_out": "BIP_OUT",
    "double_play": "BIP_OUT",
    "triple_play": "BIP_OUT",
    "sac_fly": "BIP_OUT",
    "sac_bunt": "BIP_OUT",
    "sac_fly_double_play": "BIP_OUT",
    "field_error": "OTHER_REACH",
    "fielders_choice": "OTHER_REACH",
    "catcher_interf": "OTHER_REACH",
}

EXCLUDED_EVENTS = {
    "pickoff_1b",
    "pickoff_2b",
    "pickoff_3b",
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
    "stolen_base_2b",
    "stolen_base_3b",
    "stolen_base_home",
    "wild_pitch",
    "passed_ball",
    "balk",
    "game_advisory",
    "no_play",
}


def map_event(event: object) -> str | None:
    if event is None or (isinstance(event, float) and pd.isna(event)):
        return None
    key = str(event).strip().lower()
    return _EVENT_TO_OUTCOME.get(key)


def mapping_report(events: pd.Series) -> dict:
    normalized = events.dropna().astype(str).str.strip().str.lower()
    counts = normalized.value_counts().sort_index()
    unmapped = counts[~counts.index.isin(_EVENT_TO_OUTCOME)].to_dict()
    mapped = counts[counts.index.isin(_EVENT_TO_OUTCOME)].to_dict()
    return {
        "mapped_events": {str(k): int(v) for k, v in mapped.items()},
        "unmapped_events": {str(k): int(v) for k, v in unmapped.items()},
        "outcome_mapping": dict(sorted(_EVENT_TO_OUTCOME.items())),
    }
