from __future__ import annotations

from research_lab.game_sim.savant_traits import (
    fetch_oaa,
    fetch_sprint_speed,
    impute_sprint_speed,
    replay_trait_season,
)


class _Response:
    def __init__(self, text: str) -> None:
        self.content = text.encode("utf-8")

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.urls.append(str(url))
        return _Response(self.responses.pop(0))


def test_sprint_speed_snapshot_builds_empirical_percentiles() -> None:
    session = _Session(
        [
            "player_id,player_name,sprint_speed,competitive_runs\n"
            "1,Fast Player,30.0,20\n"
            "2,Average Player,27.0,18\n"
            "3,Slow Player,24.0,12\n"
        ]
    )
    traits, receipt = fetch_sprint_speed(2024, session=session)
    assert traits[1].speed_percentile > traits[2].speed_percentile > traits[3].speed_percentile
    assert traits[1].sprint_speed_ft_s == 30.0
    assert receipt.season == 2024
    assert receipt.rows == 3
    assert "prior-season" in receipt.replay_policy


def test_missing_sprint_speed_is_explicitly_imputed() -> None:
    session = _Session(["player_id,sprint_speed\n1,28.0\n"])
    traits, _ = fetch_sprint_speed(2024, session=session)
    imputed = impute_sprint_speed(999, traits)
    assert imputed.imputed is True
    assert imputed.speed_percentile == 0.5


def test_oaa_snapshot_preserves_raw_outs_and_attempts() -> None:
    session = _Session(
        [
            "player_id,player_name,primary_pos_formatted,outs_above_average,attempts,team\n"
            "10,Shortstop,SS,8,240,MIL\n"
            "11,Center Fielder,CF,-2,220,SD\n"
        ]
    )
    traits, receipt = fetch_oaa(2024, session=session)
    assert traits[0].player_id == 10
    assert traits[0].outs_above_average == 8.0
    assert traits[0].attempts == 240
    assert receipt.rows == 2
    assert "prior-season" in receipt.replay_policy


def test_historical_replay_uses_previous_completed_season() -> None:
    assert replay_trait_season(2025) == 2024
    assert replay_trait_season(2026) == 2025
