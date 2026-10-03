from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

OUTCOME_LABELS: tuple[str, ...] = (
    "bip_out",
    "strikeout",
    "bb_hbp",
    "single",
    "double_triple",
    "home_run",
    "other_reach",
)

HalfInning = Literal["top", "bottom"]
TeamSide = Literal["away", "home"]


def _clip_rating(value: float, low: float = -2.5, high: float = 2.5) -> float:
    return float(min(high, max(low, value)))


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    name: str
    bats: str = "R"
    contact: float = 0.0
    power: float = 0.0
    discipline: float = 0.0
    speed: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "bats", (self.bats or "R").upper()[0])
        object.__setattr__(self, "contact", _clip_rating(self.contact))
        object.__setattr__(self, "power", _clip_rating(self.power))
        object.__setattr__(self, "discipline", _clip_rating(self.discipline))
        object.__setattr__(self, "speed", float(min(1.0, max(0.0, self.speed))))
        if not self.player_id:
            raise ValueError("player_id is required")
        if not self.name:
            raise ValueError("player name is required")


@dataclass(frozen=True)
class PitcherProfile:
    player_id: str
    name: str
    throws: str = "R"
    role: str = "reliever"
    stuff: float = 0.0
    command: float = 0.0
    contact_management: float = 0.0
    stamina: float = 0.5
    leverage: float = 0.5
    rest: float = 1.0
    expected_batters: int = 6
    max_batters: int = 10
    available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "throws", (self.throws or "R").upper()[0])
        object.__setattr__(self, "role", (self.role or "reliever").lower())
        object.__setattr__(self, "stuff", _clip_rating(self.stuff))
        object.__setattr__(self, "command", _clip_rating(self.command))
        object.__setattr__(self, "contact_management", _clip_rating(self.contact_management))
        object.__setattr__(self, "stamina", float(min(1.0, max(0.0, self.stamina))))
        object.__setattr__(self, "leverage", float(min(1.0, max(0.0, self.leverage))))
        object.__setattr__(self, "rest", float(min(1.0, max(0.0, self.rest))))
        object.__setattr__(self, "expected_batters", max(1, int(self.expected_batters)))
        object.__setattr__(self, "max_batters", max(1, int(self.max_batters)))
        if self.max_batters < self.expected_batters:
            object.__setattr__(self, "max_batters", self.expected_batters)
        if not self.player_id:
            raise ValueError("pitcher player_id is required")
        if not self.name:
            raise ValueError("pitcher name is required")


@dataclass(frozen=True)
class TeamProfile:
    team_id: str
    name: str
    lineup: tuple[PlayerProfile, ...]
    starter: PitcherProfile
    bullpen: tuple[PitcherProfile, ...] = ()
    defense: float = 0.0
    baserunning: float = 0.0

    def __post_init__(self) -> None:
        if len(self.lineup) != 9:
            raise ValueError("a team lineup must contain exactly nine hitters")
        ids = [player.player_id for player in self.lineup]
        if len(ids) != len(set(ids)):
            raise ValueError("lineup player IDs must be unique")
        pitcher_ids = [self.starter.player_id, *(pitcher.player_id for pitcher in self.bullpen)]
        if len(pitcher_ids) != len(set(pitcher_ids)):
            raise ValueError("pitcher IDs must be unique within a team")
        object.__setattr__(self, "defense", _clip_rating(self.defense))
        object.__setattr__(self, "baserunning", _clip_rating(self.baserunning))


@dataclass(frozen=True)
class GameMatchup:
    away: TeamProfile
    home: TeamProfile
    venue: str = "Unknown Park"
    park_factor: float = 1.0
    weather_run_factor: float = 1.0
    game_type: str = "R"

    def __post_init__(self) -> None:
        object.__setattr__(self, "park_factor", float(min(1.35, max(0.70, self.park_factor))))
        object.__setattr__(self, "weather_run_factor", float(min(1.30, max(0.75, self.weather_run_factor))))


@dataclass
class BaseRunner:
    player_id: str
    name: str
    speed: float
    responsible_pitcher_id: str | None
    earned: bool = True
    automatic: bool = False

    @classmethod
    def from_batter(cls, batter: PlayerProfile, responsible_pitcher_id: str | None) -> "BaseRunner":
        return cls(
            player_id=batter.player_id,
            name=batter.name,
            speed=batter.speed,
            responsible_pitcher_id=responsible_pitcher_id,
        )


@dataclass
class PitcherLine:
    pitcher_id: str
    name: str
    team_side: TeamSide
    is_starter: bool
    entry_inning: int
    entry_half: HalfInning
    exit_inning: int | None = None
    exit_half: HalfInning | None = None
    batters_faced: int = 0
    batters_since_entry: int = 0
    outs_recorded: int = 0
    runs_allowed: int = 0
    earned_runs: int = 0
    hits_allowed: int = 0
    walks_hbp: int = 0
    strikeouts: int = 0
    home_runs: int = 0

    @property
    def innings_pitched(self) -> float:
        return self.outs_recorded / 3.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["innings_pitched"] = round(self.innings_pitched, 3)
        return payload


@dataclass
class GameEvent:
    inning: int
    half: HalfInning
    batting_side: TeamSide
    batting_team: str
    batter_id: str
    batter_name: str
    pitcher_id: str
    pitcher_name: str
    outcome: str
    outs_before: int
    outs_after: int
    bases_before: tuple[str | None, str | None, str | None]
    bases_after: tuple[str | None, str | None, str | None]
    runs_scored: int
    away_score: int
    home_score: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GameState:
    inning: int = 1
    half: HalfInning = "top"
    outs: int = 0
    bases: list[BaseRunner | None] = field(default_factory=lambda: [None, None, None])
    away_score: int = 0
    home_score: int = 0
    away_lineup_index: int = 0
    home_lineup_index: int = 0
    plate_appearances: int = 0
    complete: bool = False
    tie: bool = False

    def score_for(self, side: TeamSide) -> int:
        return self.away_score if side == "away" else self.home_score

    def opponent_score_for(self, side: TeamSide) -> int:
        return self.home_score if side == "away" else self.away_score

    def add_runs(self, side: TeamSide, runs: int) -> None:
        if side == "away":
            self.away_score += int(runs)
        else:
            self.home_score += int(runs)

    def base_ids(self) -> tuple[str | None, str | None, str | None]:
        return tuple(runner.player_id if runner is not None else None for runner in self.bases)


@dataclass(frozen=True)
class PAContext:
    batter: PlayerProfile
    pitcher: PitcherProfile
    batting_side: TeamSide
    inning: int
    half: HalfInning
    outs: int
    bases: tuple[BaseRunner | None, BaseRunner | None, BaseRunner | None]
    batting_score: int
    fielding_score: int
    lineup_position: int
    times_through_order: int
    pitcher_batters_faced: int
    pitcher_runs_allowed: int
    pitcher_fatigue: float
    park_factor: float
    weather_run_factor: float
    batting_team_baserunning: float
    fielding_team_defense: float

    @property
    def score_diff(self) -> int:
        return self.batting_score - self.fielding_score

    @property
    def platoon_advantage(self) -> bool:
        batter_side = self.batter.bats
        pitcher_side = self.pitcher.throws
        return batter_side == "S" or batter_side != pitcher_side


@dataclass(frozen=True)
class SimulationConfig:
    regulation_innings: int = 9
    max_innings: int = 15
    automatic_runner_in_extras: bool = True
    three_batter_minimum: bool = True
    max_plate_appearances: int = 220
    record_events: bool = True

    def __post_init__(self) -> None:
        if self.regulation_innings < 1:
            raise ValueError("regulation_innings must be positive")
        if self.max_innings < self.regulation_innings:
            raise ValueError("max_innings cannot be less than regulation_innings")
        if self.max_plate_appearances < 18:
            raise ValueError("max_plate_appearances is too small")
