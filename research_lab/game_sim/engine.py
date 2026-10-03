from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .manager import ManagerPolicy
from .models import (
    BaseRunner,
    GameEvent,
    GameMatchup,
    GameState,
    HalfInning,
    OUTCOME_LABELS,
    PAContext,
    PitcherLine,
    PitcherProfile,
    SimulationConfig,
    TeamProfile,
    TeamSide,
)
from .probability import PAProbabilityProvider, normalize_probabilities
from .transitions import apply_outcome


@dataclass
class _TeamRuntime:
    profile: TeamProfile
    current_pitcher: PitcherProfile
    used_pitcher_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.used_pitcher_ids.add(self.current_pitcher.player_id)


@dataclass
class GameResult:
    seed: int
    away_team: str
    home_team: str
    away_score: int
    home_score: int
    winner: TeamSide | str
    innings_played: int
    plate_appearances: int
    inning_runs: dict[str, dict[int, int]]
    pitcher_lines: dict[str, dict[str, Any]]
    pitcher_appearances: dict[str, list[str]]
    outcome_counts: dict[str, int]
    events: list[dict[str, Any]]
    provider_name: str
    provider_validation_status: str
    ended_by_plate_appearance_cap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "away_team": self.away_team,
            "home_team": self.home_team,
            "away_score": self.away_score,
            "home_score": self.home_score,
            "winner": self.winner,
            "innings_played": self.innings_played,
            "plate_appearances": self.plate_appearances,
            "inning_runs": self.inning_runs,
            "pitcher_lines": self.pitcher_lines,
            "pitcher_appearances": self.pitcher_appearances,
            "outcome_counts": self.outcome_counts,
            "events": self.events,
            "provider_name": self.provider_name,
            "provider_validation_status": self.provider_validation_status,
            "ended_by_plate_appearance_cap": self.ended_by_plate_appearance_cap,
        }


class GameSimulator:
    def __init__(
        self,
        provider: PAProbabilityProvider,
        config: SimulationConfig | None = None,
        manager_policy: ManagerPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or SimulationConfig()
        self.manager = manager_policy or ManagerPolicy(
            three_batter_minimum=self.config.three_batter_minimum
        )

    def simulate(
        self,
        matchup: GameMatchup,
        seed: int,
        *,
        record_events: bool | None = None,
    ) -> GameResult:
        rng = np.random.default_rng(seed)
        should_record = self.config.record_events if record_events is None else record_events
        state = GameState()
        runtimes: dict[TeamSide, _TeamRuntime] = {
            "away": _TeamRuntime(matchup.away, matchup.away.starter),
            "home": _TeamRuntime(matchup.home, matchup.home.starter),
        }
        lines: dict[str, PitcherLine] = {}
        events: list[GameEvent] = []
        inning_runs: dict[str, defaultdict[int, int]] = {
            "away": defaultdict(int),
            "home": defaultdict(int),
        }
        outcome_counts: Counter[str] = Counter()
        extra_runner_placed: set[tuple[int, HalfInning]] = set()
        ended_by_cap = False

        while not state.complete:
            if state.plate_appearances >= self.config.max_plate_appearances:
                state.complete = True
                state.tie = state.away_score == state.home_score
                ended_by_cap = True
                break

            batting_side: TeamSide = "away" if state.half == "top" else "home"
            fielding_side: TeamSide = "home" if batting_side == "away" else "away"
            batting = runtimes[batting_side]
            fielding = runtimes[fielding_side]

            self._place_automatic_runner_if_needed(
                state,
                batting.profile,
                batting_side,
                extra_runner_placed,
            )

            lineup_index = (
                state.away_lineup_index if batting_side == "away" else state.home_lineup_index
            )
            batter = batting.profile.lineup[lineup_index % 9]
            pitcher = fielding.current_pitcher
            line = self._ensure_pitcher_line(
                lines,
                pitcher,
                fielding_side,
                state.inning,
                state.half,
                is_starter=(pitcher.player_id == fielding.profile.starter.player_id),
            )

            context = PAContext(
                batter=batter,
                pitcher=pitcher,
                batting_side=batting_side,
                inning=state.inning,
                half=state.half,
                outs=state.outs,
                bases=tuple(state.bases),  # type: ignore[arg-type]
                batting_score=state.score_for(batting_side),
                fielding_score=state.score_for(fielding_side),
                lineup_position=(lineup_index % 9) + 1,
                times_through_order=(line.batters_faced // 9) + 1,
                pitcher_batters_faced=line.batters_faced,
                pitcher_runs_allowed=line.runs_allowed,
                pitcher_fatigue=self.manager.fatigue(pitcher, line),
                park_factor=matchup.park_factor,
                weather_run_factor=matchup.weather_run_factor,
                batting_team_baserunning=batting.profile.baserunning,
                fielding_team_defense=fielding.profile.defense,
            )
            probabilities = normalize_probabilities(self.provider.probabilities(context))
            probability_vector = np.asarray(
                [probabilities[label] for label in OUTCOME_LABELS], dtype=float
            )
            outcome = str(rng.choice(OUTCOME_LABELS, p=probability_vector))
            outcome_counts[outcome] += 1

            outs_before = state.outs
            bases_before = state.base_ids()
            transition = apply_outcome(
                outcome=outcome,
                bases=state.bases,
                batter=batter,
                responsible_pitcher_id=pitcher.player_id,
                outs_before=outs_before,
                batting_team_baserunning=batting.profile.baserunning,
                fielding_team_defense=fielding.profile.defense,
                rng=rng,
            )

            scored_runners = list(transition.scored_runners)
            if (
                state.half == "bottom"
                and state.inning >= self.config.regulation_innings
                and state.home_score <= state.away_score
                and state.home_score + len(scored_runners) > state.away_score
                and outcome != "home_run"
            ):
                winning_runs_needed = state.away_score - state.home_score + 1
                scored_runners = scored_runners[:winning_runs_needed]

            credited_outs = min(transition.outs_added, 3 - outs_before)
            line.batters_faced += 1
            line.batters_since_entry += 1
            line.outs_recorded += credited_outs
            if outcome == "strikeout":
                line.strikeouts += 1
            elif outcome == "bb_hbp":
                line.walks_hbp += 1
            elif outcome in {"single", "double_triple", "home_run"}:
                line.hits_allowed += 1
                if outcome == "home_run":
                    line.home_runs += 1

            self._charge_scored_runners(scored_runners, lines)
            runs = len(scored_runners)
            state.add_runs(batting_side, runs)
            inning_runs[batting_side][state.inning] += runs
            state.outs = min(3, state.outs + transition.outs_added)
            state.bases = transition.bases
            state.plate_appearances += 1
            if batting_side == "away":
                state.away_lineup_index = (state.away_lineup_index + 1) % 9
            else:
                state.home_lineup_index = (state.home_lineup_index + 1) % 9

            event = GameEvent(
                inning=state.inning,
                half=state.half,
                batting_side=batting_side,
                batting_team=batting.profile.name,
                batter_id=batter.player_id,
                batter_name=batter.name,
                pitcher_id=pitcher.player_id,
                pitcher_name=pitcher.name,
                outcome=outcome,
                outs_before=outs_before,
                outs_after=state.outs,
                bases_before=bases_before,
                bases_after=tuple(
                    runner.player_id if runner is not None else None
                    for runner in transition.bases
                ),
                runs_scored=runs,
                away_score=state.away_score,
                home_score=state.home_score,
                description=transition.description,
            )
            if should_record:
                events.append(event)

            if (
                state.half == "bottom"
                and state.inning >= self.config.regulation_innings
                and state.home_score > state.away_score
            ):
                state.complete = True
                self._close_active_lines(lines, state.inning, state.half)
                break

            if state.outs >= 3:
                self._maybe_change_pitcher(
                    fielding,
                    fielding_side,
                    batting.profile,
                    state,
                    lines,
                    inning_ended=True,
                    rng=rng,
                )
                state.bases = [None, None, None]
                state.outs = 0
                if state.half == "top":
                    if (
                        state.inning >= self.config.regulation_innings
                        and state.home_score > state.away_score
                    ):
                        state.complete = True
                    else:
                        state.half = "bottom"
                else:
                    if (
                        state.inning >= self.config.regulation_innings
                        and state.home_score != state.away_score
                    ):
                        state.complete = True
                    elif state.inning >= self.config.max_innings:
                        state.complete = True
                        state.tie = state.home_score == state.away_score
                    else:
                        state.inning += 1
                        state.half = "top"
                if state.complete:
                    self._close_active_lines(lines, state.inning, state.half)
                    break
            else:
                self._maybe_change_pitcher(
                    fielding,
                    fielding_side,
                    batting.profile,
                    state,
                    lines,
                    inning_ended=False,
                    rng=rng,
                )

        winner: TeamSide | str
        if state.away_score > state.home_score:
            winner = "away"
        elif state.home_score > state.away_score:
            winner = "home"
        else:
            winner = "tie"

        pitcher_appearances = {
            side: [
                line.pitcher_id
                for line in lines.values()
                if line.team_side == side
            ]
            for side in ("away", "home")
        }
        return GameResult(
            seed=int(seed),
            away_team=matchup.away.name,
            home_team=matchup.home.name,
            away_score=state.away_score,
            home_score=state.home_score,
            winner=winner,
            innings_played=state.inning,
            plate_appearances=state.plate_appearances,
            inning_runs={
                side: dict(sorted(values.items()))
                for side, values in inning_runs.items()
            },
            pitcher_lines={
                pitcher_id: line.to_dict()
                for pitcher_id, line in lines.items()
            },
            pitcher_appearances=pitcher_appearances,
            outcome_counts=dict(outcome_counts),
            events=[event.to_dict() for event in events],
            provider_name=getattr(self.provider, "name", type(self.provider).__name__),
            provider_validation_status=getattr(
                self.provider,
                "validation_status",
                "unknown",
            ),
            ended_by_plate_appearance_cap=ended_by_cap,
        )

    def _ensure_pitcher_line(
        self,
        lines: dict[str, PitcherLine],
        pitcher: PitcherProfile,
        fielding_side: TeamSide,
        inning: int,
        half: HalfInning,
        is_starter: bool,
    ) -> PitcherLine:
        line = lines.get(pitcher.player_id)
        if line is None:
            line = PitcherLine(
                pitcher_id=pitcher.player_id,
                name=pitcher.name,
                team_side=fielding_side,
                is_starter=is_starter,
                entry_inning=inning,
                entry_half=half,
            )
            lines[pitcher.player_id] = line
        return line

    def _maybe_change_pitcher(
        self,
        fielding: _TeamRuntime,
        fielding_side: TeamSide,
        batting_team: TeamProfile,
        state: GameState,
        lines: dict[str, PitcherLine],
        inning_ended: bool,
        rng: np.random.Generator,
    ) -> None:
        current = fielding.current_pitcher
        line = lines.get(current.player_id)
        if line is None:
            return
        if not self.manager.should_remove(
            current,
            line,
            state,
            fielding_side,
            inning_ended,
            rng,
        ):
            return
        next_index = (
            state.away_lineup_index if fielding_side == "home" else state.home_lineup_index
        )
        next_batter = batting_team.lineup[next_index % 9]
        replacement = self.manager.select_reliever(
            fielding.profile.bullpen,
            fielding.used_pitcher_ids,
            state,
            fielding_side,
            next_batter,
            rng,
        )
        if replacement is None:
            return
        line.exit_inning = state.inning
        line.exit_half = state.half
        fielding.current_pitcher = replacement
        fielding.used_pitcher_ids.add(replacement.player_id)

    def _place_automatic_runner_if_needed(
        self,
        state: GameState,
        batting_team: TeamProfile,
        batting_side: TeamSide,
        placed: set[tuple[int, HalfInning]],
    ) -> None:
        key = (state.inning, state.half)
        if (
            not self.config.automatic_runner_in_extras
            or state.inning <= self.config.regulation_innings
            or key in placed
            or state.outs != 0
            or any(state.bases)
        ):
            return
        lineup_index = (
            state.away_lineup_index if batting_side == "away" else state.home_lineup_index
        )
        runner_profile = batting_team.lineup[(lineup_index - 1) % 9]
        state.bases[1] = BaseRunner(
            player_id=runner_profile.player_id,
            name=runner_profile.name,
            speed=runner_profile.speed,
            responsible_pitcher_id=None,
            earned=False,
            automatic=True,
        )
        placed.add(key)

    @staticmethod
    def _charge_scored_runners(
        scored_runners: list[BaseRunner],
        lines: dict[str, PitcherLine],
    ) -> None:
        for runner in scored_runners:
            pitcher_id = runner.responsible_pitcher_id
            if pitcher_id is None or pitcher_id not in lines:
                continue
            line = lines[pitcher_id]
            line.runs_allowed += 1
            if runner.earned:
                line.earned_runs += 1

    @staticmethod
    def _close_active_lines(
        lines: dict[str, PitcherLine],
        inning: int,
        half: HalfInning,
    ) -> None:
        for line in lines.values():
            if line.exit_inning is None:
                line.exit_inning = inning
                line.exit_half = half
