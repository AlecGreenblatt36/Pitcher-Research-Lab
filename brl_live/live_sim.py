"""Continue a game from a live state with the unchanged engine rules.

The locked engine's `simulate` always starts from the first pitch. `LiveSimulator.simulate_from`
runs the same plate-appearance loop (copied from research_lab.game_sim.engine.GameSimulator.simulate,
rule for rule) but starts from an observed state: inning, half, outs, runners, score, the lineup
spots due up, the pitchers on the mound with their lines so far, and the pitchers already used.
Probabilities, runner transitions and pitching changes are the engine's own.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from research_lab.game_sim.engine import GameSimulator, GameResult, _TeamRuntime
from research_lab.game_sim.models import (BaseRunner, GameEvent, GameMatchup, GameState, OUTCOME_LABELS,
                                          PAContext, PitcherLine, PitcherProfile)
from research_lab.game_sim.probability import normalize_probabilities
from research_lab.game_sim.transitions import apply_outcome


@dataclass
class PitcherSoFar:
    pitcher_id: str
    batters_faced: int = 0
    outs_recorded: int = 0
    runs_allowed: int = 0
    earned_runs: int = 0
    hits_allowed: int = 0
    walks_hbp: int = 0
    strikeouts: int = 0
    home_runs: int = 0
    entry_inning: int = 1
    entry_half: str = 'top'
    is_starter: bool = True


@dataclass
class LiveStart:
    inning: int
    half: str                       # 'top' or 'bottom'
    outs: int
    bases: tuple                    # three player ids or None (first, second, third)
    away_score: int
    home_score: int
    away_lineup_index: int          # 0-8, spot due up for the away team
    home_lineup_index: int
    current_pitcher: dict           # side -> pitcher id on the mound for that side's defense
    used_pitchers: dict             # side -> ordered list of pitcher ids used so far
    lines: dict = field(default_factory=dict)   # pitcher id -> PitcherSoFar
    automatic_runner_placed: bool = False       # true when the extra-inning runner is already on

    def validate(self, matchup: GameMatchup) -> None:
        if self.half not in ('top', 'bottom'):
            raise ValueError('half must be top or bottom')
        if not 1 <= self.inning <= 30 or not 0 <= self.outs <= 2:
            raise ValueError('inning or outs out of range')
        if len(self.bases) != 3:
            raise ValueError('three bases required')
        for side in ('away', 'home'):
            team = getattr(matchup, side)
            if side not in self.current_pitcher or side not in self.used_pitchers:
                raise ValueError('current and used pitchers required for both sides')
            known = {team.starter.player_id, *(p.player_id for p in team.bullpen)}
            if self.current_pitcher[side] not in known:
                raise ValueError(f'{side} current pitcher is not in the matchup')
        if not 0 <= self.away_lineup_index <= 8 or not 0 <= self.home_lineup_index <= 8:
            raise ValueError('lineup index out of range')


class LiveSimulator(GameSimulator):
    """GameSimulator that can start from a LiveStart. `simulate` is unchanged."""

    def _runtime(self, team, side: str, start: LiveStart) -> _TeamRuntime:
        pitchers = {team.starter.player_id: team.starter, **{p.player_id: p for p in team.bullpen}}
        current = pitchers[start.current_pitcher[side]]
        runtime = _TeamRuntime(team, current)
        runtime.used_pitcher_ids.update(start.used_pitchers[side])
        return runtime

    def _initial_lines(self, matchup: GameMatchup, start: LiveStart) -> dict:
        lines = {}
        for side in ('away', 'home'):
            team = getattr(matchup, side)
            names = {team.starter.player_id: team.starter.name, **{p.player_id: p.name for p in team.bullpen}}
            # Only pitchers who have already thrown get a line; the engine opens the current
            # pitcher's line on his first plate appearance, exactly as from the first pitch.
            for pid in start.used_pitchers[side]:
                so_far = start.lines.get(pid)
                if so_far is None:
                    continue
                line = PitcherLine(pitcher_id=pid, name=names.get(pid, pid), team_side=side,
                                   is_starter=(pid == team.starter.player_id),
                                   entry_inning=so_far.entry_inning, entry_half=so_far.entry_half)
                line.batters_faced = so_far.batters_faced
                line.batters_since_entry = so_far.batters_faced
                line.outs_recorded = so_far.outs_recorded
                line.runs_allowed = so_far.runs_allowed
                line.earned_runs = so_far.earned_runs
                line.hits_allowed = so_far.hits_allowed
                line.walks_hbp = so_far.walks_hbp
                line.strikeouts = so_far.strikeouts
                line.home_runs = so_far.home_runs
                if pid != start.current_pitcher[side]:
                    line.exit_inning, line.exit_half = start.inning, start.half
                lines[pid] = line
        return lines

    def simulate_from(self, matchup: GameMatchup, seed: int, start: LiveStart, *, record_events: bool = False) -> GameResult:
        start.validate(matchup)
        if self.steals is not None:
            matchup = self.steals.with_speeds(matchup)
        rng = np.random.default_rng(seed)
        state = GameState(inning=start.inning, half=start.half, outs=start.outs,
                          away_score=start.away_score, home_score=start.home_score,
                          away_lineup_index=start.away_lineup_index, home_lineup_index=start.home_lineup_index)
        batting_side_start = 'away' if start.half == 'top' else 'home'
        batting_team = getattr(matchup, batting_side_start)
        hitters = {p.player_id: p for p in batting_team.lineup}
        responsible = start.current_pitcher['home' if batting_side_start == 'away' else 'away']
        for i, pid in enumerate(start.bases):
            if pid is None:
                continue
            hitter = hitters.get(str(pid))
            state.bases[i] = BaseRunner(player_id=str(pid), name=hitter.name if hitter else str(pid),
                                        speed=hitter.speed if hitter else 0.5, responsible_pitcher_id=responsible)
        runtimes = {'away': self._runtime(matchup.away, 'away', start), 'home': self._runtime(matchup.home, 'home', start)}
        lines = self._initial_lines(matchup, start)
        events: list[GameEvent] = []
        inning_runs = {'away': defaultdict(int), 'home': defaultdict(int)}
        outcome_counts: Counter = Counter()
        extra_runner_placed = {(start.inning, start.half)} if start.automatic_runner_placed or any(start.bases) or start.outs else set()
        ended_by_cap = False
        pa_at_start = 0

        # ---- from here the loop is the engine's, unchanged ----
        while not state.complete:
            if state.plate_appearances >= self.config.max_plate_appearances:
                state.complete = True
                state.tie = state.away_score == state.home_score
                ended_by_cap = True
                break
            batting_side = 'away' if state.half == 'top' else 'home'
            fielding_side = 'home' if batting_side == 'away' else 'away'
            batting = runtimes[batting_side]
            fielding = runtimes[fielding_side]
            self._place_automatic_runner_if_needed(state, batting.profile, batting_side, extra_runner_placed)
            lineup_index = state.away_lineup_index if batting_side == 'away' else state.home_lineup_index
            batter = batting.profile.lineup[lineup_index % 9]
            pitcher = fielding.current_pitcher
            line = self._ensure_pitcher_line(lines, pitcher, fielding_side, state.inning, state.half,
                                             is_starter=(pitcher.player_id == fielding.profile.starter.player_id))
            context = PAContext(
                batter=batter, pitcher=pitcher, batting_side=batting_side, inning=state.inning, half=state.half,
                outs=state.outs, bases=tuple(state.bases), batting_score=state.score_for(batting_side),
                fielding_score=state.score_for(fielding_side), lineup_position=(lineup_index % 9) + 1,
                times_through_order=(line.batters_faced // 9) + 1, pitcher_batters_faced=line.batters_faced,
                pitcher_runs_allowed=line.runs_allowed, pitcher_fatigue=self.manager.fatigue(pitcher, line),
                park_factor=matchup.park_factor, weather_run_factor=matchup.weather_run_factor,
                batting_team_baserunning=batting.profile.baserunning, fielding_team_defense=fielding.profile.defense)
            probabilities = normalize_probabilities(self.provider.probabilities(context))
            probability_vector = np.asarray([probabilities[label] for label in OUTCOME_LABELS], dtype=float)
            outcome = str(rng.choice(OUTCOME_LABELS, p=probability_vector))
            outcome_counts[outcome] += 1
            outs_before = state.outs
            bases_before = state.base_ids()
            transition = (self.transitions.apply if getattr(self, 'transitions', None) is not None else apply_outcome)(outcome=outcome, bases=state.bases, batter=batter,
                                       responsible_pitcher_id=pitcher.player_id, outs_before=outs_before,
                                       batting_team_baserunning=batting.profile.baserunning,
                                       fielding_team_defense=fielding.profile.defense, rng=rng)
            scored_runners = list(transition.scored_runners)
            if (state.half == 'bottom' and state.inning >= self.config.regulation_innings
                    and state.home_score <= state.away_score
                    and state.home_score + len(scored_runners) > state.away_score and outcome != 'home_run'):
                scored_runners = scored_runners[:state.away_score - state.home_score + 1]
            credited_outs = min(transition.outs_added, 3 - outs_before)
            line.batters_faced += 1
            line.batters_since_entry += 1
            line.outs_recorded += credited_outs
            if outcome == 'strikeout':
                line.strikeouts += 1
            elif outcome == 'bb_hbp':
                line.walks_hbp += 1
            elif outcome in {'single', 'double_triple', 'home_run'}:
                line.hits_allowed += 1
                if outcome == 'home_run':
                    line.home_runs += 1
            self._charge_scored_runners(scored_runners, lines)
            runs = len(scored_runners)
            state.add_runs(batting_side, runs)
            inning_runs[batting_side][state.inning] += runs
            state.outs = min(3, state.outs + transition.outs_added)
            state.bases = transition.bases
            state.plate_appearances += 1
            if batting_side == 'away':
                state.away_lineup_index = (state.away_lineup_index + 1) % 9
            else:
                state.home_lineup_index = (state.home_lineup_index + 1) % 9
            if record_events:
                events.append(GameEvent(
                    inning=state.inning, half=state.half, batting_side=batting_side, batting_team=batting.profile.name,
                    batter_id=batter.player_id, batter_name=batter.name, pitcher_id=pitcher.player_id, pitcher_name=pitcher.name,
                    outcome=outcome, outs_before=outs_before, outs_after=state.outs, bases_before=bases_before,
                    bases_after=tuple(r.player_id if r is not None else None for r in transition.bases),
                    runs_scored=runs, away_score=state.away_score, home_score=state.home_score, description=transition.description))
            if (state.half == 'bottom' and state.inning >= self.config.regulation_innings
                    and state.home_score > state.away_score):
                state.complete = True
                self._close_active_lines(lines, state.inning, state.half)
                break
            if self.steals is not None and state.outs < 3:
                self._steal_step(state, batting_side, batting, fielding, lines, rng, record_events, events)
            if state.outs >= 3:
                self._maybe_change_pitcher(fielding, fielding_side, batting.profile, state, lines, inning_ended=True, rng=rng)
                state.bases = [None, None, None]
                state.outs = 0
                if state.half == 'top':
                    if state.inning >= self.config.regulation_innings and state.home_score > state.away_score:
                        state.complete = True
                    else:
                        state.half = 'bottom'
                else:
                    if state.inning >= self.config.regulation_innings and state.home_score != state.away_score:
                        state.complete = True
                    elif state.inning >= self.config.max_innings:
                        state.complete = True
                        state.tie = state.home_score == state.away_score
                    else:
                        state.inning += 1
                        state.half = 'top'
                if state.complete:
                    self._close_active_lines(lines, state.inning, state.half)
                    break
            else:
                self._maybe_change_pitcher(fielding, fielding_side, batting.profile, state, lines, inning_ended=False, rng=rng)
        # ---- end of the engine loop ----

        winner = 'away' if state.away_score > state.home_score else 'home' if state.home_score > state.away_score else 'tie'
        return GameResult(
            seed=int(seed), away_team=matchup.away.name, home_team=matchup.home.name,
            away_score=state.away_score, home_score=state.home_score, winner=winner,
            innings_played=state.inning, plate_appearances=state.plate_appearances - pa_at_start,
            inning_runs={side: dict(sorted(v.items())) for side, v in inning_runs.items()},
            pitcher_lines={pid: line.to_dict() for pid, line in lines.items()},
            pitcher_appearances={side: [l.pitcher_id for l in lines.values() if l.team_side == side] for side in ('away', 'home')},
            outcome_counts=dict(outcome_counts), events=[e.to_dict() for e in events],
            provider_name=getattr(self.provider, 'name', type(self.provider).__name__),
            provider_validation_status=getattr(self.provider, 'validation_status', 'unknown'),
            ended_by_plate_appearance_cap=ended_by_cap)
