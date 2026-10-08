"""In-game update: continue the game from its current state in many worlds.

Each cycle of the live worker re-simulates every game in progress from the observed state
(brl_live.live_feed) with the engine's own rules (brl_live.live_sim) and the same
simulation-time adjustments as pregame forecasts. The output is a current win chance, the
most likely final score and run distributions. It is a snapshot that is replaced every cycle,
never a forecast version, and it is not scored on the record page.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from .boxscore import ADJUST, adjusted_provider, manager_for, running_events_for, steal_model_for, transitions_for
from .live_sim import LiveSimulator

LIVE_WORLDS = 2000


def _state_key(game_pk: int, state: dict) -> int:
    text = f"{game_pk}|{state['inning']}|{state['half']}|{state['outs']}|{state['bases']}|{state['away_score']}|{state['home_score']}|{state['due_up']}|{state['current_pitcher']}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _distribution(counter: Counter, n: int) -> dict:
    return {'n': n, 'counts': [[int(k), int(v)] for k, v in sorted(counter.items())]}


def run_live_update(engine, matchup, start, state: dict, game: dict, full_history, *, n_worlds: int = LIVE_WORLDS,
                    settings=ADJUST, updated_at: str = '', environment=None, teams=None) -> dict:
    provider, hook, labels = adjusted_provider(engine.provider, full_history, game['date'], settings, environment, teams)
    steals = steal_model_for(settings, matchup, game['date'])
    if steals is not None:
        labels = list(labels) + [steals.describe()]
    kernel = transitions_for(settings)
    if kernel is not None:
        labels = list(labels) + [kernel.name]
    running = running_events_for(settings)
    if running is not None:
        labels = list(labels) + [running.name]
    manager, choice_label = manager_for(engine.manager, settings, teams)
    if choice_label:
        labels = list(labels) + [choice_label]
    sim = LiveSimulator(provider, config=engine.config, manager_policy=manager, steals=steals, transitions=kernel,
                        running_events=running)
    seeds = np.random.default_rng(_state_key(game['game_pk'], state)).integers(0, np.iinfo(np.int32).max, size=n_worlds, dtype=np.int64)
    home = ties = 0
    pairs: Counter = Counter(); away_runs: Counter = Counter(); home_runs: Counter = Counter(); innings: Counter = Counter()
    for seed in seeds:
        if hook:
            hook(int(seed))
        r = sim.simulate_from(matchup, int(seed), start)
        if r.winner == 'home':
            home += 1
        elif r.winner == 'tie':
            ties += 1
        pairs[(r.away_score, r.home_score)] += 1
        away_runs[r.away_score] += 1; home_runs[r.home_score] += 1; innings[r.innings_played] += 1
    n = len(seeds)
    p_home = (home + 0.5 * ties) / n
    mean_a = sum(k * v for k, v in away_runs.items()) / n; mean_h = sum(k * v for k, v in home_runs.items()) / n
    # Most likely final score; ties between equally common scores go to the one nearest the means.
    best = max(pairs.items(), key=lambda kv: (kv[1], -abs(kv[0][0] - mean_a) - abs(kv[0][1] - mean_h)))
    return {
        'schema': 'brl.live-update.v1', 'game_pk': int(game['game_pk']), 'date': game['date'], 'updated_at': updated_at,
        'n_worlds': n, 'home_win_probability': round(p_home, 4), 'tie_share': round(ties / n, 4),
        'projected_final': {'away': int(best[0][0]), 'home': int(best[0][1]), 'share': round(best[1] / n, 4)},
        'expected_final': {'away': round(mean_a, 2), 'home': round(mean_h, 2)},
        'team_run_distributions': {'away': _distribution(away_runs, n), 'home': _distribution(home_runs, n)},
        'innings_distribution': _distribution(innings, n),
        'state': {k: state[k] for k in ('inning', 'half', 'outs', 'bases', 'between_innings', 'away_score', 'home_score',
                                        'batting_side', 'balls', 'strikes', 'detailed_state', 'innings')},
        'on_mound': {s: state['current_pitcher'][s] for s in ('away', 'home')},
        'due_up': {s: matchup.away.lineup[state['due_up']['away']].name if s == 'away' else matchup.home.lineup[state['due_up']['home']].name for s in ('away', 'home')},
        'pitcher_names': {pid: p.name for side in (matchup.away, matchup.home) for p in (side.starter, *side.bullpen) for pid in [p.player_id]},
        'adjustments': labels,
        'note': 'Snapshot of the game continued from its current state; replaced every cycle and not scored.',
    }
