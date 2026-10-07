"""Public-layer win chance blend and an independent live track record.

Blend
-----
The headline win chance is the equal-weight average, on the log-odds scale, of the
simulator's win share and the team-strength model's win chance. On the full
out-of-sample 2026 regular-season replay (2,427 games, every input prior-date) the
average scored better than either parent on Brier and log loss:

    team model 0.24576, simulator 0.24691, equal-weight blend 0.24497

Weights are fixed at one half each; nothing here is fitted on live games. When a
forecast has no team-model probability the blend is the simulator alone.

Track record
------------
For every final game, the last forecast version published before the observed first
pitch is scored (Brier and log loss) for the simulator, the team model and the blend,
next to a coin flip and always-pick-home at the prior-season home rate. Games without a
pregame publication are listed as unscored, never backfilled.
"""
from __future__ import annotations

import math

HOME_RATE_PRIOR = 0.527  # 2023-2025 regular-season home win rate from the seed history
EPS = 1e-4


def _logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, float(p)))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def blend_probability(p_sim, p_team) -> float:
    """Equal-weight log-odds average; falls back to the simulator when the team model is absent."""
    if p_sim is None:
        raise ValueError('simulator probability required')
    if p_team is None:
        return float(p_sim)
    return _sigmoid(0.5 * _logit(p_sim) + 0.5 * _logit(p_team))


def _ts(value: str):
    from datetime import datetime, timezone
    v = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if v.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return v.astimezone(timezone.utc)


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = min(1 - 1e-6, max(1e-6, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def build_record(ledger: dict) -> dict:
    forecasts = ledger.get('forecasts', {}) or {}
    publications = ledger.get('publications', {}) or {}
    actuals = ledger.get('actuals', {}) or {}
    blend = {}
    for ident, f in forecasts.items():
        blend[ident] = round(blend_probability(f.get('home_win_probability'), f.get('team_baseline_probability')), 6)

    games, unscored = [], []
    by_game: dict[str, list] = {}
    for ident, f in forecasts.items():
        by_game.setdefault(str(f['game_pk']), []).append((ident, f))
    for pk, actual in actuals.items():
        versions = by_game.get(str(pk), [])
        if not versions:
            continue
        first_pitch = actual.get('first_pitch_observed_at')
        eligible = []
        for ident, f in versions:
            pub = publications.get(ident)
            if not pub or not first_pitch:
                continue
            if _ts(f['saved_at']) <= _ts(pub['published_at']) < _ts(first_pitch):
                eligible.append((_ts(pub['published_at']), ident, f))
        if not eligible:
            unscored.append({'game_pk': int(pk), 'reason': 'no forecast published before first pitch'})
            continue
        _, ident, f = max(eligible, key=lambda x: x[0])
        y = 1.0 if actual['home'] > actual['away'] else 0.0
        p_sim = float(f['home_win_probability'])
        p_team = f.get('team_baseline_probability')
        p_team = None if p_team is None else float(p_team)
        p_blend = blend[ident]
        games.append({
            'game_pk': int(pk), 'date': f.get('date'), 'away': f['away'].get('abbr'), 'home': f['home'].get('abbr'),
            'version': f.get('version'), 'forecast_id': ident, 'home_won': bool(y),
            'actual': {'away': actual['away'], 'home': actual['home']},
            'p_sim': round(p_sim, 4), 'p_team': None if p_team is None else round(p_team, 4), 'p_blend': round(p_blend, 4),
            'brier': {'sim': _brier(p_sim, y), 'team': None if p_team is None else _brier(p_team, y), 'blend': _brier(p_blend, y),
                      'coin': 0.25, 'home': _brier(HOME_RATE_PRIOR, y)},
            'log_loss': {'sim': _logloss(p_sim, y), 'team': None if p_team is None else _logloss(p_team, y), 'blend': _logloss(p_blend, y),
                         'coin': math.log(2), 'home': _logloss(HOME_RATE_PRIOR, y)},
        })

    def mean(key, metric):
        vals = [g[metric][key] for g in games if g[metric].get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    rows = [('coin', 'Coin flip'), ('home', 'Always pick the home team'), ('team', 'Team model'),
            ('sim', 'Simulator'), ('blend', 'Simulator + team model')]
    ladder = []
    for key, name in rows:
        b = mean(key, 'brier')
        ladder.append({'key': key, 'name': name, 'n': sum(1 for g in games if g['brier'].get(key) is not None),
                       'brier': None if b is None else round(b, 5),
                       'log_loss': None if mean(key, 'log_loss') is None else round(mean(key, 'log_loss'), 5),
                       'better_than_coin_pct': None if b is None else round((0.25 - b) / 0.25 * 100, 2)})
    return {
        'schema': 'brl.record.v1',
        'headline': 'blend',
        'method': ('Headline win chance is the equal-weight log-odds average of the simulator and the team model. '
                   'Each final game scores the last forecast version published before the observed first pitch.'),
        'home_rate_prior': HOME_RATE_PRIOR,
        'n_scored': len(games),
        'ladder': ladder,
        'games': sorted(games, key=lambda g: (g['date'] or '', g['game_pk'])),
        'unscored': unscored,
        'blend': blend,
    }
