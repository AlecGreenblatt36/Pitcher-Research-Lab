"""Public-layer headline win chance and an independent live track record.

Headline
--------
The headline recipe is versioned in brl_live/headline_params.json (brl_live/headline.py).
Through October 7, 2026 it was the equal-weight log-odds average of the simulator and the
team model. From October 8 it is our independent model (the simulator, the team model, the
starters and the team ratings, weighted to match the market's closing lines on the 2025 and
2026 regular seasons) combined with the market's pregame line when one was captured before
first pitch (the several-books average when read, else DraftKings), on the log-odds scale with the weights in
headline_params.json (60% market, 40% our model from October 8). A game is scored with the
recipe in force at its first pitch, so a change never rewrites a started game.

Track record
------------
For every final game, the last forecast version published before the observed first
pitch is scored (Brier and log loss) for the simulator, the team model, our model, the
headline and the market, next to a coin flip and always-pick-home at the prior-season home
rate. Games without a pregame publication are listed as unscored, never backfilled.
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
    market = ledger.get('market', {}) or {}
    boxes = ledger.get('box_scores', {}) or {}

    from .headline import version_at, ours as ours_probability, headline as headline_probability

    def team_probability(ident, f):
        """Our team model saved with the box when present; otherwise the runtime's baseline."""
        tm = (boxes.get(ident) or {}).get('team_model') or {}
        if tm.get('p_home') is not None:
            return float(tm['p_home'])
        return f.get('team_baseline_probability')

    def team_model_of(ident, f):
        tm = (boxes.get(ident) or {}).get('team_model') or {}
        if tm.get('p_home') is not None:
            return tm
        p = f.get('team_baseline_probability')
        return {'p_home': p} if p is not None else None

    def pregame_market(pk, before=None):
        """The last market line captured while the game was pregame (and before first pitch when known):
        the several-books average when at least three books were read, else the single book."""
        mk = market.get(str(pk))
        if not mk or mk.get('p_home') is None or not mk.get('captured_at'):
            return None
        if before is not None and not _ts(mk['captured_at']) < _ts(before):
            return None
        if mk.get('p_home_cons') is not None and (mk.get('n_books') or 0) >= 3:
            return float(mk['p_home_cons'])
        return float(mk['p_home'])

    # The version in force: at first pitch for games that have started, now for the rest.
    first_pitch_of = {str(pk): a.get('first_pitch_observed_at') for pk, a in actuals.items()}
    blend, ours, recipe = {}, {}, {}
    for ident, f in forecasts.items():
        pk = str(f['game_pk'])
        fp = first_pitch_of.get(pk)
        version = version_at(fp) if fp else version_at()
        p_o, how = ours_probability(f.get('home_win_probability'), team_model_of(ident, f), version)
        p_h, how_h = headline_probability(p_o, pregame_market(pk, fp), version)
        ours[ident] = round(p_o, 6); blend[ident] = round(p_h, 6); recipe[ident] = {'version': version['name'], 'ours': how, 'headline': how_h}

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
        p_team = team_probability(ident, f)
        p_team = None if p_team is None else float(p_team)
        p_runtime = f.get('team_baseline_probability')
        p_blend = blend[ident]
        p_ours = ours[ident]
        p_market = pregame_market(pk, first_pitch)
        games.append({
            'game_pk': int(pk), 'date': f.get('date'), 'away': f['away'].get('abbr'), 'home': f['home'].get('abbr'),
            'version': f.get('version'), 'forecast_id': ident, 'home_won': bool(y),
            'actual': {'away': actual['away'], 'home': actual['home']},
            'p_sim': round(p_sim, 4), 'p_team': None if p_team is None else round(p_team, 4), 'p_blend': round(p_blend, 4), 'p_ours': round(p_ours, 4),
            'p_market': None if p_market is None else round(p_market, 4), 'headline_version': recipe[ident]['version'],
            'p_team_runtime': None if p_runtime is None else round(float(p_runtime), 4),
            'brier': {'sim': _brier(p_sim, y), 'team': None if p_team is None else _brier(p_team, y), 'blend': _brier(p_blend, y),
                      'ours': _brier(p_ours, y), 'market': None if p_market is None else _brier(p_market, y),
                      'coin': 0.25, 'home': _brier(HOME_RATE_PRIOR, y)},
            'log_loss': {'sim': _logloss(p_sim, y), 'team': None if p_team is None else _logloss(p_team, y), 'blend': _logloss(p_blend, y),
                         'ours': _logloss(p_ours, y), 'market': None if p_market is None else _logloss(p_market, y),
                         'coin': math.log(2), 'home': _logloss(HOME_RATE_PRIOR, y)},
        })

    def mean(key, metric):
        vals = [g[metric][key] for g in games if g[metric].get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    rows = [('coin', 'Coin flip'), ('home', 'Always pick the home team'), ('market', 'Betting market'), ('team', 'Team model'),
            ('sim', 'Simulator (results only)'), ('ours', 'Our model (taught on past closing lines)'), ('blend', 'Headline (with the market line)')]
    ladder = []
    for key, name in rows:
        b = mean(key, 'brier')
        ladder.append({'key': key, 'name': name, 'n': sum(1 for g in games if g['brier'].get(key) is not None),
                       'brier': None if b is None else round(b, 5),
                       'log_loss': None if mean(key, 'log_loss') is None else round(mean(key, 'log_loss'), 5),
                       'better_than_coin_pct': None if b is None else round((0.25 - b) / 0.25 * 100, 2)})
    # Where our independent model and the market disagreed by five points or more: who was closer.
    split = [g for g in games if g.get('p_market') is not None and abs(g['p_ours'] - g['p_market']) >= 0.05]
    disagree = {'threshold': 0.05, 'n': len(split),
                'ours_brier': round(sum(g['brier']['ours'] for g in split) / len(split), 5) if split else None,
                'market_brier': round(sum(g['brier']['market'] for g in split) / len(split), 5) if split else None,
                'our_side_won': sum(1 for g in split if (g['p_ours'] > g['p_market']) == g['home_won']) if split else 0}
    # Line movement: for games where our final pregame view differed from the first line we saw, did the line
    # move toward our number by first pitch? (CLV-02 measured it on past seasons; this is the live tally.)
    moves = {'threshold_logit': 0.1, 'n': 0, 'toward_ours': 0}
    for g in games:
        mk = market.get(str(g['game_pk'])) or {}
        consensus = mk.get('p_home_cons') is not None and (mk.get('n_books') or 0) >= 3
        first, last = (mk.get('first_p_home_cons') if consensus else mk.get('first_p_home')), g.get('p_market')
        if first is None or last is None or not mk.get('first_captured_at') or abs(last - first) < 1e-4:
            continue
        gap = _logit(g['p_ours']) - _logit(first)
        if abs(gap) <= 0.1:
            continue
        moves['n'] += 1
        moves['toward_ours'] += int((last - first) * gap > 0)
    now_v = version_at()
    return {
        'schema': 'brl.record.v3',
        'headline_now': {'name': now_v['name'], 'kind': now_v.get('kind'), 'market': (now_v.get('headline') or {}).get('market'), 'ours': (now_v.get('headline') or {}).get('ours')},
        'line_moves': moves,
        'market_note': ('Betting market rows use the last ESPN scoreboard moneyline captured while the game was pregame, vig removed. '
                        'Our model never uses it; the headline combines the two when a pregame line exists.'),
        'headline': 'blend',
        'method': ('Headline win chance: our independent model combined with the pregame market line when one was captured, '
                   'with the recipe in force at first pitch (equal-weight simulator and team model before October 8, 2026). '
                   'Each final game scores the last forecast version published before the observed first pitch.'),
        'disagreements': disagree,
        'ours': ours,
        'home_rate_prior': HOME_RATE_PRIOR,
        'n_scored': len(games),
        'ladder': ladder,
        'games': sorted(games, key=lambda g: (g['date'] or '', g['game_pk'])),
        'unscored': unscored,
        'blend': blend,
        'recipe': recipe,
    }
