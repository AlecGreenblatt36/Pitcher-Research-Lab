"""Postseason series outlook for the page: the series score, our win chance for each remaining game and the chance
each team advances.

The current game uses our headline forecast (and the page switches to the live win chance once it starts). Games
not yet forecast use the series' strength estimate: the average, over every game of this series we have forecast, of
our headline log-odds for the higher seed with the home edge taken out, plus or minus the home edge for where the
game is played. The home edge is the league's (home teams won 52.8% of 2023 to 2025 regular-season games). The page
runs the series out from the current score with these chances; nothing here is fitted to series results.

Inputs are the public ledger fields the page already shows (schedule context with the series record, forecasts,
actual finals) and the record's headline numbers. No player data.
"""
from __future__ import annotations

import math

POSTSEASON = ('F', 'D', 'L', 'W')
# Which seed is at home in each game: Wild Card all at the higher seed, Division Series 2-2-1, LCS and World Series 2-3-2.
HOME_PATTERN = {3: 'HHH', 5: 'HHLLH', 7: 'HHLLLHH'}
HOME_EDGE = math.log(0.528 / 0.472)


def _logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, float(p)))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def series_probability(best_of: int, wins_high: int, wins_low: int, p_high: dict) -> dict:
    """Chance the higher seed advances, and the chance of each final series score, from the current score.
    p_high: game number -> the higher seed's chance of winning that game."""
    need = best_of // 2 + 1
    states = {(wins_high, wins_low): 1.0}
    finals: dict = {}
    while states:
        nxt: dict = {}
        for (a, b), pr in states.items():
            if a >= need or b >= need:
                finals[(a, b)] = finals.get((a, b), 0.0) + pr
                continue
            p = float(p_high[a + b + 1])
            nxt[(a + 1, b)] = nxt.get((a + 1, b), 0.0) + pr * p
            nxt[(a, b + 1)] = nxt.get((a, b + 1), 0.0) + pr * (1 - p)
        states = nxt
    return {'high': sum(v for (a, b), v in finals.items() if a >= need),
            'finals': {f'{a}-{b}': v for (a, b), v in sorted(finals.items())}}


def series_outlook(ledger: dict, record: dict) -> dict:
    forecasts = ledger.get('forecasts') or {}
    contexts = ledger.get('context') or {}
    actuals = ledger.get('actuals') or {}
    live = ledger.get('live') or {}
    blend = (record or {}).get('blend') or {}
    latest: dict = {}
    for ident, f in forecasts.items():
        if f.get('game_type') not in POSTSEASON:
            continue
        pk = str(f['game_pk'])
        rank = (int(f.get('version') or 0), str(f.get('saved_at') or ''))
        if pk not in latest or rank > latest[pk][0]:
            latest[pk] = (rank, ident, f)

    def pair(f):
        return tuple(sorted((f['away']['abbr'], f['home']['abbr'])))

    out = []
    for pk, ctx in contexts.items():
        if str(pk) not in latest:
            continue
        _, ident, f = latest[str(pk)]
        best_of, n = int(ctx.get('games_in_series') or 0), int(ctx.get('series_game_number') or 0)
        pattern = HOME_PATTERN.get(best_of)
        if not pattern or not 1 <= n <= best_of:
            continue
        away, home = f['away']['abbr'], f['home']['abbr']
        high, low = (home, away) if pattern[n - 1] == 'H' else (away, home)
        rec = ctx.get('records') or {}
        wins = {away: int((rec.get('away') or {}).get('wins') or 0), home: int((rec.get('home') or {}).get('wins') or 0)}
        actual = actuals.get(str(pk))
        if actual is not None and actual.get('away') is not None and wins[away] + wins[home] < n:
            # A final the schedule's series record has not counted yet.
            wins[away if actual['away'] > actual['home'] else home] += 1
        # Strength: every game of this series we have forecast (same teams, same round and season).
        logits = []
        for _, (_, gid, g) in latest.items():
            if pair(g) != pair(f) or g.get('game_type') != f.get('game_type') or str(g.get('date'))[:4] != str(f.get('date'))[:4]:
                continue
            p_home = blend.get(gid, g.get('home_win_probability'))
            if p_home is None:
                continue
            z = _logit(p_home)
            logits.append((z - HOME_EDGE) if g['home']['abbr'] == high else (-z + HOME_EDGE))
        strength = sum(logits) / len(logits) if logits else 0.0
        state = 'final' if actual is not None and actual.get('away') is not None else 'live' if str(pk) in live else 'pregame'
        games = []
        played = wins[high] + wins[low]
        need = best_of // 2 + 1
        done = max(wins.values()) >= need
        for k in range(1, best_of + 1):
            if (k <= played and k != n) or (done and k > n):
                continue
            at = high if pattern[k - 1] == 'H' else low
            g = {'n': k, 'home': at, 'away': low if at == high else high}
            if k == n:
                g.update(game_pk=int(pk), state=state, p_home=round(float(blend.get(ident, f.get('home_win_probability'))), 4), source='forecast')
                if state == 'final':
                    g['winner'] = away if actual['away'] > actual['home'] else home
            else:
                z = strength + (HOME_EDGE if at == high else -HOME_EDGE)
                g.update(state='future', p_home=round(_sigmoid(z) if at == high else 1 - _sigmoid(z), 4), source='series estimate')
            games.append(g)
        p_high = {g['n']: (g['p_home'] if g['home'] == high else 1 - g['p_home']) for g in games}
        chance = None if done else series_probability(best_of, wins[high], wins[low], p_high)
        out.append({'id': f"{str(f.get('date'))[:4]}-{ctx.get('series_description')}-{high}-{low}",
                    'round': ctx.get('series_description'), 'game_type': f.get('game_type'), 'best_of': best_of, 'date': ctx.get('date'),
                    'high': high, 'low': low, 'names': {away: f['away'].get('name'), home: f['home'].get('name')},
                    'wins': {high: wins[high], low: wins[low]}, 'current_game': n, 'games': games,
                    'strength': round(strength, 4), 'home_edge': round(HOME_EDGE, 4), 'forecast_games': len(logits),
                    'advance': None if done else {high: round(chance['high'], 4), low: round(1 - chance['high'], 4)},
                    'winner': (high if wins[high] >= need else low) if done else None})
    out.sort(key=lambda s: (s['game_type'], s['round'] or '', s['high']))
    return {'schema': 'brl.series.v1', 'home_edge': round(HOME_EDGE, 4),
            'method': ('Each remaining game: our headline forecast when it exists (the live win chance once it starts), '
                       'otherwise the average of our headline numbers in this series with the home edge taken out, plus or '
                       "minus the league's home edge for where the game is played. The series is run out from the current score."),
            'series': out}
