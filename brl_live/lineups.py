"""A team's batting order before its lineup is posted (LEDGER LINEUP-01 and LINEUP-02).

Used by the simulator (brl_engine/runtime/cloud/contracts.py) and the game plans (tools/brl_report.py), and measured on
every 2025 and 2026 game by the report lane's lineup study (tools/brl_report.py lineup_study).

A prior game is (day, game_pk, order, opp_hand, batted): day an ordinal, order the nine starters in batting order,
opp_hand 'R' or 'L' for the opposing starter, batted everyone who batted for the team in that game.

Rules:
  last  the team's most recent lineup (the rule through October 10, 2026).
  hand  its most recent lineup against a starter of the same hand as today's, within HAND_DAYS; else 'last'.
  freq  its usual lineup against that hand: over its last FREQ_GAMES games against that hand within FREQ_DAYS, the nine
        with the most starts (among hitters who batted in one of the team's last FREQ_ACTIVE games or started the latest
        of those games), in their average batting spot; else 'last'.
"""
from __future__ import annotations

RULE = 'hand'          # LINEUP-01 passed (October 10, 2026, run 38082910875); LINEUP-02 ('freq') missed on batting spots
RULES = ('last', 'hand', 'freq')
HAND_DAYS = 30
FREQ_GAMES = 10
FREQ_DAYS = 45
FREQ_ACTIVE = 5


def _who(hand: str) -> str:
    return 'a lefty' if hand == 'L' else 'a righty'


def project(prior, today: int, hand: str | None, rule: str | None = None):
    """(order of nine ids, source words) from the team's finished games before today; (None, None) without one."""
    rule = rule or RULE
    if rule not in RULES:
        raise ValueError('unknown lineup rule ' + str(rule))
    games = sorted((g for g in prior if g[0] < today and len(g[2]) == 9), key=lambda g: (g[0], g[1]))
    if not games:
        return None, None
    last = list(games[-1][2])
    if rule == 'last' or hand not in ('R', 'L'):
        return last, 'last game'
    same = [g for g in games if g[3] == hand]
    if rule == 'hand':
        recent = [g for g in same if g[0] >= today - HAND_DAYS]
        return (list(recent[-1][2]), 'last game against ' + _who(hand)) if recent else (last, 'last game')
    window = [g for g in same if g[0] >= today - FREQ_DAYS][-FREQ_GAMES:]
    if not window:
        return last, 'last game'
    active = set(window[-1][2])
    for g in games[-FREQ_ACTIVE:]:
        active.update(g[4])
    starts, spot_sum, latest = {}, {}, {}
    for k, g in enumerate(window):
        for i, pid in enumerate(g[2]):
            starts[pid] = starts.get(pid, 0) + 1
            spot_sum[pid] = spot_sum.get(pid, 0) + i + 1
            latest[pid] = k
    cand = [pid for pid in starts if pid in active]
    pick = sorted(cand, key=lambda q: (-starts[q], -latest[q], spot_sum[q] / starts[q], q))[:9]
    if len(pick) < 9:
        return list(window[-1][2]), 'last game against ' + _who(hand)
    order = sorted(pick, key=lambda q: (spot_sum[q] / starts[q], -starts[q], q))
    return order, 'usual lineup against ' + _who(hand)


def games_from_history(hist, abbr: str) -> list:
    """The team's prior games from the simulator's plate-appearance history (one row per plate appearance)."""
    import pandas as pd
    top = hist['inning_topbot'].astype(str).str.lower().str.startswith('top')
    bats = hist[((hist['away_team'] == abbr) & top) | ((hist['home_team'] == abbr) & ~top)]
    out = []
    for pk, g in bats.sort_values(['game_pk', 'at_bat_number']).groupby('game_pk', sort=False):
        order = list(dict.fromkeys(int(b) for b in g['batter'].tolist()))[:9]
        day = pd.Timestamp(str(g['date_key'].iloc[0])[:10]).toordinal()
        hand = str(g['p_throws'].iloc[0] or 'R').upper()[:1]
        out.append((day, int(pk), order, 'L' if hand == 'L' else 'R', {int(b) for b in g['batter'].tolist()}))
    return out
