"""KEYS-02: are the game page's "What decides it" chances right on real games?

The page shows, from a game's simulated worlds, how often a thing happens (a team scores first, its starter gets 18 or
more outs, a hitter homers or gets two or more hits) and the team's win chance when it does and when it does not
(KEYS-01). Replays with starter_lines and hitter_lines record the same counts per team-game (starters[side]['keys']:
first, first and won, won, starter 18+ outs, that and won; hitters[i]['kw']: worlds won with his home run, with his 2+
hits). This scores them against the real games, per season:

  share   the mean simulated chance of the event against how often it happened;
  if      over team-games where it happened, the mean simulated win chance given it against the real win rate;
  else    the same where it did not happen.

The page shifts both chances to the published pick on the log-odds scale; the replay has the simulator's own chance
only, whose mix is the simulator's win chance, so this checks the raw conditionals. Model outputs and box-score facts
only.

    python3 tools/brl_keys_check.py --season 2025='<glob>' --season 2026='<glob>' --out <file>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path


def first_inning_scored(starts, final):
    """The inning a side first scored in, from its score at the start of each inning it batted and its final runs."""
    seq = sorted((int(i), int(v)) for i, v in (starts or []))
    if not seq or final <= 0:
        return None
    for k, (inn, v) in enumerate(seq):
        nxt = seq[k + 1][1] if k + 1 < len(seq) else final
        if v == 0 and nxt > 0:
            return inn
    return seq[0][0] if seq[0][1] > 0 else None


def real_first(g):
    fs = ((g.get('innings') or {}).get('actual_first_scores')) or {}
    if not fs:
        return 'unknown'
    ia = first_inning_scored(fs.get('away'), int(g['away_runs']))
    ih = first_inning_scored(fs.get('home'), int(g['home_runs']))
    if ia is None and ih is None:
        return None
    if ih is None or (ia is not None and ia <= ih):     # the road team bats first in each inning
        return 'away'
    return 'home'


class Cell:
    def __init__(self):
        self.n = 0; self.p = 0.0; self.hit = 0
        self.n_if = 0; self.p_if = 0.0; self.w_if = 0
        self.n_else = 0; self.p_else = 0.0; self.w_else = 0

    def add(self, share, p_if, p_else, happened, won):
        self.n += 1; self.p += share; self.hit += int(happened)
        if happened and p_if is not None:
            self.n_if += 1; self.p_if += p_if; self.w_if += int(won)
        if not happened and p_else is not None:
            self.n_else += 1; self.p_else += p_else; self.w_else += int(won)

    def out(self):
        f = lambda a, b: round(100.0 * a / b, 2) if b else None
        return {'n': self.n, 'share_sim': f(self.p, self.n), 'share_real': f(self.hit, self.n),
                'if_n': self.n_if, 'if_sim': f(self.p_if, self.n_if), 'if_real': f(self.w_if, self.n_if),
                'else_n': self.n_else, 'else_sim': f(self.p_else, self.n_else), 'else_real': f(self.w_else, self.n_else)}


def cond(k_and, k_event, k_won, n):
    """Win chance given the event and given not, from world counts (None when a side has no worlds)."""
    p_if = k_and / k_event if k_event else None
    p_else = (k_won - k_and) / (n - k_event) if n - k_event else None
    return p_if, p_else


def season_check(games, min_share=0.08, max_share=0.92):
    cells = {k: Cell() for k in ('first', 'sp6', 'hr', 'h2')}
    shown = {k: Cell() for k in ('sp6', 'hr', 'h2')}        # only events in the page's 8% to 92% window
    skipped = 0
    for g in games:
        st, n = g.get('starters') or {}, int(g['n'])
        if g['home_runs'] == g['away_runs']:
            skipped += 1; continue
        winner = 'home' if g['home_runs'] > g['away_runs'] else 'away'
        rf = real_first(g)
        for side in ('away', 'home'):
            k = (st.get(side) or {}).get('keys')
            if not k:
                continue
            won = winner == side
            if rf != 'unknown':
                p_if, p_else = cond(k[1], k[0], k[2], n)
                cells['first'].add(k[0] / n, p_if, p_else, rf == side, won)
            outs = ((st[side].get('actual') or {}).get('outs'))
            if outs is not None:
                p_if, p_else = cond(k[4], k[3], k[2], n)
                for c in (cells['sp6'],) + ((shown['sp6'],) if min_share <= k[3] / n <= max_share else ()):
                    c.add(k[3] / n, p_if, p_else, int(outs) >= 18, won)
            for x in g.get('hitters') or []:
                if x.get('side') != side or x.get('actual') is None or 'kw' not in x:
                    continue
                a = x['actual']
                hr_w = n - x['hr_hist'][0]; h2_w = n - x['h_hist'][0] - x['h_hist'][1]
                for key, ev_w, kw, happened in (('hr', hr_w, x['kw'][0], a['hr'] >= 1), ('h2', h2_w, x['kw'][1], a['h'] >= 2)):
                    p_if, p_else = cond(kw, ev_w, k[2], n)
                    for c in (cells[key],) + ((shown[key],) if min_share <= ev_w / n <= max_share else ()):
                        c.add(ev_w / n, p_if, p_else, happened, won)
    return {'games_skipped_tied': skipped, 'all': {k: c.out() for k, c in cells.items()}, 'shown_window': {k: c.out() for k, c in shown.items()}}


def load_games(paths):
    seen, out = set(), []
    for f in paths:
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                g = json.loads(line)
                if g['game_pk'] not in seen:
                    seen.add(g['game_pk']); out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = {}
    for spec in a.season:
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat)); assert files, pat
        out[lab] = season_check(load_games(files))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for lab, r in out.items():
        for scope in ('all', 'shown_window'):
            for k, c in r[scope].items():
                print(lab, scope, k, 'n', c['n'], 'share %s/%s' % (c['share_sim'], c['share_real']),
                      'if %s/%s (%d)' % (c['if_sim'], c['if_real'], c['if_n']), 'else %s/%s (%d)' % (c['else_sim'], c['else_real'], c['else_n']))


if __name__ == '__main__':
    main()
