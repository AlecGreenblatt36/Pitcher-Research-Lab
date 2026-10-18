"""LEG-01 against KALSHI-01: our chance for each Kalshi market the replays price, against Kalshi's last pregame price, on
the same player-games and games.

Hitter props (KXMLBHIT 1+ and 2+ hits, KXMLBHR 1+ home run, KXMLBTB 2+ total bases) are matched by game id and player id to
the replay's hitter lines, starter strikeouts (KXMLBKS, every line) to the replay's starter ladder, game winners (KXMLBGAME)
by game id and team. For each kind: Brier of ours and of Kalshi's price, the paired difference with a date
bootstrap, calibration of both by bin, and where we and Kalshi differ by 5 points or more, how often our side was right
and what buying our side at Kalshi's price would have returned (Kalshi's trading fee, 0.07 x p x (1 - p) per contract,
taken off; Kalshi lists these series with a fee multiplier of 0.5, so the return at half that fee is shown too). Kalshi prices are the volume-weighted price of the final pregame hour, else the last pregame trade; markets
with neither are left out and counted.

    python3 tools/brl_leg_vs_kalshi.py --replay <glob of leg01 rows> --kalshi <kalshi jsonl.gz> [--kalshi ...] --out <file.json>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from brl_player_cal import own_at_least, own_tb2  # noqa: E402
from brl_leg_book import pmf, over  # noqa: E402

HIT_STATS = {'hits': 'hits', 'hit': 'hits', 'home runs': 'hr', 'home run': 'hr', 'total bases': 'tb', 'total base': 'tb'}
K_STATS = ('strikeouts', 'strikeout')

BINS = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]


def fee(p):
    return 0.07 * p * (1 - p)


def kalshi_rows(paths):
    out = []
    for pat in paths:
        for f in sorted(glob.glob(pat)):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    r = json.loads(line)
                    price = r.get('vwap_hour') if r.get('vwap_hour') is not None else r.get('last')
                    if price is None or r.get('result') not in ('yes', 'no'):
                        continue
                    r['price'] = float(price); r['y'] = int(r['result'] == 'yes')
                    out.append(r)
    return out


def replay_index(paths):
    hit, game, start, by_teams = {}, {}, {}, defaultdict(list)
    for pat in paths:
        for f in sorted(glob.glob(pat)):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    g = json.loads(line)
                    pk = int(g['game_pk'])
                    n = g.get('n') or 1
                    game[pk] = {'date': g['date'], 'p_home': g['home_wins'] / max(n - (g.get('ties') or 0), 1), 'home': g['home'], 'away': g['away']}
                    by_teams[(g['date'], str(g['away']) + str(g['home']))].append(pk)
                    for x in g.get('hitters') or []:
                        hit[(pk, int(x['batter']))] = x
                    for side in ('away', 'home'):
                        st = (g.get('starters') or {}).get(side) or {}
                        if st.get('pitcher'):
                            start[(pk, int(st['pitcher']))] = st
    return hit, game, start, by_teams


def compare(pairs, draws=2000, seed=5):
    """pairs: (date, ours, kalshi price, outcome). Brier of each, paired difference with a date bootstrap, calibration, and
    the disagreement test with Kalshi's fee."""
    if not pairs:
        return None
    d = np.array([p[0] for p in pairs]); po = np.array([p[1] for p in pairs], float); pk = np.array([p[2] for p in pairs], float); y = np.array([p[3] for p in pairs], float)
    diff = (po - y) ** 2 - (pk - y) ** 2
    keys, inv = np.unique(d, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(keys)); n = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    cal = []
    for lo, hi in BINS:
        m_o = (po >= lo) & (po < hi); m_k = (pk >= lo) & (pk < hi)
        cal.append({'bin': f'{lo:.1f}-{min(hi, 1):.1f}', 'ours_n': int(m_o.sum()), 'ours_stated': round(float(po[m_o].mean()), 4) if m_o.any() else None,
                    'ours_hit': round(float(y[m_o].mean()), 4) if m_o.any() else None, 'kalshi_n': int(m_k.sum()),
                    'kalshi_stated': round(float(pk[m_k].mean()), 4) if m_k.any() else None, 'kalshi_hit': round(float(y[m_k].mean()), 4) if m_k.any() else None})
    out = {'n': int(len(y)), 'dates': int(len(keys)), 'brier_ours': round(float(((po - y) ** 2).mean()), 5), 'brier_kalshi': round(float(((pk - y) ** 2).mean()), 5),
           'difference': round(float(diff.mean()), 6), 'ci': [round(float(np.quantile(boot, .025)), 6), round(float(np.quantile(boot, .975)), 6)],
           'calibration': cal, 'disagree': {}}
    for gap in (0.03, 0.05, 0.10):
        m = np.abs(po - pk) >= gap
        if not m.any():
            continue
        buy_yes = po > pk                                   # we think yes is cheap: buy yes at pk; else buy no at 1 - pk
        cost = np.where(buy_yes, pk, 1 - pk)
        win = np.where(buy_yes, y, 1 - y)
        profit = (win - cost - fee(cost))[m]
        profit_half = (win - cost - fee(cost) / 2)[m]
        km, im = np.unique(d[m], return_inverse=True)
        sp = np.bincount(im, weights=profit, minlength=len(km)); sc = np.bincount(im, weights=cost[m], minlength=len(km))
        bidx = rng.integers(0, len(km), size=(draws, len(km)))
        bret = sp[bidx].sum(1) / sc[bidx].sum(1)
        out['disagree'][f'{int(gap * 100)}pts'] = {'n': int(m.sum()), 'our_side_right': round(float(win[m].mean()), 4),
                                                   'price_paid_mean': round(float(cost[m].mean()), 4),
                                                   'return_per_dollar': round(float(profit.sum() / cost[m].sum()), 4),
                                                   'return_ci': [round(float(np.quantile(bret, .025)), 4), round(float(np.quantile(bret, .975)), 4)],
                                                   'return_per_dollar_half_fee': round(float(profit_half.sum() / cost[m].sum()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', action='append', required=True)
    ap.add_argument('--kalshi', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    hit, game, start, by_teams = replay_index(a.replay)
    rows = kalshi_rows(a.kalshi)
    pairs = defaultdict(list); counts = defaultdict(int)
    for r in rows:
        pk = int(r.get('game_pk') or 0)
        stat = str(r.get('stat') or '')
        if r.get('kind') == 'prop' and stat in HIT_STATS:
            what, k = HIT_STATS[stat], int(r['at_least'])
            x = hit.get((pk, int(r.get('player_id') or 0)))
            if x is None:
                counts[f'{what}_no_replay_row'] += 1; continue
            if not (x.get('actual') or {}).get('pa'):           # Kalshi settles these at a fair price when he has no plate appearance
                counts[f'{what}_did_not_bat'] += 1; continue
            if what == 'hits':
                ours = own_at_least(x['h_hist'], k, x.get('spot'))
            elif what == 'hr':
                ours = own_at_least(x['hr_hist'], k, x.get('spot'))
            elif k == 2 and 'hs_hist' in x:
                ours = own_tb2(x['hs_hist'], x.get('spot'))
            elif k == 1:
                ours = own_at_least(x['h_hist'], 1, x.get('spot'))
            else:
                counts[f'tb{k}_not_priced'] += 1; continue
            pairs[f"{what}{k}+"].append((r['date'], ours, r['price'], r['y']))
        elif r.get('kind') == 'prop' and stat in K_STATS:
            k = int(r['at_least'])
            st = start.get((pk, int(r.get('player_id') or 0)))
            if st is None or 'k' not in st:
                counts['k_not_our_starter'] += 1; continue      # Kalshi settles at a fair price when he does not start
            ours = over(pmf(st['k']), k - 0.5)
            band = 'low' if k <= 4 else ('mid' if k <= 6 else 'high')
            pairs[f'starter_k{band}'].append((r['date'], ours, r['price'], r['y']))
            pairs['starter_k_all'].append((r['date'], ours, r['price'], r['y']))
        elif r.get('series') == 'KXMLBGAME':
            g = game.get(pk)
            if g is not None and str(g['away']) + str(g['home']) != r.get('teams'):
                # rows fetched before October 18 matched a game by start time alone and could name a neighbour that
                # started the same minute; the date and the two team codes find the right one (one game that day)
                twins = by_teams.get((r['date'], r.get('teams')), [])
                if len(twins) != 1:
                    counts['game_teams_unmatched'] += 1; continue
                pk = twins[0]; g = game[pk]; counts['game_rematched_by_teams'] += 1
            if g is None:
                counts['game_no_replay_row'] += 1; continue
            team = str(r['ticker']).rsplit('-', 1)[-1]
            teams = str(r.get('teams') or '')
            # the event's team string is away then home; the market's suffix names the team the yes side backs
            is_home = teams.endswith(team) and not teams.startswith(team)
            is_away = teams.startswith(team) and not teams.endswith(team)
            if not (is_home or is_away):
                counts['game_side_unknown'] += 1; continue
            if is_away:
                counts['game_away_market_skipped'] += 1; continue   # each game once: its home team's market (yes or no)
            ours = g['p_home'] if is_home else 1 - g['p_home']
            pairs['winner'].append((r['date'], ours, r['price'], r['y']))
        else:
            counts['other_market'] += 1
    out = {'schema': 'brl.leg-vs-kalshi.v1', 'experiment': 'LEG-01', 'kalshi_rows_priced': len(rows), 'unmatched': dict(counts),
           'kinds': {k: compare(v) for k, v in pairs.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (None if v is None else {kk: v[kk] for kk in ('n', 'brier_ours', 'brier_kalshi', 'difference', 'ci', 'disagree')}) for k, v in out['kinds'].items()}, indent=1))
    print('unmatched', dict(counts))


if __name__ == '__main__':
    main()
