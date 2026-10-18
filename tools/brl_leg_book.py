"""LEG-01: the leg book. Every bet leg the simulator prices, scored on full-season replays against what happened.

Reads the per-game rows of replays run with hitter and starter lines (research/replay-<tag>-<run>-s<k>of<n>.jsonl.gz on the
ledger branch: model outputs and box-score facts only) and, where we hold them, the market's closing lines for the same games
(research/market-books-2026-*.jsonl.gz, research/market-sbr-*.jsonl.gz). For each leg family (a market, a side and a line) it
reports calibration by bin of the stated chance, the Brier score, a logistic recalibration fitted on one season and applied to
the other, which (family, bin) cells are verified (LEDGER LEG-01), cross-game parlays of verified legs, a daily "best legs"
pick, and win and total chances against the closing line.

    python3 tools/brl_leg_book.py --season 2025=<glob> --season 2026=<glob> [--market-books <jsonl.gz>] [--market-sbr <jsonl.gz>]
        --out <file.json>

Totals use the convolution of the two teams' run distributions (the replay rows keep the margin and each team's runs, not the
joint total); TOTALS-04 used the same approximation. Hitter chances are computed the way the page computes them
(tools/brl_player_cal.own_at_least, each of the slot's events kept with the spot's share)."""
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
from brl_player_cal import own_at_least, own_tb2, fit_map, apply_map, logit  # noqa: E402

MAXM = 15
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]
TOTAL_LINES = (6.5, 7.5, 8.5, 9.5, 10.5, 11.5)
TEAM_LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
K_LINES = (2.5, 3.5, 4.5, 5.5, 6.5, 7.5)
OUTS_LINES = (11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5)
H_LINES = (3.5, 4.5, 5.5, 6.5)
BB_LINES = (0.5, 1.5, 2.5)


RUN_SPREAD, RUN_MAX, K_SPREAD = 1.0954, 40, 1.0693      # the page's constants (brl_live/page/template.html)


def nb_runs(h):
    """The page's nbRuns: a team's runs as a negative binomial with the simulated mean and RUN_SPREAD times its variance."""
    h = np.asarray(h, float); t = h.sum(); k = np.arange(len(h))
    m = float((k * h).sum() / t); var = float(((k - m) ** 2 * h).sum() / t)
    sd = math.sqrt(var) or 1.0
    m = max(m, 0.05); v = sd * sd * RUN_SPREAD
    ks = np.arange(RUN_MAX)
    if v <= m * 1.0001:
        lp = ks * math.log(m) - m - np.array([math.lgamma(x + 1) for x in ks])
    else:
        r = m * m / (v - m); q = r / (r + m)
        lp = np.array([math.lgamma(x + r) - math.lgamma(r) - math.lgamma(x + 1) + r * math.log(q) + x * math.log(1 - q) for x in ks])
    e = np.exp(lp - lp.max())
    return e / e.sum()


def k_shape(h):
    """The page's kShape: a starter's strikeouts spread by K_SPREAD (one step each way with weight w, the mass below zero
    folded onto zero)."""
    h = np.asarray(h, float); nz = np.nonzero(h)[0]
    n = int(nz.max()) + 2 if len(nz) else len(h) + 1
    p = np.zeros(n); p[:len(h)] = h[:n] / h.sum()
    k = np.arange(n); m = float((k * p).sum()); v = float(((k - m) ** 2 * p).sum())
    w = min(0.25, (K_SPREAD - 1) * v / 2)
    q = (1 - 2 * w) * p
    q[1:] += w * p[:-1]; q[:-1] += w * p[1:]; q[0] += w * p[0]
    return q


def pmf(h):
    h = np.asarray(h, float)
    return h / h.sum() if h.sum() else h


def over(p, line):
    """P(X > line) for a pmf over 0..len-1 (the last cell holds its value and above)."""
    k = int(math.floor(line)) + 1
    return float(p[k:].sum()) if k < len(p) else 0.0


def push_free_over(p, line):
    """P(over | no push) for a whole-number line; the plain over for a half line."""
    if float(line).is_integer():
        k = int(line)
        po = float(p[k + 1:].sum()); pu = float(p[:k].sum())
        return po / (po + pu) if po + pu else float('nan')
    return over(p, line)


def two_sided(fam, p, y, base):
    """Both sides of a binary leg; the caller keeps only the favored side when binning."""
    return [dict(base, fam=fam + ':yes', p=p, y=int(y)), dict(base, fam=fam + ':no', p=1.0 - p, y=1 - int(y))]


def cumulative_at(scores, inning):
    """Runs through `inning` from innings.actual_first_scores ([inning, cumulative runs] pairs)."""
    best = None
    for i, c in scores or []:
        if int(i) <= inning:
            best = int(c)
    return best


def game_legs(g, as_shown=False):
    """Every leg of one replayed game. as_shown: team runs and totals through the page's negative binomial (nbRuns) and the
    starter's strikeouts through its spread (kShape), so the chances are the ones the page states."""
    n = g.get('n') or 1
    base = {'date': g['date'], 'game_pk': int(g['game_pk'])}
    hr, ar = int(g['home_runs']), int(g['away_runs'])
    legs = []
    ties = g.get('ties') or 0
    p_home = g['home_wins'] / max(n - ties, 1)
    legs += two_sided('win:home', p_home, hr > ar, base)
    hp, ap = (nb_runs(g['home_hist']), nb_runs(g['away_hist'])) if as_shown else (pmf(g['home_hist']), pmf(g['away_hist']))
    if 'margin_hist' in g:
        mh = pmf(g['margin_hist']); idx = np.arange(-MAXM, MAXM + 1)
        legs += two_sided('runline:home-1.5', float(mh[idx >= 2].sum()), hr - ar >= 2, base)
        legs += two_sided('runline:away-1.5', float(mh[idx <= -2].sum()), ar - hr >= 2, base)
    tp = np.convolve(hp, ap)
    for L in TOTAL_LINES:
        legs += two_sided(f'total:over{L}', over(tp, L), hr + ar > L, base)
    for side, p_, r_ in (('home', hp, hr), ('away', ap, ar)):
        for L in TEAM_LINES:
            legs += two_sided(f'team_total:{side}:over{L}', over(p_, L), r_ > L, base)
    fi = g.get('first_inning')
    afs = (g.get('innings') or {}).get('actual_first_scores') or {}
    a1, h1 = cumulative_at(afs.get('away'), 1), cumulative_at(afs.get('home'), 1)
    if fi and a1 is not None and h1 is not None:
        legs += two_sided('first_inning:run_scored', 1.0 - fi[0] / n, (a1 + h1) > 0, base)
    a5, h5 = cumulative_at(afs.get('away'), 5), cumulative_at(afs.get('home'), 5)
    if 'f5_margin_hist' in g and a5 is not None and h5 is not None and len(afs.get('home') or []) >= 5:
        fm = pmf(g['f5_margin_hist']); idx = np.arange(-MAXM, MAXM + 1)
        legs += two_sided('f5:home_leads', float(fm[idx > 0].sum()), h5 > a5, base)
        legs += two_sided('f5:away_leads', float(fm[idx < 0].sum()), a5 > h5, base)
        ft = pmf(g['f5_total_hist'])
        for L in (3.5, 4.5, 5.5):
            legs += two_sided(f'f5_total:over{L}', over(ft, L), a5 + h5 > L, base)
    for x in g.get('hitters') or []:
        a = x.get('actual') or {}
        if not a.get('pa'):
            continue
        b = dict(base, player=int(x['batter']))
        spot = x.get('spot')
        legs += two_sided('hitter:hits1+', own_at_least(x['h_hist'], 1, spot), (a.get('h') or 0) >= 1, b)
        legs += two_sided('hitter:hits2+', own_at_least(x['h_hist'], 2, spot), (a.get('h') or 0) >= 2, b)
        legs += two_sided('hitter:hr', own_at_least(x['hr_hist'], 1, spot), (a.get('hr') or 0) >= 1, b)
        legs += two_sided('hitter:k1+', own_at_least(x['k_hist'], 1, spot), (a.get('k') or 0) >= 1, b)
        legs += two_sided('hitter:bb1+', own_at_least(x['bb_hist'], 1, spot), (a.get('bb') or 0) >= 1, b)
        if 'hs_hist' in x and 'xbh' in a:
            legs += two_sided('hitter:tb2+', own_tb2(x['hs_hist'], spot), (a.get('h') or 0) >= 2 or (a.get('xbh') or 0) >= 1, b)
    for side in ('home', 'away'):
        s = (g.get('starters') or {}).get(side) or {}
        act = s.get('actual')
        if not act:
            continue
        b = dict(base, player=int(s.get('pitcher') or 0))
        for field, lines, af in (('k', K_LINES, 'k'), ('outs', OUTS_LINES, 'outs'), ('h', H_LINES, 'h'), ('bb', BB_LINES, 'bb')):
            if field in s and af in act:
                p_ = k_shape(s[field]) if (as_shown and field == 'k') else pmf(s[field])
                for L in lines:
                    legs += two_sided(f'starter:{field}:over{L}', over(p_, L), act[af] > L, b)
    return legs


def load(globs):
    rows = []
    for pat in globs:
        for f in sorted(glob.glob(pat)):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    g = json.loads(line)
                    if g.get('home_runs') is None or (g.get('game_type') or 'R') != 'R':
                        continue
                    rows.append(g)
    return rows


def family_of(f):
    return f.rsplit(':', 1)[0]


def bin_table(legs):
    """Favored side only (p >= 0.5), by family and bin: n, mean stated, hit rate, standard error."""
    cells = defaultdict(lambda: [0, 0.0, 0])
    for l in legs:
        if l['p'] < 0.5 or not np.isfinite(l['p']):
            continue
        for lo, hi in BINS:
            if lo <= l['p'] < hi:
                c = cells[(l['fam'], lo)]
                c[0] += 1; c[1] += l['p']; c[2] += l['y']
                break
    out = {}
    for (fam, lo), (n, sp, sy) in cells.items():
        p_, y_ = sp / n, sy / n
        se = math.sqrt(max(p_ * (1 - p_), 1e-9) / n)
        out[f'{fam}|{lo:.1f}'] = {'n': n, 'stated': round(p_, 4), 'hit': round(y_, 4), 'gap_points': round(100 * (p_ - y_), 2),
                                  'se_points': round(100 * se, 2), 'ok': bool(abs(p_ - y_) <= max(0.02, 1.96 * se) and n >= 300)}
    return out


def brier(legs):
    by = defaultdict(lambda: [0, 0.0])
    for l in legs:
        if l['fam'].endswith(':yes') and np.isfinite(l['p']):
            b = by[family_of(l['fam'])]; b[0] += 1; b[1] += (l['p'] - l['y']) ** 2
    return {k: {'n': n, 'brier': round(s / n, 5)} for k, (n, s) in by.items()}


def recalibrate(legs_fit, legs_apply, draws=1000, seed=1):
    """Per family: logit map fitted on one season's yes-legs, applied to the other's; Brier change with a date bootstrap."""
    def arr(legs, fam):
        sel = [l for l in legs if l['fam'] == fam + ':yes' and np.isfinite(l['p'])]
        return np.array([l['date'] for l in sel]), np.array([l['p'] for l in sel]), np.array([l['y'] for l in sel], float)
    fams = sorted({family_of(l['fam']) for l in legs_apply})
    out = {}
    rng = np.random.default_rng(seed)
    for fam in fams:
        _, pf, yf = arr(legs_fit, fam); d_, pa, ya = arr(legs_apply, fam)
        if len(pf) < 500 or len(pa) < 500 or yf.min() == yf.max():
            continue
        a, b = fit_map(pf, yf)
        pc = apply_map(pa, a, b)
        diff = (pc - ya) ** 2 - (pa - ya) ** 2
        keys, inv = np.unique(d_, return_inverse=True)
        s = np.bincount(inv, weights=diff, minlength=len(keys)); n = np.bincount(inv, minlength=len(keys)).astype(float)
        idx = rng.integers(0, len(keys), size=(draws, len(keys)))
        boot = s[idx].sum(1) / n[idx].sum(1)
        out[fam] = {'a': round(a, 4), 'b': round(b, 4), 'brier_change': round(float(diff.mean()), 6),
                    'ci': [round(float(np.quantile(boot, .025)), 6), round(float(np.quantile(boot, .975)), 6)]}
    return out


def parlays(legs, verified, ks=(2, 3, 5), draws=20000, seed=7):
    """Random cross-game parlays of verified legs at 70% and up: stated product against the hit rate."""
    pool = [l for l in legs if l['p'] >= 0.7 and verified.get(f"{l['fam']}|{[lo for lo, hi in BINS if lo <= l['p'] < hi][0]:.1f}")]
    by_game = defaultdict(list)
    for l in pool:
        by_game[l['game_pk']].append(l)
    games = list(by_game)
    rng = np.random.default_rng(seed)
    out = {'pool_legs': len(pool), 'games': len(games)}
    for k in ks:
        if len(games) < k:
            continue
        prods, hits = [], []
        for _ in range(draws):
            gs = rng.choice(len(games), size=k, replace=False)
            ls = [by_game[games[i]][rng.integers(len(by_game[games[i]]))] for i in gs]
            prods.append(np.prod([l['p'] for l in ls])); hits.append(all(l['y'] for l in ls))
        prods, hits = np.array(prods), np.array(hits, float)
        se = math.sqrt(prods.mean() * (1 - prods.mean()) / draws)
        out[f'{k}_legs'] = {'stated': round(float(prods.mean()), 4), 'hit': round(float(hits.mean()), 4), 'se': round(se, 4)}
    return out


def daily_best(legs, verified, k=5, min_p=0.7):
    """Each date: the k legs with the highest stated chance, at most one per game, from verified cells. Selection by the
    highest stated chance is where a model's overconfidence shows (the winner's curse), so this is the honest daily test."""
    by_date = defaultdict(list)
    for l in legs:
        if l['p'] >= min_p:
            cell = f"{l['fam']}|{[lo for lo, hi in BINS if lo <= l['p'] < hi][0]:.1f}"
            if verified.get(cell):
                by_date[l['date']].append(l)
    stated, hit, all_hit, prod = [], [], [], []
    for d, ls in by_date.items():
        ls = sorted(ls, key=lambda l: -l['p'])
        pick, seen = [], set()
        for l in ls:
            if l['game_pk'] in seen:
                continue
            pick.append(l); seen.add(l['game_pk'])
            if len(pick) == k:
                break
        if len(pick) < k:
            continue
        stated += [l['p'] for l in pick]; hit += [l['y'] for l in pick]
        prod.append(np.prod([l['p'] for l in pick])); all_hit.append(all(l['y'] for l in pick))
    if not stated:
        return {}
    return {'days': len(prod), 'legs': len(stated), 'leg_stated': round(float(np.mean(stated)), 4), 'leg_hit': round(float(np.mean(hit)), 4),
            'parlay_stated': round(float(np.mean(prod)), 4), 'parlay_hit': round(float(np.mean(all_hit)), 4)}


def devig(a, b):
    def imp(o):
        o = float(o)
        return 100 / (o + 100) if o > 0 else -o / (-o + 100)
    x, y = imp(a), imp(b)
    return x / (x + y)


def market_rows(books_path=None, sbr_path=None, season=None):
    m = {}
    if sbr_path:
        with gzip.open(sbr_path, 'rt') as fh:
            for line in fh:
                r = json.loads(line)
                if season and int(r.get('season') or 0) != season:
                    continue
                d = {}
                if r.get('p_close_avg') is not None:
                    d['p_home'] = float(r['p_close_avg'])
                t = r.get('dk_total_close')
                if t and t[1] is not None and t[2] is not None:
                    d['total_line'] = float(t[0]); d['p_over'] = devig(t[1], t[2])
                m[int(r['game_pk'])] = d
    if books_path:
        with gzip.open(books_path, 'rt') as fh:
            for line in fh:
                r = json.loads(line)
                d = m.setdefault(int(r['game_pk']), {})
                if r.get('p_cons_close') is not None:
                    d['p_home'] = float(r['p_cons_close'])
                if r.get('total_cons') is not None and r.get('p_over_cons') is not None:
                    d['total_line'] = float(r['total_cons']); d['p_over'] = float(r['p_over_cons'])
    return m


def versus_market(rows, market, draws=2000, seed=3):
    """Win and over-the-closing-line chances, ours against the market's, on the same games (date bootstrap)."""
    win, tot = [], []
    for g in rows:
        mk = market.get(int(g['game_pk']))
        if not mk:
            continue
        hr, ar = int(g['home_runs']), int(g['away_runs'])
        n = g.get('n') or 1
        if 'p_home' in mk:
            win.append((g['date'], g['home_wins'] / max(n - (g.get('ties') or 0), 1), mk['p_home'], float(hr > ar)))
        if 'p_over' in mk:
            L = mk['total_line']
            if float(L).is_integer() and hr + ar == L:
                continue
            tp = np.convolve(pmf(g['home_hist']), pmf(g['away_hist']))
            tot.append((g['date'], push_free_over(tp, L), mk['p_over'], float(hr + ar > L)))
    rng = np.random.default_rng(seed)
    out = {}
    for name, data in (('win', win), ('over_close', tot)):
        if not data:
            continue
        d = np.array([x[0] for x in data]); po = np.array([x[1] for x in data]); pm = np.array([x[2] for x in data]); y = np.array([x[3] for x in data])
        diff = (po - y) ** 2 - (pm - y) ** 2
        keys, inv = np.unique(d, return_inverse=True)
        s = np.bincount(inv, weights=diff, minlength=len(keys)); nn = np.bincount(inv, minlength=len(keys)).astype(float)
        idx = rng.integers(0, len(keys), size=(draws, len(keys)))
        boot = s[idx].sum(1) / nn[idx].sum(1)
        agree = np.abs(po - pm) >= 0.05
        right = ((po > pm) == (y > 0.5))[agree]
        out[name] = {'games': int(len(y)), 'ours': round(float(((po - y) ** 2).mean()), 5), 'market': round(float(((pm - y) ** 2).mean()), 5),
                     'difference': round(float(diff.mean()), 6), 'ci': [round(float(np.quantile(boot, .025)), 6), round(float(np.quantile(boot, .975)), 6)],
                     'disagree_5pts': int(agree.sum()), 'ours_side_right_when_disagree': round(float(right.mean()), 4) if agree.any() else None,
                     'edge': edge_returns(d, po, pm, y, rng)}
    return out


HOLD = 0.045        # a typical two-way market's overround: each side costs about 2.25% more than its fair chance


def edge_returns(d, po, pm, y, rng, gaps=(0.03, 0.05, 0.08), draws=2000):
    """Where our chance and the market's differ by at least each gap: buy our side at the market's fair (no-vig) price,
    and at that price plus a typical hold. Return per dollar staked, with a date bootstrap of the fair-price return. This
    is the test of knowing something the price does not: a side that wins more often than its price says."""
    out = {}
    for gap in gaps:
        m = np.abs(po - pm) >= gap
        if m.sum() < 30:
            continue
        yes = po > pm
        cost = np.where(yes, pm, 1 - pm)[m]; win = np.where(yes, y, 1 - y)[m]; dd = d[m]
        cost_h = np.minimum(cost * (1 + HOLD / 2), 0.99)
        keys, inv = np.unique(dd, return_inverse=True)
        pr = np.bincount(inv, weights=win - cost, minlength=len(keys)); st = np.bincount(inv, weights=cost, minlength=len(keys))
        idx = rng.integers(0, len(keys), size=(draws, len(keys)))
        boot = pr[idx].sum(1) / st[idx].sum(1)
        out[f'{int(round(gap * 100))}pts'] = {'bets': int(m.sum()), 'our_side_won': round(float(win.mean()), 4), 'price_mean': round(float(cost.mean()), 4),
                                               'return_fair': round(float((win - cost).sum() / cost.sum()), 4),
                                               'return_fair_ci': [round(float(np.quantile(boot, .025)), 4), round(float(np.quantile(boot, .975)), 4)],
                                               'return_with_hold': round(float((win - cost_h).sum() / cost_h.sum()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', action='append', required=True, help='YEAR=glob of replay rows')
    ap.add_argument('--market-books'); ap.add_argument('--market-sbr')
    ap.add_argument('--out', required=True)
    ap.add_argument('--as-shown', action='store_true', help='runs and strikeouts reshaped as the page states them')
    a = ap.parse_args()
    seasons = {}
    for s in a.season:
        y, pat = s.split('=', 1)
        seasons[int(y)] = load([pat])
    legs = {y: [l for g in rows for l in game_legs(g, a.as_shown)] for y, rows in seasons.items()}
    out = {'schema': 'brl.leg-book.v1', 'experiment': 'LEG-01', 'variant': 'as shown on the page' if a.as_shown else 'simulator as run', 'seasons': {}}
    for y in seasons:
        out['seasons'][y] = {'games': len(seasons[y]), 'legs': len(legs[y]), 'bins': bin_table(legs[y]), 'brier': brier(legs[y])}
    ys = sorted(seasons)
    if len(ys) == 2:
        verified = {}
        b0, b1 = out['seasons'][ys[0]]['bins'], out['seasons'][ys[1]]['bins']
        for cell in set(b0) & set(b1):
            verified[cell] = bool(b0[cell]['ok'] and b1[cell]['ok'])
        out['verified'] = sorted(c for c, v in verified.items() if v)
        out['confident'] = sorted(c for c in out['verified'] if float(c.rsplit('|', 1)[1]) >= 0.7)
        out['recalibration'] = {f'{ys[0]}->{ys[1]}': recalibrate(legs[ys[0]], legs[ys[1]]), f'{ys[1]}->{ys[0]}': recalibrate(legs[ys[1]], legs[ys[0]])}
        for y in ys:
            out['seasons'][y]['parlays'] = parlays(legs[y], verified)
            out['seasons'][y]['daily_best'] = {k: daily_best(legs[y], verified, k=k) for k in (3, 5, 8)}
    for y in ys:
        mk = market_rows(a.market_books if y == 2026 else None, a.market_sbr if y != 2026 else None, season=y)
        if mk:
            out['seasons'][y]['versus_market'] = versus_market(seasons[y], mk)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({y: {k: v for k, v in out['seasons'][y].items() if k in ('games', 'legs', 'parlays', 'daily_best', 'versus_market')} for y in ys}, indent=1))
    if 'confident' in out:
        print('confident cells:', len(out['confident']), out['confident'][:40])


if __name__ == '__main__':
    main()
