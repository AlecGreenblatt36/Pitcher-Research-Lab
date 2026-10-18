"""LEG-01 exploratory (designed after a first look; EDGE-01 is its forward test): the moneyline where the simulator and the
market differ by 3 or 5 points, buying the simulator's side at the books' fair closing and opening prices, at DraftKings'
prices with its margin, and at the best price across books. Profit per unit staked with a date bootstrap. Metrics only.

    python3 -I tools/brl_leg_win_edge.py <folder with replay-leg01-*.jsonl.gz, market-books-2026-*.jsonl.gz, market-sbr-*.jsonl.gz>
"""
import gzip, json, glob, sys, ast
import numpy as np
S = sys.argv[1]

def imp(o):
    o = float(o); return 100 / (o + 100) if o > 0 else -o / (-o + 100)
def payout(o):        # profit per 1 staked at American odds o
    o = float(o); return o / 100 if o > 0 else 100 / -o

def rows(season):
    out = {}
    for f in sorted(glob.glob(f'{S}/replay-leg01-{season}-*.jsonl.gz')):
        for line in gzip.open(f, 'rt'):
            g = json.loads(line); n = g.get('n') or 1
            out[int(g['game_pk'])] = (g['date'], g['home_wins'] / max(n - (g.get('ties') or 0), 1), int(g['home_runs']) > int(g['away_runs']))
    return out

def boot(d, prof, stake, rng, draws=2000):
    keys, inv = np.unique(d, return_inverse=True)
    p = np.bincount(inv, weights=prof, minlength=len(keys)); s = np.bincount(inv, weights=stake, minlength=len(keys))
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    b = p[idx].sum(1) / s[idx].sum(1)
    return [round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]

def study(season, games, market):
    rng = np.random.default_rng(11)
    res = {}
    for label, pick in market.items():
        for gap in (0.03, 0.05):
            D, P, ST, W, FAV = [], [], [], [], []
            for pk, (date, po, hw) in games.items():
                m = pick.get(pk)
                if not m: continue
                pm, odds_home, odds_away = m
                if abs(po - pm) < gap: continue
                home = po > pm
                o = odds_home if home else odds_away
                if o is None or abs(float(o)) < 100: continue
                win = hw if home else not hw
                D.append(date); ST.append(1.0); P.append(payout(o) if win else -1.0); W.append(win); FAV.append((pm >= .5) == home)
            if len(P) < 30: continue
            P = np.array(P); ST = np.array(ST); FAV = np.array(FAV); W = np.array(W, float)
            res[f'{label} {int(gap*100)}pts'] = {'bets': len(P), 'won': round(float(W.mean()), 4), 'roi': round(float(P.sum() / ST.sum()), 4), 'ci': boot(np.array(D), P, ST, rng),
                                                 'on_favorites': int(FAV.sum()), 'roi_favorites': round(float(P[FAV].mean()), 4) if FAV.any() else None,
                                                 'roi_underdogs': round(float(P[~FAV].mean()), 4) if (~FAV).any() else None}
    return res

def fair_odds(p):
    return -100 * p / (1 - p) if p >= .5 else 100 * (1 - p) / p

out = {}
g26 = rows(2026)
mk = {'close fair': {}, 'close DK': {}, 'close best book': {}, 'open fair': {}, 'open DK': {}}
for line in gzip.open(glob.glob(f'{S}/market-books-2026-*.jsonl.gz')[0], 'rt'):
    r = json.loads(line); pk = int(r['game_pk'])
    if r.get('p_cons_close') is not None:
        pc = r['p_cons_close']; mk['close fair'][pk] = (pc, fair_odds(pc), fair_odds(1 - pc))
    lines = r.get('lines') or {}
    if isinstance(lines, str): lines = ast.literal_eval(lines)
    dk = lines.get('draftkings')
    if dk and r.get('p_dk_close') is not None: mk['close DK'][pk] = (r['p_dk_close'], dk[0], dk[1])
    if lines and r.get('p_cons_close') is not None:
        ok = lambda o: o is not None and abs(float(o)) >= 100
        hs = [v[0] for v in lines.values() if v and ok(v[0])]; aw = [v[1] for v in lines.values() if v and ok(v[1])]
        if hs and aw: mk['close best book'][pk] = (r['p_cons_close'], max(hs, key=payout), max(aw, key=payout))
    if r.get('p_cons_open') is not None:
        po_ = r['p_cons_open']; mk['open fair'][pk] = (po_, fair_odds(po_), fair_odds(1 - po_))
out['2026'] = study(2026, g26, mk)
g25 = rows(2025)
mk = {'close fair': {}, 'close DK': {}, 'open fair': {}, 'open DK': {}}
for line in gzip.open(sorted(glob.glob(f'{S}/market-sbr-*.jsonl.gz'))[-1], 'rt'):
    r = json.loads(line)
    if int(r.get('season') or 0) != 2025: continue
    pk = int(r['game_pk'])
    if r.get('p_close_avg') is not None: mk['close fair'][pk] = (r['p_close_avg'], fair_odds(r['p_close_avg']), fair_odds(1 - r['p_close_avg']))
    dk = r.get('dk_close')
    if isinstance(dk, str): dk = ast.literal_eval(dk)
    if dk and r.get('p_close_dk') is not None: mk['close DK'][pk] = (r['p_close_dk'], dk[0], dk[1])
    if r.get('p_open_avg') is not None: mk['open fair'][pk] = (r['p_open_avg'], fair_odds(r['p_open_avg']), fair_odds(1 - r['p_open_avg']))
    dko = r.get('dk_open')
    if isinstance(dko, str): dko = ast.literal_eval(dko)
    if dko and r.get('p_open_dk') is not None: mk['open DK'][pk] = (r['p_open_dk'], dko[0], dko[1])
out['2025'] = study(2025, g25, mk)
print(json.dumps(out, indent=1))
