"""LEG-02, plus-money mode: same-game slips a bettor can actually place, at plus-money prices, from the simulation's own
worlds (Alec's October 23 specification).

The leg universe is the markets books put in same-game parlays: starter strikeouts and outs recorded (over and under),
hits allowed and walks allowed, a hitter's hits, total bases, home run and strikeouts (over lines), the game total, team
totals, the moneyline and the run line. No first-five lines (our first-five totals failed their check), no exchange-only
"no home run" contracts. Every leg must clear an odds floor (fair price no worse than -150) and be backed by a verified
record: its own (family, chance band) cell, or its other side's, held within 2 points on two seasons re-run (LEG-01).

Slips of 2 and 3 such legs (one per player) are priced in the worlds where every leg hit, each leg scaled to its stated
chance, and kept when the fair price is +150 or better. They are ranked by how much the worlds beat multiplying (legs that
rise together), because that is where a book's parlay pricing can be off; with no public feed of parlay prices, the book's
quote is typed in and turned into the gap, the expected value and a quarter-Kelly stake. Each quote is logged with our
chance so the record can be scored after the game.

    python3 tools/brl_sgp_plus.py <predictions.json> [--game <pk>] [--top 20] [--floor -150] [--min-slip +150]
        [--quote "leg + leg = +450"] ...   [--log research/sgp/quotes.jsonl]

The honest part: our chance for a slip is a model number scaled to verified single-leg records; the slip chances themselves
are LEG-02's forward test, and nothing here is an edge until the logged quotes are scored.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from brl_leg_board_score import _script, _fn, FUNCS  # noqa: E402
from brl_sgp_scan import slip_chance, american_to_p, p_to_american, kalshi_for, kind  # noqa: E402
import brl_leg_book as lb  # noqa: E402

BOOK_FAMS = ('starter:k:', 'starter:outs:', 'starter:h:', 'starter:bb:', 'hitter:hits1+', 'hitter:hits2+', 'hitter:tb2+', 'hitter:hr', 'hitter:k1+',
             'total:', 'team_total:', 'win:home', 'runline:')
CUSHION = 0.10          # the book must pay for at least 10% more than our chance says (our slip chances are a forward test)


def all_legs(f: dict, box: dict, market: dict) -> list[dict]:
    """Every leg the page prices (both sides), no board filter."""
    import subprocess
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('RUN_SPREAD', 'LANCZOS', 'K_SPREAD', 'SLOT_SHARE', 'SUFFIX', 'LEG_LINES'))
    js = 'var D = ' + json.dumps({'market': market}) + ';' + consts + ''.join(_fn(s, n) for n in FUNCS if n not in ('legBin',)) + \
        'console.log(JSON.stringify(gameLegs(' + json.dumps(f) + ', ' + json.dumps(box) + ')));'
    return json.loads(subprocess.run(['node'], input=js, capture_output=True, text=True, check=True).stdout)


def verified_cells() -> set:
    book = json.loads((ROOT / 'research/leg01/book-as-shown.json').read_text())
    return set(book['verified'])


def cell_of(fam: str, p: float) -> str:
    """The (family, side, band) cell for a leg with chance p: the leg's own when p >= .5, else the other side's."""
    if p < 0.5:
        fam = fam[:-4] + ':no' if fam.endswith(':yes') else fam[:-3] + ':yes'; p = 1 - p
    band = [lo for lo, hi in lb.BINS if lo <= p < hi]
    return f'{fam}|{band[0]:.1f}' if band else ''


def book_text(l: dict, f: dict) -> str:
    base = re.sub(r':(yes|no)$', '', l['fam']); yes = l['fam'].endswith(':yes'); who = l.get('who') or ''
    line = (re.search(r'over([\d.]+)', base) or [None, None])[1]
    if base.startswith('hitter:'):
        k = base.split(':')[1]
        if k == 'hr':
            return f"{who} {'to hit a home run' if yes else 'no home run'}"
        what = {'hits1+': '0.5 hits', 'hits2+': '1.5 hits', 'tb2+': '1.5 total bases', 'k1+': '0.5 strikeouts'}[k]
        return f"{who} {'over' if yes else 'under'} {what}"
    if base.startswith('starter:'):
        stat = {'k': 'strikeouts', 'outs': 'outs recorded', 'h': 'hits allowed', 'bb': 'walks allowed'}[base.split(':')[1]]
        return f"{who} {'over' if yes else 'under'} {line} {stat}"
    if base.startswith('total:'):
        return f"{'Over' if yes else 'Under'} {line} runs"
    if base.startswith('team_total:'):
        side = base.split(':')[1]
        return f"{f[side]['abbr']} team total {'over' if yes else 'under'} {line}"
    if base == 'win:home':
        return f"{f['home']['abbr'] if yes else f['away']['abbr']} moneyline"
    if base.startswith('runline:'):
        side = base.split(':')[1][:4]
        fav = f[side]['abbr']; dog = f['away' if side == 'home' else 'home']['abbr']
        return f"{fav} -1.5" if yes else f"{dog} +1.5"
    return l['text']


def eligible(legs: list[dict], f: dict, floor_p: float, ver: set) -> list[dict]:
    out = []
    for l in legs:
        if not any(l['fam'].startswith(b) for b in BOOK_FAMS):
            continue
        if not (0 < l['p'] <= floor_p):
            continue
        c = cell_of(l['fam'], l['p'])
        if c not in ver:
            continue
        l = dict(l); l['book'] = book_text(l, f); l['cell'] = c
        out.append(l)
    return out


def market_group(l: dict) -> str:
    """Legs a book will not parlay together, or that say the same thing: one leg per group in a slip."""
    base = re.sub(r':(yes|no)$', '', l['fam'])
    if base == 'win:home' or base.startswith('runline:'):
        return 'result'
    if base.startswith('total:') or base.startswith('team_total:'):
        return 'runs:' + (base.split(':')[1] if base.startswith('team_total:') else 'game')
    return base.rsplit(':over', 1)[0] + '|' + str(l.get('pid') or '')


def bit_matrix(box: dict, legs: list[dict]):
    """(M: legs x worlds, 0/1 uint8 for each leg's own side; scale: stated chance over the worlds' share)."""
    from brl_sgp_scan import leg_bits
    rows, scale, keep = [], [], []
    for i, l in enumerate(legs):
        v = leg_bits(box, l)
        if v is None:
            continue
        share = float(v.mean())
        rows.append(v.astype(np.uint8)); scale.append(l['p'] / share if share > 0 else 1.0); keep.append(i)
    return np.array(rows, dtype=np.uint8), np.array(scale), keep


def slips(box: dict, legs: list[dict], lo=0.10, hi=0.40, top=60):
    """2- and 3-leg slips among the likeliest `top` eligible legs, counted in the worlds. Legs that all but imply each
    other (a run line with its moneyline) and legs of one market group never share a slip."""
    legs = sorted(legs, key=lambda l: -l['p'])[:top]
    M, scale, keep = bit_matrix(box, legs)
    legs = [legs[i] for i in keep]; n = M.shape[1]; m = len(legs)
    if m < 2:
        return []
    Mf = M.astype(np.float32)
    J2 = (Mf @ Mf.T) / n                                              # share of worlds where both hit
    raw = M.mean(1); p = np.array([l['p'] for l in legs]); groups = [market_group(l) for l in legs]
    ok = np.ones((m, m), bool)
    for i in range(m):
        for j in range(m):
            if i == j or groups[i] == groups[j] or J2[i, j] >= 0.95 * min(raw[i], raw[j]):
                ok[i, j] = False
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            if not ok[i, j]:
                continue
            pj = min(J2[i, j] * scale[i] * scale[j], p[i], p[j]); prod = p[i] * p[j]
            if lo <= pj <= hi:
                out.append({'legs': [legs[i], legs[j]], 'p': float(pj), 'raw': float(J2[i, j]), 'product': float(prod), 'lift': float(pj / prod)})
            both = (M[i] & M[j]).astype(np.float32)
            if j + 1 >= m:
                continue
            c3 = (Mf[j + 1:] @ both) / n                              # share where i, j and each later k hit
            for t, k in enumerate(range(j + 1, m)):
                if not (ok[i, k] and ok[j, k]):
                    continue
                pk_ = min(c3[t] * scale[i] * scale[j] * scale[k], p[i], p[j], p[k]); prod3 = p[i] * p[j] * p[k]
                if lo <= pk_ <= hi:
                    out.append({'legs': [legs[i], legs[j], legs[k]], 'p': float(pk_), 'raw': float(c3[t]), 'product': float(prod3), 'lift': float(pk_ / prod3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('--game'); ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--floor', type=int, default=-150, help='worst fair price allowed for a leg')
    ap.add_argument('--min-slip', type=int, default=150, help='the slip must be at least this plus-money at fair value')
    ap.add_argument('--legs', type=int, default=60, help='how many of the likeliest eligible legs enter the slips');
    ap.add_argument('--quote', action='append', default=[]); ap.add_argument('--log', default=str(ROOT / 'research/sgp/quotes.jsonl'))
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text()); ver = verified_cells()
    floor_p = american_to_p(a.floor); slip_hi = american_to_p(a.min_slip)
    latest = {}
    for fid, f in data['forecasts'].items():
        pk = str(f['game_pk'])
        if f['date'] == data['date'] and (not a.game or pk == a.game) and (pk not in latest or f['version'] > latest[pk][1]['version']):
            latest[pk] = (fid, f)
    for pk, (fid, f) in sorted(latest.items()):
        box = (data.get('box_scores') or {}).get(fid)
        if not box or not box.get('world_legs'):
            print(f"{f['away']['abbr']} at {f['home']['abbr']}: no world bits in this box"); continue
        legs = eligible(all_legs(f, box, data.get('market') or {}), f, floor_p, ver)
        for l in legs:
            l['kalshi'] = kalshi_for(data, pk, l, f)
        print(f"\n{f['away']['abbr']} at {f['home']['abbr']} (v{f['version']}, lineups {f['lineup_status']['away']}/{f['lineup_status']['home']}): "
              f"{len(legs)} legs at {a.floor} or better with a verified record")
        print(f"{'leg (book wording)':44s} {'ours':>6s} {'fair':>6s} {'min odds to bet':>15s}  Kalshi ask")
        for l in sorted(legs, key=lambda l: -l['p'])[:a.top]:
            need = p_to_american(l['p'] * (1 - CUSHION))
            kq = f"{l['kalshi']['ask']*100:.0f}c" if l.get('kalshi') else ''
            print(f"{l['book']:44.44s} {l['p']*100:5.1f}% {p_to_american(l['p']):>6s} {need:>15s}  {kq}")
        found = slips(box, legs, hi=slip_hi, top=a.legs)
        by_lift = sorted(found, key=lambda s: (-s['lift'], -s['p'])); by_p = sorted(found, key=lambda s: -s['p'])
        for title, rows_ in ((f"legs that rise together (the worlds beat multiplying most; a book that multiplies with a thin correlation charge can be off here)", by_lift),
                             ("likeliest plus-money slips", by_p)):
            print(f"\n{len(found)} slips of 2 or 3 legs at {a.min_slip:+d} or better. {title}:")
            print(f"{'slip':88s} {'worlds':>7s} {'mult.':>6s} {'fair':>6s} {'min odds':>8s}")
            for sl in rows_[:a.top]:
                text = ' + '.join(l['book'] for l in sl['legs'])
                print(f"{text:88.88s} {sl['p']*100:6.1f}% {sl['product']*100:5.1f}% {p_to_american(sl['p']):>6s} {p_to_american(sl['p'] * (1 - CUSHION)):>8s}")
        # the scan's top slips, kept so they can be scored after the game (LEG-02's forward test, plus-money side)
        out = Path(a.log).parent / f"plus_{pk}_v{f['version']}.json"; out.parent.mkdir(parents=True, exist_ok=True)
        keep = lambda rows_: [{'legs': [{'fam': l['fam'], 'pid': l.get('pid') or '', 'text': l['text'], 'book': l['book'], 'p': round(l['p'], 4)} for l in sl['legs']],
                               'p': round(sl['p'], 4), 'product': round(sl['product'], 4), 'raw': round(sl['raw'], 4)} for sl in rows_[:30]]
        out.write_text(json.dumps({'game_pk': int(pk), 'date': f['date'], 'version': f['version'], 'lineups': f['lineup_status'], 'saved_at': datetime.now(timezone.utc).isoformat(),
                                   'floor': a.floor, 'min_slip': a.min_slip, 'cushion': CUSHION, 'by_lift': keep(by_lift), 'by_chance': keep(by_p)}, indent=1))
        now = datetime.now(timezone.utc).isoformat()
        for q in a.quote:
            m = re.match(r'^(.*)=\s*([+-]?\d+)\s*$', q)
            if not m:
                print('quote format: "leg + leg = +450"'); continue
            want = [t.strip().lower() for t in m.group(1).split('+')]; odds = float(m.group(2)); combo = []
            for w in want:
                hit = [l for l in legs if l['book'].lower() == w or l['text'].lower() == w]
                if not hit:
                    print(f'  no eligible leg named "{w}"'); combo = None; break
                combo.append(hit[0])
            if not combo:
                continue
            sc = slip_chance(box, combo)
            if not sc:
                continue
            p, raw, prod = sc; pm = american_to_p(odds); b = (1 - pm) / pm
            ev = p * b - (1 - p); kelly = (p * (b + 1) - 1) / b; need = p_to_american(p * (1 - CUSHION))
            verdict = 'clears the bar' if pm <= p * (1 - CUSHION) else ('fair by our number but inside the cushion' if pm <= p else 'worse than our number')
            print(f"  quote {' + '.join(l['book'] for l in combo)} at {int(odds):+d}: ours {p*100:.1f}% (fair {p_to_american(p)}), the book's {pm*100:.1f}%, "
                  f"gap {100*(p-pm):+.1f} points, EV {ev*100:+.1f} cents a dollar, quarter Kelly {max(kelly,0)*25:.1f}% of bankroll; needs {need} or better: {verdict}")
            rec = Path(a.log); rec.parent.mkdir(parents=True, exist_ok=True)
            with rec.open('a') as fh:
                fh.write(json.dumps({'at': now, 'game_pk': int(pk), 'date': f['date'], 'version': f['version'], 'legs': [l['text'] for l in combo],
                                     'book_text': [l['book'] for l in combo], 'ours': round(p, 4), 'product': round(prod, 4), 'quote': odds, 'book_p': round(pm, 4),
                                     'ev': round(ev, 4), 'verdict': verdict}) + '\n')


if __name__ == '__main__':
    main()
