"""LEG-02: the same-game slip scanner. For each of today's games whose box carries the per-world leg bits
(brl_live.boxscore.WorldLog), the chance of 2- and 3-leg slips counted in the simulated worlds where every leg hit, next
to the multiplied chance, with the legs' two-season records. Slips come from the board's legs (verified cells at 70% and
up) and from three templates: a starter going deep with the other side's bottom of the order quiet, an early exit with the
other side scoring, and strikeouts stacked on both sides of the matchup. A book's quote for a slip turns into the edge and
a quarter-Kelly stake; without a quote the fair price is shown, because no public feed lists same-game parlay prices.

    python3 tools/brl_sgp_scan.py <predictions.json> [--game <game_pk>] [--top 20]
        [--quote "Leg text + Leg text = +450"] ...   (American odds a book offers for that slip)

Each leg's share of worlds is scaled to the chance the page states for it (the page spreads totals and strikeouts and
thins a hitter's slot to his own share), so the worlds give the dependence and the stated chances the level. The joint
chances are LEG-02's, registered for a forward check; the legs' own chances are LEG-01's, verified on two seasons.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from brl_leg_board_score import page_legs  # noqa: E402


def unpack_bits(text: str, n: int):
    """The yes bits of one leg back from the payload's base64 (brl_live.boxscore.unpack_bits, without the runtime import)."""
    import base64
    return np.unpackbits(np.frombuffer(base64.b64decode(text), dtype=np.uint8))[:n].astype(bool)


def american_to_p(odds: float) -> float:
    o = float(odds)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def p_to_american(p: float) -> str:
    if p <= 0 or p >= 1:
        return '-'
    v = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    return ('+' if v > 0 else '-') + str(int(round(abs(v))))


def leg_bits(box: dict, leg: dict):
    wl = box.get('world_legs')
    if not wl:
        return None
    base = re.sub(r':(yes|no)$', '', leg['fam']); pid = str(leg.get('pid') or '')
    for (fam, lpid, hits), text in zip(wl['legs'], wl['bits']):
        if fam == base and str(lpid or '') == pid:
            v = unpack_bits(text, int(wl['n']))
            return v if leg['fam'].endswith(':yes') else ~v
    return None


def slip_chance(box: dict, legs: list[dict]):
    """(chance scaled to the stated legs, the worlds' raw share, the product of the stated chances) or None."""
    rows = []
    for l in legs:
        v = leg_bits(box, l)
        if v is None:
            return None
        rows.append(v)
    both = np.logical_and.reduce(rows); raw = float(both.mean()); product = float(np.prod([l['p'] for l in legs]))
    scale = 1.0
    for l, v in zip(legs, rows):
        share = float(v.mean())
        scale *= l['p'] / share if share > 0 else 1.0
    return min(raw * scale, min(l['p'] for l in legs)), raw, product


def kind(fam: str) -> str:
    return re.sub(r'over[\d.]+', '', fam).replace('-1.5', '')


def templates(legs: list[dict], f: dict, box: dict) -> list[tuple[str, list[dict]]]:
    """Named slips from the three correlation templates, built from the board's legs only (verified cells)."""
    out = []
    by = lambda pred: [l for l in legs if pred(l)]
    spots = {}
    for side in ('away', 'home'):
        for b in (box.get('teams') or {}).get(side, {}).get('batting') or []:
            spots[str(b['player_id'])] = (side, int(b.get('spot') or 0))
    for side, opp in (('home', 'away'), ('away', 'home')):
        sp = str(((box.get('starters') or {}).get(side) or {}).get('player_id') or '')
        deep = [l for l in by(lambda l: l['fam'].startswith('starter:outs:') and l['fam'].endswith(':yes') and str(l.get('pid')) == sp)]
        deep.sort(key=lambda l: -float(re.search(r'over([\d.]+)', l['fam']).group(1)))
        quiet = [l for l in by(lambda l: l['fam'] in ('hitter:tb2+:no', 'hitter:hits2+:no') and spots.get(str(l.get('pid')), ('', 0))[0] == opp and spots[str(l['pid'])][1] >= 7)]
        quiet.sort(key=lambda l: -l['p'])
        if deep and quiet:
            out.append(('A: ' + deep[0]['who'] + ' goes deep, bottom of the order quiet', [deep[0]] + quiet[:2]))
        short = [l for l in by(lambda l: l['fam'].startswith('starter:outs:') and l['fam'].endswith(':no') and str(l.get('pid')) == sp)]
        short.sort(key=lambda l: float(re.search(r'over([\d.]+)', l['fam']).group(1)))
        score = [l for l in by(lambda l: l['fam'].startswith('team_total:' + opp + ':') and l['fam'].endswith(':yes'))]
        score.sort(key=lambda l: -float(re.search(r'over([\d.]+)', l['fam']).group(1)))
        if short and score:
            out.append(('B: ' + short[0]['who'] + ' out early, ' + f[opp]['abbr'] + ' scores', [short[0], score[0]]))
        ks = [l for l in by(lambda l: l['fam'].startswith('starter:k:') and l['fam'].endswith(':yes') and str(l.get('pid')) == sp)]
        ks.sort(key=lambda l: -float(re.search(r'over([\d.]+)', l['fam']).group(1)))
        whiff = [l for l in by(lambda l: l['fam'] == 'hitter:k1+:yes' and spots.get(str(l.get('pid')), ('', 0))[0] == opp)]
        whiff.sort(key=lambda l: -l['p'])
        if ks and whiff:
            out.append(('C: ' + ks[0]['who'] + ' strikeouts stacked', [ks[0]] + whiff[:2]))
    return out


def lead_legs(legs: list[dict], max_n: int = 24) -> list[dict]:
    """The board's leading legs: one per player (or team-level kind), the likeliest line each."""
    seen, out = set(), []
    for l in sorted(legs, key=lambda l: -l['p']):
        who = (l.get('pid') or kind(l['fam']))
        if who in seen:
            continue
        seen.add(who); out.append(l)
        if len(out) >= max_n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('--game'); ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--quote', action='append', default=[], help='"leg text + leg text = +450"')
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text())
    latest = {}
    for fid, f in data['forecasts'].items():
        pk = str(f['game_pk'])
        if f['date'] == data['date'] and (not a.game or pk == a.game) and (pk not in latest or f['version'] > latest[pk][1]['version']):
            latest[pk] = (fid, f)
    for pk, (fid, f) in sorted(latest.items()):
        box = (data.get('box_scores') or {}).get(fid)
        if not box or not box.get('world_legs'):
            print(f"{f['away']['abbr']} at {f['home']['abbr']}: no world bits in this box (built before LEG-02)"); continue
        legs = page_legs(f, box, data.get('market') or {})
        for l in legs:
            l['text'] = l['text']
        print(f"\n{f['away']['abbr']} at {f['home']['abbr']}, forecast v{f['version']} ({box['world_legs']['n']:,} worlds; lineups {f['lineup_status']['away']}/{f['lineup_status']['home']})")
        rows = []
        for name, combo in templates(legs, f, box):
            sc = slip_chance(box, combo)
            if sc:
                rows.append((name, combo, sc))
        lead = lead_legs(legs)
        for k in (2, 3):
            for combo in itertools.combinations(lead, k):
                sc = slip_chance(box, list(combo))
                if sc:
                    rows.append(('', list(combo), sc))
        rows.sort(key=lambda r: -r[2][0])
        shown = [r for r in rows if r[0]][:6] + [r for r in rows if not r[0]][:a.top]
        print(f"{'slip':70s} {'worlds':>7s} {'multiplied':>10s} {'fair odds':>9s}")
        for name, combo, (p, raw, prod) in shown:
            text = ' + '.join(l['text'] for l in combo)
            print(f"{(name + ': ' if name else '') + text:70.70s} {p*100:6.1f}% {prod*100:9.1f}% {p_to_american(p):>9s}")
        for q in a.quote:
            m = re.match(r'^(.*)=\s*([+-]?\d+)\s*$', q)
            if not m:
                print('quote format: "leg + leg = +450"'); continue
            want = [t.strip().lower() for t in m.group(1).split('+')]; odds = float(m.group(2))
            combo = []
            for w in want:
                hit = [l for l in legs if l['text'].lower() == w]
                if not hit:
                    print(f'  no board leg named "{w}"'); combo = None; break
                combo.append(hit[0])
            if not combo:
                continue
            sc = slip_chance(box, combo)
            if not sc:
                continue
            p, raw, prod = sc; pm = american_to_p(odds); b = (1 - pm) / pm
            ev = p * b - (1 - p); kelly = (p * (b + 1) - 1) / b
            print(f"  quote {m.group(1).strip()} at {int(odds):+d}: ours {p*100:.1f}%, the book's {pm*100:.1f}%, gap {100*(p-pm):+.1f} points, "
                  f"expected value {ev*100:+.1f} cents a dollar, quarter Kelly {max(kelly,0)*25:.1f}% of bankroll")


if __name__ == '__main__':
    main()
