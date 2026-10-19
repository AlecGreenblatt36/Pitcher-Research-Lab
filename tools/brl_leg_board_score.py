"""The leg board, scored after the games: the legs the page listed before first pitch (verified cells at 70% and up, from
the latest forecast's box) against the final box score. Uses the page's own functions (brl_live/page/template.html,
gameLegs and LEG_TABLE) through node, so the legs are exactly the ones shown. Hitter legs are void when he had no plate
appearance, starter legs when he did not start.

    python3 tools/brl_leg_board_score.py <page data json (window.BRL)> [game_pk ...]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'brl_live/page/template.html'
FUNCS = ('dist', 'total', 'mean', 'sd', 'lgam', 'nbRuns', 'kShape', 'ownAtLeast', 'mmShare', 'lastName', 'shortNames', 'marketProb',
         'pmfOf', 'overArr', 'convArr', 'legBin', 'gameLegs')


def _script():
    return TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]


def _fn(s, name):
    m = re.search(r'\n  function ' + name + r'\(.*?\n  \}\n', s, re.S)
    one = re.search(r'\n  function ' + name + r'\([^\n]*\}\n', s)
    if one and (not m or len(one.group(0)) < len(m.group(0))):
        return one.group(0)
    return m.group(0)


def page_legs(forecast: dict, box: dict, market: dict) -> list[dict]:
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('RUN_SPREAD', 'LANCZOS', 'K_SPREAD', 'SLOT_SHARE', 'SUFFIX', 'LEG_LINES', 'LEG_TABLE', 'LEG_MIN'))
    js = 'var D = ' + json.dumps({'market': market}) + ';' + consts + ''.join(_fn(s, n) for n in FUNCS) + \
        'var out = gameLegs(' + json.dumps(forecast) + ', ' + json.dumps(box) + ').filter(function (l) { if (l.p < LEG_MIN) return false; var c = LEG_TABLE && LEG_TABLE.cells[l.fam + "|" + legBin(l.p)]; if (c && c[6]) { l.rec = c; return true; } return false; });' + \
        'console.log(JSON.stringify(out));'
    return json.loads(subprocess.run(['node'], input=js, capture_output=True, text=True, check=True).stdout)


def outcome(leg: dict, actual: dict, score: dict):
    """True, False, or None (void) for the leg's own side."""
    fam = leg['fam']; side_yes = fam.endswith(':yes'); base = fam.rsplit(':', 1)[0]
    h, a = int(score['home']), int(score['away'])
    inn = actual.get('innings') or {}
    def runs_through(n):
        return sum(int(((inn.get(sd) or {}).get(str(i)) or {}).get('R') or 0) for sd in ('away', 'home') for i in range(1, n + 1))
    line = lambda: float(re.search(r'over([\d.]+)', base).group(1))
    ev = None
    if base == 'win:home':
        ev = h > a
    elif base == 'runline:home-1.5':
        ev = h - a >= 2
    elif base == 'runline:away-1.5':
        ev = a - h >= 2
    elif base.startswith('total:'):
        ev = h + a > line()
    elif base.startswith('team_total:'):
        ev = (h if ':home:' in base else a) > line()
    elif base == 'first_inning:run_scored':
        ev = runs_through(1) > 0
    elif base.startswith('f5_total:'):
        ev = runs_through(5) > line()
    elif base.startswith('hitter:'):
        row = next((r for sd in ('away', 'home') for r in (actual.get('batting') or {}).get(sd) or [] if str(r.get('player_id')) == str(leg.get('pid'))), None)
        if not row or not row.get('PA'):
            return None
        k = base.split(':')[1]
        ev = {'hits1+': row['H'] >= 1, 'hits2+': row['H'] >= 2, 'hr': row['HR'] >= 1, 'k1+': row['K'] >= 1, 'bb1+': row['BB'] >= 1,
              'tb2+': row['H'] >= 2 or (row.get('2B', 0) + row.get('3B', 0) + row['HR']) >= 1}[k]
    elif base.startswith('starter:'):
        row = None
        for sd in ('away', 'home'):
            rows = (actual.get('pitching') or {}).get(sd) or []
            if rows and str(rows[0].get('player_id')) == str(leg.get('pid')):
                row = rows[0]
        if not row:
            return None
        stat = base.split(':')[1]
        ev = {'k': row['K'], 'outs': row['outs'], 'h': row['H'], 'bb': row['BB']}[stat] > line()
    if ev is None:
        return None
    return ev if side_yes else not ev


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    want = set(sys.argv[2:])
    latest = {}
    for fid, f in data['forecasts'].items():
        pk = str(f['game_pk'])
        if (not want or pk in want) and (pk not in latest or f['version'] > latest[pk][1]['version']):
            latest[pk] = (fid, f)
    for pk, (fid, f) in sorted(latest.items()):
        actual = (data.get('actual_boxes') or {}).get(pk); score = (data.get('actuals') or {}).get(pk)
        box = (data.get('box_scores') or {}).get(fid)
        if not actual or not score or not box:
            continue
        legs = page_legs(f, box, data.get('market') or {})
        res = [(l, outcome(l, actual, score)) for l in legs]
        graded = [(l, r) for l, r in res if r is not None]
        hit = sum(1 for _, r in graded if r)
        said = sum(l['p'] for l, _ in graded)
        print(json.dumps({'game_pk': pk, 'legs': len(legs), 'graded': len(graded), 'hit': hit, 'expected_hits': round(said, 1),
                          'misses': [(l['text'], round(l['p'], 3)) for l, r in graded if not r]}, indent=1))


if __name__ == '__main__':
    main()
