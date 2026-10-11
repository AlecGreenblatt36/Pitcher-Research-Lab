"""The Players page's leaderboards draw one button per percentile row (PCT_ROWS), so every row needs a short label
(LD_SHORT) and its good direction (LD_DIR), the same direction the cards rank it by (tools/brl_report.py PCT_HITTER,
PCT_PITCHER). A row added to the percentile panel without them showed as a blank button."""
import json
import re
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'
REPORT = Path(__file__).parents[1] / 'tools/brl_report.py'


def _obj(script, name):
    m = re.search(r'\n  var ' + name + r' = (\{.*?\});\n', script, re.S)
    assert m, name
    text = re.sub(r'(\w+):', r'"\1":', m.group(1))
    return json.loads(text)


def test_every_percentile_row_has_a_leaderboard_label_and_direction():
    s = TEMPLATE.read_text(encoding='utf-8')
    rows = json.loads(re.sub(r'(\w+):', r'"\1":', re.search(r'\n  var PCT_ROWS = (\{.*?\n  \});\n', s, re.S).group(1)))
    short, direction = _obj(s, 'LD_SHORT'), _obj(s, 'LD_DIR')
    ranked = {}
    for kind, name in (('hitter', 'PCT_HITTER'), ('pitcher', 'PCT_PITCHER')):
        src = re.search(name + r' = (\(.*?\))\n', REPORT.read_text(encoding='utf-8')).group(1)
        ranked[kind] = dict(re.findall(r"\('(\w+)', (-?1)\)", src))
    for kind in ('hitter', 'pitcher'):
        for r in rows[kind]:
            assert short[kind].get(r[0]), (kind, r[0])
            assert str(direction[kind].get(r[0])) == ranked[kind][r[0]], (kind, r[0])
