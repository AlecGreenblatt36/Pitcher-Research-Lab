"""In-game win chance starts from the pregame headline and the gap fades as the game runs out (page helpers)."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')

RUNNER = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const a = html.indexOf('/* win-anchor:start */'), b = html.indexOf('/* win-anchor:end */');
const code = 'var HEADLINE = 0.5, WE = null;\nfunction homeProb(f) { return HEADLINE; }\n' + html.slice(a, b) +
  '\nmodule.exports = { winAt, leftShare, anchorGap, anchored, leagueBlend, setHeadline: function (h) { HEADLINE = h; }, setWE: function (w) { WE = w; } };';
const m = { exports: {} };
new Function('module', code)(m);
const t = { shape: [9, 2, 3, 8, 13], lead_min: -6, cells: new Array(9 * 2 * 3 * 8 * 13).fill(43.0) };
const out = { share: [m.exports.leftShare(1, 'top', 0), m.exports.leftShare(5, 'bottom', 2), m.exports.leftShare(9, 'bottom', 3), m.exports.leftShare(10, 'top', 1)] };
m.exports.setHeadline(0.48);
const f = { home_win_probability: 0.44 };
const gap = m.exports.anchorGap(f, t);
out.gap = gap;
out.start = m.exports.anchored(m.exports.winAt(t, 1, 'top', 0, [], 0), gap, 1, 'top', 0);
out.mid = m.exports.anchored(0.43, gap, 5, 'top', 0);
out.late = m.exports.anchored(0.43, gap, 9, 'bottom', 2);
out.certain = [m.exports.anchored(1, gap, 5, 'top', 0), m.exports.anchored(0, gap, 5, 'top', 0)];
out.sim_gap = m.exports.anchorGap(f, null);
out.blend_none = m.exports.leagueBlend(0.6, f, 5, 'top', 0, [], 0);               // league table not loaded: unchanged
const we = { shape: [9, 2, 3, 8, 13], lead_min: -6, cells: new Array(9 * 2 * 3 * 8 * 13).fill(53) };
we.cells[(((4 * 2 + 0) * 3 + 0) * 8 + 0) * 13 + 7] = 70;                          // top 5th, none out, empty, home up 1
m.exports.setWE(we);
out.blend_start = m.exports.leagueBlend(0.48, f, 1, 'top', 0, [], 0);             // both anchored to 0.48 at first pitch
out.blend_mid = m.exports.leagueBlend(0.6, f, 5, 'top', 0, [], 1);
out.blend_certain = [m.exports.leagueBlend(1, f, 9, 'bottom', 2, [], 3), m.exports.leagueBlend(0, f, 9, 'bottom', 2, [], -3)];
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_in_game_numbers_start_from_the_headline(tmp_path):
    (tmp_path / 'run.js').write_text(RUNNER)
    res = subprocess.run([NODE, str(tmp_path / 'run.js'), str(ROOT / 'brl_live/page/template.html')], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr[-2000:]
    got = json.loads(res.stdout)
    lg = lambda p: math.log(p / (1 - p))
    assert got['share'][0] == 1 and abs(got['share'][1] - (54 - 29) / 54) < 1e-12
    assert abs(got['share'][2] - 6 / 54) < 1e-12                                    # tied after nine: the tenth is left
    assert abs(got['share'][3] - 5 / 54) < 1e-12                                   # extra innings: what is left of the inning
    assert abs(got['gap'] - (lg(0.48) - lg(0.43))) < 1e-12 and abs(got['start'] - 0.48) < 1e-12
    assert abs(lg(got['mid']) - (lg(0.43) + got['gap'] * math.sqrt(30 / 54))) < 1e-12
    assert 0.43 < got['late'] < got['mid'] < got['start']                          # the gap fades as the game runs out
    assert got['certain'] == [1, 0] and abs(got['sim_gap'] - (lg(0.48) - lg(0.44))) < 1e-12
    # Averaged with the league's record for the same situation, anchored the same way (LIVE-04).
    assert got['blend_none'] == 0.6
    assert abs(got['blend_start'] - 0.48) < 1e-12
    league = lg(0.70) + (lg(0.48) - lg(0.53)) * math.sqrt(30 / 54)
    assert abs(lg(got['blend_mid']) - (0.5 * lg(0.6) + 0.5 * league)) < 1e-9
    assert got['blend_certain'] == [1, 0]                                           # a decided game stays decided
