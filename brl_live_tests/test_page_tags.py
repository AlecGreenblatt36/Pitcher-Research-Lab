"""The plans' hitter tags (TAGS-01): the two-strike tags read his change against his own chase in the other counts, net
of the league's change, so a hitter who chases a lot everywhere is not tagged again for two strikes."""
import re
import subprocess
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'


def page_functions(*names):
    script = TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]
    out = []
    for name in names:
        m = re.search(r'\n  function ' + re.escape(name) + r'\(.*?\n  \}\n', script, re.S)
        assert m, name
        out.append(m.group(0))
    return '\n'.join(out)


def _run(suffix):
    functions = page_functions('hitterTags', 'twoStrikeShift')
    prefix = "const assert=require('assert');var TWO_SHIFT=0.065;"
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_two_strike_tags_read_the_change_not_the_level():
    # league (his side): 30% chase before two strikes, 42% with two strikes (a 12-point rise)
    lg = "{chase_rate:0.29,whiff_rate:0.24,counts:{R:{first:[1000,300,0,400,120],ahead:[1000,450,0,450,135],behind:[1000,480,0,350,105],two:[1000,520,0,500,210]}}}"
    # a big chaser who rises like everyone else: Chases a lot, no two-strike tag
    chaser = "{side:'R',chase_rate:0.36,whiff_rate:0.24,counts:{first:[300,90,0,120,45],ahead:[300,135,0,135,50],behind:[300,144,0,105,40],two:[300,160,0,150,74]}}"
    # a hitter at the league level before two strikes who jumps 25 points: Expands with two strikes
    expands = "{side:'R',chase_rate:0.30,whiff_rate:0.24,counts:{first:[300,90,0,120,36],ahead:[300,135,0,135,40],behind:[300,144,0,105,32],two:[300,170,0,150,83]}}"
    # one who barely moves: Holds his zone with two strikes
    holds = "{side:'R',chase_rate:0.29,whiff_rate:0.24,counts:{first:[300,90,0,120,36],ahead:[300,135,0,135,40],behind:[300,144,0,105,32],two:[300,150,0,150,47]}}"
    _run("const lg=" + lg + ";"
         "let t=hitterTags(" + chaser + ",lg);assert.ok(t.indexOf('Chases a lot')>=0);assert.ok(t.indexOf('Expands with two strikes')<0&&t.indexOf('Holds his zone with two strikes')<0);"
         "t=hitterTags(" + expands + ",lg);assert.ok(t.indexOf('Expands with two strikes')>=0,t);"
         "t=hitterTags(" + holds + ",lg);assert.ok(t.indexOf('Holds his zone with two strikes')>=0,t);"
         "const thin={side:'R',counts:{first:[50,10,0,20,6],ahead:[50,20,0,20,6],behind:[50,20,0,20,6],two:[50,30,0,40,30]}};"
         "assert.equal(twoStrikeShift(thin.counts,lg.counts.R),null);")


def test_line_score_adds_errors_for_real_games_only():
    functions = page_functions('lineScore', 'samplePlayed')
    prefix = ("const assert=require('assert');function esc(x){return String(x);}"
              "const f={away:{abbr:'CWS'},home:{abbr:'CLE'}};")
    suffix = ("const real={innings:{away:{'1':{R:0,H:1}},home:{'1':{R:2,H:2}}},score:{away:0,home:2},errors:{away:1,home:0},plays:[],_played:9};"
              "let h=lineScore(f,real);assert.ok(h.indexOf('>E</th>')>0&&h.indexOf('has-e')>0);"
              "const sim={innings:{away:{'1':{R:0,H:1}},home:{'1':{R:2,H:2}}},score:{away:0,home:2},plays:[],_played:9};"
              "h=lineScore(f,sim);assert.ok(h.indexOf('>E</th>')<0);")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)
