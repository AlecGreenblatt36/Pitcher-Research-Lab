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


def test_runners_to_hold_and_the_running_lines():
    functions = page_functions('stealThreat', 'runLine', 'runAgainstLine')
    prefix = "const assert=require('assert');const lg={run:{att_per_on1:0.096,sb_pct:0.77,att_per_bf:0.0232}};"
    suffix = ("assert.ok(stealThreat({season:2026,sb:23,cs:5,on1:149,sprint:28.9},lg));"
              "assert.ok(!stealThreat({season:2026,sb:2,cs:3,on1:124,sprint:26.7},lg));"
              "assert.ok(!stealThreat({season:2026,sb:5,cs:1,on1:20},lg));"
              "assert.equal(runLine({season:2026,sb:23,cs:5,on1:149,sprint:28.9},lg),'On the bases in 2026: 23 steals in 28 tries, sprint speed 28.9 ft/s (average about 27.4). Hold him on first.');"
              "assert.equal(runLine({season:2026,sb:0,cs:0,on1:60},lg),'On the bases in 2026: no steal tries in 60 times on first.');"
              "assert.equal(runAgainstLine({season:2026,sb:12,cs:2,bf:600},lg,'Misiorowski'),'The running game against Misiorowski in 2026: 12 steals in 14 tries over 600 batters (2.3 tries per 100 batters; most pitchers 2.3, 77% safe).');")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_count_mix_line_reads_the_count_and_side():
    functions = page_functions('countMixLine')
    prefix = "const assert=require('assert');const pc={usage:{R:{first:[60,30,10],ahead:[50,40,10],behind:[70,20,10],two:[44,50,6]},L:{first:[10,10,5],two:[30,20,50]}}};"
    suffix = ("assert.equal(countMixLine(pc,{side:'R'},{balls:1,strikes:2}),'With two strikes to righties: breaking ball 50%, fastball 44%, offspeed 6%.');"
              "assert.equal(countMixLine(pc,{side:'R'},{balls:0,strikes:0}),'On the first pitch to righties: fastball 60%, breaking ball 30%, offspeed 10%.');"
              "assert.equal(countMixLine(pc,{side:'L'},{balls:0,strikes:0}),'');"
              "assert.equal(countMixLine(pc,{side:'L'},{balls:2,strikes:2}),'With two strikes to lefties: offspeed 50%, fastball 30%, breaking ball 20%.');"
              "assert.equal(countMixLine(null,{side:'R'},{balls:0,strikes:0}),'');")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)
