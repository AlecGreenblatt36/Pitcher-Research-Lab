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
    functions = page_functions('hitterTags', 'twoStrikeShift', 'famWhiffShift', 'pullShift', 'gbShift', 'vertShift', 'horizShift', 'hzSplit', 'hzCells', 'hzTotal', 'hzLeague', 'hzLeagueFor')
    prefix = "const assert=require('assert');var TWO_SHIFT=0.065;var FAM_SHIFT={breaking:0.052,offspeed:0.067},FAM_FLOOR={breaking:100,offspeed:60};var VERT_SHIFT={miss:0.09,hard:0.10},VERT_FLOOR={miss:80,hard:50},HORZ_SHIFT={miss:0.043,hard:0.13};"
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


def test_pitch_type_miss_tags_read_against_his_own_fastball():
    functions = page_functions('hitterTags', 'twoStrikeShift', 'famWhiffShift', 'pullShift', 'gbShift', 'vertShift', 'horizShift', 'hzSplit', 'hzCells', 'hzTotal', 'hzLeague', 'hzLeagueFor')
    prefix = ("const assert=require('assert');var TWO_SHIFT=0.065;var FAM_SHIFT={breaking:0.052,offspeed:0.067},FAM_FLOOR={breaking:100,offspeed:60};var VERT_SHIFT={miss:0.09,hard:0.10},VERT_FLOOR={miss:80,hard:50},HORZ_SHIFT={miss:0.043,hard:0.13};"
              "function cells(sw,mi){return [[1,sw*2,sw,mi,0,0,0]];}"
              "function zones(fb,br,os){return {R:{all:cells(fb[0]+br[0]+os[0],fb[1]+br[1]+os[1]),fastball:cells(fb[0],fb[1]),breaking:cells(br[0],br[1]),offspeed:cells(os[0],os[1])},L:{all:cells(0,0),fastball:cells(0,0),breaking:cells(0,0),offspeed:cells(0,0)}};}"
              "const lg={chase_rate:0.29,whiff_rate:0.24,zones:{R:zones([1000,180],[600,180],[300,90])}};")
    suffix = ("let t=hitterTags({side:'R',zones:zones([300,54],[200,90],[100,30])},lg);assert.ok(t.indexOf('Misses breaking balls')>=0,t);"
              "t=hitterTags({side:'R',zones:zones([300,54],[200,40],[100,30])},lg);assert.ok(t.indexOf('Handles breaking balls')>=0,t);"
              "t=hitterTags({side:'R',zones:zones([300,54],[50,40],[100,30])},lg);assert.ok(t.indexOf('Misses breaking balls')<0&&t.indexOf('Handles breaking balls')<0,t);")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_pull_tags_read_his_pull_side():
    functions = page_functions('pullShift', 'hzLeague')     # hzLeague brings isSwitch and hzSideFor with it
    prefix = "const assert=require('assert');const lg={spray:{R:{gb:[500,300,200],air:[300,400,300]},L:{gb:[200,300,500],air:[300,400,300]}}};"
    suffix = ("assert.ok(Math.abs(pullShift({side:'R',spray:{gb:[60,25,15],air:[30,40,30]}},lg,'gb')-0.1)<1e-9);"          # 60% pulled against 50%
              "assert.ok(Math.abs(pullShift({side:'L',spray:{gb:[15,25,60],air:[30,40,30]}},lg,'gb')-0.1)<1e-9);"          # a lefty pulls to right field
              "assert.equal(pullShift({side:'R',spray:{gb:[20,10,10],air:[30,40,30]}},lg,'gb'),null);")                   # under 60 grounders
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_ground_ball_tag_reads_his_share_against_the_league():
    functions = page_functions('gbShift')
    prefix = "const assert=require('assert');const lg={spray:{R:{gb:[300,200,100],air:[150,250,150],popups:50}}};"      # 50% ground balls
    suffix = ("assert.ok(Math.abs(gbShift({side:'R',spray:{gb:[50,40,20],air:[20,30,20],popups:20}},lg)-0.05)<1e-9);"    # 110 of 200: 55%
              "assert.equal(gbShift({side:'R',spray:{gb:[10,10,10],air:[10,10,10],popups:0}},lg),null);")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_pitcher_tags_from_counts_against_the_league():
    functions = page_functions('pitcherTags')
    prefix = ("const assert=require('assert');var PTAG_RULES=[['Throws first-pitch strikes','fps',0.05,100],['Lives in the zone','zone',0.04,300],['Works off the plate','zone',-0.04,300],"
              "['Gets chases','chase',0.04,150],['Fastballs when behind','fb_behind',0.12,100],['Spins it when behind','fb_behind',-0.12,100],['Starts with spin','fb_first',-0.15,100]];"
              "const lg={ptags:{fps:0.6,zone:0.5,chase:0.3,fb_behind:0.6,fb_first:0.55}};")
    suffix = ("let t=pitcherTags({ptags:{fps:[130,200],zone:[300,500],chase:[60,200],fb_behind:[80,100],fb_first:[70,200]}},lg);"
              "assert.deepEqual(t,['Throws first-pitch strikes','Lives in the zone','Fastballs when behind','Starts with spin']);"
              "t=pitcherTags({ptags:{fps:[50,80],zone:[100,250],chase:[20,100],fb_behind:[40,100],fb_first:[100,200]}},lg);"
              "assert.deepEqual(t,['Spins it when behind']);"                          # the rest under their floors
              "assert.deepEqual(pitcherTags({},lg),[]);")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_live_count_tip_follows_the_count():
    functions = page_functions('liveCountTip', 'twoStrikeShift')
    prefix = ("const assert=require('assert');var TWO_SHIFT=0.065;function pct(v){return Math.round(100*v)+'%';}"
              "const lg={counts:{R:{first:[1000,300,0,400,120],ahead:[1000,450,0,450,135],behind:[1000,480,0,350,105],two:[1000,520,0,500,210]}}};"
              "const hc={side:'R',counts:{first:[300,45,0,120,36],ahead:[300,135,0,135,40],behind:[300,200,0,105,32],two:[300,170,0,150,83]}};")
    suffix = ("assert.ok(liveCountTip(hc,lg,{balls:0,strikes:0}).indexOf('takes the first pitch')>0);"
              "assert.ok(liveCountTip(hc,lg,{balls:1,strikes:2}).indexOf('expands more than most')>0);"
              "assert.ok(liveCountTip(hc,lg,{balls:2,strikes:0}).indexOf('aggressive')>0);"
              "assert.equal(liveCountTip(hc,lg,{balls:1,strikes:1}),'');")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_starter_lines_card_and_postseason_forms():
    # the card form has no leading space (the card puts a breakable space before it); the header's postseason line is short
    functions = page_functions('spText')
    prefix = ("const assert=require('assert');var SP_LINES={'1:2026':{R:{wins:16,losses:5,era:'1.80',strikeOuts:252},"
              "P:{gamesPlayed:2,wins:1,losses:0,era:'1.50'}},'2:2026':{R:{wins:12,losses:7,era:'2.72',strikeOuts:186},P:{gamesPlayed:0}}};")
    suffix = ("assert.equal(spText('1','2026',false,'card'),'(16-5, 1.80)');"
              "assert.equal(spText('1','2026',true,'hero'),'16-5, 1.80 ERA, 252 K\\nPostseason: 1-0, 1.50');"
              "assert.equal(spText('2','2026',true,'hero'),'12-7, 2.72 ERA, 186 K');"
              "assert.equal(spText('1','2026',true,'wl'),'16-5, 1.80 ERA');")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_short_names_tell_two_hernandezes_apart():
    functions = page_functions('lastName', 'shortNames')
    prefix = "const assert=require('assert');var SUFFIX=/^(Jr\\.?|Sr\\.?|II|III|IV)$/;"
    suffix = ("const n=shortNames([{player_id:'1',name:'Teoscar Hernández'},{player_id:'2',name:'Enrique Hernández'},{player_id:'3',name:'Mookie Betts'}]);"
              "assert.equal(n['1'],'T. Hernández');assert.equal(n['2'],'E. Hernández');assert.equal(n['3'],'Betts');")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_vertical_tags_from_the_zone_counts():
    # TAGS-07 and TAGS-08: misses and hard contact up in the zone against down, net of the league's same difference
    functions = page_functions('hitterTags', 'twoStrikeShift', 'famWhiffShift', 'pullShift', 'gbShift', 'vertShift', 'horizShift', 'hzSplit', 'hzCells', 'hzTotal', 'hzLeague', 'hzLeagueFor')
    prefix = ("const assert=require('assert');var TWO_SHIFT=0.065;var FAM_SHIFT={breaking:0.052,offspeed:0.067},FAM_FLOOR={breaking:100,offspeed:60};"
              "var VERT_SHIFT={miss:0.09,hard:0.10},VERT_FLOOR={miss:80,hard:50},HORZ_SHIFT={miss:0.043,hard:0.13};"
              # cells [zone, pitches, swings, misses, ab, h, tb, hard, measured]; zones 1-3 up, 7-9 down
              "function cells(upSw,upMiss,dnSw,dnMiss,upHard,upMeas,dnHard,dnMeas){var out=[];[1,2,3,4,5,6,7,8,9,11,12,13,14].forEach(function(z){"
              "var up=z<=3,dn=z>=7&&z<=9;out.push([z,100,up?upSw/3:dn?dnSw/3:30,up?upMiss/3:dn?dnMiss/3:6,10,3,4,up?upHard/3:dn?dnHard/3:5,up?upMeas/3:dn?dnMeas/3:12]);});return out;}"
              "const lg={zones:{R:{R:{all:cells(300,60,300,48,60,150,69,150)},L:{all:cells(300,60,300,48,60,150,69,150)}}}};")
    suffix = ("let up={side:'R',zones:{R:{all:cells(60,21,60,6,15,30,15,30)},L:{all:cells(60,9,60,6,15,30,15,30)}}};"   # misses up 25%, down 10%: +15 vs league +4
              "let t=hitterTags(up,lg);assert.ok(t.indexOf('Misses up in the zone')>=0,t);assert.ok(t.indexOf('Hits the high strike hard')<0,t);"
              "let lowHard={side:'R',zones:{R:{all:cells(60,12,60,9,9,30,24,30)},L:{all:cells(60,12,60,9,9,30,24,30)}}};"   # hard 30% up, 80% down
              "t=hitterTags(lowHard,lg);assert.ok(t.indexOf('Hits the low strike hard')>=0,t);"
              "let thin={side:'R',zones:{R:{all:cells(30,15,30,3,5,10,5,10)}}};"
              "assert.equal(vertShift(thin,lg,'miss'),null);assert.equal(vertShift(thin,lg,'hard'),null);"
              "let old={side:'R',zones:{R:{all:cells(60,21,60,6,15,30,15,30).map(function(c){return c.slice(0,7);})}}};"
              "assert.equal(vertShift(old,lg,'hard'),null);")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_plan_when_early_and_look_back():
    functions = page_functions('planWhen')
    prefix = "const assert=require('assert');var D={date:'2026-10-11'};"
    suffix = (
        # a plan for tomorrow built tonight (after 8 p.m. Eastern carries tomorrow's UTC date): early
        "let w=planWhen({date:'2026-10-12',asof:'2026-10-11',built_at:'2026-10-12T01:30:00.123456+00:00'});assert.ok(w.early&&!w.retro,JSON.stringify(w));"
        # game day, before the night's pitches are in (as of the day before): neither
        "w=planWhen({date:'2026-10-11',asof:'2026-10-10',built_at:'2026-10-11T05:35:06.165523+00:00'});assert.ok(!w.early&&!w.retro,JSON.stringify(w));"
        # the next day's regrade of a game-day plan (as of its own date): neither
        "w=planWhen({date:'2026-10-10',asof:'2026-10-10',built_at:'2026-10-11T06:35:00+00:00'});assert.ok(!w.early&&!w.retro,JSON.stringify(w));"
        # a re-run built after the season from an earlier date: a look back
        "w=planWhen({date:'2026-09-29',asof:'2026-09-01',built_at:'2026-10-10T19:00:00+00:00'});assert.ok(!w.early&&w.retro,JSON.stringify(w));")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_arsenal_changes_from_last_season():
    functions = page_functions('arsenalChanges', 'listNames')
    prefix = ("const assert=require('assert');var TYPE_SAY={FF:'four-seam fastball',SI:'sinker',FC:'cutter',SL:'slider',CH:'changeup',CU:'curveball',EP:'eephus'};")
    suffix = ("const pc={arsenal_season:2026,arsenal:[{type:'FF',share:0.64,speed:100.6},{type:'FC',share:0.14,speed:95.8},{type:'SL',share:0.11,speed:91.8},{type:'CU',share:0.10,speed:88.0},{type:'CH',share:0.01,speed:91.0}],"
              "arsenal_shares:{FF:0.64,FC:0.14,SL:0.11,CU:0.10,CH:0.01},"
              "arsenal_prev:{season:2025,pitches:1500,types:[{type:'FF',share:0.52,speed:99.1},{type:'SL',share:0.20,speed:90.5},{type:'CH',share:0.12,speed:91.5},{type:'CU',share:0.10,speed:87.5}]}};"
              # a type left off this season's table for few pitches but thrown 3.6% is not dropped
              "const few={arsenal_season:2026,arsenal:[{type:'FF',share:0.9,speed:95}],arsenal_shares:{FF:0.9,CU:0.036,SL:0.064},"
              "arsenal_prev:{season:2025,pitches:600,types:[{type:'FF',share:0.85,speed:95},{type:'CU',share:0.09,speed:80},{type:'SL',share:0.06,speed:85}]}};"
              "assert.equal(arsenalChanges(few),'');"
              # cards without the complete shares say no added or dropped
              "assert.ok(arsenalChanges(Object.assign({},pc,{arsenal_shares:null})).indexOf('dropped')<0);"
              "const t=arsenalChanges(pc);"
              "assert.ok(t.indexOf('Since 2025: ')===0,t);assert.ok(t.indexOf('added a cutter (14% of his pitches)')>0,t);"
              "assert.ok(t.indexOf('dropped his changeup (12% in 2025)')>0,t);"
              "assert.ok(t.indexOf('four-seam fastball 1.5 mph faster, thrown 12 points more often')>0,t);"
              "assert.equal(arsenalChanges({arsenal_season:null,arsenal:pc.arsenal,arsenal_prev:pc.arsenal_prev}),'');"
              "assert.equal(arsenalChanges({arsenal_season:2026,arsenal:pc.arsenal}),'');")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_hitter_changes_from_last_season():
    functions = page_functions('hitterChanges')
    s = TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]
    prefix = "const assert=require('assert');" + re.search(r'\n  var HC_RATES = .*?;\n', s, re.S).group(0)
    suffix = ("const prev={pa:600,outside:1100,chases:340,swings:1300,misses:300,k:130,bb:50,bbe:420,hard:170};"
              "const cur={pa:550,outside:1000,chases:230,swings:1200,misses:282,k:121,bb:48,bbe:400,hard:220};"
              "const t=hitterChanges({line_seasons:{season:2026,cur:cur,prev:prev}});"
              "assert.ok(t.indexOf('Since 2025: chase rate down 7.9 points (31% to 23%; changes like it held about 60% the next season)')===0,t);"
              "assert.ok(t.indexOf('hard-hit rate up 14.5 points (40% to 55%; changes like it held about 80% the next season)')>0,t);"
              # HCHANGE-01: walk-rate changes are no longer called out
              "assert.ok(hitterChanges({line_seasons:{season:2026,cur:Object.assign({},cur,{bb:110}),prev:prev}}).indexOf('walk rate')<0);"
              # a 4-point move on these samples is under three standard errors: not said
              "assert.equal(hitterChanges({line_seasons:{season:2026,cur:Object.assign({},cur,{chases:300,hard:170}),prev:prev}}),'');"
              "assert.ok(t.indexOf('miss rate')<0&&t.indexOf('strikeout rate')<0,t);"
              "assert.equal(hitterChanges({}),'');")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)


def test_switch_hitters_follow_the_side_they_take():
    functions = page_functions('isSwitch', 'hzSideFor', 'hzLeagueFor', 'hzLeague', 'hzCells', 'pullShift', 'batsWords')
    prefix = "const assert=require('assert');"
    suffix = ("const sw={side:'L',side_by_hand:{R:'L',L:'R'}},one={side:'L'};"
              "assert.ok(isSwitch(sw)&&!isSwitch(one));"
              "assert.equal(hzSideFor(sw,'R'),'L');assert.equal(hzSideFor(sw,'L'),'R');assert.equal(hzSideFor(sw,'all'),'S');assert.equal(hzSideFor(one,'L'),'L');"
              "assert.equal(batsWords(sw),'switch-hits');assert.equal(batsWords(one),'bats left');"
              # league cells: right-handed batters against lefties for his at-bats against lefties; both hands added for 'all'
              "const cell=function(n){return {all:[[1,n,0,0,0,0,0]]};};"
              "const lg={zones:{R:{R:cell(10),L:cell(20)},L:{R:cell(30),L:cell(40)}},spray:{R:{gb:[60,25,15]},L:{gb:[15,25,60]}}};"
              "assert.equal(hzLeagueFor(lg,sw,'L','all')[0][1],20);assert.equal(hzLeagueFor(lg,sw,'R','all')[0][1],30);assert.equal(hzLeagueFor(lg,sw,'all','all')[0][1],50);"
              # pull share from his own frame against the two sides' league pull shares averaged (0.6)
              "const v=pullShift({side:'L',side_by_hand:{R:'L',L:'R'},spray_pull:{gb:[70,20,10]}},lg,'gb');assert.ok(Math.abs(v-0.10)<1e-9,v);"
              "assert.equal(pullShift({side:'L',side_by_hand:{R:'L',L:'R'}},lg,'gb'),null);")
    subprocess.run(['node', '-e', prefix + functions + suffix], check=True)



def test_horizontal_tags_follow_the_side_he_takes():
    # TAGS-09 (misses away) and TAGS-10 (hard contact inside or away), each hand read with the side he takes against it
    functions = page_functions('horizShift', 'hzSplit', 'hzCells', 'hzLeague')
    prefix = "const assert=require('assert');var HORZ_SHIFT={miss:0.043,hard:0.13};"
    # cells [zone, pitches, swings, misses, ab, h, tb, hard, measured]; for a righty zones 1, 4, 7 are inside
    suffix = ("function cells(inSw,inMiss,outSw,outMiss,inHard,inMeas,outHard,outMeas,lefty){var zi=lefty?[3,6,9]:[1,4,7],zo=lefty?[1,4,7]:[3,6,9],o=[];"
              "[1,2,3,4,5,6,7,8,9,11,12,13,14].forEach(function(z){var i=zi.indexOf(z)>=0,a=zo.indexOf(z)>=0;o.push([z,100,i?inSw/3:a?outSw/3:30,i?inMiss/3:a?outMiss/3:6,10,3,4,i?inHard/3:a?outHard/3:5,i?inMeas/3:a?outMeas/3:12]);});return {all:o};}"
              "const lg={zones:{R:{R:cells(300,36,300,57,90,240,102,240,false),L:cells(300,36,300,57,90,240,102,240,false)},L:{R:cells(300,36,300,57,90,240,102,240,true),L:cells(300,36,300,57,90,240,102,240,true)}}};"
              # a righty who misses away far more than inside: 12% inside, 30% away (league 12% and 19%)
              "let v=horizShift({side:'R',zones:{R:cells(60,7.2,60,18,15,30,15,30,false),L:cells(60,7.2,60,18,15,30,15,30,false)}},lg,'miss');assert.ok(v<-0.043,v);"
              # a switch hitter whose hard contact is inside from both sides: left-handed against righties, right-handed against lefties
              "const sw={side:'L',side_by_hand:{R:'L',L:'R'},zones:{R:cells(60,7,60,11,24,40,8,40,true),L:cells(60,7,60,11,24,40,8,40,false)}};"
              "v=horizShift(sw,lg,'hard');assert.ok(v>0.13,v);"
              "function isSwitch(card){var b=card&&card.side_by_hand;return !!(b&&b.R&&b.L&&b.R!==b.L);}"
              "assert.equal(horizShift({side:'R',zones:{all:[]}},lg,'miss'),null);")
    prefix_iso = "function hzSideFor(card,hand){if(!card)return 'R';var b=card.side_by_hand;if(b&&b.R&&b.L&&b.R!==b.L)return hand==='R'||hand==='L'?b[hand]:'S';return card.side==='L'?'L':'R';}"
    subprocess.run(['node', '-e', prefix + prefix_iso + functions + suffix], check=True)
