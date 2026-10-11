"""Player pages: the season line and the last games from MLB's public stats (display only). The page reads the answer
the stats probe recorded: a person's stats by type (season, gameLog), each game a split with its date, opponent, home or
away, the team's result and the box line; AVG and ERA through that game."""
import re
import subprocess
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'


def _script():
    return TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]


def _fn(script, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\(.*?\n  \}\n', script, re.S)
    assert m, name
    return m.group(0)


def _run(suffix):
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('GL_ROUND', 'TEAM_BY_ID'))
    fns = ''.join(_fn(s, n) for n in ('psParse', 'glNum', 'glAvg', 'glDec', 'glPostLine', 'glSorted', 'psStrip', 'glTable'))
    prefix = ("const assert=require('assert');function esc(x){return String(x==null?'':x);}"
              "function ip(outs){outs=Math.round(outs);return Math.floor(outs/3)+'.'+(outs%3);}"
              "function shortDate(d){return d.slice(5);}function nickname(t){var p=String(t).split(' ');return p[p.length-1];}")
    subprocess.run(['node', '-e', prefix + consts + fns + suffix], check=True)


GAME = ("function g(date,type,home,win,opp,stat,num){return {date:date,gameType:type,isHome:home,isWin:win,opponent:{id:opp,name:'Team '+opp},"
        "game:{gamePk:Math.floor(Math.random()*1e9),gameNumber:num||1},stat:stat};}")


def test_parse_picks_the_person_and_a_traded_players_whole_season():
    _run("const doc={people:[{id:7,stats:[{type:{displayName:'season'},splits:[{team:{id:1},stat:{avg:'.250'}},{team:{id:2},stat:{avg:'.300'}},{numTeams:2,stat:{avg:'.280'}}]},"
         "{type:{displayName:'gameLog'},splits:[{date:'2026-09-01',stat:{hits:1}},{stat:{hits:2}},{date:'2026-09-02',stat:null}]}]},{id:8,stats:[{type:{displayName:'gameLog'},splits:[{date:'2026-09-03',stat:{}}]}]}]};"
         "const r=psParse(doc,'7');assert.equal(r.season.avg,'.280');assert.equal(r.games.length,1);assert.equal(r.games[0].date,'2026-09-01');"
         "assert.deepEqual(psParse(null,7),{season:null,games:[]});assert.equal(psParse({people:[{id:7,stats:[{type:{displayName:'season'},splits:[{stat:{avg:'.2'}}]}]}]},7).season.avg,'.2');")


def test_outs_from_the_outs_count_or_the_innings_string():
    _run("assert.equal(glOuts({outs:20,inningsPitched:'1.0'}),20);assert.equal(glOuts({inningsPitched:'6.2'}),20);assert.equal(glOuts({inningsPitched:'7'}),21);"
         "assert.equal(glOuts({inningsPitched:'x'}),0);assert.equal(glOuts({}),0);assert.equal(glOuts(null),0);"
         "assert.equal(glDec({wins:1,blownSaves:1}),'W, BS');assert.equal(glDec({saves:1,holds:1}),'S');assert.equal(glDec({holds:1}),'H');assert.equal(glDec({}),'');")


def test_game_log_latest_first_ten_rows_round_tags_and_totals():
    _run(GAME +
         "const games=[];for(let i=1;i<=12;i++)games.push(g('2026-09-'+String(i).padStart(2,'0'),'R',i%2===0,i%3!==0,137,{atBats:4,runs:0,hits:i%2,homeRuns:0,rbi:0,baseOnBalls:0,strikeOuts:1,stolenBases:0,avg:'.250'}));"
         "games.push(g('2026-10-03','D',true,true,144,{atBats:3,runs:1,hits:2,homeRuns:1,rbi:2,baseOnBalls:1,strikeOuts:0,stolenBases:0,avg:'.667'}));"
         "const h=glTable({season:null,games:games},'hitting');"
         "assert.ok(h.indexOf('Last 10 games')>0);"
         "const rows=h.split('<tr').slice(2);assert.equal(rows.length,11);"  # ten games and the totals row
         "assert.ok(rows[0].indexOf('10-03')>0&&rows[0].indexOf('<small>DS</small>')>0&&rows[0].indexOf('vs ATL')>0,rows[0]);"
         "assert.ok(rows[1].indexOf('09-12')>0&&rows[9].indexOf('09-04')>0&&h.indexOf('09-03')<0);"
         "assert.ok(rows[1].indexOf('vs SF')>0&&rows[2].indexOf('@ SF')>0);"
         # totals: 3 + 9*4 = 39 at-bats; hits 2 + (12,10,8,6,4 even: 0; 11,9,7,5: 1) = 2+4 = 6; AVG .154
         "assert.ok(rows[10].indexOf('<td>39</td>')>0&&rows[10].indexOf('<td>6</td>')>0&&rows[10].indexOf('.154')>0,rows[10]);"
         "assert.equal(glTable({season:null,games:[]},'hitting'),'');")


def test_doubleheader_and_pitcher_rows():
    _run(GAME +
         "const st=function(o,er,w){return {outs:o,inningsPitched:'0.0',hits:3,runs:er,earnedRuns:er,baseOnBalls:1,strikeOuts:6,homeRuns:0,numberOfPitches:90,wins:w,losses:0,saves:0,holds:0,blownSaves:0,gamesStarted:1,era:'3.00'};};"
         "const games=[g('2026-09-05','R',true,true,121,st(18,2,1),1),g('2026-09-05','R',true,false,121,st(3,0,0),2),g('2026-09-01','R',false,true,158,st(21,1,0))];"
         "const h=glTable({games:games},'pitching');const rows=h.split('<tr').slice(2);"
         "assert.ok(rows[0].indexOf('<td>1.0</td>')>0&&rows[1].indexOf('<td>6.0</td>')>0,rows[0]);"  # game 2 of the doubleheader first
         "assert.ok(rows[1].indexOf('>W<')>0&&rows[2].indexOf('@ MIL')>0);"
         "assert.ok(rows[3].indexOf('<td>14.0</td>')>0&&rows[3].indexOf('1.93')>0,rows[3]);")  # 42 outs, 3 ER: 27*3/42


def test_strip_starter_reliever_and_postseason_line():
    _run(GAME +
         "let h=psStrip({season:{gamesPitched:30,gamesStarted:30,wins:12,losses:7,era:'2.72',inningsPitched:'185.1',strikeOuts:221,whip:'0.98'},games:[]},'pitching','2026');"
         "assert.ok(h.indexOf('<dt>GS</dt><dd>30</dd>')>0&&h.indexOf('<dt>IP</dt><dd>185.1</dd>')>0&&h.indexOf('<dt>W-L</dt><dd>12-7</dd>')>0,h);"
         "h=psStrip({season:{gamesPitched:60,gamesStarted:0,wins:3,losses:2,era:'2.10',saves:31,strikeOuts:70,whip:'1.01'},games:[]},'pitching','2026');"
         "assert.ok(h.indexOf('<dt>SV</dt><dd>31</dd>')>0&&h.indexOf('<dt>GS</dt>')<0);"
         "const post=[g('2026-10-03','D',true,true,144,{atBats:4,hits:2,homeRuns:1,rbi:2}),g('2026-10-05','D',false,false,144,{atBats:3,hits:0,homeRuns:0,rbi:0})];"
         "h=psStrip({season:{avg:'.274',homeRuns:22,rbi:70,runs:66,stolenBases:12,ops:'.811'},games:post},'hitting','2026');"
         "assert.ok(h.indexOf('<dt>OPS</dt><dd>.811</dd>')>0&&h.indexOf('Postseason: 2 games, 2 for 7 (.286), 1 HR, 2 RBI.')>0,h);"
         "assert.equal(psStrip({season:null,games:[]},'hitting','2026'),'');"
         "assert.equal(glPostLine([g('2026-10-03','D',true,true,144,{outs:20,earnedRuns:1,gamesStarted:1,wins:1,losses:0,strikeOuts:7})],'pitching'),'Postseason: 1 start, 1-0, 6.2 IP, 1 ER, 7 SO, 1.35 ERA.');")


def test_movement_chart_dots_rings_and_thin_arsenals():
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('TYPE_COLOR', 'TYPE_NAME'))
    prefix = "const assert=require('assert');function esc(x){return String(x==null?'':x);}function pct(p){return Math.round(p*100)+'%';}"
    js = ("const pc={arsenal_season:2026,arsenal:[{type:'FF',share:0.5,v_mov:10.1,h_arm:2.3},{type:'SL',share:0.3,v_mov:1.2,h_arm:-4.5},{type:'CU',share:0.01,v_mov:-8,h_arm:-6},{type:'CH',share:0.19,v_mov:null,h_arm:8}]};"
          "let h=movementChart(pc,{arsenal:{FF:{v_mov:9.6,h_arm:4.4},SL:{speed:86}}});"
          "assert.equal((h.match(/class=\"mv-pt\"/g)||[]).length,2);"            # a pitch under 2% or without movement gets no dot
          "assert.equal((h.match(/class=\"mv-lg\"/g)||[]).length,1);"            # a ring only where the league has movement
          "assert.ok(h.indexOf('Rings:')>0&&h.indexOf('This season (2026) only.')>0);"
          "assert.ok(h.indexOf('Four-seam 10.1 in. rise, 2.3 in. arm side; Slider 1.2 in. rise, 4.5 in. glove side')>0,h.slice(0,400));"
          "h=movementChart(pc,{});assert.equal((h.match(/class=\"mv-lg\"/g)||[]).length,0);assert.ok(h.indexOf('Rings:')<0);"
          "assert.equal(movementChart({arsenal:[{type:'FF',share:1,v_mov:10,h_arm:3}]},{}),'');assert.equal(movementChart(null,{}),'');")
    subprocess.run(['node', '-e', prefix + consts + _fn(s, 'movementChart') + js], check=True)


def test_fastball_speed_by_game_and_the_last_start_note():
    s = _script()
    consts = ''.join(re.search(r'\n  var ' + v + r' = .*?;\n', s, re.S).group(0) for v in ('TYPE_NAME', 'TYPE_SAY'))
    velo_read = re.search(r'\n  var VELO_READ = function.*?\n  \};\n', s, re.S).group(0)
    prefix = ("const assert=require('assert');function esc(x){return String(x==null?'':x);}function shortDate(d){return d.slice(5);}")
    js = ("const pc={velo_type:'FF',velo_games:[['2026-09-14',96.7,42,true],['2026-09-19',96.9,38,true],['2026-09-30',96.4,12,false],['2026-10-03',95.4,41,true]],"
          "velo_last:{type:'FF',date:'2026-10-03',mph:95.4,base:96.8,delta:-1.4,earlier_starts:25}};"
          "let h=velChart(pc);assert.equal((h.match(/class=\"vel-pt\"/g)||[]).length,3);assert.equal((h.match(/vel-pt rel/g)||[]).length,1);"
          "assert.ok(h.indexOf('Four-seam speed by game')>0&&h.indexOf('earlier starts this season')>0);"
          "assert.equal(velChart({velo_games:pc.velo_games.slice(0,2)}),'');"
          "h=velChart({velo_type:'SI',velo_games:pc.velo_games,velo_season:{season:2026,mph:96.1,n:900}});assert.ok(h.indexOf('Dashed: his 2026 average')>0&&h.indexOf('>96.1<')>0&&h.indexOf('with 8 or more of them')>0,h);"
          "h=velChart({velo_type:'SI',velo_games:pc.velo_games});assert.ok(h.indexOf('Dashed: the average of the games shown')>0);"
          "let n=velLastNote(pc);assert.ok(n.indexOf('1.4 mph below his earlier starts this season (96.8)')>0&&n.indexOf('struck out about 2 fewer')>0,n);"
          "n=velLastNote({velo_last:{type:'SI',date:'2026-10-03',mph:95.4,base:94.2,delta:1.2}});assert.ok(n.indexOf('sinker')>0&&n.indexOf('above')>0&&n.indexOf('1.3 per 100')>0,n);"
          "assert.equal(velLastNote({velo_last:{type:'FF',date:'2026-10-03',mph:95.4,base:95.9,delta:-0.5}}),'');assert.equal(velLastNote({}),'');")
    subprocess.run(['node', '-e', prefix + consts + velo_read + _fn(s, 'velChart') + _fn(s, 'velLastNote') + js], check=True)


def test_last_three_starts_line():
    s = _script()
    js = (GAME + "const st=function(o,er,k,bb,gs){return {outs:o,earnedRuns:er,strikeOuts:k,baseOnBalls:bb,gamesStarted:gs};};"
          "const games=[g('2026-09-01','R',true,true,121,st(18,2,7,1,1)),g('2026-09-06','R',true,true,121,st(21,0,9,2,1)),g('2026-09-08','R',true,true,121,st(3,1,1,0,0)),"
          "g('2026-09-12','R',true,true,121,st(15,3,5,3,1)),g('2026-10-03','D',true,true,144,st(18,1,7,2,1))];"
          "const h=psLast3({games:games});"
          "assert.ok(h.indexOf('18.0 IP, 4 ER, 21 K, 7 BB')===0,h);"        # Oct 3, Sep 12 and Sep 6: 54 outs; the relief outing left out
          "assert.ok(h.indexOf('2.00 ERA')>0,h);assert.equal(psLast3({games:[]}),'\\u2013');"
          "assert.ok(psLast3({games:games.slice(0,2)}).indexOf('Only 2 starts: 13.0 IP, 2 ER')===0,psLast3({games:games.slice(0,2)}));")
    fns = ''.join(_fn(s, n) for n in ('glNum', 'glAvg', 'glSorted', 'psLast3'))
    prefix = ("const assert=require('assert');function ip(outs){outs=Math.round(outs);return Math.floor(outs/3)+'.'+(outs%3);}")
    subprocess.run(['node', '-e', prefix + fns + js], check=True)


def test_bullpen_rates():
    s = _script()
    prefix = ("const assert=require('assert');function pctColor(p){return {bg:'B'+p,fg:'F'};}")
    js = ("assert.equal(penRate(null,'k'),'\\u2013');assert.equal(penRate({line:{pa:80,k:30,bb:5}},'k'),'\\u2013');"
          "assert.equal(penRate({line:{pa:200,k:50,bb:16}},'k'),'25.0');"
          "assert.ok(penRate({line:{pa:400,k:120,bb:20},pct:{k:91,bb:70}},'bb').indexOf('background:B70')>0);")
    subprocess.run(['node', '-e', prefix + _fn(s, 'lineRate') + _fn(s, 'penRate') + js], check=True)


def test_american_odds_from_a_win_chance():
    s = _script()
    js = ("assert.equal(amOdds(0.5),'\\u2212100');assert.equal(amOdds(0.6),'\\u2212150');assert.equal(amOdds(0.4),'+150');assert.equal(amOdds(0.25),'+300');"
          "assert.equal(amOdds(null),'\\u2013');assert.equal(amOdds(1),'\\u2013');assert.equal(amText(-113),'\\u2212113');assert.equal(amText(105),'+105');assert.equal(amText(null),'\\u2013');")
    subprocess.run(['node', '-e', "const assert=require('assert');" + _fn(s, 'amOdds') + _fn(s, 'amText') + js], check=True)


def test_hitter_season_line_text():
    s = _script()
    js = ("assert.equal(hlText({avg:'.274',homeRuns:22,ops:'.811',plateAppearances:540},'2026'),'2026: .274 AVG, 22 HR, .811 OPS in 540 PA');"
          "assert.equal(hlText({avg:'.200',homeRuns:0},'2026'),'2026: .200 AVG, 0 HR, \\u2013 OPS');assert.equal(hlText(null,'2026'),'');assert.equal(hlText({homeRuns:3},'2026'),'');")
    subprocess.run(['node', '-e', "const assert=require('assert');" + _fn(s, 'hlText') + js], check=True)


def test_line_rate_reads_expected_against():
    s = _script()
    js = ("assert.equal(lineRate({ab:200,xh:50,xtb:80},'xavg'),0.25);assert.equal(lineRate({ab:200,xh:50,xtb:80},'xslg'),0.4);"
          "assert.equal(lineRate({ab:200,h:50},'xavg'),null);")
    subprocess.run(['node', '-e', "const assert=require('assert');" + _fn(s, 'lineRate') + js], check=True)
