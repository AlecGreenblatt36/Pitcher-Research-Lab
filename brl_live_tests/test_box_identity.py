"""The page must show a box score only for the latest forecast version of a game."""
from pathlib import Path
import re,subprocess

TEMPLATE=Path(__file__).parents[1]/'brl_live/page/template.html'

def page_functions(*names):
    script=TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]
    out=[]
    for name in names:
        m=re.search(r'\n  function '+re.escape(name)+r'\(.*?\n  \}\n',script,re.S)
        assert m,name
        out.append(m.group(0))
    return '\n'.join(out)

def test_exact_box_version_and_live_status():
    functions=page_functions('withId','latestForecasts','versionsFor','boxFor','gameState')
    prefix="const assert=require('assert');const D={forecasts:{a:{game_pk:1,saved_at:'2026-10-06T18:00:00Z'},b:{game_pk:1,saved_at:'2026-10-06T19:00:00Z'}},box_scores:{a:{game_pk:1}},actuals:{},status:{1:{state:'In Progress'}}};"
    suffix=("assert.equal(boxFor(1),undefined);D.box_scores.b={game_pk:1};assert.equal(boxFor(1).game_pk,1);assert.equal(boxFor(2),undefined);"
            "assert.equal(gameState(1),'live');D.status[1].state='Preview';assert.equal(gameState(1),'pregame');D.actuals[1]={};assert.equal(gameState(1),'final');"
            "assert.equal(versionsFor(1).length,2);")
    subprocess.run(['node','-e',prefix+functions+suffix],check=True)

def test_latest_version_wins_over_save_time():
    functions=page_functions('withId','latestForecasts')
    prefix="const assert=require('assert');const D={forecasts:{a:{game_pk:1,version:2,saved_at:'2026-10-06T18:00:00Z'},b:{game_pk:1,version:1,saved_at:'2026-10-06T19:00:00Z'}}};"
    suffix="assert.equal(latestForecasts()[1]._id,'a');"
    subprocess.run(['node','-e',prefix+functions+suffix],check=True)

def test_roles_saved_with_the_forecast_drive_the_version_buttons():
    functions=page_functions('variantsFor')
    prefix=("const assert=require('assert');var ROLE_LABELS=[['projected','proj','Projected game'],['high','high','High scoring'],['low','low','Low scoring'],['upset','upset','Upset']];"
            "function favorite(){return 'home';}"
            "const box={sample_roles:{projected:40,high:7,low:900,upset:3},sample_indices:[40,7,900,3,11],samples:[1,2,3,4,5].map(function(i){return {score:{away:0,home:i},plays:[],pitching:{away:[],home:[]}};})};")
    suffix="const v=variantsFor(box,{});assert.deepEqual(v.map(function(x){return x.key+':'+x.i;}),['proj:0','high:1','low:2','upset:3']);"
    subprocess.run(['node','-e',prefix+functions+suffix],check=True)

def page_vars(*names):
    script=TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]
    out=[]
    for name in names:
        m=re.search(r'\n  var '+re.escape(name)+r' = .*?;\n',script,re.S)
        assert m,name
        out.append(m.group(0))
    return '\n'.join(out)

def test_play_wording_uses_drawn_contact():
    code=page_vars('HOMER','SPOT','FIELDER','SHAPE','OUT_VERB','DP_ROUTE')+page_functions('contactWords','playParts','listNames')
    prefix="const assert=require('assert');function lastName(n){return n.split(' ').pop();}"
    cases=[({'box_outcome':'single','contact':{'t':'L','loc':8}},"Betts singles on a line drive to center field."),
           ({'box_outcome':'single','contact':{'t':'G','loc':6}},"Betts singles on a ground ball to shortstop."),
           ({'box_outcome':'single'},"Betts singles."),
           ({'box_outcome':'double','contact':{'t':'F','loc':9}},"Betts doubles on a fly ball to right field."),
           ({'box_outcome':'home_run','runs_scored':2,'contact':{'t':'F','loc':7,'dist':412}},"Betts hits a two-run homer to left field (412 ft)."),
           ({'box_outcome':'bip_out','description':'x put the ball in play for an out.','contact':{'t':'G','loc':4}},"Betts grounds out to second base."),
           ({'box_outcome':'bip_out','description':'x put the ball in play for an out.','contact':{'t':'F','loc':8}},"Betts flies out to center field."),
           ({'box_outcome':'bip_out','description':'x put the ball in play for an out.','contact':{'t':'F','loc':6}},"Betts pops out to shortstop."),
           ({'box_outcome':'bip_out','description':'x put the ball in play for an out.','contact':{'t':'L','loc':1}},"Betts lines out to the pitcher."),
           ({'box_outcome':'bip_out','description':'x put the ball in play for an out.','contact':{'t':None,'loc':5}},"Betts is out to third base."),
           ({'box_outcome':'bip_out','description':'x put the ball in play for an out.'},"Betts is out."),
           ({'box_outcome':'bip_out','description':'x grounded into a double play.','outs_before':0,'outs_after':2,'contact':{'t':'G','loc':6}},"Betts grounds into a double play, shortstop to second to first."),
           ({'box_outcome':'bip_out','description':'x drove in a run on a sacrifice fly.','contact':{'t':'F','loc':9},'scoring_players':['7']},"Betts hits a sacrifice fly to right field, Smith scores."),
           ({'box_outcome':'other_reach','description':'x reached on an error or other play.','contact':{'t':'G','loc':6}},"Betts reaches on an error by the shortstop."),
           ({'box_outcome':'other_reach','description':"x reached on a fielder's choice.",'contact':{'t':'G','loc':4}},"Betts reaches on a fielder's choice to second base."),
           ({'box_outcome':'strikeout','contact':None},"Betts strikes out.")]
    import json
    checks=''.join("assert.equal(playParts(Object.assign({batter_id:'1',batter_name:'Mookie Betts'},%s),{'7':'Will Smith'},{}).text,%s);"%(json.dumps(p),json.dumps(t)) for p,t in cases)
    subprocess.run(['node','-e',prefix+code+checks],check=True)

def test_series_and_record_context_lines():
    code=page_vars('SERIES','SERIES_SHORT')+page_functions('contextFor','isPostseason','contextLine','weatherLine','recordText')
    prefix="const assert=require('assert');const D={context:{'1':{venue:'Progressive Field',series_description:'Division Series',series_game_number:3,games_in_series:5,records:{away:{wins:2,losses:0},home:{wins:0,losses:2}},weather:{temp:'71',condition:'Clear',wind:'8 mph, Out To CF'}},'2':{venue:'Yankee Stadium',records:{away:{wins:88,losses:70},home:{wins:95,losses:63}}}}};"
    checks=("const f={game_pk:1,game_type:'D',away:{abbr:'CLE'},home:{abbr:'CWS'}};"
            "assert.equal(contextLine(f,contextFor(1),true),'Division Series Game 3 \\u00b7 CLE leads 2-0 \\u00b7 Progressive Field');"
            "assert.equal(contextLine(f,contextFor(1),false),'Division Series Game 3 \\u00b7 CLE leads 2-0');"
            "assert.equal(weatherLine(contextFor(1)),'71\\u00b0F, Clear, wind 8 mph, Out To CF');"
            "assert.equal(recordText(f,contextFor(1),'away'),'');"
            "const g={game_pk:2,game_type:'R',away:{abbr:'TB'},home:{abbr:'NYY'}};"
            "assert.equal(contextLine(g,contextFor(2),true),'Yankee Stadium');assert.equal(recordText(g,contextFor(2),'home'),'95-63');"
            "assert.equal(contextLine(g,null,true),'');assert.equal(weatherLine(null),'');"
            "const t={game_pk:1,game_type:'L',away:{abbr:'A'},home:{abbr:'H'}};D.context['1'].records={away:{wins:1,losses:1},home:{wins:1,losses:1}};D.context['1'].games_in_series=7;"
            "assert.equal(contextLine(t,contextFor(1),false),'LCS Game 3 \\u00b7 Series tied 1-1');")
    subprocess.run(['node','-e',prefix+code+checks],check=True)
