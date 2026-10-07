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
