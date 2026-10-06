from pathlib import Path
import ast,subprocess

def test_exact_box_version_and_live_status():
    source=Path(__file__).parents[1]/'brl_live/box_page.py'
    tree=ast.parse(source.read_text())
    js=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JS' for t in n.targets))
    functions='\n'.join(line for line in js.splitlines() if line.startswith(('function versions(','function latest(','function boxFor(','function gameState(')))
    prefix="const assert=require('assert');const D={forecasts:{a:{game_pk:1,saved_at:'2026-10-06T18:00:00Z'},b:{game_pk:1,saved_at:'2026-10-06T19:00:00Z'}},box_scores:{a:{game_pk:1}},actuals:{},status:{1:{state:'In Progress'}}};"
    suffix="assert.equal(boxFor(1),undefined);D.box_scores.b={game_pk:1};assert.equal(boxFor(1)[0],'b');assert.equal(gameState(1),'Live');D.status[1].state='Preview';assert.equal(gameState(1),'Pregame');D.actuals[1]={};assert.equal(gameState(1),'Final');"
    subprocess.run(['node','-e',prefix+functions+suffix],check=True)
