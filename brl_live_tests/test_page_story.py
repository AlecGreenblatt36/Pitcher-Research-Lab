"""KEYS-01 and WATCH-01 on the page: the What decides it chances add up to the pick, the cards and the header say the
game in plain words, and the play-by-play player steps through a simulated game from before first pitch to the final."""
import json
import re
import subprocess
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'


def _script():
    return TEMPLATE.read_text(encoding='utf-8').split('<script>')[2].split('</script>')[0]


def _fn(s, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\(.*?\n  \}\n', s, re.S); assert m, name; return m.group(0)


def _line(s, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\([^\n]*\}\n', s); assert m, name; return m.group(0)


def _var(s, name):
    m = re.search(r'\n  var ' + re.escape(name) + r' = .*?;\n', s); assert m, name; return m.group(0)


STUBS = ("function nickname(n){return n;} function winProb(f,s){return s==='home'?f.p:1-f.p;} function favorite(f){return f.p>=.5?'home':'away';}"
         "function homeProb(f){return f.p;} function ordinal(n){return n+'th';} function playParts(p){return {text:p.description,tag:'X',cls:''};}")


def _run(js, names_line=(), names_fn=()):
    s = _script()
    src = _line(s, 'esc') + _line(s, 'pct') + ''.join(_line(s, n) for n in names_line) + ''.join(_fn(s, n) for n in names_fn)
    return json.loads(subprocess.run(['node', '-e', STUBS + src + js], check=True, capture_output=True, text=True).stdout)


def test_key_chances_mix_to_the_pick():
    js = ("var f={p:0.51,away:{name:'Dodgers'},home:{name:'Brewers'}};var out=[];"
          "[{side:'home',share:0.46,win_if:0.63,win_else:0.31},{side:'away',share:0.19,win_if:0.69,win_else:0.51},{side:'away',share:0.52,win_if:1.0,win_else:0.0}].forEach(function(k){"
          "var a=keyAligned(f,k);out.push([a[0],a[1],k.share*a[0]+(1-k.share)*a[1],winProb(f,k.side),a[0]>a[1]]);});process.stdout.write(JSON.stringify(out));")
    out = _run(js, names_fn=('keyAligned',))
    for a, b, mix, target, order in out:
        assert abs(mix - target) < 0.002 and order and 0 < b < a < 1


def test_card_call_words():
    js = ("var f={p:0.51,away:{name:'Dodgers'},home:{name:'Brewers'}},g={p:0.60,away:{name:'Dodgers'},home:{name:'Brewers'}};"
          "process.stdout.write(JSON.stringify([cardCall(f,null,null),cardCall(f,{home:3,away:5},null),cardCall(g,{home:3,away:5},null),cardCall(g,{home:4,away:1},null),"
          "cardCall(g,null,{state:{home_score:1,away_score:2},home_win_probability:0.55}),cardCall(g,null,{state:{home_score:2,away_score:2},home_win_probability:0.6})]));")
    pre, flip, miss, hit, live, tied = _run(js, names_fn=('storyHeadline', 'cardCall'))
    assert 'A coin flip, leaning the Brewers' in pre
    assert 'Dodgers won 5-3. We had it a coin flip.' in flip and 'hit' not in flip and 'miss' not in flip
    assert 'card-call miss' in miss and 'We had the Brewers at 60%' in miss
    assert 'card-call hit' in hit and 'Brewers won 4-1. We had them at 60%.' in hit
    assert 'Dodgers lead 2-1. Brewers 55% to win from here.' in live and 'Tied 2-2. Brewers 60% to win from here.' in tied


def test_header_story_while_live_and_after():
    js = ("var g={p:0.6,away:{name:'Dodgers'},home:{name:'Brewers'}};"
          "process.stdout.write(JSON.stringify([heroStory(g,null,{state:{home_score:1,away_score:2},home_win_probability:0.55}),heroStory(g,null,{state:{home_score:3,away_score:1},home_win_probability:0.8}),"
          "heroStory(g,{home:2,away:6},null),heroStory(g,null,null)]));")
    behind, ahead, final, none = _run(js, names_fn=('heroStory',))
    assert 'The Dodgers lead 2-1, but the Brewers still win 55 times in 100 from here.' in behind
    assert 'The Brewers lead 3-1 and win 80 times in 100 from here.' in ahead
    assert 'Dodgers win 6-2' in final and none == ''


def test_watch_steps_from_before_first_pitch_to_the_final():
    s = _script()
    plays = [{'inning': 1, 'half': 'top', 'outs_before': 0, 'outs_after': 0, 'bases_before': [None, None, None], 'bases_after': [None, None, None],
              'away_score': 1, 'home_score': 0, 'runs_scored': 1, 'box_outcome': 'home_run', 'description': 'Betts homers.', 'contact': {'t': 'F', 'loc': 7, 'dist': 401}},
             {'inning': 1, 'half': 'top', 'outs_before': 0, 'outs_after': 1, 'bases_before': [None, None, None], 'bases_after': [None, None, None],
              'away_score': 1, 'home_score': 0, 'runs_scored': 0, 'box_outcome': 'bip_out', 'description': 'Freeman grounds out.', 'contact': {'t': 'G', 'loc': 4, 'dist': 20}}]
    js = (_var(s, 'WATCH_ANGLE') + _var(s, 'WATCH_FT') +
          "var f={p:0.51,away:{name:'Dodgers',abbr:'LAD'},home:{name:'Brewers',abbr:'MIL'}},sm={plays:" + json.dumps(plays) + "},pts=[{p:.51},{p:.5},{p:.4},{p:.42},{p:0}];"
          "process.stdout.write(JSON.stringify([watchState(f,sm,pts,0,{},{}),watchState(f,sm,pts,1,{},{}),watchState(f,sm,pts,2,{},{})]));")
    start, one, end = _run(js, names_fn=('watchField', 'watchState'))
    assert 'Before first pitch' in start and 'Brewers 51% to win' in start and '0 of 2' in start
    assert 'Top of the 1th' in one and 'Betts homers.' in one and 'Brewers 60%' not in one and 'Dodgers 60% to win' in one and 'wt-ball hit' in one
    assert 'Final' in end and 'Dodgers win 1-0.' in end and '2 of 2' in end
    # a grounder is drawn at the fielder, not at its short carry
    assert 'wt-flight gb' in end
