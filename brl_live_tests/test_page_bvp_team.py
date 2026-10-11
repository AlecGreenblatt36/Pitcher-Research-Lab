"""Two preview tables read from MLB's public stats in the browser. The career line against tonight's starter (bvpRows,
bvpTable): a hitter's meetings summed over the answer's splits, AVG and OPS from them, a lineup total, and hitters who never
faced him named as such. The team stats (tsTeams, tsTable): each club's hitting and pitching lines from the all-clubs
answer, each row ranked among the clubs in the right direction, the better of the two in bold."""
import json
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


def _line(script, name):
    m = re.search(r'\n  function ' + re.escape(name) + r'\([^\n]*\}\n', script)
    assert m, name
    return m.group(0)


def _run(js):
    s = _script()
    rows = re.search(r'\n  var TS_ROWS = .*?\]\];\n', s, re.S).group(0)
    fns = ''.join(_line(s, n) for n in ('esc', 'ordinal')) + ''.join(_fn(s, n) for n in ('bvpRows', 'bvpTable', 'tsTeams', 'tsTable'))
    out = subprocess.run(['node', '-e', rows + fns + js], check=True, capture_output=True, text=True)
    return out.stdout


def _split(pa, ab, h, hr, bb, k, tb, hbp=0, sf=0):
    return {'gameType': 'R', 'stat': {'plateAppearances': pa, 'atBats': ab, 'hits': h, 'homeRuns': hr, 'baseOnBalls': bb, 'strikeOuts': k,
                                      'totalBases': tb, 'hitByPitch': hbp, 'sacFlies': sf}}


def test_career_lines_sum_splits_and_total_the_lineup():
    doc = {'people': [
        {'id': 1, 'stats': [{'type': {'displayName': 'vsPlayerTotal'}, 'splits': [_split(10, 8, 3, 1, 2, 2, 7), _split(2, 2, 1, 0, 0, 1, 1)]}]},
        {'id': 2, 'stats': [{'type': {'displayName': 'vsPlayer'}, 'splits': [_split(50, 50, 50, 50, 0, 0, 200)]},          # per season: not summed in
                            {'type': {'displayName': 'vsPlayerTotal'}, 'splits': [_split(5, 4, 0, 0, 1, 3, 0)]}]}]}
    names = [[1, 1, 'Betts'], [2, 2, 'Freeman'], [3, 3, 'Smith']]
    html = _run('var rows=bvpRows(' + json.dumps(doc) + ');process.stdout.write(JSON.stringify([rows,bvpTable(rows,' + json.dumps(names) + ')]));')
    rows, table = json.loads(html)
    assert rows['1'] == {'pa': 12, 'ab': 10, 'h': 4, 'hr': 1, 'bb': 2, 'k': 3, 'hbp': 0, 'sf': 0, 'tb': 8}
    assert rows['2']['pa'] == 5 and '3' not in rows
    # Betts: AVG .400, OBP 6/12 = .500, SLG .800, OPS 1.300
    assert '<td>12</td><td>4</td><td>1</td><td>2</td><td>3</td><td>.400</td><td>1.300</td>' in table
    assert 'Smith</th><td colspan="7" class="bvp-none">Never faced him</td>' in table
    # lineup: 17 PA, 14 AB, 4 H, 1 HR, 3 BB, 6 K, 8 TB: AVG .286, OBP 7/17, SLG 8/14
    tot = 7 / 17 + 8 / 14
    assert 'Lineup</th><td>17</td><td>4</td><td>1</td><td>3</td><td>6</td><td>.286</td><td>' + ('%.3f' % tot).lstrip('0') + '</td>' in table


def _teams():
    hit, pit = [], []
    for i, (runs, ops, hr, so, pa, era, whip, k9, w, l) in enumerate([(800, '.760', 200, 1300, 6100, '3.50', '1.15', '9.50', 95, 67),
                                                                       (700, '.700', 150, 1400, 6000, '4.20', '1.30', '8.00', 80, 82),
                                                                       (750, '.730', 180, 1200, 6050, '3.90', '1.20', '8.80', 88, 74)]):
        hit.append({'team': {'id': 100 + i}, 'stat': {'gamesPlayed': 162, 'runs': runs, 'avg': '.250', 'ops': ops, 'homeRuns': hr, 'strikeOuts': so,
                                                    'plateAppearances': pa, 'stolenBases': 100}})
        pit.append({'team': {'id': 100 + i}, 'stat': {'gamesPlayed': 162, 'runs': 700 - 20 * i, 'era': era, 'whip': whip, 'strikeoutsPer9Inn': k9,
                                                    'walksPer9Inn': '3.00', 'wins': w, 'losses': l}})
    return {'stats': [{'group': {'displayName': 'hitting'}, 'splits': hit}, {'group': {'displayName': 'pitching'}, 'splits': pit}]}


def test_team_table_ranks_each_row_the_right_way_and_bolds_the_better():
    html = _run('var t=tsTeams(' + json.dumps(_teams()) + ');process.stdout.write(tsTable(t,["100","101"],["LAD","MIL"]));')
    row = lambda label: re.search(r'<th scope="row">' + re.escape(label) + r'</th>(.*?)</tr>', html).group(1)
    assert row('Record') == '<td class="ts-win">95-67<small>1st</small></td><td>80-82<small>3rd</small></td>'
    assert row('Runs per game') == '<td class="ts-win">4.94<small>1st</small></td><td>4.32<small>3rd</small></td>'
    # lower is better: ERA 3.50 ranks 1st, 4.20 3rd; strikeout % 1300/6100 = 21.3% ranks 2nd (the third club strikes out less)
    assert row('ERA') == '<td class="ts-win">3.50<small>1st</small></td><td>4.20<small>3rd</small></td>'
    assert row('Strikeout %') == '<td class="ts-win">21.3%<small>2nd</small></td><td>23.3%<small>3rd</small></td>'
    # runs allowed: 700/162 = 4.32 against 680/162 = 4.20; the second club allowed fewer
    assert row('Runs allowed per game') == '<td>4.32<small>3rd</small></td><td class="ts-win">4.20<small>2nd</small></td>'
    # a tie is bold for neither
    assert row('Walks per 9') == '<td>3.00<small>1st</small></td><td>3.00<small>1st</small></td>'


def test_a_group_without_its_block_name_is_read_from_its_keys_and_a_missing_team_hides_the_table():
    doc = _teams()
    for st in doc['stats']:
        st.pop('group')
    out = json.loads(_run('var t=tsTeams(' + json.dumps(doc) + ');process.stdout.write(JSON.stringify([Object.keys(t["100"]).sort(),tsTable(t,["100","999"],["A","B"])]));'))
    assert out == [['hitting', 'pitching'], '']


def test_career_lines_count_regular_season_meetings_only():
    post = dict(_split(4, 4, 2, 1, 0, 1, 5), gameType='P')
    doc = {'people': [{'id': 7, 'stats': [{'type': {'displayName': 'vsPlayerTotal'}, 'splits': [_split(6, 5, 1, 0, 1, 2, 1), post]}]}]}
    rows = json.loads(_run('process.stdout.write(JSON.stringify(bvpRows(' + json.dumps(doc) + ')));'))
    assert rows['7']['pa'] == 6 and rows['7']['hr'] == 0
