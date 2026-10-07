"""The page's own reading of the official feed (live-client block in the template) matches the saved runs' reading."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from brl_live.real_game import plays_from_feed, box_from_feed

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')

RUNNER = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const a = html.indexOf('/* live-client:start */'), b = html.indexOf('/* live-client:end */');
const code = html.slice(a, b) + '\nmodule.exports = { lcPlays, lcBox, lcSituation };';
const m = { exports: {} };
new Function('module', 'URLSearchParams', 'location', code)(m, URLSearchParams, { search: '', hostname: 'example.org' });
const feed = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
process.stdout.write(JSON.stringify({ plays: m.exports.lcPlays(feed), box: m.exports.lcBox(feed.liveData.boxscore), sit: m.exports.lcSituation(feed) }));
"""


def run_page_reader(tmp_path, feed):
    (tmp_path / 'feed.json').write_text(json.dumps(feed))
    (tmp_path / 'run.js').write_text(RUNNER)
    out = subprocess.run([NODE, str(tmp_path / 'run.js'), str(ROOT / 'brl_live/page/template.html'), str(tmp_path / 'feed.json')],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_page_reads_plays_and_lines_like_the_saved_runs(tmp_path):
    from brl_live_tests.test_real_game import feed as make_feed
    feed = make_feed()
    feed['liveData']['linescore'].update({'currentInning': 1, 'inningState': 'Bottom', 'outs': 1, 'balls': 2, 'strikes': 1,
                                          'offense': {'batter': {'id': 12, 'fullName': 'Batter 12'}, 'first': {'id': 11}},
                                          'defense': {'pitcher': {'id': 60, 'fullName': 'Pitcher 60'}, 'batter': {'id': 3, 'fullName': 'Batter 3'}}})
    got = run_page_reader(tmp_path, feed)
    want = json.loads(json.dumps(plays_from_feed(feed)))
    assert got['plays'] == want
    box = box_from_feed(feed)
    assert got['box']['batting'] == box['batting'] and got['box']['pitching'] == box['pitching']
    sit = got['sit']
    assert sit['started'] and not sit['final'] and sit['state']['half'] == 'bottom' and sit['state']['outs'] == 1
    assert sit['state']['bases'] == ['11', None, None] and sit['state']['away_score'] == 2 and sit['state']['balls'] == 2
    assert sit['due_up']['home'] == 'Batter 12' and sit['on_mound']['away'] == '60' and sit['pitcher_names']['60'] == 'Pitcher 60'
    assert sit['current'] is None
    # the at-bat in progress: its pitches so far, read like a completed play's
    cur = json.loads(json.dumps(feed['liveData']['plays']['allPlays'][-1]))      # the open plate appearance (four balls so far)
    feed['liveData']['plays']['currentPlay'] = cur
    sit = run_page_reader(tmp_path, feed)['sit']
    assert sit['current']['pitches'] == [['FF', 93, 'B', -0.5 + 0.25 * k, 2.0 + 0.3 * k] for k in range(4)] and sit['current']['zone'] == [3.4, 1.6]


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_page_reads_between_innings_and_finals(tmp_path):
    from brl_live_tests.test_real_game import feed as make_feed
    feed = make_feed()
    feed['liveData']['linescore'].update({'currentInning': 3, 'inningState': 'Middle', 'outs': 3})
    sit = run_page_reader(tmp_path, feed)['sit']
    assert sit['state']['between_innings'] and sit['state']['half'] == 'bottom' and sit['state']['inning'] == 3 and sit['state']['outs'] == 0
    feed['liveData']['linescore'].update({'currentInning': 9, 'inningState': 'End'})
    feed['gameData']['status'] = {'abstractGameState': 'Final', 'detailedState': 'Final'}
    sit = run_page_reader(tmp_path, feed)['sit']
    assert sit['final'] and sit['started'] and sit['state']['inning'] == 10
    feed['liveData']['linescore'] = {'teams': {'away': {}, 'home': {}}}
    feed['gameData']['status'] = {'abstractGameState': 'Preview', 'detailedState': 'Pre-Game'}
    sit = run_page_reader(tmp_path, feed)['sit']
    assert not sit['started'] and not sit['final']
