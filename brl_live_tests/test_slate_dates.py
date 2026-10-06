import copy,json
from datetime import datetime,timezone
from brl_live.refreshed_page import render_page
from cloud.contracts import score_versions


def empty_ledger():
    return {'date':'2026-10-06','forecasts':{},'publications':{},'actuals':{},'status':{}}


def test_previous_day_unforecast_result_not_waiting_card(tmp_path):
    ledger=empty_ledger()
    ledger['status']={'yesterday':{'date':'2026-10-05','state':'final'},
                      'today':{'date':'2026-10-06','state':'Scheduled',
                               'away':'Today Away','home':'Today Home'}}
    before=copy.deepcopy(ledger)
    scores=score_versions({}, {}, {})
    path=render_page(ledger,scores,tmp_path)
    html=path.read_text()
    assert html.count('<article class="card">')==1
    assert 'Today Away' in html and 'Today Home' in html
    public=json.loads((tmp_path/'predictions.json').read_text())
    assert public['status']==before['status']
    assert ledger==before


def test_no_today_game_does_not_show_old_placeholder(tmp_path):
    ledger=empty_ledger()
    ledger['status']={'yesterday':{'date':'2026-10-05','state':'final'}}
    path=render_page(ledger,score_versions({}, {}, {}),tmp_path)
    assert '<article class="card">' not in path.read_text()
    assert 'Away team' not in path.read_text()
