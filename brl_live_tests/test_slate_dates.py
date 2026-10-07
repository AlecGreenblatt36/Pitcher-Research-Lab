import copy, json
from brl_live.refreshed_page import render_page
from cloud.contracts import score_versions


def empty_ledger():
    return {'date': '2026-10-06', 'forecasts': {}, 'publications': {}, 'actuals': {}, 'status': {}}


def test_previous_day_results_do_not_become_cards_and_nothing_is_rewritten(tmp_path):
    ledger = empty_ledger()
    ledger['status'] = {'yesterday': {'date': '2026-10-05', 'state': 'final'},
                        'today': {'date': '2026-10-06', 'state': 'Scheduled', 'away': 'Today Away', 'home': 'Today Home'}}
    before = copy.deepcopy(ledger)
    path = render_page(ledger, score_versions({}, {}, {}), tmp_path)
    public = json.loads((tmp_path / 'predictions.json').read_text())
    assert public['status'] == before['status'] and public['forecasts'] == {}
    assert ledger == before


def test_no_forecast_means_no_placeholder_cards(tmp_path):
    ledger = empty_ledger()
    ledger['status'] = {'yesterday': {'date': '2026-10-05', 'state': 'final'}}
    render_page(ledger, score_versions({}, {}, {}), tmp_path)
    public = json.loads((tmp_path / 'predictions.json').read_text())
    assert public['forecasts'] == {} and public['box_scores'] == {} and public['date'] == '2026-10-06'
