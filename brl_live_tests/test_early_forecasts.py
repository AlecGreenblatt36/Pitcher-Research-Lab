"""Tomorrow's games get an early forecast only when both probable starters are announced, inside a time budget."""
from datetime import datetime, timezone


def _game(pk, state='Preview', away_sp=11, home_sp=12):
    teams = {}
    for side, sp in (('away', away_sp), ('home', home_sp)):
        teams[side] = {'team': {'id': pk * 10 + (1 if side == 'home' else 0), 'name': side.title() + ' ' + str(pk)}}
        if sp:
            teams[side]['probablePitcher'] = {'id': sp, 'fullName': 'Pitcher ' + str(sp)}
    return {'gamePk': pk, 'gameDate': '2026-10-11T00:08:00Z', 'gameType': 'L', 'status': {'abstractGameState': state}, 'teams': teams}


def test_early_forecasts_need_both_starters_and_respect_the_budget(tmp_path):
    from cloud.runner import LocalStore
    from cloud.security import key_bytes
    from brl_live.box_runner import BoxRunner

    asked = []

    class Net:
        def json(self, url):
            asked.append(url)
            assert 'date=2026-10-11' in url and 'probablePitcher' in url
            return {'dates': [{'games': [_game(1), _game(2, home_sp=None), _game(3, state='Final'), _game(4)]}]}, {'finished_at': '2026-10-10T14:00:00+00:00'}

    store = LocalStore(tmp_path, key_bytes('ab' * 32))
    clock = lambda: datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
    runner = BoxRunner(Net(), store, object(), clock=clock, run_id='t')
    seen = []
    runner.process = lambda pk, item: seen.append((pk, item['state'], item['game_type']))
    runner.early_iteration()
    assert seen == [(1, 'Preview', 'L'), (4, 'Preview', 'L')]
    rec = store.ledger['early_receipt']
    assert rec['date'] == '2026-10-11' and rec['processed'] == [1, 4] and rec['skipped'] == {'2': 'probable starters not announced'}

    # out of time: nothing is simulated, every eligible game waits for the next run
    seen.clear()
    runner.early_iteration(budget=-1)
    assert seen == [] and store.ledger['early_receipt']['skipped'] == {'1': 'left for the next run', '2': 'probable starters not announced', '4': 'left for the next run'}

    # one game's failure never stops the others
    def flaky(pk, item):
        if pk == 1:
            raise ValueError('feed hiccup')
        seen.append(pk)
    runner.process = flaky
    runner.early_iteration()
    assert seen == [4] and store.ledger['early_receipt']['skipped']['1'].startswith('ValueError')


def test_a_long_day_leaves_no_time_for_tomorrow(tmp_path, monkeypatch):
    from cloud.runner import LocalStore
    from cloud.security import key_bytes
    import brl_live.box_runner as br

    class Net:
        def json(self, url):
            if 'probablePitcher' in url:
                return {'dates': [{'games': [_game(7)]}]}, {'finished_at': '2026-10-10T14:00:00+00:00'}
            return {'dates': []}, {'finished_at': '2026-10-10T14:00:00+00:00'}

    store = LocalStore(tmp_path, key_bytes('ab' * 32))
    runner = br.BoxRunner(Net(), store, object(), clock=lambda: datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc), run_id='t')
    seen = []
    runner.process = lambda pk, item: seen.append(pk)
    monkeypatch.setattr(br, 'EARLY_RUN_LIMIT_SECONDS', -1)
    runner.iteration()
    assert seen == [] and store.ledger['early_receipt']['skipped'] == {'7': 'left for the next run'}
    monkeypatch.setattr(br, 'EARLY_RUN_LIMIT_SECONDS', 2700)
    runner.iteration()
    assert seen == [7] and store.ledger['early_receipt']['processed'] == [7]


def test_a_game_that_ended_after_midnight_is_fetched_again(tmp_path):
    """DQ-01: a game still in progress at the last run of its date is processed again on later runs until final; a game
    moved to a later date (back in Preview) is left to that date."""
    from cloud.runner import LocalStore
    from cloud.security import key_bytes
    from brl_live.box_runner import BoxRunner

    class Net:
        def json(self, url):
            assert 'date=2026-10-07' in url
            g1 = {**_game(849827, state='Final'), 'gameDate': '2026-10-08T01:38:00Z'}
            g2 = {**_game(5, state='Preview')}
            return {'dates': [{'games': [g1, g2]}]}, {'finished_at': '2026-10-08T14:00:00+00:00'}

    store = LocalStore(tmp_path, key_bytes('ab' * 32))
    store.ledger['status'] = {'849827': {'state': 'In Progress', 'date': '2026-10-07'}, '5': {'state': 'forecast_saved', 'date': '2026-10-07'},
                              '9': {'state': 'final', 'date': '2026-10-06'}, '10': {'state': 'forecast_saved', 'date': '2026-10-08'}}
    runner = BoxRunner(Net(), store, object(), clock=lambda: datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc), run_id='t')
    seen = []

    def process(pk, item):
        seen.append(pk)
        store.ledger['status'][str(pk)] = {'state': 'final', 'date': '2026-10-07'}
    runner.process = process
    runner.straggler_iteration()
    rec = store.ledger['straggler_receipt']
    assert seen == [849827] and rec['processed'] == {'849827': 'final'} and rec['skipped'] == {'5': 'moved to a later date'}


def test_postponed_is_not_a_final_score():
    from brl_live.box_runner import not_played
    assert not_played({'abstractGameState': 'Final', 'detailedState': 'Postponed', 'codedGameState': 'D'})
    assert not_played({'abstractGameState': 'Final', 'detailedState': 'Suspended: Rain'})
    assert not not_played({'abstractGameState': 'Final', 'detailedState': 'Final', 'codedGameState': 'F'})
    assert not not_played({'abstractGameState': 'Final', 'detailedState': 'Game Over', 'codedGameState': 'O'})
