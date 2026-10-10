"""Old boxes leave the ledger whole only in their scores: archive_old_boxes caches each box's player and skill outcome
while it still verifies, then keeps the day archive's lean copy. Checked on the real ledger of October 10, 2026 (25
boxes archived, 14.3 MB to 7.9 MB, player scores, skill scores and every rendered page file identical); here the
mechanics."""
from brl_live import box_runner as BR
from brl_live import boxscore as BS
from brl_live import edge_metrics as EM


def _box(date, pk, saved='2026-10-01T10:00:00Z'):
    return {'date': date, 'game_pk': pk, 'saved_at': saved, 'samples': [{'score': {'away': 1, 'home': 2}, 'plays': []}, {'x': 1}],
            'teams': {'away': {'batting': [{'player_id': '1', 'distributions': {'H': [1]}, 'means': {'H': 1}}], 'pitching': []},
                      'home': {'batting': [], 'pitching': []}}, 'skill_baselines': {'big': True}}


def test_archive_respects_the_slate_window_and_the_final(monkeypatch):
    calls = []
    monkeypatch.setattr(BS, 'player_box_version', lambda ident, box, pub, actual: (calls.append(('p', ident)) or ({'forecast_id': ident, 'rows': []}, None)))
    monkeypatch.setattr(EM, 'skill_box_outcome', lambda box, pub, actual: (None, None, {'rows': [], 'ok': True}))
    L = {'date': '2026-10-10', 'box_scores': {'old': _box('2026-10-07', 7), 'yesterday': _box('2026-10-09', 9), 'open': _box('2026-10-06', 6)},
         'box_publications': {}, 'actual_boxes': {'7': {'away': 1}, '9': {'away': 1}}}
    assert BR.archive_old_boxes(L, '2026-10-10') == 1
    assert L['box_scores']['old']['archived'] and len(L['box_scores']['old']['samples']) == 1 and 'skill_baselines' not in L['box_scores']['old']
    assert 'archived' not in L['box_scores']['yesterday']           # the page still shows yesterday's whole box
    assert 'archived' not in L['box_scores']['open']                # no final box: nothing to score yet
    assert L['box_cache']['old']['player'] == {'version': {'forecast_id': 'old', 'rows': []}, 'excluded': None}
    assert BR.archive_old_boxes(L, '2026-10-10') == 0


def test_scoring_error_leaves_the_box_whole(monkeypatch):
    monkeypatch.setattr(BS, 'player_box_version', lambda *a: ({'rows': []}, None))
    def boom(*a):
        raise ValueError('Frozen baseline has a different pitcher pool')
    monkeypatch.setattr(EM, 'skill_box_outcome', boom)
    L = {'date': '2026-10-10', 'box_scores': {'old': _box('2026-10-07', 7)}, 'box_publications': {}, 'actual_boxes': {'7': {}}}
    assert BR.archive_old_boxes(L, '2026-10-10') == 0 and 'archived' not in L['box_scores']['old']


def test_scorers_read_the_cache_for_archived_boxes_only(monkeypatch):
    seen = []
    monkeypatch.setattr(BS, 'player_box_version', lambda ident, box, pub, actual: (seen.append(ident) or ({'forecast_id': ident, 'game_pk': box['game_pk'], 'saved_at': box['saved_at'], 'rows': []}, None)))
    boxes = {'a': dict(_box('2026-10-07', 7), archived=True), 'b': _box('2026-10-08', 8)}
    cache = {'a': {'player': {'version': {'forecast_id': 'a', 'game_pk': 7, 'saved_at': 'x', 'rows': []}, 'excluded': None},
                   'skill': {'published': None, 'eligible': 'No frozen pregame skill baseline; never backfill after outcome', 'result': None}}}
    out = BS.score_player_boxes(boxes, {}, {'7': {}, '8': {}}, cache)
    assert seen == ['b'] and out['n_games'] == 2
    monkeypatch.setattr(EM, 'skill_box_outcome', lambda box, pub, actual: ('Missing or changed published box', None, None))
    sk = EM.score_skill_boxes(boxes, {}, {'7': {}, '8': {}}, cache)
    assert sk['excluded'] == {'a': 'No frozen pregame skill baseline; never backfill after outcome', 'b': 'Missing or changed published box'}
