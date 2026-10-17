"""HIST-01: the season backfill's pieces without the network: spans, the postseason relabel and the receipt, and the
history builder on a small Savant-shaped export (the same builder as the 2023-2026 history)."""
import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location('brl_history_backfill', Path(__file__).resolve().parents[1] / 'tools' / 'brl_history_backfill.py')
hb = importlib.util.module_from_spec(spec); spec.loader.exec_module(hb)


def test_spans_cover_the_season_in_three_day_chunks():
    sp = hb.chunks(date(2019, 3, 28), date(2019, 4, 5))
    assert sp[0] == (date(2019, 3, 28), date(2019, 3, 30)) and sp[-1] == (date(2019, 4, 3), date(2019, 4, 5)) and len(sp) == 3
    assert hb.history_path(2019) == 'private/history/pa-2019.enc' and hb.history_purpose(2019) == 'pa-history-season-2019'


def _pitches(game_pk, game_type, day):
    rows = []
    # two plate appearances: a single (two pitches) and a strikeout (one pitch)
    for ab, pitches, event in ((1, 2, 'single'), (2, 1, 'strikeout')):
        for n in range(1, pitches + 1):
            rows.append({'game_date': day, 'game_year': int(day[:4]), 'game_pk': game_pk, 'at_bat_number': ab, 'pitch_number': n,
                         'batter': 100 + ab, 'pitcher': 900, 'events': event if n == pitches else None, 'stand': 'R', 'p_throws': 'L',
                         'home_team': 'CLE', 'away_team': 'NYY', 'inning': 1, 'inning_topbot': 'Top', 'outs_when_up': 0,
                         'home_score': 0, 'away_score': 0, 'bat_score': 0, 'fld_score': 0, 'n_thruorder_pitcher': 1, 'game_type': game_type})
    return pd.DataFrame(rows)


def test_postseason_relabel_build_and_receipt(tmp_path):
    from research_lab.pa_model.data import build_plate_appearances
    reg = tmp_path / 'reg.csv.gz'; post = tmp_path / 'post.csv.gz'
    _pitches(1, 'R', '2019-06-01').to_csv(reg, index=False, compression='gzip')
    _pitches(2, 'L', '2019-10-12').to_csv(post, index=False, compression='gzip')
    types = hb.relabel_postseason([post])
    assert types == {2: 'L'}
    pa, report = build_plate_appearances([reg, post])
    pa['game_type'] = [types.get(int(pk), 'R') for pk in pa['game_pk']]
    pa['date_key'] = pd.to_datetime(pa['date_key']).dt.strftime('%Y-%m-%d')
    assert len(pa) == 4 and sorted(pa['outcome']) == ['1B', '1B', 'K', 'K'] and list(pa['game_type']) == ['R', 'R', 'L', 'L']
    games = [{'game_pk': 1, 'date': '2019-06-01', 'game_type': 'R'}, {'game_pk': 3, 'date': '2019-06-02', 'game_type': 'R'},
             {'game_pk': 2, 'date': '2019-10-12', 'game_type': 'L'}]
    r = hb.season_receipt(pa, games, 2019)
    assert r['by_game_type']['R'] == {'schedule_games': 2, 'games': 1, 'missing_games': 1, 'plate_appearances': 2}
    assert r['by_game_type']['L']['plate_appearances'] == 2 and r['date_min'] == '2019-06-01' and r['outcome_mix_pct_regular_season'] == {'1B': 50.0, 'K': 50.0}


def test_postseason_only_build_and_receipt(tmp_path):
    """POST-05: a package season's postseason alone builds and gets its own path, purpose and receipt counts."""
    from research_lab.pa_model.data import build_plate_appearances
    post = tmp_path / 'post.csv.gz'
    _pitches(2, 'D', '2024-10-05').to_csv(post, index=False, compression='gzip')
    types = hb.relabel_postseason([post])
    pa, report = build_plate_appearances([post])
    pa['game_type'] = [types.get(int(pk), 'R') for pk in pa['game_pk']]
    pa['date_key'] = pd.to_datetime(pa['date_key']).dt.strftime('%Y-%m-%d')
    pa = pa[pa['game_type'].isin(hb.POST_TYPES)].reset_index(drop=True)
    assert len(pa) == 2 and set(pa['game_type']) == {'D'}
    r = hb.season_receipt(pa, [{'game_pk': 2, 'date': '2024-10-05', 'game_type': 'D'}], 2024)
    assert r['by_game_type'] == {'D': {'schedule_games': 1, 'games': 1, 'missing_games': 0, 'plate_appearances': 2}} and r['outcome_mix_pct_regular_season'] == {}
    assert hb.postseason_path(2024) == 'private/history/post-2024.enc' and hb.postseason_purpose(2024) == 'pa-history-postseason-2024'
    assert hb.postseason_receipt_path(2024) == 'history/post-2024.json'
