"""LONG-01 at forecast time: the sealed seasons go in front of the live history for the provider only."""
import gzip
import os

import pandas as pd
import pytest

from app.safety import Blocked
from cloud.security import seal
from brl_live import talent_history as th


class Store:
    def __init__(self, files): self.files = files
    def read(self, path): return (self.files[path], 'sha') if path in self.files else None


def frame(dates, gt='R'):
    return pd.DataFrame({'date_key': dates, 'game_pk': range(len(dates)), 'at_bat_number': 1, 'batter': 1, 'pitcher': 2, 'stand': 'R', 'p_throws': 'L',
                         'park': 'NYY', 'outcome': 'K', 'age_bat': 30, 'age_pit': 28, 'game_type': gt, 'extra': 0})


def test_older_seasons_go_first_and_only_regular_season(tmp_path):
    key = os.urandom(32)
    old = pd.concat([frame(['2022-04-07', '2022-09-30']), frame(['2022-10-20'], 'W')], ignore_index=True)
    store = Store({th.history_path(2022): seal(gzip.compress(old.to_csv(index=False).encode()), key, th.history_purpose(2022))})
    live = frame(['2023-03-30', '2026-10-15'])
    out = tmp_path / 't.csv.gz'
    r = th.write(store, key, [2022], live, out)
    got = pd.read_csv(out)
    assert list(got.columns) == th.COLUMNS and got['date_key'].tolist() == ['2022-04-07', '2022-09-30', '2023-03-30', '2026-10-15']
    assert r == {'seasons': [2022], 'older_rows': 2, 'rows': 4, 'through': '2026-10-15'}
    with pytest.raises(Blocked):
        th.write(store, key, [2021], live, out)
