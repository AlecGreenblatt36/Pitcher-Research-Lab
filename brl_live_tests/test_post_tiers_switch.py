"""POST-03's switch (off): a starter's postseason factor by his strikeouts minus walks this season, when turned on."""
import pandas as pd

from brl_live import live_extension as LE
from brl_live.boxscore import ADJUST


def test_switch_is_off_and_the_tier_reads_this_regular_season():
    assert ADJUST.get('postseason_starter_tiers') is None and LE.postseason_tiers_table() is None
    rows = ([{'pitcher': 7, 'date_key': '2026-06-01', 'game_type': 'R', 'outcome': 'K'}] * 60 +
            [{'pitcher': 7, 'date_key': '2026-06-01', 'game_type': 'R', 'outcome': 'BB_HBP'}] * 10 +
            [{'pitcher': 7, 'date_key': '2026-06-01', 'game_type': 'R', 'outcome': 'BIP_OUT'}] * 180 +
            [{'pitcher': 7, 'date_key': '2025-06-01', 'game_type': 'R', 'outcome': 'K'}] * 300 +       # last season: left out
            [{'pitcher': 7, 'date_key': '2026-10-05', 'game_type': 'D', 'outcome': 'K'}] * 30)         # postseason: left out
    h = pd.DataFrame(rows)
    assert abs(LE.season_kbb(h, 7, '2026-10-11') - 50 / 250) < 1e-9
    assert LE.season_kbb(h, 8, '2026-10-11') is None
    tiers = {'cuts': [0.1518, 0.1876], 'scale': [0.81, 0.89, 0.94]}
    assert LE.starter_tier_scale(tiers, 0.20) == 0.94 and LE.starter_tier_scale(tiers, 0.17) == 0.89 and LE.starter_tier_scale(tiers, 0.10) == 0.81
    assert LE.starter_tier_scale(tiers, None) is None
