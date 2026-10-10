"""Game-day plan refresh: plans older than their date's latest forecast are rebuilt (lineups posted, a starter named)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _trigger():
    sys.path.insert(0, str(ROOT / 'tools'))
    spec = importlib.util.spec_from_file_location('brl_daily_report_trigger_test', ROOT / 'tools' / 'brl_daily_report_trigger.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_newest_forecast_per_day_today_and_tomorrow_only():
    T = _trigger()
    D = {'date': '2026-10-10', 'forecasts': {
        'a': {'date': '2026-10-10', 'saved_at': '2026-10-10T03:58:47+00:00'},
        'b': {'date': '2026-10-10', 'saved_at': '2026-10-10T21:05:00+00:00'},
        'c': {'date': '2026-10-11', 'saved_at': '2026-10-10T22:00:00+00:00'},
        'd': {'date': '2026-10-08', 'saved_at': '2026-10-08T22:00:00+00:00'}}}
    assert T.newest_forecasts(D) == {'2026-10-10': '2026-10-10T21:05:00+00:00', '2026-10-11': '2026-10-10T22:00:00+00:00'}


def test_stale_when_plans_older_or_missing():
    T = _trigger()
    newest = {'2026-10-10': '2026-10-10T21:05:00+00:00', '2026-10-11': '2026-10-10T22:00:00+00:00'}
    built = {'2026-10-10': '2026-10-10T20:50:00+00:00', '2026-10-11': None}
    assert T.stale_dates(newest, built) == ['2026-10-10', '2026-10-11']
    assert T.stale_dates(newest, {'2026-10-10': '2026-10-10T21:30:00+00:00', '2026-10-11': '2026-10-10T22:10:00+00:00'}) == []
