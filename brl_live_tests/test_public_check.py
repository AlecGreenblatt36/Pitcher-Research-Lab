import copy,importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('phonecheck',Path(__file__).parents[1]/'tools/check_brl_phone_page.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def specimen():
    f=dict.fromkeys(mod.ALLOWED_FORECAST_FIELDS)
    f.update(game_pk=1,date='2026-10-06',n_simulations=10000,
             forecast_origin='2026-10-06T18:00:00Z',saved_at='2026-10-06T18:05:00Z',
             scheduled_start='2026-10-06T22:00:00Z',home_win_probability=0.6)
    return {'actuals':{},'date':'2026-10-06','forecasts':{'x':f},
            'publications':{'x':{'commit':'a'*40,'published_at':'2026-10-06T18:06:00Z'}},
            'scores':{},'status':{'1':{'date':'2026-10-06'},'2':{'date':'2026-10-05'}}}


def test_expected_today_cards_only():
    latest,ids=mod.inspect_data(specimen())
    assert ids=={'1'} and len(latest)==1


@pytest.mark.parametrize('field,value',[('home_win_probability',float('nan')),
    ('n_simulations',9999),('saved_at','2026-10-06T22:00:00Z'),('training_raw_rows',[])])
def test_invalid_public_forecast_fails(field,value):
    data=specimen();data['forecasts']['x'][field]=value
    with pytest.raises(ValueError):mod.inspect_data(data)
