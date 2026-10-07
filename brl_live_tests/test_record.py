import math
import pytest
from brl_live.record import blend_probability,build_record,HOME_RATE_PRIOR


def test_blend_is_equal_weight_on_log_odds():
    assert blend_probability(0.5,0.5)==pytest.approx(0.5)
    assert blend_probability(0.7,0.7)==pytest.approx(0.7)
    p=blend_probability(0.8,0.5)
    z=0.5*math.log(4)  # half of logit(0.8)
    assert p==pytest.approx(1/(1+math.exp(-z)))
    assert blend_probability(0.62,None)==0.62
    with pytest.raises(ValueError):blend_probability(None,0.5)


def forecast(ident,pk,version,saved,p_sim,p_team,date='2026-10-06'):
    return ident,{'game_pk':pk,'version':version,'saved_at':saved,'date':date,'home_win_probability':p_sim,'team_baseline_probability':p_team,
                  'away':{'abbr':'LAD'},'home':{'abbr':'ATL'}}


def ledger():
    f=dict([forecast('v1',1,1,'2026-10-06T18:00:00Z',0.60,0.55),forecast('v2',1,2,'2026-10-06T20:00:00Z',0.65,0.55),
            forecast('late',1,3,'2026-10-06T23:30:00Z',0.90,0.55),forecast('other',2,1,'2026-10-06T18:00:00Z',0.40,None)])
    pubs={'v1':{'commit':'a'*40,'published_at':'2026-10-06T18:05:00Z'},'v2':{'commit':'b'*40,'published_at':'2026-10-06T20:05:00Z'},
          'late':{'commit':'c'*40,'published_at':'2026-10-06T23:35:00Z'},'other':{'commit':'d'*40,'published_at':'2026-10-06T18:05:00Z'}}
    actuals={'1':{'away':2,'home':5,'first_pitch_observed_at':'2026-10-06T23:10:00Z'},
             '2':{'away':3,'home':1,'first_pitch_observed_at':None}}
    return {'forecasts':f,'publications':pubs,'actuals':actuals}


def test_last_pregame_version_is_scored_and_late_versions_are_not():
    rec=build_record(ledger())
    assert rec['n_scored']==1
    g=rec['games'][0]
    assert g['forecast_id']=='v2' and g['version']==2 and g['home_won'] is True
    assert g['p_sim']==0.65 and g['p_team']==0.55
    assert g['brier']['sim']==pytest.approx((0.65-1)**2)
    assert g['brier']['home']==pytest.approx((HOME_RATE_PRIOR-1)**2)
    assert g['brier']['coin']==0.25
    assert rec['unscored']==[{'game_pk':2,'reason':'no forecast published before first pitch'}]


def test_ladder_and_blend_cover_every_forecast():
    rec=build_record(ledger())
    assert set(rec['blend'])=={'v1','v2','late','other'}
    assert rec['blend']['other']==0.4
    ladder={r['key']:r for r in rec['ladder']}
    assert ladder['blend']['n']==1 and ladder['team']['n']==1
    assert ladder['coin']['better_than_coin_pct']==0
    assert ladder['blend']['brier']==pytest.approx((rec['blend']['v2']-1)**2,abs=1e-5)


def test_empty_ledger_has_empty_record():
    rec=build_record({'forecasts':{},'publications':{},'actuals':{}})
    assert rec['n_scored']==0 and rec['games']==[] and all(r['brier'] is None for r in rec['ladder'])
