import numpy as np
from brl_live.world_selection import world_features,select_worlds,KEYS


def world(innings,away,home,**kw):
    f={'innings':innings,'away':away,'home':home,'away_hits':away*2,'home_hits':home*2,'away_outs':18,'home_outs':18,'away_K':5,'home_K':6,'away_PC':90,'home_PC':92}
    f.update(kw);return f


def test_features_read_the_box():
    box={'innings':{'away':{'1':{'R':1,'H':2},'10':{'R':0,'H':1}},'home':{'1':{'R':0,'H':0}}},'score':{'away':1,'home':0},
         'pitching':{'away':[{'outs':18,'K':7,'PC':95}],'home':[{'outs':15,'K':3,'PC':80}]}}
    f=world_features(box)
    assert f['innings']==10 and f['away_hits']==3 and f['home_hits']==0 and f['away_K']==7 and f['home_PC']==80
    assert set(KEYS)<=set(f)


def test_projected_is_central_nine_inning_favorite_win():
    rng=np.random.default_rng(3);feats=[]
    for i in range(4000):
        a,h=int(rng.poisson(4.0)),int(rng.poisson(4.6))
        if a==h:h+=1
        feats.append(world(9 if rng.random()>0.1 else 12,a,h,away_outs=int(rng.normal(16,3)),home_outs=int(rng.normal(17,3)),
                           away_K=int(rng.poisson(5)),home_K=int(rng.poisson(6)),away_PC=int(rng.normal(88,12)),home_PC=int(rng.normal(90,12))))
    feats.append(world(15,4,5))   # a marathon with the modal score must not be the projected game
    picks=select_worlds(feats)
    p=feats[picks['projected']]
    assert p['innings']==9 and p['home']>p['away']
    assert feats[picks['upset']]['home']<feats[picks['upset']]['away'] and feats[picks['upset']]['innings']==9
    totals=np.array([f['away']+f['home'] for f in feats])
    assert feats[picks['high']]['away']+feats[picks['high']]['home']>=np.percentile(totals,85)
    assert feats[picks['low']]['away']+feats[picks['low']]['home']<=np.percentile(totals,15)
    assert len({picks['projected'],picks['high'],picks['low'],picks['upset']})==4
    assert picks['projected']!=len(feats)-1
    assert all(i not in (picks['projected'],picks['high'],picks['low'],picks['upset']) for i in picks['runners_up'])


def test_deterministic_and_handles_missing_roles():
    feats=[world(9,2,5),world(9,3,5),world(9,1,4)]   # the favorite never loses: no upset world
    a=select_worlds(feats);b=select_worlds(list(feats))
    assert a==b and a['upset'] is None and a['projected'] is not None
