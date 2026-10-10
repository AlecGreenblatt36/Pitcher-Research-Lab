import copy
from collections import Counter
from datetime import datetime,timezone
import json
import numpy as np
import pytest
from brl_live import edge_metrics as e
from brl_live.boxscore import BoxAccumulator
from brl_live_tests.test_boxscore import matchup


def d(values):
    return {'n':len(values),'counts':[[int(x),int(n)] for x,n in sorted(Counter(values).items())]}


def row(pid,k=2,bb=1,pc=30,outs=9):
    return {'player_id':str(pid),'name':'Player '+str(pid),'K':k,'BB':bb,'PC':pc,'outs':outs}


def specimen():
    pitching={s:[row(s+'SP'),row(s+'RP',1,0,10,3)] for s in ('away','home')}
    actual={'game_pk':20,'score':{'away':3,'home':5},'team_ids':{'away':1,'home':2},
            'pitching':pitching,'first_pitch_observed_at':'2026-10-06T23:00:00Z'}
    prior=[]
    for pk in range(3):
        a=copy.deepcopy(actual);a['game_pk']=pk;a['first_pitch_observed_at']='2026-10-05T14:00:00Z'
        if pk==2:
            a['pitching']['away']=a['pitching']['away'][:1]
        prior.append({'date':'2026-10-05','available_at':'2026-10-05T19:00:00Z','box':a})
    box={'schema':'brl.box-forecast.v1','game_pk':20,'date':'2026-10-06','n_simulations':10,
         'forecast_origin':'2026-10-06T20:00:00Z','saved_at':'2026-10-06T20:05:00Z',
         'team_ids':actual['team_ids'],'starters':{s:{'player_id':s+'SP'} for s in ('away','home')},
         'team_run_distributions':{'away':d([3]*10),'home':d([5]*10)},
         'total_run_distribution':d([8]*10),'teams':{}}
    for side in ('away','home'):
        box['teams'][side]={'batting':[],'pitching':[]}
        for p in pitching[side]+[row(side+'UNUSED',0,0,0,0)]:
            r={'player_id':p['player_id'],'name':p['name'],'role':'starter' if p['player_id'].endswith('SP') else 'reliever',
               'appearance_probability':0.0 if p['player_id'].endswith('UNUSED') else 1.0,
               'distributions':{k:d([p[k]]*10) for k in ('K','BB','PC','outs')}}
            box['teams'][side]['pitching'].append(r)
    team={'game_pk':20,'prior_through':'2026-10-05','dispersion_fit':{'training_through':'2024-10-01'},
          'away_mean_runs':4.0,'home_mean_runs':4.4,'nb_alpha':0.2,'prior_games':100}
    box['skill_baselines']=e.freeze_skill_baselines(box,prior,team,'2026-10-06T19:00:00Z')
    return box,actual,prior,team


def pub(b,when=None):
    from app.common import content_hash
    return {'commit':'a'*40,'box_sha256':content_hash(b),'published_at':when or b['saved_at']}


@pytest.mark.parametrize('values,obs',[([0,1,2],3),([1,1],0),([0,8,14,4],5)])
def test_fair_crps_quadratic_oracle(values,obs):
    a=np.array(values);n=len(a)
    oracle=np.abs(a-obs).mean()-np.abs(a[:,None]-a[None,:]).sum()/(2*n*(n-1))
    assert e.ensemble_crps(d(values),obs)==pytest.approx(oracle)


def test_exact_baseline_not_incorrectly_mc_corrected():
    pmf=[[0,.4],[1,.6]]
    assert e.discrete_crps(pmf,1)==pytest.approx(.16)
    assert e.ensemble_crps(d([0]*4+[1]*6),1)==pytest.approx(.16-.24/9)


@pytest.mark.parametrize('bad',[{'n':10,'counts':[[0,9]]},{'n':2,'counts':[[0,1],[0,1]]},
    {'n':2,'counts':[[-1,2]]},{'n':2,'counts':[[0,1.5],[2,.5]]},{'n':1,'counts':[[0,1]]}])
def test_bad_histogram_rejected(bad):
    with pytest.raises(ValueError):e.histogram(bad)


@pytest.mark.parametrize('mean,alpha,y',[(4,.2,3),(8,.7,20),(1,.1,0)])
def test_nb_crps_matches_large_discrete_oracle(mean,alpha,y):
    from scipy.stats import nbinom
    r=1/alpha;p=1/(1+alpha*mean);xs=np.arange(1000);ps=nbinom.pmf(xs,r,p)
    score,bound=e.nb_crps(mean,alpha,y)
    expected=np.sum((nbinom.cdf(xs,r,p)-(xs>=y))**2)
    assert score==pytest.approx(expected,abs=1e-10)
    assert 0<=bound<1e-12


def test_high_total_probability_exact_convolution():
    from scipy.stats import nbinom
    a=nbinom.pmf(np.arange(200),5,1/(1+0.2*4))
    b=nbinom.pmf(np.arange(200),5,1/(1+0.2*5))
    assert e.nb_high_probability(4,5,.2)==pytest.approx(np.convolve(a,b)[9:].sum())


@pytest.mark.parametrize('change',['target','same_day','future_availability','duplicate','baseline_future','wrong_origin'])
def test_prior_input_gate(change):
    b,a,h,t=specimen();origin='2026-10-06T19:00:00Z'
    if change=='target':h[0]['box']['game_pk']=20
    if change=='same_day':h[0]['date']='2026-10-06'
    if change=='future_availability':h[0]['available_at']='2026-10-06T21:00:00Z'
    if change=='duplicate':h.append(h[0])
    if change=='baseline_future':t['prior_through']='2026-10-06'
    if change=='wrong_origin':origin='2026-10-07T19:00:00Z'
    with pytest.raises(ValueError):e.freeze_skill_baselines(b,h,t,origin)


def test_prior_roles_include_zero_relief_exposure():
    b,a,h,t=specimen();x=b['skill_baselines']['pitching']['away']
    assert x['awayRP']==x['awayUNUSED']
    assert x['awayRP']['appearance_probability']==pytest.approx(5/12)
    assert dict(x['awayRP']['pmf']['K'])[0]==pytest.approx(7/12)


def test_smaller_current_pool_still_valid_distribution():
    b,a,h,t=specimen()
    b['teams']['away']['pitching']=b['teams']['away']['pitching'][:2]
    for r in h:r['box']['pitching']['away'].extend([row('extra1'),row('extra2')])
    fit=e.freeze_skill_baselines(b,h,t,'2026-10-06T19:00:00Z')
    assert 0<=fit['pitching']['away']['awayRP']['appearance_probability']<=1
    assert sum(p for _,p in fit['pitching']['away']['awayRP']['pmf']['K'])==pytest.approx(1)


def test_all_seven_metrics_and_absence_counts():
    b,a,_,_=specimen();result=e.score_skill_boxes({'v':b},{'v':pub(b)},{'20':a})
    assert result['n_games']==1
    assert {x['metric'] for x in result['aggregates']}=={x[0] for x in e.METRICS}
    absent=[r for r in result['versions'][0]['rows'] if r['entity_id']=='awayUNUSED']
    assert len(absent)==5 and all(r['actual']==0 for r in absent)


def test_actual_missing_stat_unavailable_not_zero():
    b,a,_,_=specimen();a['pitching']['away'][0]['BB']=None
    out=e.score_one(b,a)
    assert len(out['missing_counts'])==1
    assert not any(r['metric']=='pitcher_BB' and r['entity_id']=='awaySP' for r in out['rows'])


def test_actual_unprojected_pitcher_visible_as_coverage_failure():
    b,a,_,_=specimen();a['pitching']['away'].append(row('surprise'))
    out=e.score_skill_boxes({'v':b},{'v':pub(b)},{'20':a})
    assert out['unprojected_pitcher_appearances']==1


@pytest.mark.parametrize('change',['hash','late','lane','baseline_late','missing_baseline','team','threshold'])
def test_publication_and_lane_gate(change):
    b,a,_,_=specimen();p=pub(b)
    if change=='hash':b['skill_baselines']['pitching']['away']['awayRP']['appearance_probability']=.8
    if change=='late':p['published_at']='2026-10-06T23:01:00Z'
    if change=='lane':b['skill_baselines']['lane']='in_game';p=pub(b)
    if change=='baseline_late':b['skill_baselines']['as_of']='2026-10-06T22:00:00Z';p=pub(b)
    if change=='missing_baseline':b.pop('skill_baselines');p=pub(b)
    if change=='team':a['team_ids']={'away':3,'home':2}
    if change=='threshold':b['skill_baselines']['high_scoring_threshold']=10;p=pub(b)
    out=e.score_skill_boxes({'v':b},{'v':p},{'20':a})
    assert out['n_games']==0 and 'v' in out['excluded']


def test_equal_game_weight_not_player_row_weight():
    versions=[{'game_pk':1,'rows':[{'metric':'pitcher_K','model_score':1.,'baseline_score':2.}]},
              {'game_pk':2,'rows':[{'metric':'pitcher_K','model_score':3.,'baseline_score':4.}]*10}]
    a=e.aggregate_rows(versions)[0]
    assert a['model_score']==2 and a['n_cases']==11 and a['n_games']==2


def test_score_pairs_keep_dependence_not_marginal_sums():
    a=BoxAccumulator(matchup());a.n=10000;a.score_pairs=[(0,10),(10,0)]*5000;a.seeds=list(range(10000))
    for side in ('away','home'):
        for kind in ('batting','pitching'):
            for val in a.hist[side][kind].values():
                for metric in val['stats']:val['stats'][metric][0]=10000
    result=a.finish()
    assert result['total_run_distribution']=={'n':10000,'counts':[[10,10000]]}
    assert result['team_run_distributions']['away']['counts']==[[0,5000],[10,5000]]


def test_all_registered_ideas_start_unfitted():
    from pathlib import Path
    registry=json.loads((Path(__file__).parents[1]/'research/edge_track/experiments.json').read_text())
    assert len(registry['experiments'])==5
    assert all(not x['retained'] and x['candidate_hash'] is None for x in registry['experiments'])
    assert all(set(k for k,_,_ in e.METRICS)<=set(x['effects']) for x in registry['experiments'])


def test_paired_report_keeps_metrics_and_shared_baseline():
    from research.edge_track.report import compare
    rows=[{'game_pk':i,'date':f'2025-07-{i:02d}','metric':metric,'entity_id':'g',
           'score':1.,'baseline_score':.8} for i in range(1,5) for metric in ('win_brier','win_log_loss','pitcher_K')]
    candidate=copy.deepcopy(rows)
    for r in candidate:r['score']=.9
    out=compare(rows,candidate,n_boot=200)
    assert out['numerical_game_gate']=='PASS' and not out['adoption_authorized']
    assert out['effects']['pitcher_K']['delta']==pytest.approx(-.1)
    candidate[0]['baseline_score']=.7
    with pytest.raises(ValueError):compare(rows,candidate,n_boot=200)


def test_paired_report_refuses_cherry_picked_rows():
    from research.edge_track.report import compare
    with pytest.raises(ValueError):compare([],[],n_boot=20)
    a=[{'game_pk':1,'date':'2025-01-01','metric':'pitcher_K','entity_id':'1','score':1.,'baseline_score':1.}]
    with pytest.raises(ValueError):compare(a,[],n_boot=20)



def test_missing_latest_baseline_cannot_revive_older_comparison():
    b,a,_,_=specimen();later=copy.deepcopy(b);later['saved_at']='2026-10-06T20:10:00Z'
    later.pop('skill_baselines')
    out=e.score_skill_boxes({'old':b,'new':later},{'old':pub(b),'new':pub(later)},{'20':a})
    assert out['n_games']==0 and len(out['versions'])==1


def test_an_early_call_the_day_before_freezes_its_baseline():
    b,a,h,t=specimen()
    fit=e.freeze_skill_baselines(b,h,t,'2026-10-05T23:30:00Z')   # 7:30 p.m. Eastern the day before the game
    assert fit['as_of']=='2026-10-05T23:30:00Z' and fit['n_prior_games']>=1
