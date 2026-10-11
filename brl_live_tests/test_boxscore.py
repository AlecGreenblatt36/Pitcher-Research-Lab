import copy,json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from brl_live.boxscore import (BookkeepingFit,ObservedSimulator,build_game_box,fair_crps,corrected_brier,parse_actual_box,score_player_boxes,BoxAccumulator)
from research_lab.game_sim.models import (PlayerProfile,PitcherProfile,TeamProfile,GameMatchup,SimulationConfig)
from research_lab.game_sim.engine import GameSimulator
from app.safety import Blocked


def matchup():
    def team(side):
        return TeamProfile(side,side,tuple(PlayerProfile(f'{side}{i}',f'{side} Hitter {i}') for i in range(9)),
                           PitcherProfile(side+'SP',side+' Starter',role='starter',expected_batters=18),
                           (PitcherProfile(side+'RP',side+' Relief'),))
    return GameMatchup(team('away'),team('home'))
class Provider:
    name='test_only';validation_status='synthetic_test_only'
    def probabilities(self,c):return dict(zip(('bip_out','strikeout','bb_hbp','single','double_triple','home_run','other_reach'),(.4,.2,.1,.16,.07,.05,.02)))

def history():
    return pd.DataFrame([{'date_key':'2026-10-01','outcome':o,'pitcher':1,'stand':hand,'terminal_event':event,'pitch_number':pc}
                         for hand in ('R','L') for o,event,pc in [('BIP_OUT','field_out',2),('K','strikeout',5),('BB_HBP','walk',6),('BB_HBP','hit_by_pitch',2),('1B','single',4),('2B_3B','double',3),('HR','home_run',5),('OTHER_REACH','field_error',2)]])

@pytest.mark.parametrize('seed',range(12))
def test_observation_does_not_change_any_engine_event(seed):
    m=matchup();config=SimulationConfig(max_innings=100,max_plate_appearances=4000)
    a=GameSimulator(Provider(),config);b=ObservedSimulator(Provider(),config)
    x=a.simulate(m,seed,record_events=True);y=b.simulate(m,seed,record_events=True)
    assert x.to_dict()==y.to_dict()
    box=build_game_box(y,m,BookkeepingFit(history(),'2026-10-06'))
    for side in ('away','home'):
        assert sum(r['R'] for r in box['batting'][side])==getattr(x,side+'_score')
        assert sum(r['H'] for r in box['batting'][side])==sum(r['H'] for r in box['pitching']['home' if side=='away' else 'away'])
    assert x.to_dict()==y.to_dict()


def test_reproducible_annotations_on_separate_rng():
    m=matchup();sim=ObservedSimulator(Provider(),SimulationConfig(max_innings=100,max_plate_appearances=4000));f=BookkeepingFit(history(),'2026-10-06')
    assert build_game_box(sim.simulate(m,78),m,f)==build_game_box(sim.simulate(m,78),m,f)


def test_future_history_cannot_change_annotation_pools():
    h=history();future=h.copy();future.date_key='2026-10-06';future.pitch_number=40
    one=BookkeepingFit(h,'2026-10-06');two=BookkeepingFit(pd.concat([h,future]),'2026-10-06')
    assert one.max_input_date<'2026-10-06';assert one.pools.keys()==two.pools.keys()
    for k in one.pools:assert np.array_equal(one.pools[k],two.pools[k])


def test_invalid_one_pitch_strikeouts_not_imputed():
    h=history();h.loc[h.outcome=='K','pitch_number']=1;fit=BookkeepingFit(h,'2026-10-06')
    with pytest.raises(Blocked):fit.draw({'pitcher_id':'1','outcome':'strikeout'},np.random.default_rng(1))

@pytest.mark.parametrize('values',[[0,0,1,2],[0,3],[2,2,2,2],[0,1,3,6,10]])
@pytest.mark.parametrize('obs',[0,1,5])
def test_fair_crps_exact_pair_formula(values,obs):
    from collections import Counter
    n=len(values);d={'n':n,'counts':list(Counter(values).items())};x=np.array(values)
    expected=np.abs(x-obs).mean()-np.abs(x[:,None]-x[None,:]).sum()/(2*n*(n-1))
    assert fair_crps(d,obs)==pytest.approx(expected)


def test_corrected_brier_formula():
    assert corrected_brier({'n':10,'counts':[[0,4],[1,6]]},1)==pytest.approx(.4**2-.6*.4/9)


def test_incomplete_distribution_rejected():
    with pytest.raises(Blocked):fair_crps({'n':10,'counts':[[0,3],[1,6]]},0)


def test_accumulator_has_zero_for_nonappearing_pitcher():
    a=BoxAccumulator(matchup());x={'seed':1,'score':{'away':0,'home':0},'innings':{'away':{},'home':{}},'batting':{'away':[],'home':[]},'pitching':{'away':[],'home':[]}}
    a.add(x)
    assert a.hist['away']['pitching']['awayRP']['stats']['outs'][0]==1
    assert a.hist['away']['pitching']['awayRP']['appeared']==0


def test_refuse_less_than_10000_worlds():
    with pytest.raises(Blocked):BoxAccumulator(matchup()).finish()


def specimen():
    b={'game_pk':1,'saved_at':'2026-10-06T18:00:00Z','team_ids':{'away':1,'home':2},'teams':{s:{'batting':[],'pitching':[]} for s in ('away','home')}}
    row={'player_id':'1','name':'Test','means':{'H':.6,'HR':.2,'K':1,'outs':9},'distributions':{k:{'n':10,'counts':[[0,4],[1,6]]} for k in ('H','HR','K','outs')}}
    b['teams']['away']['batting']=[row];b['teams']['home']['pitching']=[row]
    actual={'first_pitch_observed_at':'2026-10-06T22:00:00Z','team_ids':b['team_ids'],'batting':{'away':[],'home':[]},'pitching':{'away':[],'home':[]}}
    return b,actual


def publication(box,time=None):
    from app.common import content_hash
    return {'published_at':time or box['saved_at'],'commit':'a'*40,'box_sha256':content_hash(box)}


def test_latest_pregame_not_best_result_selected():
    b,a=specimen();later=copy.deepcopy(b);later['saved_at']='2026-10-06T20:00:00Z'
    pub={k:publication(v) for k,v in [('old',b),('new',later)]}
    s=score_player_boxes({'old':b,'new':later},pub,{'1':a})
    assert s['n_games']==1 and len(s['versions'])==2
    assert s['aggregates'][0]['n_player_games']==1


def test_late_player_forecast_not_backdated_to_win_forecast():
    b,a=specimen();s=score_player_boxes({'x':b},{'x':publication(b,'2026-10-06T22:01:00Z')},{'1':a})
    assert s['n_games']==0 and 'x' in s['excluded']


def test_missing_actual_field_is_not_zero():
    b,a=specimen();a['batting']['away']=[{'player_id':'1','H':None,'HR':0,'K':0}]
    s=score_player_boxes({'x':b},{'x':publication(b)},{'1':a})
    assert not any(r['metric']=='H' for r in s['versions'][0]['rows'])


def test_absent_final_participant_is_zero_unconditionally():
    b,a=specimen();s=score_player_boxes({'x':b},{'x':publication(b)},{'1':a})
    assert all(r['actual']==0 for r in s['versions'][0]['rows'])


def test_team_mismatch_refuses_player_scoring():
    b,a=specimen();a['team_ids']={'away':3,'home':4}
    assert score_player_boxes({'x':b},{'x':publication(b)},{'1':a})['n_games']==0


def actual_fixture():
    def team(pid,h,r,allowed_h,allowed_r):
        b=dict(plateAppearances=3,atBats=3,hits=h,doubles=0,triples=0,homeRuns=0,runs=r,rbi=r,baseOnBalls=0,hitByPitch=0,strikeOuts=1,sacFlies=0)
        p=dict(inningsPitched='3.0',numberOfPitches=40,hits=allowed_h,runs=allowed_r,baseOnBalls=0,hitBatsmen=0,strikeOuts=1,homeRuns=0,battersFaced=3)
        return {'batters':[pid],'pitchers':[pid+10],'players':{f'ID{pid}':{'person':{'id':pid,'fullName':'Test Hitter'},'battingOrder':'100','stats':{'batting':b}},f'ID{pid+10}':{'person':{'id':pid+10,'fullName':'Test Pitcher'},'stats':{'pitching':p}}},'teamStats':{'batting':{'runs':r,'hits':h}}}
    return {'gamePk':1,'gameData':{'status':{'abstractGameState':'Final'},'teams':{'away':{'id':1},'home':{'id':2}}},'liveData':{'linescore':{'teams':{'away':{'runs':1},'home':{'runs':2}},'innings':[{'num':1,'away':{'runs':1,'hits':1},'home':{'runs':2,'hits':2}}]},'boxscore':{'teams':{'away':team(1,1,1,2,2),'home':team(2,2,2,1,1)}},'plays':{'allPlays':[{'playEvents':[{'isPitch':True,'startTime':'2026-10-06T22:00:00Z'}]}]}}}


def test_actual_reader_joins_ids_and_converts_innings():
    b=parse_actual_box(actual_fixture(),1,'2026-10-07T00:00:00Z')
    assert b['pitching']['home'][0]['outs']==9
    assert b['batting']['away'][0]['player_id']=='1'


def test_actual_reader_keeps_errors_only_when_the_feed_has_them():
    assert 'errors' not in parse_actual_box(actual_fixture(),1,'2026-10-07T00:00:00Z')
    f=actual_fixture();f['liveData']['linescore']['teams']['away']['errors']=1;f['liveData']['linescore']['teams']['home']['errors']=0
    assert parse_actual_box(f,1,'2026-10-07T00:00:00Z')['errors']=={'away':1,'home':0}


def test_actual_reader_never_accepts_in_progress():
    f=actual_fixture();f['gameData']['status']['abstractGameState']='Live'
    with pytest.raises(ValueError):parse_actual_box(f,1,'2026-10-07T00:00:00Z')


def test_actual_totals_must_match_players():
    f=actual_fixture();f['liveData']['boxscore']['teams']['away']['players']['ID1']['stats']['batting']['hits']=5
    with pytest.raises(ValueError):parse_actual_box(f,1,'2026-10-07T00:00:00Z')


def test_baseball_two_outs_not_point_two_innings():
    f=actual_fixture();f['liveData']['boxscore']['teams']['home']['players']['ID12']['stats']['pitching']['inningsPitched']='5.2'
    assert parse_actual_box(f,1,'2026-10-07T00:00:00Z')['pitching']['home'][0]['outs']==17


def test_altered_distribution_rejected_by_publication_hash():
    b,a=specimen();pub=publication(b);b['teams']['away']['batting'][0]['means']['H']=3
    assert score_player_boxes({'x':b},{'x':pub},{'1':a})['n_games']==0


def test_no_publication_identity_not_scored():
    b,a=specimen();pub=publication(b);pub.pop('commit')
    assert score_player_boxes({'x':b},{'x':pub},{'1':a})['n_games']==0


@pytest.mark.parametrize('seed',range(6))
def test_box_scores_add_up_with_the_empirical_base_running_kernel(seed):
    """With the production kernel (brl_live/transitions.json) the box still adds up: runs, hits, outs and the sacrifice flies."""
    from brl_live.boxscore import transitions_for
    kernel=transitions_for({'transitions':True})
    assert kernel is not None and len(kernel.cells)>100 and transitions_for({'transitions':False}) is None
    m=matchup();sim=ObservedSimulator(Provider(),SimulationConfig(max_innings=100,max_plate_appearances=4000),transitions=kernel)
    r=sim.simulate(m,seed,record_events=True)
    box=build_game_box(r,m,BookkeepingFit(history(),'2026-10-06'))
    for side in ('away','home'):
        assert sum(x['R'] for x in box['batting'][side])==getattr(r,side+'_score')
        assert sum(x['H'] for x in box['batting'][side])==sum(x['H'] for x in box['pitching']['home' if side=='away' else 'away'])
        assert sum(x['PA'] for x in box['batting'][side])==sum(x['AB']+x['BB']+x['HBP']+x['SF'] for x in box['batting'][side])
