import json
from brl_live.box_page import render_page,trim_boxes,public_payload


def ledger():
    f={'game_pk':1,'date':'2026-10-06','version':1,'saved_at':'2026-10-06T18:00:00Z','scheduled_start':'2026-10-06T23:00:00Z',
       'home_win_probability':0.6,'team_baseline_probability':0.55,'away':{'abbr':'LAD','name':'Los Angeles Dodgers'},'home':{'abbr':'ATL','name':'Atlanta Braves'}}
    return {'date':'2026-10-06','forecasts':{'a':f},'publications':{'a':{'commit':'a'*40,'published_at':'2026-10-06T18:01:00Z'}},
            'actuals':{},'status':{'1':{'date':'2026-10-06','state':'Preview'}},
            'box_scores':{'a':{'game_pk':1,'date':'2026-10-06'},'old':{'game_pk':9,'date':'2026-09-01'},'yday':{'game_pk':8,'date':'2026-10-05'}},
            'box_publications':{'a':{},'old':{},'yday':{}},'actual_boxes':{'9':{'x':1},'8':{'x':1}},'player_scores':{},'skill_scores':{}}


def test_page_inlines_public_data_fonts_and_record(tmp_path):
    path=render_page(ledger(),{'n_games':0},tmp_path)
    html=path.read_text(encoding='utf-8')
    assert 'window.BRL=' in html and '/*DATA*/' not in html and '/*FONTS*/' not in html
    assert html.count('@font-face')==5 and 'data:font/woff2;base64,' in html
    assert '<script>' in html and '</script>' in html
    public=json.loads((tmp_path/'predictions.json').read_text())
    assert public['view_scope']=='live' and public['record']['schema']=='brl.record.v2'
    assert public['record']['blend']['a']>0.55 and public['record']['blend']['a']<0.6
    assert (tmp_path/'.nojekyll').exists()
    assert {p.name for p in tmp_path.iterdir()}=={'index.html','predictions.json','.nojekyll','days'}
    days=json.loads((tmp_path/'days/index.json').read_text())
    assert days['dates']==['2026-09-01','2026-10-05','2026-10-06']
    old=json.loads((tmp_path/'days/2026-09-01.json').read_text())
    assert old['date']=='2026-09-01' and set(old)>={'forecasts','box_scores','actuals','market'}


def test_only_recent_boxes_stay_on_the_page():
    public=public_payload(ledger(),{'n_games':0})
    assert set(public['box_scores'])=={'a','yday'}
    assert set(public['box_publications'])=={'a','yday'}
    assert set(public['actual_boxes'])=={'8'}
    assert set(public['forecasts'])=={'a'}


def test_trim_keeps_nothing_for_bad_dates():
    public=trim_boxes({'date':'not-a-date','box_scores':{'a':{'game_pk':1,'date':'2026-10-06'}},'box_publications':{'a':{}},'actual_boxes':{'1':{}}})
    assert public['box_scores']=={} and public['box_publications']=={} and public['actual_boxes']=={}


def test_data_is_escaped_against_script_breakout(tmp_path):
    l=ledger();l['forecasts']['a']['away']['name']='</script><script>alert(1)</script>'
    html=render_page(l,{'n_games':0},tmp_path).read_text(encoding='utf-8')
    assert '</script><script>alert' not in html


def test_live_snapshots_follow_the_box_window():
    l = ledger(); l['live'] = {'1': {'game_pk': 1, 'date': '2026-10-06', 'home_win_probability': 0.7}, '9': {'game_pk': 9, 'date': '2026-09-01'}}
    public = public_payload(l, {'n_games': 0})
    assert set(public['live']) == {'1'}


def test_lean_box_keeps_the_projected_game_only():
    from brl_live.box_page import lean_box
    row=lambda pid:{'player_id':pid,'name':'P '+pid,'means':{'H':1.0},'distributions':{'H':{'n':10000,'counts':[[0,5000],[1,5000]]}},'hit_probability':0.5}
    sample=lambda s:{'seed':s,'score':{'away':1,'home':2},'innings':{'away':{},'home':{}},'batting':{'away':[],'home':[]},'pitching':{'away':[],'home':[]},
                     'plays':[{'inning':1,'half':'top','batter_id':'1','batter_name':'A','pitcher_id':'2','pitcher_name':'B','box_outcome':'single','description':'x','outs_before':0,'outs_after':0,'runs_scored':0,'away_score':0,'home_score':0,'bases_before':[None]*3,'bases_after':['1',None,None],'scoring_players':[],'rbi':0,'estimated_pitches':3,'pitches':[['FF',95,'X']]}],'world_index':7,'typical':True}
    box={'schema':'brl.box-forecast.v1','game_pk':1,'date':'2026-10-06','saved_at':'x','forecast_origin':'x','team_ids':{},'starters':{},'lineup_status':{},'history_through':'x',
         'forecast_id':'a','n_simulations':10000,'sample_roles':{'projected':7,'high':8},'sample_indices':[7,8],'adjustments':[],'team_model':{'p_home':0.5},
         'teams':{'away':{'batting':[row('1')],'pitching':[row('2')]},'home':{'batting':[row('3')],'pitching':[row('4')]}},'samples':[sample(7),sample(8)],'line_score':{}}
    lean=lean_box(box)
    assert lean['archived'] and len(lean['samples'])==1 and lean['samples'][0]['seed']==7 and 'pitches' not in lean['samples'][0]['plays'][0]
    assert 'distributions' not in lean['teams']['away']['batting'][0] and lean['teams']['away']['batting'][0]['hit_probability']==0.5
    assert lean['sample_roles']=={'projected':7} and lean['sample_indices']==[7]


def test_only_the_latest_version_box_of_each_game_stays_on_the_page():
    l=ledger()
    f1=l['forecasts']['a']; f2=dict(f1,version=2,saved_at='2026-10-06T20:00:00Z')
    l['forecasts']={'a':f1,'b':f2}; l['publications']['b']=l['publications']['a']
    l['box_scores']['a']['skill_baselines']={'as_of':'x'}
    l['box_scores']['b']={'game_pk':1,'date':'2026-10-06','skill_baselines':{'as_of':'y'},'teams':{}}
    l['box_publications']['b']={}
    public=public_payload(l,{'n_games':0})
    assert set(public['box_scores'])=={'b','yday'}
    assert 'skill_baselines' not in public['box_scores']['b'] and 'teams' in public['box_scores']['b']
    assert 'skill_baselines' in l['box_scores']['b']      # the ledger copy is untouched
    assert set(public['box_publications'])=={'b','yday'}
