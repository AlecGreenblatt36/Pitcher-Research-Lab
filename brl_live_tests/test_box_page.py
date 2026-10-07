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
    assert public['view_scope']=='live' and public['record']['schema']=='brl.record.v1'
    assert public['record']['blend']['a']>0.55 and public['record']['blend']['a']<0.6
    assert (tmp_path/'.nojekyll').exists()
    assert {p.name for p in tmp_path.iterdir()}=={'index.html','predictions.json','.nojekyll'}


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
