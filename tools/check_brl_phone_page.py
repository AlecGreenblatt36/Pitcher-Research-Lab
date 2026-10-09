"""Read-only deployed page checks and phone screenshots; no private runtime/key."""
from __future__ import annotations
import argparse,json,math,re,sys
from datetime import datetime,timezone
from pathlib import Path
SITE='https://alecgreenblatt36.github.io/Pitcher-Research-Lab/'
ALLOWED_FORECAST_FIELDS=set('schema scope game_pk date game_type scheduled_start forecast_origin saved_at version lineup_status home away n_simulations seed home_win_probability probability_mcse projected_away_runs projected_home_runs team_baseline_probability market_probability model history_through postseason_regular_bullpen_logic automatic_runner github_run_id snapshot_hash'.split())
OLD_FIELDS=set('actuals date forecasts publications scores status'.split())
BOX_FIELDS=OLD_FIELDS|set('box_scores box_publications actual_boxes player_scores view_scope'.split())
OPTIONAL_FIELDS={'skill_scores','record','live','market','generated_at','context','series'}
def utc(text):
    d=datetime.fromisoformat(text.replace('Z','+00:00'))
    if d.tzinfo is None:raise ValueError('Timezone required')
    return d.astimezone(timezone.utc)
def inspect_data(data):
    fields=set(data)
    if fields!=OLD_FIELDS and not (BOX_FIELDS<=fields<=BOX_FIELDS|OPTIONAL_FIELDS):raise ValueError('Unexpected public fields')
    for ident,f in data['forecasts'].items():
        if set(f)!=ALLOWED_FORECAST_FIELDS:raise ValueError('Unexpected forecast field')
        if f['n_simulations']!=10000:raise ValueError('Not a 10,000-world forecast')
        p=f['home_win_probability']
        if not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
        if not utc(f['forecast_origin'])<=utc(f['saved_at'])<utc(f['scheduled_start']):raise ValueError('Forecast timing invalid')
        publication=data['publications'].get(ident)
        if not publication or not re.fullmatch('[0-9a-f]{40}',publication['commit']):raise ValueError('Missing publication receipt')
        if utc(publication['published_at'])>=utc(f['scheduled_start']):raise ValueError('Late publication')
    record=data.get('record') or {}
    for ident,p in (record.get('blend') or {}).items():
        if ident not in data['forecasts']:raise ValueError('Blend for an unknown forecast')
        if not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid blended probability')
    latest={}
    for ident,f in sorted(data['forecasts'].items(),key=lambda x:(x[1].get('version',0),x[1]['saved_at'],x[0])):
        if f['date']==data['date']:latest[str(f['game_pk'])]=(ident,f)
    ids={pk for pk,s in data['status'].items() if s.get('date')==data['date']}|set(latest)
    return latest,ids
def headline(data,ident,f):
    blend=(data.get('record') or {}).get('blend') or {}
    return blend.get(ident,f['home_win_probability'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='phone-check');ap.add_argument('--url',default=SITE);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    receipt={'started_at':datetime.now(timezone.utc).isoformat(),'site':a.url,'status':'started','private_data_or_secret_access':False,'checks':[]}
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser=pw.chromium.launch()
            for width in (390,1440):
                context=browser.new_context(viewport={'width':width,'height':1000})
                page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                response=page.goto(a.url,wait_until='networkidle',timeout=45000)
                assert response.status==200
                # Explicitly load the weights used by the body and headings.
                # fonts.ready waits for used faces; check(default 400) alone can
                # fail for an unused declared weight and can pass for a missing face.
                fonts=page.evaluate('''async () => {
                    const requests = ['400 16px "Atkinson Hyperlegible"',
                                      '700 24px "Barlow Condensed"'];
                    const before = Array.from(document.fonts, f => ({family:f.family,weight:f.weight,status:f.status}));
                    const loaded = await Promise.race([
                        Promise.all(requests.map(async spec => ({spec,faces:(await document.fonts.load(spec, 'Baseball 0123456789')).map(f=>({family:f.family,weight:f.weight,status:f.status}))}))),
                        new Promise((_,reject)=>setTimeout(()=>reject(new Error('Requested fonts did not load within 15 seconds')),15000))
                    ]);
                    return {before,loaded,body_family:getComputedStyle(document.body).fontFamily,
                            heading_family:getComputedStyle(document.querySelector('h1')).fontFamily};
                }''')
                receipt.setdefault('font_checks',[]).append({'width':width,**fonts})
                page.screenshot(path=str(out/f'font_evidence_{width}.png'))
                assert all(item['faces'] and all(face['status']=='loaded' for face in item['faces']) for item in fonts['loaded']),'Requested font faces unavailable'
                assert 'Atkinson Hyperlegible' in fonts['body_family'] and 'Barlow Condensed' in fonts['heading_family'],'Requested fonts not applied'
                # Deterministic screenshot positioning; do not alter page content.
                page.add_style_tag(content='html{scroll-behavior:auto!important}')
                def design_check(name):
                    result=page.evaluate('''() => {
                        const shown=e=>!!(e.offsetWidth||e.offsetHeight||e.getClientRects().length)&&getComputedStyle(e).visibility!=='hidden';
                        const text=Array.from(document.querySelectorAll('#app *')).filter(e=>shown(e)&&!e.closest('svg')&&Array.from(e.childNodes).some(n=>n.nodeType===3&&n.textContent.trim()));
                        const taps=Array.from(document.querySelectorAll('a,button,summary')).filter(shown);
                        return {minimum_text_px:Math.min(...text.map(e=>parseFloat(getComputedStyle(e).fontSize))),
                                minimum_tap_height:Math.min(...taps.map(e=>e.getBoundingClientRect().height)),
                                overflow:document.documentElement.scrollWidth>innerWidth};
                    }''')
                    receipt.setdefault('design_checks',[]).append({'width':width,'view':name,**result})
                    assert result['minimum_text_px']>=14, name+' contains small text'
                    assert result['minimum_tap_height']>=44, name+' has a small tap target'
                    assert not result['overflow'], name+' overflows'
                def go(hash_):
                    page.evaluate('h=>{location.hash=h}',hash_);page.wait_for_timeout(250)
                data=page.evaluate('window.BRL');latest,ids=inspect_data(data)
                assert data.get('view_scope')=='live','Deployed page is not the live lane'
                assert not page.evaluate('Boolean(document.body.dataset.refreshBlocked)'),'Refresh is blocked'
                go('#/')
                assert page.locator('.card[data-game]').count()==len(latest)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Horizontal page overflow'
                for ident,f in latest.values():
                    card=page.locator(f'[data-game="{f["game_pk"]}"]')
                    lv=(data.get('live') or {}).get(str(f['game_pk']))
                    p=lv['home_win_probability'] if lv and not lv.get('error') and str(f['game_pk']) not in data['actuals'] else headline(data,ident,f)
                    fav=f['home'] if p>=.5 else f['away']
                    if card.locator('span.state.live').count():
                        # In progress: the page follows the game from MLB's feed, so its number moves between saved runs.
                        assert card.count()==1 and re.search(r'\d+%',card.inner_text())
                        continue
                    assert card.count()==1 and f'{round(max(p,1-p)*100)}%' in card.inner_text() and fav['abbr'] in card.inner_text()
                design_check('slate')
                page.screenshot(path=str(out/f'slate_{width}.png'),full_page=True)
                for ident,f in latest.values():
                    pk=f['game_pk']
                    if ident not in data['box_scores']:continue
                    go(f'#/game/{pk}/summary')
                    assert page.locator('.hero').is_visible()
                    assert page.locator('table.line').count()==1,'Missing line score'
                    assert page.locator('button[data-variant]').count()>=3,'Missing simulated versions of the game'
                    assert page.locator('button[data-variant="proj"]').get_attribute('aria-pressed')=='true'
                    assert 'Starter not saved' not in page.inner_text('#app')
                    design_check('game_summary')
                    page.screenshot(path=str(out/f'game_summary_{width}.png'),full_page=True)
                    page.screenshot(path=str(out/f'game_header_{width}.png'))
                    page.locator('button[data-tab="box"]').click();page.wait_for_timeout(150)
                    # In-game and after the final the tabs open on the real game; the check reads the projected one.
                    if page.locator('button[data-boxside="proj"]').count():page.locator('button[data-boxside="proj"]').click();page.wait_for_timeout(150)
                    assert page.locator('table.box').count()==4,'Missing batting and pitching tables'
                    assert page.locator('table.box tbody tr').count()>=24,'Missing full batting/pitching boxes'
                    design_check('game_box');page.screenshot(path=str(out/f'game_box_{width}.png'),full_page=True)
                    page.locator('button[data-tab="plays"]').click();page.wait_for_timeout(150)
                    if page.locator('button[data-playside="proj"]').count():page.locator('button[data-playside="proj"]').click();page.wait_for_timeout(150)
                    assert page.locator('.half').count()>=17,'Missing half innings'
                    assert page.locator('.pa').count()>=50,'Missing plate appearances'
                    design_check('game_plays');page.screenshot(path=str(out/f'game_plays_{width}.png'),full_page=True)
                    page.locator('button[data-tab="odds"]').click();page.wait_for_timeout(150)
                    assert page.locator('.winbar').count()==1
                    assert 'Simulator alone' in page.inner_text('#app')
                    design_check('game_odds');page.screenshot(path=str(out/f'game_odds_{width}.png'),full_page=True)
                    page.locator('button[data-tab="summary"]').click();page.wait_for_timeout(150)
                    for key in ('high','low','upset'):
                        b=page.locator(f'button[data-variant="{key}"]')
                        if not b.count():continue
                        b.click();page.wait_for_timeout(150)
                        assert b.get_attribute('aria-pressed')=='true'
                        assert page.locator('table.line').count()==1
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    design_check('game_variant');page.screenshot(path=str(out/f'game_variant_{width}.png'))
                    if str(pk) in data['actuals'] and str(pk) in data['actual_boxes']:
                        page.locator('button[data-variant="proj"]').click();page.wait_for_timeout(150)
                        assert 'How close' in page.inner_text('#app'),'Missing How close section after the final'
                        page.screenshot(path=str(out/f'how_close_{width}.png'))
                    break
                go('#/record');assert 'Track record' in page.inner_text('#app')
                rec=data.get('record') or {}
                if rec.get('n_scored'):assert page.locator('.ladder li').count()>=5,'Missing record ladder'
                else:assert page.locator('.empty-state').count()==1
                design_check('record');page.screenshot(path=str(out/f'record_{width}.png'),full_page=True)
                go('#/how');assert 'How it works' in page.inner_text('#app')
                assert 'regular-season habits' in page.inner_text('#app')
                assert 'pitch-by-pitch' in page.inner_text('#app')
                design_check('how');page.screenshot(path=str(out/f'how_{width}.png'),full_page=True)
                # Players pages and the matchup report read model outputs from the data branch: recorded, with the design checks applied when they load; only page errors fail.
                try:
                    go('#/players');loaded=False
                    for _ in range(15):
                        page.wait_for_timeout(1000)
                        if page.locator('.pl-list li').count()>=20:loaded=True;break
                    info={'list_rows':page.locator('.pl-list li').count(),'loaded':loaded}
                    if loaded:
                        design_check('players');page.screenshot(path=str(out/f'players_{width}.png'),full_page=True)
                        href=page.locator('.pl-list a').first.get_attribute('href');go(href)
                        for _ in range(15):
                            page.wait_for_timeout(1000)
                            if page.locator('.rp-grid').count():break
                        info.update(card=href,card_grids=page.locator('.rp-grid').count(),card_text=re.sub(r'\s+',' ',page.inner_text('#app')[:240]))
                        if page.locator('.rp-grid').count():design_check('player_card');page.screenshot(path=str(out/f'player_card_{width}.png'),full_page=True)
                    receipt.setdefault('players',[]).append({'width':width,**info})
                except Exception as exc:
                    receipt.setdefault('players',[]).append({'width':width,'error':type(exc).__name__+': '+str(exc)[:200]})
                try:
                    f=next(iter(latest.values()))[1] if latest else None
                    if f:
                        go(f'#/game/{f["game_pk"]}/report');loaded=False
                        for _ in range(15):
                            page.wait_for_timeout(1000)
                            if page.locator('table.mu').count() or 'No matchup report' in page.inner_text('#app'):loaded=True;break
                        info={'game':f['game_pk'],'loaded':loaded,'tables':page.locator('table.mu').count(),'now_strip':page.locator('.rp-now').count()}
                        if page.locator('table.mu').count():
                            page.locator('td.rp-cell[style]').first.click();page.wait_for_timeout(300)
                            info['plan_grids']=page.locator('.rp-grid').count()
                            design_check('report');page.screenshot(path=str(out/f'report_{width}.png'),full_page=True)
                        receipt.setdefault('report',[]).append({'width':width,**info})
                except Exception as exc:
                    receipt.setdefault('report',[]).append({'width':width,'error':type(exc).__name__+': '+str(exc)[:200]})
                # Past dates: the season replay's cards come from the page's own files; the game opens from MLB's feed.
                go('#/day/2026-07-04');page.wait_for_timeout(2500)
                assert page.locator('nav.dstrip').count()==1,'Missing date strip'
                assert page.locator('a.card[href^="#/past/"]').count()>=10,'Missing past games'
                design_check('past_slate');page.screenshot(path=str(out/f'past_slate_{width}.png'),full_page=True)
                if width==390:
                    try:
                        href=page.locator('a.card[href^="#/past/"]').first.get_attribute('href')
                        page.evaluate('h=>{location.hash=h}',href)
                        loaded=False
                        for _ in range(20):
                            page.wait_for_timeout(1000)
                            if page.locator('table.line').count():loaded=True;break
                        info={'game':href,'line_score_from_mlb':loaded}
                        if loaded:
                            page.locator('button[data-tab="box"]').click();page.wait_for_timeout(300)
                            info['box_rows']=page.locator('table.box tbody tr').count()
                            page.locator('button[data-tab="odds"]').click();page.wait_for_timeout(300)
                            info['odds']='Betting market at the close' in page.inner_text('#app')
                        page.screenshot(path=str(out/'past_game_390.png'),full_page=True)
                        receipt['past_games']=info
                    except Exception as exc:
                        receipt['past_games']={'error':type(exc).__name__+': '+str(exc)[:200]}
                if width==390:
                    # Record (never fail on) whether the page followed a game in progress from MLB's feed in the browser.
                    try:
                        go('#/');page.wait_for_timeout(2500)
                        cards=page.locator('a.card:has(span.state.live)')
                        feed={'live_cards':cards.count()}
                        if cards.count():
                            pk=cards.first.get_attribute('data-game');feed['game']=pk
                            page.evaluate("pk => { location.hash = '#/game/' + pk + '/summary'; }",pk)
                            seen=False
                            for _ in range(30):
                                page.wait_for_timeout(1000)
                                text=page.inner_text('#app')
                                if 'Live from MLB' in text or 'Final from the official feed' in text:seen=True;break
                            feed.update(browser_feed_seen=seen,win_chart='Win chance through the game' in text,
                                        scoring_plays='Scoring plays so far' in text or 'Scoring plays' in text,
                                        this_at_bat=page.locator('.rp-now').count(),hero=re.sub(r'\s+',' ',text[:600]))
                            page.screenshot(path=str(out/'live_game_390.png'),full_page=True)
                        receipt['browser_feed']=feed
                    except Exception as exc:
                        receipt['browser_feed']={'error':type(exc).__name__+': '+str(exc)[:200]}
                assert not errors,errors
                receipt['checks'].append({'width':width,'font_faces_loaded':True,'no_overflow':True,'all_game_sections':True,'javascript_errors':errors})
                if all(k in data.get('scores',{}) for k in ('model_brier','team_brier','market_brier')):
                    receipt['three_brier_numbers']={k:data['scores'][k] for k in ('model_brier','team_brier','market_brier')}
                receipt['box_forecasts']=len(data['box_scores']);receipt['scored_games']=data.get('scores',{}).get('n_games')
                receipt['record']={k:rec.get(k) for k in ('n_scored','ladder')}
                context.close()
            browser.close()
        receipt['status']='PASS_DEPLOYED_BOX_SCORES'
    except Exception as exc:receipt.update(status='FAIL',error=type(exc).__name__+': '+str(exc)[:500])
    receipt['finished_at']=datetime.now(timezone.utc).isoformat();(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
    return 0 if receipt['status'].startswith('PASS') else 1
if __name__=='__main__':sys.exit(main())
