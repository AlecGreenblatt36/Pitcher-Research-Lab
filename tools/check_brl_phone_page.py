"""Read-only deployed page checks and phone screenshots; no private runtime/key."""
from __future__ import annotations
import argparse,json,math,re,sys
from datetime import datetime,timezone
from pathlib import Path
SITE='https://alecgreenblatt36.github.io/Pitcher-Research-Lab/'
ALLOWED_FORECAST_FIELDS=set('schema scope game_pk date game_type scheduled_start forecast_origin saved_at version lineup_status home away n_simulations seed home_win_probability probability_mcse projected_away_runs projected_home_runs team_baseline_probability market_probability model history_through postseason_regular_bullpen_logic automatic_runner github_run_id snapshot_hash'.split())
OLD_FIELDS=set('actuals date forecasts publications scores status'.split())
BOX_FIELDS=OLD_FIELDS|set('box_scores box_publications actual_boxes player_scores view_scope'.split())
def utc(text):
    d=datetime.fromisoformat(text.replace('Z','+00:00'))
    if d.tzinfo is None:raise ValueError('Timezone required')
    return d.astimezone(timezone.utc)
def inspect_data(data):
    if set(data) not in (OLD_FIELDS,BOX_FIELDS):raise ValueError('Unexpected public fields')
    for ident,f in data['forecasts'].items():
        if set(f)!=ALLOWED_FORECAST_FIELDS:raise ValueError('Unexpected forecast field')
        if f['n_simulations']!=10000:raise ValueError('Not a 10,000-world forecast')
        p=f['home_win_probability']
        if not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
        if not utc(f['forecast_origin'])<=utc(f['saved_at'])<utc(f['scheduled_start']):raise ValueError('Forecast timing invalid')
        publication=data['publications'].get(ident)
        if not publication or not re.fullmatch('[0-9a-f]{40}',publication['commit']):raise ValueError('Missing publication receipt')
        if utc(publication['published_at'])>=utc(f['scheduled_start']):raise ValueError('Late publication')
    latest={}
    for ident,f in sorted(data['forecasts'].items(),key=lambda x:(x[1]['saved_at'],x[0])):
        if f['date']==data['date']:latest[str(f['game_pk'])]=(ident,f)
    ids={pk for pk,s in data['status'].items() if s.get('date')==data['date']}|set(latest)
    return latest,ids

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
                page.evaluate('document.fonts.ready')
                assert page.evaluate('document.fonts.check(\'16px "Atkinson Hyperlegible"\') && document.fonts.check(\'24px "Barlow Condensed"\')'),'Requested fonts unavailable'
                data=page.evaluate('window.BRL');latest,ids=inspect_data(data)
                assert data.get('view_scope')=='live','Deployed page is not the live lane'
                assert not page.evaluate('Boolean(document.body.dataset.refreshBlocked)'),'Refresh is blocked'
                assert page.locator('.game-card').count()==len(latest)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Horizontal page overflow'
                for ident,f in latest.values():
                    card=page.locator('.game-card').filter(has_text=f['home']['name'])
                    assert card.count()==1 and f'{round(f["home_win_probability"]*100)}%' in card.inner_text()
                page.screenshot(path=str(out/f'slate_{width}.png'),full_page=True)
                if latest:
                    pk=next(iter(latest));page.locator(f'[data-game="{pk}"]').click()
                    assert page.locator('#game').is_visible()
                    assert page.locator('#game .sample-buttons button').count()==5,'Missing five full sample games'
                    assert page.locator('#game .player-row').count()>=20,'Missing full batting/pitching boxes'
                    assert 'Starter not saved' not in page.inner_text('#game')
                    page.screenshot(path=str(out/f'game_{width}.png'),full_page=True)
                    page.screenshot(path=str(out/f'game_header_{width}.png'))
                    for section in ('projectedBatting','projectedPitching'):
                        page.locator('#'+section).scroll_into_view_if_needed();page.screenshot(path=str(out/f'{section}_{width}.png'))
                    for i in range(5):
                        page.locator(f'[data-sample="{i}"]').click()
                        assert page.locator('#sample .play-half').count()>=17
                        assert page.locator('#sample .player-row').count()>=20
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                        if i==0:
                            page.screenshot(path=str(out/f'sample_{width}.png'),full_page=True)
                            page.screenshot(path=str(out/f'sample_header_{width}.png'))
                            page.locator('#sample .play-half').first.scroll_into_view_if_needed();page.screenshot(path=str(out/f'play_by_play_{width}.png'))
                        page.locator('#backGame').click()
                page.locator('#trackTab').click();assert page.locator('#track').is_visible()
                page.screenshot(path=str(out/f'track_{width}.png'),full_page=True)
                page.locator('#howTab').click();assert page.locator('#how').is_visible()
                assert 'regular-season bullpen' in page.inner_text('#how')
                assert 'pitch-by-pitch' in page.inner_text('#how')
                page.screenshot(path=str(out/f'how_{width}.png'),full_page=True)
                assert not errors
                receipt['checks'].append({'width':width,'font_faces_loaded':True,'no_overflow':True,'five_samples_opened':True,'all_game_sections':True,'javascript_errors':errors})
                receipt['three_brier_numbers']={k:data['scores'][k] for k in ('model_brier','team_brier','market_brier')}
                receipt['box_forecasts']=len(data['box_scores']);receipt['scored_games']=data['scores']['n_games']
                context.close()
            browser.close()
        receipt['status']='PASS_DEPLOYED_BOX_SCORES'
    except Exception as exc:receipt.update(status='FAIL',error=type(exc).__name__+': '+str(exc)[:500])
    receipt['finished_at']=datetime.now(timezone.utc).isoformat();(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
    return 0 if receipt['status'].startswith('PASS') else 1
if __name__=='__main__':sys.exit(main())
