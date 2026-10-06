"""Read-only normal-browser check of the deployed, prediction-only BRL page.

No key or private-runtime dependency. Writes public screenshots and a compact
receipt. It does not change forecasts, inspect raw data or trigger simulation.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re,sys
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen

SITE='https://alecgreenblatt36.github.io/Pitcher-Research-Lab/'
ALLOWED_FORECAST_FIELDS=set('schema scope game_pk date game_type scheduled_start forecast_origin saved_at version lineup_status home away n_simulations seed home_win_probability probability_mcse projected_away_runs projected_home_runs team_baseline_probability market_probability model history_through postseason_regular_bullpen_logic automatic_runner github_run_id snapshot_hash'.split())


def utc(text):
    value=datetime.fromisoformat(text.replace('Z','+00:00'))
    if value.tzinfo is None: raise ValueError('Timezone required')
    return value.astimezone(timezone.utc)


def inspect_data(data):
    if set(data)!=set('actuals date forecasts publications scores status'.split()):
        raise ValueError('Unexpected public top-level field')
    forecasts=data['forecasts'];today=data['date']
    for ident,f in forecasts.items():
        if set(f)!=ALLOWED_FORECAST_FIELDS: raise ValueError('Unexpected forecast field')
        if f['n_simulations']!=10000: raise ValueError('Not a 10,000-world forecast')
        p=f['home_win_probability']
        if not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:
            raise ValueError('Invalid probability')
        publication=data['publications'].get(ident)
        if not utc(f['forecast_origin'])<=utc(f['saved_at'])<utc(f['scheduled_start']):
            raise ValueError('Forecast timing invalid')
        if not publication or not re.fullmatch(r'[0-9a-f]{40}',publication['commit']):
            raise ValueError('Missing publication receipt')
        if utc(publication['published_at'])>=utc(f['scheduled_start']):
            raise ValueError('Late publication')
    latest={}
    for ident,f in sorted(forecasts.items(),key=lambda x:(x[1]['saved_at'],x[0])):
        if f['date']==today: latest[str(f['game_pk'])]=(ident,f)
    expected_ids={pk for pk,s in data['status'].items() if s.get('date')==today}|set(latest)
    return latest,expected_ids


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='phone-check')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'started_at':datetime.now(timezone.utc).isoformat(),'site':SITE,
            'status':'started','method':'Normal Playwright navigation of deployed HTTPS site',
            'private_data_or_secret_access':False,'checks':[]}
    try:
        req=Request(SITE+'predictions.json',headers={'Cache-Control':'no-cache','User-Agent':'BRL-public-check/1.0'})
        with urlopen(req,timeout=25) as response:
            raw=response.read(2_000_001);report['predictions_http_status']=response.status
        if len(raw)>2_000_000: raise ValueError('Public document exceeds check limit')
        data=json.loads(raw);latest,expected=inspect_data(data)
        report.update(public_json_sha256=hashlib.sha256(raw).hexdigest(),
                      forecast_versions=len(data['forecasts']),date=data['date'],
                      games_with_forecasts=len(latest),scored_games=data['scores']['n_games'],
                      latest_forecast_ids=[v[0] for v in latest.values()],
                      three_brier_numbers={k:data['scores'][k] for k in ('model_brier','team_brier','market_brier')})
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser=pw.chromium.launch()
            try:
                for width in (1440,390):
                    context=browser.new_context(viewport={'width':width,'height':1000})
                    page=context.new_page();errors=[]
                    page.on('pageerror',lambda error:errors.append(str(error)))
                    response=page.goto(SITE,wait_until='networkidle',timeout=30000)
                    assert response and response.status==200,'Page HTTP status'
                    assert 'Refresh blocked' not in page.inner_text('main'),'Published page is stale/fallback'
                    assert page.locator('.card').count()==len(expected),'Unexpected slate cards'
                    assert 'Away team' not in page.inner_text('#slate'),'Unnamed old result card'
                    for pk,(ident,f) in latest.items():
                        card=page.locator('.card').filter(has_text=f['home']['name'])
                        assert card.count()==1,'Game card identity'
                        assert f['away']['name'] in card.inner_text(),'Away team'
                        assert f"{f['home_win_probability']:.1%}" in card.inner_text(),'Win probability'
                        assert f"{f['projected_away_runs']:.1f} – {f['projected_home_runs']:.1f}" in card.inner_text(),'Projected score'
                        if f['postseason_regular_bullpen_logic']:
                            assert 'regular-season bullpen logic' in card.inner_text(),'Postseason disclosure'
                        card.locator('summary').click()
                        count=sum(str(v['game_pk'])==pk for v in data['forecasts'].values())
                        assert card.locator('.history table tr').count()==count+1,'Version retention'
                    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),'Horizontal overflow'
                    page.screenshot(path=str(out/f'slate_{width}.png'),full_page=True)
                    page.locator('#howTab').click()
                    assert page.locator('#how').is_visible() and not page.locator('#slate').is_visible()
                    assert 'history' in page.inner_text('#how').lower()
                    page.screenshot(path=str(out/f'how_{width}.png'),full_page=True)
                    page.locator('#slateTab').click()
                    assert page.locator('#slate').is_visible() and not errors
                    report['checks'].append({'width':width,'page_http_status':200,
                         'game_cards_match_public_json':True,'versions_retained':True,
                         'tabs_work':True,'no_horizontal_overflow':True,'javascript_errors':errors})
                    context.close()
            finally: browser.close()
        report['status']='PASS_DEPLOYED_PHONE_AND_DESKTOP'
    except Exception as exc:
        report.update(status='FAIL',error=type(exc).__name__+': '+str(exc)[:400])
    report['finished_at']=datetime.now(timezone.utc).isoformat()
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status'].startswith('PASS') else 1

if __name__=='__main__':sys.exit(main())
