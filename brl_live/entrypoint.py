"""Scheduled launcher for the encrypted runtime plus the reviewed refresh extension."""
from __future__ import annotations
import argparse,base64,html,json,os,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo

# The launcher lives beside bootstrap in the repository; the raw runtime does not.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from brl_live.bootstrap import api,fetch,restore,setup_page,setup_prerequisites,TAG,ASSET


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site',required=True);parser.add_argument('--runtime',required=True)
    parser.add_argument('--config',default='brl_live/runtime.json')
    args=parser.parse_args()
    repo=os.environ['GITHUB_REPOSITORY'];token=os.environ['GH_TOKEN'];now=datetime.now(timezone.utc)
    receipt={'created_at':now.isoformat(),'status':'starting','live_forecasts_created':0,'raw_data_published':False,'refresh_extension':'prior-day-v1','stage':'source_setup','secret_present':False,'runtime_authenticated_decryption':False}
    games=[]
    try:
        day=now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
        schedule=api('https://statsapi.mlb.com/api/v1/schedule?sportId=1&date='+day,None)
        games=[{'game_pk':g['gamePk'],'scheduled_start':g['gameDate'],'away':g['teams']['away']['team']['name'],'home':g['teams']['home']['team']['name']} for d in schedule.get('dates',[]) for g in d.get('games',[])]
        receipt['scheduled_games']=games;receipt['setup']=setup_prerequisites(repo,token)
        receipt['stage']='runtime_restore'
        key=os.environ.get('BRL_PA_PACKAGE_KEY','')
        receipt['secret_present']=bool(key)
        if not key:raise ValueError('Activation needed: add BRL_PA_PACKAGE_KEY and upload the encrypted runtime asset.')
        config=json.loads(Path(args.config).read_text())
        runtime=restore(fetch(f'https://github.com/{repo}/releases/download/{TAG}/{ASSET}'),key,config,args.runtime)
        receipt['runtime_authenticated_decryption']=True
        receipt['stage']='runtime_dependencies'
        env={k:v for k,v in os.environ.items() if k not in ('BRL_PA_PACKAGE_KEY','GH_TOKEN','GITHUB_TOKEN')}
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--quiet','-r',str(runtime/'requirements-cloud.txt')],env=env,check=True)
        sys.path.insert(0,str(runtime))
        from brl_live.live_extension import main as run_iteration
        receipt['stage']='history_and_forecast_iteration'
        receipt.update(run_iteration(args.site));receipt['status']='iteration_completed';receipt['stage']='complete'
    except Exception as exc:
        # Do not print exception bodies from networking, source payloads or key operations.
        reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__+': runtime/source not ready; forecast not fabricated.'
        receipt.update(status='blocked',reason=reason,error_type=type(exc).__name__)
        frames=traceback.extract_tb(exc.__traceback__)
        if frames:receipt['error_location']={'file':Path(frames[-1].filename).name,'function':frames[-1].name,'line':frames[-1].lineno}
        preserved=False
        try:
            prior={}
            for name in ('index.html','predictions.json','.nojekyll'):
                value=api(f'https://api.github.com/repos/{repo}/contents/public/{name}?ref=brl-live-data',token)
                if value.get('encoding')!='base64':raise ValueError('Prior public page unavailable')
                prior[name]=base64.b64decode(value['content'],validate=False)
            Path(args.site).mkdir(parents=True,exist_ok=True)
            for name,raw in prior.items():(Path(args.site)/name).write_bytes(raw)
            page=Path(args.site)/'index.html'
            page.write_text(page.read_text().replace('<main>','<main><div class="notice">Refresh blocked; showing the last saved forecasts. '+html.escape(reason)+'</div>',1))
            preserved=True
        except Exception:setup_page(args.site,reason,games)
        receipt['preserved_previous_forecasts']=preserved
        print('::warning::BRL live worker blocked; see receipt. A successful setup step is not a successful forecast.')
    public=Path(args.site)
    if {p.name for p in public.iterdir() if p.is_file()}!={'index.html','predictions.json','.nojekyll'} or any(p.is_dir() for p in public.iterdir()):
        raise RuntimeError('Public output file allowlist failed')
    Path('brl_live_run_receipt.json').write_text(json.dumps(receipt,indent=2))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as stream:
            stream.write('pages_ready='+str(receipt.get('setup',{}).get('pages')=='workflow').lower()+'\n')
    print(json.dumps(receipt,indent=2))
    return receipt

if __name__=='__main__':main()
