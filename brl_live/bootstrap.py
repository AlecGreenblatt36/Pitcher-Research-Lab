"""Encrypted BRL runtime bootstrap. Only the generated site is publishable.

No credentials or raw player data are contained in this public source file.
The key is a repository Actions secret, never a command argument or log value.
"""
from __future__ import annotations
import argparse,hashlib,html,io,json,os,re,subprocess,sys,zipfile,stat,base64
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

TAG='brl-live-runtime-20261006'
ASSET='brl-live-runtime-20261006.zip.enc'

def fetch(url,token=None,method='GET',value=None):
    headers={'User-Agent':'BRL-encrypted-runtime/1.0','Accept':'application/vnd.github+json'}
    if token:headers['Authorization']='Bearer '+token
    data=None if value is None else json.dumps(value).encode()
    if data is not None:headers['Content-Type']='application/json'
    with urlopen(Request(url,headers=headers,data=data,method=method),timeout=35) as response:
        raw=response.read(100_000_001)
        if len(raw)>100_000_000:raise ValueError('Response too large')
        return raw

def api(url,token,method='GET',value=None):return json.loads(fetch(url,token,method,value))

def restore(ciphertext,key_hex,config,destination):
    if not re.fullmatch('[a-fA-F0-9]{64}',key_hex.strip()):raise ValueError('Invalid key format')
    if hashlib.sha256(ciphertext).hexdigest()!=config['cipher_sha256']:raise ValueError('Encrypted package hash mismatch')
    if ciphertext[:8]!=b'BRLAESG1':raise ValueError('Package format mismatch')
    raw=AESGCM(bytes.fromhex(key_hex.strip())).decrypt(ciphertext[8:20],ciphertext[20:],b'BRL:runtime')
    if hashlib.sha256(raw).hexdigest()!=config['plaintext_sha256']:raise ValueError('Decrypted package hash mismatch')
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=set();size=0
        for item in z.infolist():
            p=(destination/item.filename).resolve();size+=item.file_size
            if not p.is_relative_to(destination) or '\\' in item.filename or item.filename in names or stat.S_ISLNK(item.external_attr>>16):raise ValueError('Unsafe package path')
            names.add(item.filename)
        if size>500_000_000:raise ValueError('Expanded package too large')
        z.extractall(destination)
    return destination

def setup_page(directory,message,games):
    esc=lambda x:html.escape(str(x),quote=True)
    cards=''.join('<article><small>Waiting for forecast</small><h2>'+esc(g['away'])+'<br>'+esc(g['home'])+'</h2><p>Win chance — · Projected score —</p><p>'+esc(g['scheduled_start'])+'</p><p>Postseason · regular-season bullpen logic</p></article>' for g in games)
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Baseball Research Lab · Daily slate</title><style>*{box-sizing:border-box}body{margin:0;background:#101917;color:#edf5ec;font:16px/1.6 system-ui}main{max-width:1000px;margin:auto;padding:30px 20px}h1{font-size:38px;margin-bottom:0}small{color:#bbe578}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,290px),1fr));gap:18px}article{border:1px solid #314c3e;border-radius:16px;padding:22px;background:#172521}h2{font-size:22px}button{font:inherit;border:1px solid #49634e;background:#bbe578;color:#101917;padding:10px 18px;border-radius:22px;cursor:pointer;margin:20px 8px 20px 0}.notice{padding:14px;border-left:3px solid #bbe578;background:#172521}p{color:#b3c5b8}.metrics{margin:24px 0}a{color:#bbe578}</style><main><small>BASEBALL RESEARCH LAB</small><h1>Today's games</h1><p>Pregame forecasts. Every version kept.</p><button onclick="show('cards')">Game cards</button><button onclick="show('how')">How to read this</button><div class="notice">'''+esc(message)+'''</div><section id="cards"><div class="cards">'''+cards+'''</div><div class="metrics">Live scored games: 0<br>Model Brier — · Team baseline — · Market —</div></section><section id="how" hidden><h2>Setup is not complete</h2><p>No live forecasts have been created. The encrypted runtime asset and its decryption secret must both be installed before the scheduled job can simulate games. Never upload the plaintext model archive or the secret key to this public repository.</p><p>Once active, checks run every 15 minutes. GitHub can delay or drop scheduled jobs. Forecasts that finish too late are withheld, not backdated. Earlier projected-lineup forecasts are retained when official lineups arrive.</p><p>Postseason games use regular-season bullpen logic; no automatic runner is used in postseason extra innings. The initial locked player history ends September 27. It is not represented as a current full-data refresh.</p><p>Historical replay scores are separate from live scores. The current anchor used 2025 calibration, so a 2025 replay is not a clean accuracy test.</p></section></main><script>function show(v){document.getElementById('cards').hidden=v!=='cards';document.getElementById('how').hidden=v!=='how'}</script></html>'''
    p=Path(directory);p.mkdir(parents=True,exist_ok=True);(p/'index.html').write_text(doc);(p/'.nojekyll').write_text('')
    (p/'predictions.json').write_text(json.dumps({'status':'setup_blocked','message':message,'games':games,'forecasts':[],'live_scored_games':0},indent=2))

def setup_prerequisites(repo,token):
    base='https://api.github.com/repos/'+repo;info={}
    try:api(base+'/releases/tags/'+TAG,token);info['release']='exists'
    except HTTPError as exc:
        if exc.code!=404:raise
        api(base+'/releases',token,'POST',{'tag_name':TAG,'target_commitish':'main','name':'BRL encrypted live runtime','body':'Upload only brl-live-runtime-20261006.zip.enc. Never upload the plaintext package or key. This release contains no published raw player data.','draft':False,'prerelease':True})
        info['release']='created_empty_for_encrypted_asset'
    try:
        pages=api(base+'/pages',token);info['pages']=pages.get('build_type','unknown');info['pages_url']=pages.get('html_url')
    except HTTPError as exc:
        if exc.code!=404:info['pages']='unavailable_http_'+str(exc.code);return info
        try:
            pages=api(base+'/pages',token,'POST',{'build_type':'workflow'})
            info['pages']='workflow';info['pages_url']=pages.get('html_url')
        except HTTPError as denied:info['pages']='enable_in_settings_http_'+str(denied.code)
    return info

def main():
    p=argparse.ArgumentParser();p.add_argument('--site',required=True);p.add_argument('--runtime',required=True);p.add_argument('--config',default='brl_live/runtime.json');args=p.parse_args()
    repo=os.environ['GITHUB_REPOSITORY'];token=os.environ['GH_TOKEN'];now=datetime.now(timezone.utc)
    receipt={'created_at':now.isoformat(),'status':'starting','live_forecasts_created':0,'raw_data_published':False}
    games=[]
    try:
        # This is schedule metadata only, never a raw pitch or player-data dump.
        from zoneinfo import ZoneInfo
        day=now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
        data=api('https://statsapi.mlb.com/api/v1/schedule?sportId=1&date='+day,None)
        games=[{'game_pk':g['gamePk'],'scheduled_start':g['gameDate'],'away':g['teams']['away']['team']['name'],'home':g['teams']['home']['team']['name']} for d in data.get('dates',[]) for g in d.get('games',[])]
        receipt['scheduled_games']=games
        receipt['setup']=setup_prerequisites(repo,token)
        key=os.environ.get('BRL_PA_PACKAGE_KEY','')
        if not key:raise ValueError('Activation needed: add BRL_PA_PACKAGE_KEY and upload the encrypted runtime asset.')
        config=json.loads(Path(args.config).read_text())
        source=f'https://github.com/{repo}/releases/download/{TAG}/{ASSET}'
        runtime=restore(fetch(source),key,config,args.runtime)
        # Package installation receives no decryption key or repository credential.
        env={k:v for k,v in os.environ.items() if k not in ('BRL_PA_PACKAGE_KEY','GH_TOKEN','GITHUB_TOKEN')}
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--quiet','-r',str(runtime/'requirements-cloud.txt')],env=env,check=True)
        sys.path.insert(0,str(runtime))
        from cloud.runner import main as run_iteration
        sys.argv=['cloud.runner','--public-dir',args.site]
        receipt.update(run_iteration());receipt['status']='iteration_completed'
    except Exception as exc:
        message=str(exc) if isinstance(exc,ValueError) else type(exc).__name__+': encrypted runtime/source unavailable; no forecast fabricated.'
        receipt['status']='blocked';receipt['reason']=message
        preserved=False
        try:
            prior={}
            for name in ('index.html','predictions.json','.nojekyll'):
                item=api('https://api.github.com/repos/'+repo+'/contents/public/'+name+'?ref=brl-live-data',token)
                if item.get('encoding')!='base64':raise ValueError('Prior public page unavailable')
                prior[name]=base64.b64decode(item['content'])
            Path(args.site).mkdir(parents=True,exist_ok=True)
            for name,raw in prior.items():(Path(args.site)/name).write_bytes(raw)
            page=Path(args.site)/'index.html';page.write_text(page.read_text().replace('<main>','<main><div class="notice">Refresh blocked; showing the last saved forecasts. '+html.escape(message)+'</div>',1))
            preserved=True
        except Exception:setup_page(args.site,message,games)
        receipt['preserved_previous_forecasts']=preserved
        print('::warning::BRL live forecasts blocked; see public activation status and receipt.')
    # A whitelist protects deployment even when a runtime command fails.
    files={p.name for p in Path(args.site).iterdir() if p.is_file()}
    if files!={'index.html','predictions.json','.nojekyll'} or any(p.is_dir() for p in Path(args.site).iterdir()):raise RuntimeError('Public artifact allowlist failed')
    Path('brl_live_run_receipt.json').write_text(json.dumps(receipt,indent=2))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('pages_ready='+str(receipt.get('setup',{}).get('pages')=='workflow').lower()+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
