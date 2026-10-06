"""Bounded official-source capture for an isolated postseason-policy experiment.
Raw payloads are encrypted before persistent writes; logs contain counts only.
No model fitting, forecast publication, or changes to the live data branch.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
import gzip, hashlib, json, os, secrets, time, urllib.request
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

OUT=Path('sprint1_capture');OUT.mkdir(exist_ok=True)
KEY=bytes.fromhex(os.environ['BRL_PA_PACKAGE_KEY'].strip())

def utc(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def get(url):
    errors=[]
    for attempt in range(2):
        started=datetime.now(timezone.utc).isoformat()
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'BRL-postseason-policy/1.0'}),timeout=25) as r:
                raw=r.read(18_000_001)
                if r.status!=200 or len(raw)>18_000_000:raise ValueError('response size/status')
            return json.loads(raw),{'url':url,'capture_started_at':started,'captured_at':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(raw).hexdigest(),'attempt':attempt+1}
        except Exception as e:
            errors.append(type(e).__name__)
            if attempt==0:time.sleep(1)
    raise RuntimeError('Source unavailable after two attempts: '+','.join(errors))

def fetch_game(item):
    pk=item['gamePk'];out={'game_pk':pk,'schedule':item}
    base=f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
    try:
        final,rec=get(base);out.update(final=final,final_receipt=rec)
        if final.get('gamePk')!=pk:raise ValueError('game identity')
        pitches=[e['startTime'] for p in final.get('liveData',{}).get('plays',{}).get('allPlays',[]) for e in p.get('playEvents',[]) if e.get('isPitch') and e.get('startTime')]
        if not pitches:raise ValueError('missing first pitch')
        out['first_pitch']=min(pitches,key=utc)
        origin=utc(item['gameDate'])-timedelta(minutes=15)
        out['forecast_origin']=origin.isoformat()
        if origin>=utc(out['first_pitch']):raise ValueError('fixed origin not before first pitch')
        stamps,stamp_rec=get(base+'/timestamps')
        eligible=[]
        for s in stamps:
            try:
                t=datetime.strptime(s,'%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
                if t<=origin:eligible.append((t,s))
            except (ValueError,TypeError):pass
        if not eligible:raise ValueError('no pre-origin timecode')
        stamp=max(eligible)[1]
        pre,prec=get(base+'?timecode='+stamp)
        if pre.get('gamePk')!=pk:raise ValueError('pregame identity')
        if pre.get('gameData',{}).get('status',{}).get('abstractGameState')!='Preview':raise ValueError('snapshot is not Preview')
        for play in pre.get('liveData',{}).get('plays',{}).get('allPlays',[]):
            if any(e.get('isPitch') or e.get('details',{}).get('isScoringPlay') for e in play.get('playEvents',[])):raise ValueError('pregame snapshot has game events')
        out.update(pregame=pre,pregame_receipt=prec,timecode=stamp,status='captured')
    except Exception as e:out.update(status='blocked',error=str(e)[:200])
    raw=gzip.compress(canonical(out),mtime=0);nonce=secrets.token_bytes(12)
    cipher=b'SPR1GCM1'+nonce+AESGCM(KEY).encrypt(nonce,raw,('BRL:sprint1:'+str(pk)).encode())
    (OUT/(str(pk)+'.enc')).write_bytes(cipher)
    return {'game_pk':pk,'year':item['officialDate'][:4],'status':out['status'],'error':out.get('error'),'cipher_sha256':hashlib.sha256(cipher).hexdigest()}

def main():
    all_games={};schedule_receipts=[]
    for year in (2023,2024,2025):
        data,rec=get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}&startDate={year}-01-01&endDate={year}-12-31&gameTypes=F,D,L,W')
        schedule_receipts.append(rec)
        for day in data.get('dates',[]):
            for g in day.get('games',[]):
                if g['gameType'] in ('F','D','L','W') and g['status']['abstractGameState']=='Final':
                    if g['gamePk'] in all_games:raise ValueError('duplicate schedule game')
                    all_games[g['gamePk']]=g
    if not 60<=len(all_games)<=160:raise ValueError('unexpected postseason coverage count')
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(fetch_game,[all_games[k] for k in sorted(all_games)]))
    receipt={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'historical official-source capture, not prospective forecasts','schedule_receipts':schedule_receipts,'n_games':len(results),'games':results,'raw_published':False,'key_logged':False}
    (OUT/'receipt.json').write_bytes(canonical(receipt))
    print(json.dumps({'n_games':len(results),'captured':sum(r['status']=='captured' for r in results),'blocked':sum(r['status']=='blocked' for r in results)}))
if __name__=='__main__':main()
