"""Bounded read-only official-MLB source check; no fitting or forecast creation."""
import gzip, hashlib, json, datetime, pathlib, urllib.request
OUT=pathlib.Path('source_probe');OUT.mkdir(exist_ok=True)
receipts=[]
def fetch(name,url):
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    rec={'name':name,'url':url,'requested_at':start}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'BaseballResearchLab-source-check/1.0'})
        with urllib.request.urlopen(req,timeout=20) as response:
            raw=response.read(15000001)
            rec.update(http_status=response.status,content_type=response.headers.get('Content-Type'),final_url=response.url)
        if len(raw)>15000000:raise ValueError('response exceeds 15MB')
        data=json.loads(raw)
        with gzip.open(OUT/(name+'.json.gz'),'wb') as f:f.write(raw)
        rec.update(status='RETRIEVED_JSON',sha256=hashlib.sha256(raw).hexdigest(),n_bytes=len(raw))
    except Exception as exc:
        rec.update(status='FAILED',error=type(exc).__name__+': '+str(exc));data=None
    rec['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipts.append(rec)
    print(json.dumps(rec));return data

def main():
    today=datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    fetch('current_schedule','https://statsapi.mlb.com/api/v1/schedule?sportId=1&date='+today)
    for pk in (849830,822679):
        url=f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
        final=fetch(f'{pk}_current',url)
        if not final:continue
        if final.get('gamePk')!=pk:continue
        gd=final.get('gameData',{})
        print(json.dumps({'game_pk':pk,'datetime':gd.get('datetime'),'status':gd.get('status'),'teams':{s:gd.get('teams',{}).get(s,{}).get('name') for s in ('away','home')},'linescore':final.get('liveData',{}).get('linescore',{}).get('teams')}))
        ts=fetch(f'{pk}_timestamps',url+'/timestamps')
        firsts=[e['startTime'] for p in final.get('liveData',{}).get('plays',{}).get('allPlays',[]) for e in p.get('playEvents',[]) if e.get('isPitch') and e.get('startTime')]
        if not firsts or not isinstance(ts,list):continue
        first=min(datetime.datetime.fromisoformat(v.replace('Z','+00:00')) for v in firsts)
        before=[]
        for stamp in ts:
            try:
                t=datetime.datetime.strptime(stamp,'%Y%m%d_%H%M%S').replace(tzinfo=datetime.timezone.utc)
                if t<first-datetime.timedelta(minutes=1):before.append((t,stamp))
            except (TypeError,ValueError):pass
        if before:
            stamp=max(before)[1]
            pre=fetch(f'{pk}_pregame_'+stamp,url+'?timecode='+stamp)
            if pre:
                print(json.dumps({'game_pk':pk,'timecode':stamp,'metaData':pre.get('metaData'),'status':pre.get('gameData',{}).get('status'),'probablePitchers':pre.get('gameData',{}).get('probablePitchers'),'n_plays':len(pre.get('liveData',{}).get('plays',{}).get('allPlays',[]))}))
    doc={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Official source accessibility and archive inspection only; no forecasts or model evaluation','requests':receipts}
    (OUT/'receipt.json').write_text(json.dumps(doc,indent=2))
if __name__=='__main__':main()
