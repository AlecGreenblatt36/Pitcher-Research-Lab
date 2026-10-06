"""Daily public-source history refresh, private storage, no model fitting.

Each completed day is reconciled against MLB's official PA identities/results.
The full CSV and official feeds go only into authenticated encrypted snapshots.
The immutable packaged history is NEVER overwritten. A separate combined file
is materialized in the ephemeral private runtime for current forecasts only.
"""
from __future__ import annotations
import base64, gzip, hashlib, io, json, math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

SCHEMA = 'brl.daily-history.v1'
VALID_TYPES = {'R', 'F', 'D', 'L', 'W'}
# Exactly the retained seven-outcome mapping; no new category or fitting.
EVENTS = {
 'strikeout':'K','strikeout_double_play':'K','walk':'BB_HBP','intent_walk':'BB_HBP',
 'hit_by_pitch':'BB_HBP','single':'1B','double':'2B_3B','triple':'2B_3B',
 'home_run':'HR','field_out':'BIP_OUT','force_out':'BIP_OUT',
 'grounded_into_double_play':'BIP_OUT','fielders_choice_out':'BIP_OUT',
 'double_play':'BIP_OUT','triple_play':'BIP_OUT','sac_fly':'BIP_OUT',
 'sac_bunt':'BIP_OUT','sac_fly_double_play':'BIP_OUT','field_error':'OTHER_REACH',
 'fielders_choice':'OTHER_REACH','catcher_interf':'OTHER_REACH'}
EXCLUDED = {'pickoff_1b','pickoff_2b','pickoff_3b','caught_stealing_2b',
 'caught_stealing_3b','caught_stealing_home','stolen_base_2b','stolen_base_3b',
 'stolen_base_home','pickoff_caught_stealing_2b','pickoff_caught_stealing_3b','pickoff_caught_stealing_home','wild_pitch','passed_ball','balk','game_advisory','no_play'}
REQUIRED = {'game_date','game_pk','at_bat_number','pitch_number','batter','pitcher',
 'events','stand','p_throws','home_team','away_team','inning','inning_topbot',
 'outs_when_up','on_1b','on_2b','on_3b','home_score','away_score','game_type'}

class RefreshBlocked(ValueError):
    """Only controlled, non-player-specific error text may be public."""

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(raw): return hashlib.sha256(raw).hexdigest()

def timestamp(s):
    t=datetime.fromisoformat(str(s).replace('Z','+00:00'))
    if t.tzinfo is None: raise RefreshBlocked('Source time requires a timezone')
    return t.astimezone(timezone.utc)

def yesterday(now):
    if now.tzinfo is None: raise RefreshBlocked('Clock needs timezone')
    return now.astimezone(ZoneInfo('America/New_York')).date()-timedelta(days=1)

def csv_url(day):
    return 'https://baseballsavant.mlb.com/statcast_search/csv?'+urlencode({
     'all':'true','type':'details','player_type':'pitcher','group_by':'name',
     'hfGT':'R|F|D|L|W|','game_date_gt':day,'game_date_lt':day,
     'min_pitches':'0','min_results':'0','min_abs':'0'})

class PublicSource:
    def __init__(self,clock=None):
        self.clock=clock or (lambda:datetime.now(timezone.utc))
    def get(self,url):
        started=self.clock().isoformat()
        allowed=('https://statsapi.mlb.com/','https://baseballsavant.mlb.com/')
        if not url.startswith(allowed): raise RefreshBlocked('Non-allowlisted history source')
        with urlopen(Request(url,headers={'User-Agent':'BRL-daily-history/1.0'}),timeout=60) as response:
            if response.status!=200 or not response.url.startswith(allowed):
                raise RefreshBlocked('History source response/redirect rejected')
            raw=response.read(30_000_001)
        if len(raw)>30_000_000: raise RefreshBlocked('History source exceeds daily size limit')
        return raw,{'url':url,'requested_at':started,'finished_at':self.clock().isoformat(),
                    'sha256':digest(raw),'n_bytes':len(raw),'http_status':200}


def official_labels(feed,day,captured):
    gd=feed['gameData'];live=feed['liveData'];pk=int(feed['gamePk'])
    if gd['datetime']['officialDate']!=day or gd['game']['type'] not in VALID_TYPES:
        raise RefreshBlocked('Wrong history date/game type')
    if gd['status'].get('detailedState') not in ('Final','Completed Early'):
        raise RefreshBlocked('History game is not a completed final')
    totals={}
    for side in ('away','home'):
        a=live['linescore']['teams'][side]['runs']
        b=live['boxscore']['teams'][side]['teamStats']['batting']['runs']
        if type(a) is not int or a!=b or a<0: raise RefreshBlocked('History final score reconciliation failed')
        totals[side]=a
    if totals['away']==totals['home']:raise RefreshBlocked('Tied final history game unsupported')
    labels={};endings=[];count_by_side={'away':0,'home':0}
    for play in live['plays']['allPlays']:
        event=play.get('result',{}).get('eventType')
        if not play.get('about',{}).get('isComplete'):
            continue
        if event in EXCLUDED:continue
        if event not in EVENTS:raise RefreshBlocked('Unknown completed official PA outcome')
        k=(pk,int(play['about']['atBatIndex'])+1)
        if k in labels:raise RefreshBlocked('Duplicate official PA identity')
        ended=timestamp(play['about']['endTime'])
        if ended>captured:raise RefreshBlocked('History label not resolved before capture')
        endings.append(ended)
        labels[k]=(EVENTS[event],int(play['matchup']['batter']['id']),int(play['matchup']['pitcher']['id']))
        count_by_side['away' if play['about']['halfInning'].lower()=='top' else 'home']+=1
    if not labels:raise RefreshBlocked('Final game has no completed PAs')
    for side in ('away','home'):
        n=live['boxscore']['teams'][side]['teamStats']['batting'].get('plateAppearances')
        if type(n) is not int or n!=count_by_side[side]:
            raise RefreshBlocked('Official PA count disagrees with box score')
    row={'game_pk':pk,'date':day,'game_type':gd['game']['type'],
         'away_id':int(gd['teams']['away']['id']),'home_id':int(gd['teams']['home']['id']),
         'away_runs':totals['away'],'home_runs':totals['home'],
         'label_resolved_at':max(endings).isoformat()}
    return labels,row


def collapse_verified(raw,day,labels,game_types,available_at):
    if not labels:
        # Empty-day input must be empty/CSV with zero completed PA rows.
        if not raw.strip():return pd.DataFrame(),{'pitch_rows':0,'pa_rows':0,'excluded_terminal_rows':0}
    try:frame=pd.read_csv(io.BytesIO(raw),low_memory=False)
    except pd.errors.EmptyDataError:
        if labels:raise RefreshBlocked('Empty Statcast response for played games')
        return pd.DataFrame(),{'pitch_rows':0,'pa_rows':0,'excluded_terminal_rows':0}
    if not REQUIRED<=set(frame):raise RefreshBlocked('Statcast required schema missing')
    if frame.empty and not labels:return pd.DataFrame(),{'pitch_rows':0,'pa_rows':0,'excluded_terminal_rows':0}
    if not frame['game_date'].astype(str).str[:10].eq(day).all():raise RefreshBlocked('Statcast returned another date')
    for col in ('game_pk','at_bat_number','pitch_number','batter','pitcher','inning','outs_when_up'):
        v=pd.to_numeric(frame[col],errors='coerce')
        if v.isna().any() or not np.isfinite(v).all() or (v!=np.floor(v)).any():
            raise RefreshBlocked('Statcast identifier/state is not finite integer')
        frame[col]=v.astype('int64')
    keys=['game_pk','at_bat_number'];pkkeys=keys+['pitch_number']
    if frame.duplicated(pkkeys).any():raise RefreshBlocked('Duplicate Statcast pitch identity')
    if not frame['stand'].isin(['R','L']).all() or not frame['p_throws'].isin(['R','L']).all():
        raise RefreshBlocked('Missing/unresolved pitch handedness')
    if any(game_types.get(int(pk))!=gt for pk,gt in zip(frame.game_pk,frame.game_type)):
        raise RefreshBlocked('Statcast games disagree with completed official slate')
    frame=frame.sort_values(pkkeys,kind='mergesort')
    terminal=frame[frame['events'].notna()].copy()
    if terminal.duplicated(keys).any():raise RefreshBlocked('Multiple terminal events per PA')
    unknown=set(terminal.events)-set(EVENTS)-EXCLUDED
    if unknown:raise RefreshBlocked('Unknown completed Statcast outcome')
    excluded=int(terminal.events.isin(EXCLUDED).sum())
    terminal=terminal[terminal.events.isin(EVENTS)].copy()
    found={(int(row.game_pk),int(row.at_bat_number)):(EVENTS[row.events],int(row.batter),int(row.pitcher))
           for row in terminal.itertuples()}
    if found!=labels:raise RefreshBlocked('Statcast/official PA identity, outcome or participant mismatch')
    first=frame.groupby(keys,sort=False).head(1).drop(columns=['events'])
    last=terminal[keys+['events','pitch_number']].rename(columns={'events':'terminal_event','pitch_number':'pa_pitches'})
    pa=first.merge(last,on=keys,how='inner',validate='one_to_one')
    # Match the original anchor history representation; no EV/LA from this PA enters a forecast.
    pa['outcome']=pa['terminal_event'].map(EVENTS)
    pa['season']=int(day[:4]);pa['date_key']=day;pa['game_date']=day
    pa['platoon']=(pa.stand==pa.p_throws).astype(int)
    pa['is_home_batter']=(pa.inning_topbot.str.lower()=='bot').astype(int)
    for base in ('1b','2b','3b'):pa['runner_'+base]=pa['on_'+base].notna().astype(int)
    if 'bat_score_diff' not in pa:
        pa['bat_score_diff']=np.where(pa.is_home_batter==1,pa.home_score-pa.away_score,pa.away_score-pa.home_score)
    for col,default,lo,hi in [('bat_score_diff',0,-10,10),('outs_when_up',0,0,2),('inning',1,1,20),
       ('n_thruorder_pitcher',1,1,8),('age_bat',np.nan,18,50),('age_pit',np.nan,18,50),
       ('batter_days_since_prev_game',np.nan,0,30),('pitcher_days_since_prev_game',np.nan,0,30)]:
        v=pa[col] if col in pa else pd.Series(default,index=pa.index)
        pa[col]=pd.to_numeric(v,errors='coerce').fillna(default).clip(lo,hi)
    pa['bat_score']=np.where(pa.is_home_batter==1,pa.home_score,pa.away_score)
    pa['fld_score']=np.where(pa.is_home_batter==1,pa.away_score,pa.home_score)
    pa['game_year']=int(day[:4])
    pa['park']=pa.home_team;pa['matchup_key']=pa.batter.astype(str)+'_'+pa.pitcher.astype(str)
    pa['first_available_time']=available_at
    keep=['game_date','game_pk','at_bat_number','pitch_number','batter','pitcher','stand','p_throws',
       'home_team','away_team','inning','inning_topbot','outs_when_up','on_1b','on_2b','on_3b',
       'home_score','away_score','bat_score','fld_score','game_year','game_type','terminal_event','pa_pitches','outcome','season','date_key',
       'platoon','is_home_batter','runner_1b','runner_2b','runner_3b','bat_score_diff',
       'n_thruorder_pitcher','age_bat','age_pit','batter_days_since_prev_game',
       'pitcher_days_since_prev_game','park','matchup_key','first_available_time']
    return pa[keep].sort_values(keys),{'pitch_rows':len(frame),'pa_rows':len(pa),
       'excluded_terminal_rows':excluded,'schema_header_sha256':digest(canonical(list(frame.columns)))}


def capture_day(day,source,now):
    if date.fromisoformat(day)>yesterday(now):raise RefreshBlocked('History refresh cannot include today or future dates')
    schedule_raw,schedule_receipt=source.get('https://statsapi.mlb.com/api/v1/schedule?sportId=1&date='+day)
    schedule=json.loads(schedule_raw);games={};excluded=[]
    for block in schedule.get('dates',[]):
        if block['date']!=day:raise RefreshBlocked('Schedule returned another date')
        for game in block['games']:
            if game['gameType'] not in VALID_TYPES:continue
            if game['gamePk'] in games:raise RefreshBlocked('Duplicate schedule game')
            status=game['status'].get('detailedState','')
            if status in ('Postponed','Cancelled','Cancelled: Rain','Suspended'):
                if status=='Suspended':raise RefreshBlocked('Unresolved suspended history game')
                excluded.append({'game_pk':int(game['gamePk']),'status':status});continue
            if status not in ('Final','Completed Early'):raise RefreshBlocked('Prior-day game not final; retry later')
            games[int(game['gamePk'])]=game
    labels={};results=[];feeds=[];receipts=[schedule_receipt]
    for pk,game in sorted(games.items()):
        raw,receipt=source.get(f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live')
        feed=json.loads(raw)
        if feed.get('gamePk')!=pk:raise RefreshBlocked('Official final feed game mismatch')
        current,score=official_labels(feed,day,timestamp(receipt['finished_at']))
        labels.update(current);results.append(score)
        feeds.append({'raw_base64':base64.b64encode(raw).decode(),'receipt':receipt})
        receipts.append(receipt)
    raw,receipt=source.get(csv_url(day));receipts.append(receipt)
    available=max(timestamp(r['finished_at']) for r in receipts).isoformat()
    pa,summary=collapse_verified(raw,day,labels,{pk:g['gameType'] for pk,g in games.items()},available)
    if any(timestamp(r['requested_at'])>timestamp(r['finished_at']) for r in receipts):
        raise RefreshBlocked('Capture timestamps out of order')
    doc={'schema':SCHEMA,'day':day,'captured_at':available,'results':results,'excluded_games':excluded,
         'pa_csv':pa.to_csv(index=False) if len(pa) else '',
         'statcast':{'raw_base64':base64.b64encode(raw).decode(),'receipt':receipt},
         'official_schedule':{'raw_base64':base64.b64encode(schedule_raw).decode(),'receipt':schedule_receipt},
         'official_feeds':feeds,'summary':dict(summary,played_games=len(games),excluded_games=len(excluded))}
    return doc


def read_private(store,kind,ident):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    pair=store.read(f'private/{kind}/{ident}.enc')
    if pair is None:raise RefreshBlocked('Missing encrypted history object')
    encrypted=pair[0]
    if encrypted[:8]!=b'BRLAESG1':raise RefreshBlocked('Invalid encrypted history format')
    payload=AESGCM(store.key).decrypt(encrypted[8:20],encrypted[20:],('BRL:private:'+kind+':'+ident).encode())
    doc=json.loads(gzip.decompress(payload))
    if digest(canonical(doc))!=ident:raise RefreshBlocked('Encrypted history content identity mismatch')
    return doc


def ensure_history(store,base_path,base_rows,workdir,source,now):
    """Return an immutable combined history path and strictly prior-date baseline rows."""
    root=Path(workdir);root.mkdir(parents=True,exist_ok=True)
    base_path=Path(base_path);base_hash=digest(base_path.read_bytes())
    base=pd.read_csv(base_path,low_memory=False);last=str(base.date_key.max())[:10]
    target=yesterday(now).isoformat()
    if last>target:raise RefreshBlocked('Packaged history extends beyond prior day')
    pointer=store.read('history_pointer.json')
    state={'schema':'brl.history-manifest.v1','base_sha256':base_hash,'days':{}}
    if pointer:
        state=read_private(store,'history-manifests',json.loads(pointer[0])['revision'])
        if state['base_sha256']!=base_hash:raise RefreshBlocked('History manifest uses another anchor corpus')
    start=min(date.fromisoformat(last)+timedelta(days=1), date.fromisoformat(max(r['date'] for r in base_rows))+timedelta(days=1)).isoformat()
    # Include the baseline's last day and the package's last day for reconciliation;
    # preserve packaged PA rows verbatim and append only strictly later dates.
    day=date.fromisoformat(start);added=0;documents=[]
    while day.isoformat()<=target:
        key=day.isoformat();entry=state['days'].get(key)
        if entry:doc=read_private(store,'history-days',entry)
        else:
            doc=capture_day(key,source,now)
            ident=digest(canonical(doc));store.private('history-days',ident,doc)
            state['days'][key]=ident;added+=1
            partial_id=digest(canonical(state));store.private('history-manifests',partial_id,state)
            store.put('history_pointer.json',canonical({'revision':partial_id,'status':'partial'}))
        if doc['day']!=key or timestamp(doc['captured_at'])>source.clock():
            raise RefreshBlocked('History snapshot date/availability mismatch')
        documents.append(doc);day+=timedelta(days=1)
    pieces=[base];results={r['game_pk']:dict(r) for r in base_rows}
    for doc in documents:
        for row in doc['results']:
            # Retain inherited regular-season-only baseline definition.
            if row['game_type']=='R':results[row['game_pk']]={k:v for k,v in row.items() if k not in ('game_type','label_resolved_at')}
        if doc['pa_csv'] and doc['day']>last:
            rows=pd.read_csv(io.StringIO(doc['pa_csv']),low_memory=False)
            if not rows.date_key.astype(str).eq(doc['day']).all():raise RefreshBlocked('Derived history date mismatch')
            if any(timestamp(t)>source.clock() for t in rows.first_available_time):raise RefreshBlocked('Derived history unavailable')
            pieces.append(rows)
    combined=pd.concat(pieces,ignore_index=True,sort=False)
    if combined.duplicated(['game_pk','at_bat_number']).any():raise RefreshBlocked('Duplicate PA on history merge')
    combined=combined.sort_values(['date_key','game_pk','at_bat_number'],kind='mergesort')
    if str(combined.date_key.max())[:10]>target:raise RefreshBlocked('Combined history crosses cutoff')
    # Content-addressed, deterministic bytes. Original anchor/history stays intact.
    raw=gzip.compress(combined.to_csv(index=False).encode(),mtime=0)
    path=root/(digest(raw)+'.csv.gz')
    if not path.exists():path.write_bytes(raw)
    state.update(checked_through=target,history_sha256=digest(raw),n_rows=len(combined),
                 materialized_at=source.clock().isoformat())
    ident=digest(canonical(state));store.private('history-manifests',ident,state)
    store.put('history_pointer.json',canonical({'revision':ident,'checked_through':target}))
    if digest(base_path.read_bytes())!=base_hash:raise RefreshBlocked('Packaged history was altered')
    receipt={'schema':'brl.history-refresh.receipt.v1','checked_through':target,
       'latest_played_date':str(combined.date_key.max())[:10], 'newly_captured_days':added,
       'captured_days':len(documents),'appended_pa_rows':len(combined)-len(base),
       'history_sha256':digest(raw),'base_sha256':base_hash,'manifest_revision':ident,'available_at':state['materialized_at'],
       'model_fitted':False,'raw_published':False,'status':'complete'}
    return path,list(results.values()),receipt
