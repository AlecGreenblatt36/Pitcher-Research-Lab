"""Can a browser on the public page read the official MLB feeds directly? Records the CORS answer.

Fetches a few statsapi endpoints with an Origin header like the page's and records the status,
the Access-Control-Allow-Origin answer, the payload size and how long it took, plus the size of the
same live feed trimmed with the fields parameter. Writes diagnostics/cors_probe.json on the ledger
branch. No secrets, no player data beyond public schedule identifiers.
"""
from __future__ import annotations
import base64, json, os, time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ORIGIN = 'https://alecgreenblatt36.github.io'


def get(url, origin=ORIGIN, method='GET'):
    headers = {'Origin': origin, 'User-Agent': 'Mozilla/5.0 (BRL probe)', 'Accept': 'application/json'}
    if method == 'OPTIONS':
        headers['Access-Control-Request-Method'] = 'GET'
    t = time.time()
    try:
        with urlopen(Request(url, headers=headers, method=method), timeout=30) as r:
            body = r.read()
            return {'status': r.status, 'allow_origin': r.headers.get('Access-Control-Allow-Origin'), 'allow_methods': r.headers.get('Access-Control-Allow-Methods'),
                    'cache_control': r.headers.get('Cache-Control'), 'encoding': r.headers.get('Content-Encoding'), 'bytes': len(body), 'seconds': round(time.time() - t, 2)}, body
    except HTTPError as exc:
        return {'status': exc.code, 'allow_origin': exc.headers.get('Access-Control-Allow-Origin'), 'seconds': round(time.time() - t, 2)}, b''
    except Exception as exc:
        return {'error': type(exc).__name__ + ': ' + str(exc)[:120], 'seconds': round(time.time() - t, 2)}, b''


def main():
    out = {'schema': 'brl.cors-probe.v1', 'at': datetime.now(timezone.utc).isoformat(), 'origin': ORIGIN, 'checks': {}}
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    info, body = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day}')
    out['checks']['schedule'] = info
    pk = None
    try:
        games = [g for d in json.loads(body).get('dates', []) for g in d.get('games', [])]
        live = [g for g in games if g['status']['abstractGameState'] == 'Live'] or games
        pk = live[0]['gamePk'] if live else None
    except Exception:
        pass
    pk = pk or 849833
    out['game_pk'] = pk
    for name, url in (('feed_live', f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'),
                      ('feed_live_fields', f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live?fields=gameData,status,abstractGameState,detailedState,liveData,linescore,currentInning,inningState,outs,balls,strikes,teams,away,home,runs,hits,errors,innings,num,offense,defense,first,second,third,batter,pitcher,id,fullName,battingOrder,plays,allPlays,about,atBatIndex,halfInning,inning,isComplete,result,eventType,description,awayScore,homeScore,rbi,matchup,count,playEvents,isPitch,details,code,type,pitchData,startSpeed,coordinates,pX,pZ,strikeZoneTop,strikeZoneBottom,hitData,launchSpeed,totalDistance,trajectory,location,coordX,coordY,runners,movement,start,end,isOut,playIndex,credits,position'),
                      ('linescore', f'https://statsapi.mlb.com/api/v1/game/{pk}/linescore'),
                      ('timestamps', f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live/timestamps')):
        info, _ = get(url)
        out['checks'][name] = info
    out['checks']['preflight'] = get(f'https://statsapi.mlb.com/api/v1/game/{pk}/linescore', method='OPTIONS')[0]
    text = json.dumps(out, indent=1)
    print(text)
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GH_TOKEN')
    if repo and token:
        url = f'https://api.github.com/repos/{repo}/contents/diagnostics/cors_probe.json'
        headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-probe/1.0', 'Content-Type': 'application/json'}
        sha = None
        try:
            with urlopen(Request(url + '?ref=brl-live-data', headers=headers), timeout=30) as r:
                sha = json.loads(r.read()).get('sha')
        except HTTPError:
            pass
        payload = {'message': 'BRL: cors probe', 'content': base64.b64encode(text.encode()).decode(), 'branch': 'brl-live-data'}
        if sha:
            payload['sha'] = sha
        for attempt in range(4):
            try:
                with urlopen(Request(url, headers=headers, data=json.dumps(payload).encode(), method='PUT'), timeout=30) as r:
                    r.read(); break
            except HTTPError as exc:
                if exc.code != 409 or attempt == 3:
                    raise
                time.sleep(3)


if __name__ == '__main__':
    main()
