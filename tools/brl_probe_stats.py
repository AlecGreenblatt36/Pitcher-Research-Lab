"""Can the page show each probable starter's season line and each team's record the way every preview does?

Asks statsapi, with an Origin header like the page's, for (1) the probable starters' season pitching lines through the
people endpoint with a stats hydration, (2) the schedule with probable pitchers and team records, and (3) what the live
feed's box score carries for a probable starter before first pitch; records the CORS answer, the status and the shape
(keys and the public stat values for the two starters). Writes diagnostics/stats_probe.json on the ledger branch. Public
MLB facts only; no secrets.
"""
from __future__ import annotations
import base64, json, os, time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ORIGIN = 'https://alecgreenblatt36.github.io'
API = 'https://statsapi.mlb.com/api/'


def get(url):
    headers = {'Origin': ORIGIN, 'User-Agent': 'Mozilla/5.0 (BRL probe)', 'Accept': 'application/json'}
    t = time.time()
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as r:
            body = r.read()
            return {'status': r.status, 'allow_origin': r.headers.get('Access-Control-Allow-Origin'), 'bytes': len(body), 'seconds': round(time.time() - t, 2)}, json.loads(body)
    except HTTPError as exc:
        return {'status': exc.code, 'allow_origin': exc.headers.get('Access-Control-Allow-Origin'), 'seconds': round(time.time() - t, 2)}, None
    except Exception as exc:
        return {'error': type(exc).__name__ + ': ' + str(exc)[:160], 'seconds': round(time.time() - t, 2)}, None


def pick(d, keys):
    return {k: d.get(k) for k in keys if isinstance(d, dict) and k in d}


STAT_KEYS = ('gamesPlayed', 'gamesStarted', 'wins', 'losses', 'era', 'inningsPitched', 'strikeOuts', 'baseOnBalls', 'whip', 'saves', 'homeRuns')


def main():
    out = {'schema': 'brl.stats-probe.v1', 'at': datetime.now(timezone.utc).isoformat(), 'origin': ORIGIN, 'checks': {}}
    now = datetime.now(timezone.utc)
    game, day = None, None
    for k in range(0, 4):
        d = (now + timedelta(days=k)).strftime('%Y-%m-%d')
        info, doc = get(API + 'v1/schedule?sportId=1&date=' + d + '&hydrate=probablePitcher,team')
        games = [g for x in (doc or {}).get('dates', []) for g in x.get('games', [])]
        withp = [g for g in games if all(((g.get('teams') or {}).get(s) or {}).get('probablePitcher') for s in ('away', 'home'))]
        if withp:
            game, day = withp[0], d
            out['checks']['schedule'] = info
            break
    if not game:
        out['error'] = 'no game with both probable starters in the next four days'
    else:
        teams = game['teams']
        out['game'] = {'game_pk': game['gamePk'], 'date': day, 'game_type': game.get('gameType'), 'series': pick(game, ('seriesDescription', 'seriesGameNumber', 'gamesInSeries'))}
        out['schedule_team_keys'] = sorted((teams.get('away') or {}).keys())
        out['schedule_records'] = {s: {'leagueRecord': (teams.get(s) or {}).get('leagueRecord'), 'seriesNumber': (teams.get(s) or {}).get('seriesNumber'),
                                       'team_record': ((teams.get(s) or {}).get('team') or {}).get('record')} for s in ('away', 'home')}
        ids = [teams[s]['probablePitcher']['id'] for s in ('away', 'home')]
        season = int(day[:4])
        for name, hyd in (('people_season', f'stats(group=[pitching],type=[season],season={season})'),
                          ('people_season_regular', f'stats(group=[pitching],type=[season],season={season},gameType=R)'),
                          ('people_season_post', f'stats(group=[pitching],type=[season],season={season},gameType=P)')):
            info, doc = get(API + 'v1/people?personIds=' + ','.join(str(i) for i in ids) + '&hydrate=' + quote(hyd, safe='()=,[]'))
            rows = []
            for p in (doc or {}).get('people', []):
                for st in p.get('stats') or []:
                    for sp in st.get('splits') or []:
                        rows.append({'id': p.get('id'), 'group': (st.get('group') or {}).get('displayName'), 'type': (st.get('type') or {}).get('displayName'),
                                     'gameType': sp.get('gameType'), 'season': sp.get('season'), 'stat': pick(sp.get('stat') or {}, STAT_KEYS),
                                     'split_keys': sorted(sp.keys())})
            info['rows'] = rows
            out['checks'][name] = info
        info, doc = get(API + f"v1.1/game/{game['gamePk']}/feed/live")
        if doc:
            bx = ((doc.get('liveData') or {}).get('boxscore') or {}).get('teams') or {}
            seen = {}
            for s in ('away', 'home'):
                pl = (bx.get(s) or {}).get('players') or {}
                p = pl.get('ID' + str(teams[s]['probablePitcher']['id'])) or {}
                ss = p.get('seasonStats') or {}
                seen[s] = {'player_keys': sorted(p.keys()), 'season_pitching': pick(ss.get('pitching') or {}, STAT_KEYS)}
            gd_teams = (doc.get('gameData') or {}).get('teams') or {}
            info['probables'] = seen
            info['gamedata_records'] = {s: (gd_teams.get(s) or {}).get('record') for s in ('away', 'home')}
        out['checks']['feed_live'] = info
    text = json.dumps(out, indent=1)
    print(text)
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GH_TOKEN')
    if repo and token:
        url = f'https://api.github.com/repos/{repo}/contents/diagnostics/stats_probe.json'
        headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-probe/1.0', 'Content-Type': 'application/json'}
        sha = None
        try:
            with urlopen(Request(url + '?ref=brl-live-data', headers=headers), timeout=30) as r:
                sha = json.loads(r.read()).get('sha')
        except HTTPError:
            pass
        payload = {'message': 'BRL: stats probe', 'content': base64.b64encode(text.encode()).decode(), 'branch': 'brl-live-data'}
        if sha:
            payload['sha'] = sha
        for attempt in range(6):
            try:
                with urlopen(Request(url, headers=headers, data=json.dumps(payload).encode(), method='PUT'), timeout=30) as r:
                    r.read(); break
            except HTTPError as exc:
                if exc.code != 409 or attempt == 5:
                    raise
                time.sleep(4)


if __name__ == '__main__':
    main()
