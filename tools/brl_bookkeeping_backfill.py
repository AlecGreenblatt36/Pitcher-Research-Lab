"""Backfill a season's pitch sequences and batted-ball contact for box-score bookkeeping.

Runs in GitHub Actions. For every completed game of the season so far (regular season and
postseason, from the official schedule) it reads the official play-by-play and keeps, per
plate appearance, only what the box-score bookkeeping uses: pitcher, batter, batter hand,
the result, the pitch list (count, result code, type, speed) and the batted-ball record
(shape, fielder, distance, exit velocity). The rows are gzipped, sealed with AES-256-GCM
(purpose bookkeeping-season-<year>) and committed to private/bookkeeping/season-<year>.enc
on the ledger branch; a small public receipt with counts and hashes goes next to it.

The job resumes: games already in the sealed file are not fetched again, so a rerun only
adds the days played since. Plaintext never leaves the runner; the key is never printed.
"""
from __future__ import annotations
import base64, gzip, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
from brl_live.pitch_bridge import pitch_list, contact_of  # noqa: E402
from brl_live.bookkeeping_season import SCHEMA, STUDY_SCHEMA, season_path, season_purpose, receipt_path, study_path, study_purpose  # noqa: E402
from cloud.security import seal, unseal, key_bytes, sha  # noqa: E402


def _outcomes_module():
    # The event-to-outcome table lives in the engine's pa_model package, whose __init__ pulls in
    # the fitting stack (joblib, scikit-learn); load the one file directly so this job needs
    # none of it.
    import importlib.util
    path = ROOT / 'brl_engine' / 'runtime' / 'research_lab' / 'pa_model' / 'outcomes.py'
    spec = importlib.util.spec_from_file_location('brl_outcomes', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


map_event = _outcomes_module().map_event

API = 'https://statsapi.mlb.com/api/v1'
GAME_TYPES = {'R', 'F', 'D', 'L', 'W'}        # regular season and the four postseason rounds
WORKERS = 4


def get_json(url: str, token: str | None = None, method: str = 'GET', payload=None, tries: int = 5):
    headers = {'User-Agent': 'BaseballResearchLab/2.0 (+public research)', 'Accept': 'application/json'}
    body = None
    if token:
        headers['Authorization'] = 'Bearer ' + token
    if payload is not None:
        body = json.dumps(payload).encode(); headers['Content-Type'] = 'application/json'
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers=headers, data=body, method=method), timeout=60) as response:
                raw = response.read()
            return json.loads(raw) if raw else {}
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in (409, 422) or attempt == tries - 1:
                raise
        except URLError:
            if attempt == tries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))


def put(repo: str, token: str, path: str, raw: bytes, branch: str, message: str):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    existing = get_json(url + '?ref=' + branch, token)
    payload = {'message': message, 'content': base64.b64encode(raw).decode(), 'branch': branch}
    if existing and existing.get('sha'):
        payload['sha'] = existing['sha']
    return get_json(url, token, 'PUT', payload)


def read_blob(repo: str, token: str, path: str, branch: str) -> bytes | None:
    value = get_json(f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}', token)
    if not value:
        return None
    if value.get('encoding') == 'base64' and value.get('content'):
        return base64.b64decode(''.join(value['content'].split()))
    blob = get_json(f'https://api.github.com/repos/{repo}/git/blobs/{value["sha"]}', token)
    return base64.b64decode(''.join(blob['content'].split()))


def completed_games(year: int, through: str) -> list[dict]:
    doc = get_json(f'{API}/schedule?sportId=1&startDate={year}-03-01&endDate={through}&fields=dates,date,games,gamePk,gameType,officialDate,status,abstractGameState,detailedState')
    games = []
    for d in (doc or {}).get('dates') or []:
        for g in d.get('games') or []:
            status = g.get('status') or {}
            if g.get('gameType') not in GAME_TYPES or status.get('abstractGameState') != 'Final':
                continue
            if not str(status.get('detailedState') or '').startswith(('Final', 'Completed Early', 'Game Over')):
                continue
            games.append({'game_pk': int(g['gamePk']), 'date': str(g.get('officialDate') or d['date'])[:10], 'game_type': g['gameType']})
    return games


def _num(value, digits=2):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if v == v else None


PITCH_FIELDS = ('type', 'code', 'balls', 'strikes', 'start_speed', 'end_speed', 'spin_rate', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                'release_x', 'release_z', 'extension', 'zone')
HIT_FIELDS = ('launch_speed', 'launch_angle', 'distance', 'coord_x', 'coord_y', 'trajectory', 'hardness', 'location')


def physics(play: dict) -> dict:
    """Pitch-by-pitch physics and the batted-ball measurement of one play (the research dataset)."""
    pitches = []
    balls = strikes = 0
    hit = None
    for e in play.get('playEvents') or []:
        if e.get('isPitch') is not True:
            continue
        d = e.get('details') or {}
        pd_ = e.get('pitchData') or {}
        co = pd_.get('coordinates') or {}
        br = pd_.get('breaks') or {}
        pitches.append([str(((d.get('type') or {}).get('code')) or '').upper() or None, str(d.get('code') or ''), balls, strikes,
                        _num(pd_.get('startSpeed'), 1), _num(pd_.get('endSpeed'), 1), _num(br.get('spinRate'), 0),
                        _num(co.get('pfxX')), _num(co.get('pfxZ')), _num(co.get('pX')), _num(co.get('pZ')),
                        _num(co.get('x0')), _num(co.get('z0')), _num(pd_.get('extension')), pd_.get('zone')])
        c = e.get('count') or {}
        balls, strikes = int(c.get('balls', balls)), int(c.get('strikes', strikes))
        hd = e.get('hitData')
        if hd:
            hc = hd.get('coordinates') or {}
            hit = [_num(hd.get('launchSpeed'), 1), _num(hd.get('launchAngle'), 1), _num(hd.get('totalDistance'), 0),
                   _num(hc.get('coordX'), 1), _num(hc.get('coordY'), 1), hd.get('trajectory'), hd.get('hardness'), hd.get('location')]
    return {'pitches': pitches, 'hit': hit}


def extract(doc: dict, research: list | None = None) -> list[dict]:
    rows = []
    for play in (doc or {}).get('allPlays') or []:
        result = play.get('result') or {}
        outcome = map_event(result.get('eventType', ''))
        if outcome is None or not (play.get('about') or {}).get('isComplete'):
            continue
        m = play.get('matchup') or {}
        about = play.get('about') or {}
        batter = (m.get('batter') or {}).get('id')
        rows.append({'i': int(about['atBatIndex']), 'o': outcome, 'e': str(result.get('eventType') or ''),
                     'p': int(m['pitcher']['id']), 'b': None if batter is None else int(batter), 's': str((m.get('batSide') or {}).get('code') or 'R'),
                     'n': sum(e.get('isPitch') is True for e in play.get('playEvents') or []),
                     'pt': pitch_list(play), 'c': contact_of(play)})
        if research is not None:
            ph = physics(play)
            research.append({'i': int(about['atBatIndex']), 'inning': about.get('inning'), 'half': 'top' if about.get('isTopInning') else 'bottom',
                             'o': outcome, 'e': str(result.get('eventType') or ''), 'p': int(m['pitcher']['id']),
                             'b': None if batter is None else int(batter), 's': str((m.get('batSide') or {}).get('code') or 'R'),
                             't': str((m.get('pitchHand') or {}).get('code') or ''), 'pitches': ph['pitches'], 'hit': ph['hit']})
    return rows


def fetch_game(game: dict) -> tuple[int, dict, dict]:
    doc = get_json(f'{API}/game/{game["game_pk"]}/playByPlay')
    fetched = datetime.now(timezone.utc).isoformat()
    research = []
    rows = extract(doc, research)
    return game['game_pk'], {'date': game['date'], 'game_type': game['game_type'], 'fetched_at': fetched, 'rows': rows}, \
        {'date': game['date'], 'game_type': game['game_type'], 'fetched_at': fetched, 'rows': research}


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key = key_bytes(os.environ['BRL_PA_PACKAGE_KEY'])
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    now = datetime.now(timezone.utc)
    year = int(os.environ.get('BRL_SEASON') or now.astimezone(ZoneInfo('America/New_York')).year)
    through = (now.astimezone(ZoneInfo('America/New_York')).date() - timedelta(days=1)).isoformat()
    existing = read_blob(repo, token, season_path(year), branch)
    doc = {'schema': SCHEMA, 'year': year, 'games': {}}
    if existing is not None:
        doc = json.loads(gzip.decompress(unseal(existing, key, season_purpose(year))))
        if doc.get('schema') != SCHEMA or int(doc.get('year')) != year:
            raise ValueError('Season bookkeeping identity mismatch')
    study = {'schema': STUDY_SCHEMA, 'year': year, 'games': {}}
    existing_study = read_blob(repo, token, study_path(year), branch)
    if existing_study is not None:
        study = json.loads(gzip.decompress(unseal(existing_study, key, study_purpose(year))))
        if study.get('schema') != STUDY_SCHEMA or int(study.get('year')) != year:
            raise ValueError('Season study identity mismatch')
    games = completed_games(year, through)
    todo = [g for g in games if str(g['game_pk']) not in doc['games'] or str(g['game_pk']) not in study['games']]
    print(json.dumps({'year': year, 'through': through, 'completed_games': len(games), 'already_sealed': len(doc['games']), 'to_fetch': len(todo)}))
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for game, outcome in zip(todo, pool.map(lambda g: _safe(fetch_game, g), todo)):
            if isinstance(outcome, Exception):
                failures.append({'game_pk': game['game_pk'], 'error': type(outcome).__name__ + ': ' + str(outcome)[:120]})
                continue
            pk, value, research = outcome
            if value['rows']:
                doc['games'][str(pk)] = value
                study['games'][str(pk)] = research
    doc['through'] = through
    doc['sealed_at'] = datetime.now(timezone.utc).isoformat()
    plain = gzip.compress(json.dumps(doc, separators=(',', ':'), sort_keys=True).encode(), mtime=0)
    cipher = seal(plain, key, season_purpose(year))
    put(repo, token, season_path(year), cipher, branch, f'BRL: season {year} bookkeeping through {through}')
    study['through'] = through
    study['sealed_at'] = doc['sealed_at']
    study_plain = gzip.compress(json.dumps(study, separators=(',', ':'), sort_keys=True).encode(), mtime=0)
    study_cipher = seal(study_plain, key, study_purpose(year))
    put(repo, token, study_path(year), study_cipher, branch, f'BRL: season {year} pitch physics through {through}')
    n_rows = sum(len(g['rows']) for g in doc['games'].values())
    with_pitches = sum(1 for g in doc['games'].values() for r in g['rows'] if r['pt'])
    with_contact = sum(1 for g in doc['games'].values() for r in g['rows'] if r['c'])
    receipt = {'schema': 'brl.bookkeeping-season-receipt.v1', 'year': year, 'through': through, 'games': len(doc['games']), 'plate_appearances': n_rows,
               'with_pitch_list': with_pitches, 'with_contact': with_contact, 'fetched_this_run': len(todo) - len(failures), 'failures': failures[:50],
               'cipher_sha256': sha(cipher), 'cipher_bytes': len(cipher), 'plaintext_gzip_bytes': len(plain), 'sealed_at': doc['sealed_at'],
               'path': season_path(year), 'format': 'AES-256-GCM; BRLAESG1 header; 12-byte nonce; AAD BRL:' + season_purpose(year) + '; gzip JSON inside',
               'study': {'path': study_path(year), 'purpose': study_purpose(year), 'games': len(study['games']), 'plate_appearances': sum(len(g['rows']) for g in study['games'].values()),
                         'pitches': sum(len(r['pitches']) for g in study['games'].values() for r in g['rows']),
                         'with_hit_measurement': sum(1 for g in study['games'].values() for r in g['rows'] if r['hit']),
                         'cipher_sha256': sha(study_cipher), 'cipher_bytes': len(study_cipher), 'pitch_fields': list(PITCH_FIELDS), 'hit_fields': list(HIT_FIELDS)}}
    put(repo, token, receipt_path(year), json.dumps(receipt, indent=1).encode(), branch, f'BRL: season {year} bookkeeping receipt')
    print(json.dumps({k: receipt[k] for k in ('games', 'plate_appearances', 'with_pitch_list', 'with_contact', 'fetched_this_run', 'cipher_bytes')} | {'failures': len(failures)}))


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception as exc:      # recorded in the receipt; the rest of the season still seals
        return exc


def run():
    try:
        main()
    except Exception as exc:
        # Logs are not always readable from outside; the receipt on the ledger branch says what happened.
        import traceback
        frames = traceback.extract_tb(exc.__traceback__)
        receipt = {'schema': 'brl.bookkeeping-season-receipt.v1', 'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)[:300],
                   'where': [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in frames[-5:]],
                   'failed_at': datetime.now(timezone.utc).isoformat()}
        try:
            repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']
            year = int(os.environ.get('BRL_SEASON') or datetime.now(ZoneInfo('America/New_York')).year)
            put(repo, token, receipt_path(year), json.dumps(receipt, indent=1).encode(), os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data'), 'BRL: season bookkeeping failed')
        except Exception:
            pass
        raise


if __name__ == '__main__':
    run()
