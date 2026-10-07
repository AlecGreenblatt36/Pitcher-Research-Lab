"""Network with receipts, the Git-backed ledger store, and the daily iteration runner.

Ledger layout on the data branch (unchanged from the inherited runtime):
  ledger.json                      everything the page needs plus status, publications, actuals, boxes
  forecasts/<id>.json              immutable public forecast records
  box_forecasts/<sha>.json         immutable public box payloads
  private/<kind>/<id>.enc          sealed source snapshots, inputs and results (AES-GCM, purpose-bound)
  public/index.html, predictions.json, .nojekyll   last rendered page (restored when a run is blocked)
"""
from __future__ import annotations

import base64, hashlib, json, os, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from app.common import canonical, content_hash
from app.safety import Blocked
from app.results import parse_final  # re-exported for the public layer
from .contracts import utcnow, live_inputs
from .security import seal, unseal

BRANCH = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
API = 'https://api.github.com'
SCHEDULE = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}'
MAX_API_WAIT = 25 * 60   # the longest a run waits out GitHub rate limiting or an outage, in seconds


def api_pause(exc, attempt: int, waited: float):
    """Seconds to wait before retrying a failed GitHub API call, or None when the failure is final.

    Rate limiting (429, or 403 with the rate-limit headers) waits for the reset the response
    names; server errors and dropped connections back off and retry. 404, 409 and 422 are
    answers, not failures, and go straight back to the caller.
    """
    if isinstance(exc, HTTPError):
        if exc.code in (404, 409, 422) or exc.code < 400:
            return None
        headers = exc.headers or {}
        if exc.code in (403, 429):
            retry_after = headers.get('Retry-After')
            remaining = headers.get('X-RateLimit-Remaining')
            reset = headers.get('X-RateLimit-Reset')
            if retry_after and str(retry_after).strip().isdigit():
                pause = float(retry_after) + 2
            elif remaining == '0' and reset and str(reset).strip().isdigit():
                pause = max(5.0, float(reset) - time.time() + 3)
            elif exc.code == 429:
                pause = 60.0 * (attempt + 1)
            else:
                return None            # a real permission error
        elif exc.code >= 500:
            pause = 2.0 + 6.0 * attempt       # a server error on one request is usually gone on the next
        else:
            return None
    elif isinstance(exc, (URLError, TimeoutError, ConnectionError, OSError)):
        pause = 2.0 + 5.0 * attempt
    else:
        return None
    if waited + pause > MAX_API_WAIT:
        return None
    return pause


class Network:
    """JSON over HTTPS with a receipt (url, clocks, byte count, sha256) for every fetch."""
    user_agent = 'BaseballResearchLab/2.0 (+public research)'

    def json(self, url: str):
        started = utcnow().isoformat()
        request = Request(url, headers={'User-Agent': self.user_agent, 'Accept': 'application/json'})
        with urlopen(request, timeout=40) as response:
            raw = response.read(100_000_001)
        if len(raw) > 100_000_000:
            raise Blocked('Response too large')
        finished = utcnow().isoformat()
        return json.loads(raw), {'url': url, 'started_at': started, 'finished_at': finished, 'bytes': len(raw),
                                'sha256': hashlib.sha256(raw).hexdigest()}


class GitStore:
    """Read/write files on the ledger branch through the GitHub contents API; sealed private blobs."""

    def __init__(self, repo: str, token: str, key: bytes):
        self.repo, self.token, self.key = repo, token, key
        self.read_audit = getattr(self, 'read_audit', None) or {}
        self._ensure_branch()
        saved = self.read('ledger.json')
        self.ledger = json.loads(saved[0]) if saved else {'date': None, 'forecasts': {}, 'publications': {}, 'actuals': {}, 'status': {}}

    def request(self, path: str, method: str = 'GET', data=None):
        headers = {'Authorization': 'Bearer ' + self.token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-store/2.0'}
        body = None
        if data is not None:
            body = json.dumps(data).encode(); headers['Content-Type'] = 'application/json'
        audit = self.read_audit if isinstance(getattr(self, 'read_audit', None), dict) else {}
        waited = 0.0
        for attempt in range(12):
            audit['requests'] = audit.get('requests', 0) + 1
            try:
                with urlopen(Request(API + '/repos/' + self.repo + path, headers=headers, data=body, method=method), timeout=40) as response:
                    raw = response.read()
                    remaining = response.headers.get('X-RateLimit-Remaining')
                if remaining is not None and str(remaining).isdigit():
                    audit['rate_limit_remaining'] = int(remaining)
                return json.loads(raw) if raw else {}
            except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
                pause = api_pause(exc, attempt, waited)
                if pause is None:
                    raise
                # The receipt shows what failed (status code or exception) and how long the run waited for it.
                label = str(exc.code) if isinstance(exc, HTTPError) else type(exc).__name__
                failures = audit.setdefault('failures', {})
                failures[label] = failures.get(label, 0) + 1
                audit['retries'] = audit.get('retries', 0) + 1
                audit['waited_seconds'] = round(audit.get('waited_seconds', 0.0) + pause, 1)
                time.sleep(pause); waited += pause
        raise Blocked('GitHub API unavailable')

    def _ensure_branch(self):
        try:
            self.request('/git/ref/heads/' + BRANCH)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            default = self.request('')['default_branch']
            sha = self.request('/git/ref/heads/' + default)['object']['sha']
            self.request('/git/refs', 'POST', {'ref': 'refs/heads/' + BRANCH, 'sha': sha})

    def read(self, path: str):
        try:
            value = self.request('/contents/' + quote(path, safe='/') + '?ref=' + BRANCH)
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        if not isinstance(value, dict):
            raise Blocked('Git object is not a file')
        if value.get('encoding') == 'base64' and isinstance(value.get('content'), str):
            raw = base64.b64decode(''.join(value['content'].split()))
        else:
            blob = self.request('/git/blobs/' + value['sha'])
            raw = base64.b64decode(''.join(blob['content'].split()))
        return raw, value['sha']

    def put(self, path: str, raw: bytes, immutable: bool = False):
        # Several jobs write to the ledger branch (live runs, backfills, research); a 409 means the
        # branch moved between reading the file's sha and writing, so re-read and try again.
        for attempt in range(6):
            existing = self.read(path)
            if existing is not None:
                if existing[0] == raw:
                    return None
                if immutable:
                    raise Blocked('Immutable object conflict: ' + path)
            payload = {'message': 'BRL: save ' + path.split('/')[0], 'content': base64.b64encode(raw).decode(), 'branch': BRANCH}
            if existing is not None:
                payload['sha'] = existing[1]
            try:
                return self.request('/contents/' + quote(path, safe='/'), 'PUT', payload)
            except HTTPError as exc:
                if exc.code != 409 or attempt == 5:
                    raise
                time.sleep(2 + 3 * attempt)

    def private(self, kind: str, ident: str, value: dict):
        purpose = 'private:' + kind + ':' + ident
        self.put('private/' + kind + '/' + ident + '.enc', seal(canonical(value), self.key, purpose), immutable=True)

    def publish(self, forecast: dict) -> str:
        ident = content_hash(forecast)
        reply = self.put('forecasts/' + ident + '.json', canonical(forecast), immutable=True)
        if reply is None:
            raise Blocked('Forecast already published; no fresh publication receipt')
        commit = reply['commit']
        self.ledger['forecasts'][ident] = forecast
        self.ledger['publications'][ident] = {'commit': commit['sha'], 'published_at': utcnow().isoformat(),
                                              'server_commit_time': ((commit.get('committer') or {}).get('date'))}
        return ident

    def persist(self):
        self.put('ledger.json', canonical(self.ledger))


class LocalStore:
    """Same interface on a local directory, for tests and dry runs."""

    def __init__(self, path, key: bytes, clock=utcnow):
        self.root, self.key, self.clock = Path(path), key, clock
        self.root.mkdir(parents=True, exist_ok=True)
        self.read_audit = {}
        saved = self.read('ledger.json')
        self.ledger = json.loads(saved[0]) if saved else {'date': None, 'forecasts': {}, 'publications': {}, 'actuals': {}, 'status': {}}

    def read(self, path):
        p = self.root / path
        if not p.exists():
            return None
        raw = p.read_bytes()
        return raw, hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()

    def put(self, path, raw, immutable=False):
        p = self.root / path
        if p.exists():
            if p.read_bytes() == raw:
                return None
            if immutable:
                raise Blocked('Immutable object conflict: ' + path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        return {'commit': {'sha': hashlib.sha1(raw).hexdigest(), 'committer': {'date': self.clock().isoformat()}}}

    def private(self, kind, ident, value):
        self.put('private/' + kind + '/' + ident + '.enc', seal(canonical(value), self.key, 'private:' + kind + ':' + ident), immutable=True)

    def publish(self, f):
        ident = content_hash(f)
        reply = self.put('forecasts/' + ident + '.json', canonical(f), immutable=True)
        if reply is None:
            raise Blocked('Forecast already published')
        self.ledger['forecasts'][ident] = f
        self.ledger['publications'][ident] = {'commit': reply['commit']['sha'], 'published_at': self.clock().isoformat(), 'server_commit_time': None}
        return ident

    def persist(self):
        self.put('ledger.json', canonical(self.ledger))


def fingerprint_from_forecasts(store, pk) -> list:
    """(ident, forecast) for every saved version of this game, oldest first."""
    items = [(ident, f) for ident, f in store.ledger['forecasts'].items() if int(f['game_pk']) == int(pk)]
    return sorted(items, key=lambda x: (x[1].get('version', 0), x[1].get('saved_at', '')))


def read(store, path):
    return store.read(path)


def schedule_context(g: dict) -> dict:
    """What the schedule says about a game beyond who plays: venue, series, game number, records."""
    teams = g.get('teams') or {}
    def record(side):
        r = (teams.get(side) or {}).get('leagueRecord') or {}
        try:
            return {'wins': int(r.get('wins')), 'losses': int(r.get('losses'))}
        except (TypeError, ValueError):
            return None
    out = {'venue': ((g.get('venue') or {}).get('name')), 'series_description': g.get('seriesDescription'), 'description': g.get('description'),
           'series_game_number': g.get('seriesGameNumber'), 'games_in_series': g.get('gamesInSeries'), 'double_header': g.get('doubleHeader'),
           'game_number': g.get('gameNumber'), 'day_night': g.get('dayNight'), 'records': {'away': record('away'), 'home': record('home')}}
    return {k: v for k, v in out.items() if v not in (None, '', {})}


class Runner:
    """One pass over the day's schedule (Eastern date); each game goes through process()."""

    def __init__(self, network, store, simulator, clock=utcnow, run_id='local'):
        self.net, self.store, self.sim, self.clock, self.run_id = network, store, simulator, clock, run_id

    def today(self) -> str:
        return self.clock().astimezone(ZoneInfo('America/New_York')).date().isoformat()

    def schedule(self, day: str) -> list:
        doc, _ = self.net.json(SCHEDULE.format(date=day))
        games = []
        for d in doc.get('dates') or []:
            for g in d.get('games') or []:
                games.append({'game_pk': int(g['gamePk']), 'scheduled_start': g.get('gameDate'), 'state': (g.get('status') or {}).get('abstractGameState'),
                              'away': ((g.get('teams') or {}).get('away') or {}).get('team', {}).get('name'),
                              'home': ((g.get('teams') or {}).get('home') or {}).get('team', {}).get('name'), 'game_type': g.get('gameType'),
                              'context': schedule_context(g)})
        return games


    def iteration(self):
        day = self.today()
        self.store.ledger['date'] = day
        errors = {}
        for item in self.schedule(day):
            try:
                self.process(item['game_pk'], item)
            except Exception as exc:
                # One game's problem never blocks the others: it is recorded with its location.
                import traceback
                frames = traceback.extract_tb(exc.__traceback__)
                where = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in frames[-4:]]
                errors[str(item['game_pk'])] = {'error': type(exc).__name__ + ': ' + str(exc)[:200], 'where': where}
                self.store.ledger.setdefault('status', {})[str(item['game_pk'])] = {'state': 'blocked', 'date': day, 'reason': type(exc).__name__ + ': ' + str(exc)[:200],
                                                                                   'where': where, 'checked_at': self.clock().isoformat()}
        self.store.ledger['iteration'] = {'run_id': self.run_id, 'finished_at': self.clock().isoformat(), 'blocked': errors}
        self.store.persist()

    def process(self, pk, item):
        raise NotImplementedError
