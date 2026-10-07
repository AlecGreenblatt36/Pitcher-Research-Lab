"""GitHub API failures are waited out, not turned into silent gaps: rate limiting waits for the reset,
server errors back off, and the answers (404, 409, 422, a real 403) go straight back to the caller."""
import importlib.util
import io
import time
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from cloud import runner

ROOT = Path(__file__).resolve().parents[1]


def http_error(code, headers=None):
    msg = Message()
    for k, v in (headers or {}).items():
        msg[k] = str(v)
    return HTTPError('https://api.github.com/x', code, 'err', msg, io.BytesIO(b''))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('pause_for', [runner.api_pause, load('brl_live/bootstrap.py', 'brl_bootstrap_t').pause_for,
                                       load('tools/brl_chain_next.py', 'brl_chain_t').pause_for])
def test_pause_policy(pause_for):
    # answers come straight back
    for code in (404, 409, 422, 400):
        assert pause_for(http_error(code), 0, 0.0) is None
    assert pause_for(http_error(403), 0, 0.0) is None                           # a real permission error
    # primary rate limit: wait for the reset the response names
    reset = int(time.time()) + 90
    pause = pause_for(http_error(403, {'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': reset}), 0, 0.0)
    assert 85 <= pause <= 100
    # secondary rate limit: Retry-After wins, otherwise a growing wait
    assert pause_for(http_error(429, {'Retry-After': '30'}), 0, 0.0) == 32
    assert pause_for(http_error(429), 1, 0.0) == 120
    # server errors and dropped connections back off
    assert pause_for(http_error(502), 0, 0.0) > 0 and pause_for(http_error(503), 2, 0.0) > pause_for(http_error(503), 0, 0.0)
    assert pause_for(URLError('reset'), 0, 0.0) > 0
    # the total wait is bounded
    assert pause_for(http_error(429, {'Retry-After': '3600'}), 0, 0.0) is None
    assert pause_for(http_error(503), 0, 40 * 60) is None


def test_store_request_retries_then_returns(monkeypatch):
    calls, sleeps = [], []
    answers = [http_error(503), http_error(429, {'Retry-After': '1'}), b'{"ok": true}']

    class Response:
        def __init__(self, raw):
            self.raw = raw; self.headers = {'X-RateLimit-Remaining': '812'}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return self.raw

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Response(answer)

    monkeypatch.setattr(runner, 'urlopen', fake_urlopen)
    monkeypatch.setattr(runner.time, 'sleep', lambda s: sleeps.append(s))
    store = runner.GitStore.__new__(runner.GitStore)
    store.repo, store.token, store.read_audit = 'o/r', 't', {}
    assert store.request('/contents/x') == {'ok': True}
    assert len(calls) == 3 and sleeps == [2.0, 3.0]
    assert store.read_audit['requests'] == 3 and store.read_audit['retries'] == 2 and store.read_audit['rate_limit_remaining'] == 812
    assert store.read_audit['failures'] == {'503': 1, '429': 1}


def test_store_request_passes_answers_through(monkeypatch):
    def fake_urlopen(request, timeout=0):
        raise http_error(409)
    monkeypatch.setattr(runner, 'urlopen', fake_urlopen)
    store = runner.GitStore.__new__(runner.GitStore)
    store.repo, store.token, store.read_audit = 'o/r', 't', {}
    with pytest.raises(HTTPError) as caught:
        store.request('/contents/x', 'PUT', {'a': 1})
    assert caught.value.code == 409 and store.read_audit['requests'] == 1
