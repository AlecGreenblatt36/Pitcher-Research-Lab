"""The ledger store writes by blob sha: no download of the file it replaces, nothing written when the bytes match,
immutable objects never replaced, and a 409 retried."""
import base64, hashlib, io
from urllib.error import HTTPError
import pytest
from cloud import runner as R
from app.safety import Blocked


def blob_sha(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class FakeApi:
    def __init__(self, files, conflicts=0):
        self.files, self.conflicts, self.calls = dict(files), conflicts, []

    def __call__(self, path, method='GET', data=None):
        self.calls.append((method, path.split('?')[0]))
        name = path.split('/contents/')[1].split('?')[0]
        if method == 'GET':
            if name not in self.files:
                raise HTTPError(path, 404, 'Not Found', {}, io.BytesIO(b''))
            raw = self.files[name]
            return {'sha': blob_sha(raw), 'size': len(raw), 'content': '', 'encoding': 'none'}   # a large file: no content
        if self.conflicts:
            self.conflicts -= 1
            raise HTTPError(path, 409, 'Conflict', {}, io.BytesIO(b''))
        if name in self.files:
            assert data.get('sha') == blob_sha(self.files[name])
        self.files[name] = base64.b64decode(data['content'])
        return {'commit': {'sha': 'c' * 40}}


def store(api):
    s = object.__new__(R.GitStore)
    s.repo, s.token, s.key, s.read_audit = 'o/r', 't', b'k' * 32, {}
    s.request = api
    return s


def test_put_compares_by_sha_and_never_downloads(monkeypatch):
    monkeypatch.setattr(R.time, 'sleep', lambda s: None)
    api = FakeApi({'ledger.json': b'old'}, conflicts=2)
    s = store(api)
    assert s.put('ledger.json', b'new')['commit']['sha'] == 'c' * 40
    assert api.files['ledger.json'] == b'new'
    assert all('/git/blobs/' not in p for _, p in api.calls)              # metadata only
    assert s.put('ledger.json', b'new') is None                            # the same bytes: nothing written
    assert s.put('fresh.json', b'x')['commit']                             # a new file
    with pytest.raises(Blocked):
        s.put('fresh.json', b'y', immutable=True)
    assert s.put('fresh.json', b'x', immutable=True) is None
