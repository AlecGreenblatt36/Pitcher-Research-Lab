import base64
import os
from urllib.error import HTTPError
import pytest
from app.safety import Blocked
from cloud.security import seal, unseal
from brl_live.verified_store import VerifiedGitStore, checked_content, git_blob_sha


def meta(raw):
    return {'encoding': 'base64', 'sha': git_blob_sha(raw), 'size': len(raw),
            'content': base64.encodebytes(raw).decode()}


def store_for(inline, blob):
    store = object.__new__(VerifiedGitStore)
    store.read_audit = {'contents_verified': 0, 'blob_fallbacks': 0, 'bytes_verified': 0}
    calls = []
    def request(path):
        calls.append(path)
        value = inline if path.startswith('/contents/') else blob
        if isinstance(value, Exception):
            raise value
        return value
    store.request = request
    return store, calls


@pytest.mark.parametrize('raw', [b'', b'abc', bytes(range(256)), b'x' * 1_100_000])
def test_correct_inline_unchanged(raw):
    store, calls = store_for(meta(raw), AssertionError('unexpected fallback'))
    assert store.read('sample.enc') == (raw, git_blob_sha(raw))
    assert len(calls) == 1


@pytest.mark.parametrize('failure', ['encoding_none', 'empty_content', 'wrong_length', 'wrong_same_size', 'bad_base64'])
def test_corrupt_or_missing_inline_recovers_from_pinned_blob(failure):
    key = os.urandom(32)
    encrypted = seal(b'private-inputs', key, 'test')
    inline = meta(encrypted)
    if failure == 'encoding_none': inline.update(encoding='none', content='')
    elif failure == 'empty_content': inline['content'] = ''
    elif failure == 'wrong_length': inline['content'] = base64.b64encode(b'x' * (len(encrypted)+330)).decode()
    elif failure == 'wrong_same_size': inline['content'] = base64.b64encode(b'x' * len(encrypted)).decode()
    else: inline['content'] = '###not-base64###'
    store, calls = store_for(inline, meta(encrypted))
    actual, _ = store.read('private/test.enc')
    assert actual == encrypted and unseal(actual, key, 'test') == b'private-inputs'
    assert calls[-1] == '/git/blobs/' + git_blob_sha(encrypted)
    assert len(calls) == 2 and store.read_audit['blob_fallbacks'] == 1


@pytest.mark.parametrize('failure', ['sha', 'size', 'payload', 'type', 'encoding'])
def test_invalid_fallback_fails_closed(failure):
    raw = b'abc'; inline = meta(raw); inline['encoding'] = 'none'
    blob = meta(raw)
    if failure == 'sha': blob['sha'] = '0'*40
    elif failure == 'size': blob['size'] = 1
    elif failure == 'payload': blob['content'] = base64.b64encode(b'xyz').decode()
    elif failure == 'type': blob = []
    else: blob['encoding'] = 'none'
    store, calls = store_for(inline, blob)
    with pytest.raises(Blocked, match='integrity check failed'): store.read('test')
    assert len(calls) == 2


@pytest.mark.parametrize('code', [403, 404, 500])
def test_http_errors_not_misrepresented(code):
    store, calls = store_for(HTTPError('test', code, 'redacted', None, None), None)
    if code == 404: assert store.read('test') is None
    else:
        with pytest.raises(HTTPError): store.read('test')
    assert len(calls) == 1


@pytest.mark.parametrize('updates', [{'size': -1}, {'size': True}, {'size': 100_000_001}, {'sha': 'bad'}])
def test_bad_metadata_fails_without_fetch(updates):
    inline = meta(b'abc'); inline.update(updates)
    store, calls = store_for(inline, None)
    with pytest.raises(Blocked): store.read('test')
    assert len(calls) == 1
