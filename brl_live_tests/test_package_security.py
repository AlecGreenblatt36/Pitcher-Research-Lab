"""The sealed data package: a wrong key, a flipped byte or a changed purpose is refused (AES-GCM's tag), a payload without
the header is refused, the zip extractor refuses paths outside the folder, symbolic links and repeated names, and the
public receipt never quotes data values (REVIEW-01's security follow-ups, October 18)."""
import base64
import hashlib
import io
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
sys.path.insert(0, str(ROOT))
from cloud.security import seal, unseal, key_bytes, extract_verified, sha  # noqa: E402
import brl_engine.entrypoint as ep  # noqa: E402

KEY = bytes(range(32))


def test_round_trip_and_tampering_refused():
    raw = b'plate appearances'
    sealed = seal(raw, KEY, 'data-package')
    assert unseal(sealed, KEY, 'data-package') == raw
    with pytest.raises(InvalidTag):
        unseal(sealed, bytes(32), 'data-package')                       # wrong key
    flipped = bytearray(sealed); flipped[-1] ^= 1
    with pytest.raises(InvalidTag):
        unseal(bytes(flipped), KEY, 'data-package')                     # one flipped bit in the tag
    body = bytearray(sealed); body[25] ^= 1
    with pytest.raises(InvalidTag):
        unseal(bytes(body), KEY, 'data-package')                        # one flipped bit in the ciphertext
    with pytest.raises(InvalidTag):
        unseal(sealed, KEY, 'runtime')                                  # sealed for another purpose
    with pytest.raises(ValueError, match='format mismatch'):
        unseal(b'NOTBRL00' + sealed[8:], KEY, 'data-package')


def test_key_format_checked_without_echoing_it():
    with pytest.raises(ValueError) as e:
        key_bytes('zz' * 32)
    assert 'zz' not in str(e.value)
    assert key_bytes('AB' * 32) == bytes([0xAB]) * 32


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode:
                info.external_attr = mode << 16
            z.writestr(info, data)
    return buf.getvalue()


def test_extractor_refuses_unsafe_archives(tmp_path):
    good = _zip([('data/a.txt', b'x', None)])
    assert (extract_verified(good, sha(good), tmp_path / 'ok') / 'data' / 'a.txt').read_bytes() == b'x'
    with pytest.raises(ValueError, match='hash mismatch'):
        extract_verified(good, '0' * 64, tmp_path / 'h')
    for bad in (_zip([('../escape.txt', b'x', None)]),
                _zip([('data/link', b'/etc/passwd', stat.S_IFLNK | 0o777)]),
                _zip([('a\\b.txt', b'x', None)])):
        with pytest.raises(ValueError, match='Unsafe package path'):
            extract_verified(bad, sha(bad), tmp_path / 'bad')
    assert not (tmp_path / 'escape.txt').exists()


def test_restore_package_verifies_both_hashes_and_extracts_safely(tmp_path, monkeypatch):
    plain = _zip([('data/x.csv', b'1,2', None)])
    cipher = seal(plain, KEY, 'data-package')
    manifest = {'branch': 'b', 'path': 'private/data-package.enc', 'purpose': 'data-package',
                'cipher_sha256': hashlib.sha256(cipher).hexdigest(), 'plaintext_sha256': hashlib.sha256(plain).hexdigest()}
    monkeypatch.setattr(ep, 'api', lambda url, token: {'encoding': 'base64', 'content': base64.b64encode(cipher).decode()})
    out = ep.restore_package('o/r', 't', KEY.hex(), manifest, tmp_path / 'd')
    assert (out / 'data' / 'x.csv').read_bytes() == b'1,2'
    with pytest.raises(ValueError, match='Data package hash mismatch'):
        ep.restore_package('o/r', 't', KEY.hex(), dict(manifest, cipher_sha256='0' * 64), tmp_path / 'e')
    with pytest.raises(InvalidTag):
        ep.restore_package('o/r', 't', bytes(32).hex(), manifest, tmp_path / 'f')
    evil = _zip([('../../outside.txt', b'x', None)])
    evil_c = seal(evil, KEY, 'data-package')
    monkeypatch.setattr(ep, 'api', lambda url, token: {'encoding': 'base64', 'content': base64.b64encode(evil_c).decode()})
    with pytest.raises(ValueError, match='Unsafe package path'):
        ep.restore_package('o/r', 't', KEY.hex(), dict(manifest, cipher_sha256=hashlib.sha256(evil_c).hexdigest(),
                                                       plaintext_sha256=hashlib.sha256(evil).hexdigest()), tmp_path / 'g' / 'h')


def test_receipt_errors_never_quote_values():
    assert ep.safe_error(ValueError('Unsafe package path')) == 'ValueError: Unsafe package path'
    msg = ep.safe_error(KeyError("'Aaron Judge' row 6600123"))
    assert 'Judge' not in msg and '6600123' not in msg and msg.startswith('KeyError')
    assert '?ref=' not in ep.redact('HTTP Error 404: https://api.github.com/repos/o/r/contents/x?ref=brl-live-data')
    assert ep.redact('no PA history before cutoff 2026-10-18') == 'no PA history before cutoff 2026-10-18'
