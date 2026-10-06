"""Verify encrypted/public Git objects before consumption; never alter their bytes.

A contents response is accepted only when both its byte count and Git object
identity match. Missing, truncated or inconsistent inline payloads are fetched
once by immutable blob SHA, then checked again. No weakening of AES-GCM checks.
"""
from __future__ import annotations
import base64
import binascii
import hashlib
import re
from urllib.error import HTTPError
from urllib.parse import quote

from app.safety import Blocked
from cloud.runner import GitStore, BRANCH


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()


def checked_content(value: dict, expected_sha: str, expected_size: int):
    """Return verified bytes or None, without including payloads in diagnostics."""
    if not isinstance(value, dict) or value.get('encoding') != 'base64':
        return None
    if value.get('sha') != expected_sha or value.get('size') != expected_size:
        return None
    text = value.get('content')
    if not isinstance(text, str):
        return None
    try:
        raw = base64.b64decode(''.join(text.split()), validate=True)
    except (ValueError, binascii.Error):
        return None
    if len(raw) != expected_size or git_blob_sha(raw) != expected_sha:
        return None
    return raw


class VerifiedGitStore(GitStore):
    """Same write/publish behavior as the locked runtime; stricter transport IO."""
    def __init__(self, *args, **kwargs):
        self.read_audit = {'contents_verified': 0, 'blob_fallbacks': 0, 'bytes_verified': 0}
        super().__init__(*args, **kwargs)

    def read(self, path):
        try:
            metadata = self.request('/contents/' + quote(path, safe='/') + '?ref=' + BRANCH)
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        if not isinstance(metadata, dict):
            raise Blocked('Git object metadata is not a file')
        sha, size = metadata.get('sha'), metadata.get('size')
        if (not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha)
                or type(size) is not int or not 0 <= size <= 100_000_000):
            raise Blocked('Git object identity/size metadata invalid')
        raw = checked_content(metadata, sha, size)
        if raw is None:
            self.read_audit['blob_fallbacks'] += 1
            # Exactly one fallback for this object; no recursive retry loop.
            blob = self.request('/git/blobs/' + sha)
            raw = checked_content(blob, sha, size)
            if raw is None:
                raise Blocked('Git object integrity check failed after blob fetch')
        else:
            self.read_audit['contents_verified'] += 1
        self.read_audit['bytes_verified'] += len(raw)
        return raw, sha
