from __future__ import annotations
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

# Working root for private, decrypted material (never inside the repository checkout).
ROOT = Path(os.environ.get('BRL_WORK_ROOT') or Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'brl-work')


def now():
    return datetime.now(timezone.utc)


def iso():
    return now().isoformat()


def timestamp(value):
    """Timezone-aware UTC datetime from an ISO string (Z allowed); naive values are refused."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError('Timestamp must include timezone')
        return value.astimezone(timezone.utc)
    v = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if v.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return v.astimezone(timezone.utc)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def content_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data, exclusive=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and p.exists():
        raise FileExistsError(str(p))
    p.write_bytes(canonical(data))
