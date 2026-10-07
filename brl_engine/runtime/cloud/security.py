"""AES-256-GCM sealing with the same wire format as the inherited runtime:
8-byte header BRLAESG1, 12-byte random nonce, ciphertext+tag; the purpose string is the AAD."""
from __future__ import annotations
import hashlib, io, os, re, stat, zipfile
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

HEADER = b'BRLAESG1'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def key_bytes(value: str) -> bytes:
    value = (value or '').strip()
    if not re.fullmatch('[a-fA-F0-9]{64}', value):
        raise ValueError('Invalid key format')
    return bytes.fromhex(value)


def seal(raw: bytes, key: bytes, purpose: str) -> bytes:
    nonce = os.urandom(12)
    return HEADER + nonce + AESGCM(key).encrypt(nonce, raw, purpose.encode())


def unseal(raw: bytes, key: bytes, purpose: str) -> bytes:
    if raw[:8] != HEADER:
        raise ValueError('Sealed payload format mismatch')
    return AESGCM(key).decrypt(raw[8:20], raw[20:], purpose.encode())


def extract_verified(raw: bytes, expected_hash: str, destination):
    if sha(raw) != expected_hash:
        raise ValueError('Package hash mismatch')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = set(); size = 0
        for item in z.infolist():
            p = (destination / item.filename).resolve(); size += item.file_size
            if not p.is_relative_to(destination) or '\\' in item.filename or item.filename in names or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Unsafe package path')
            names.add(item.filename)
        if size > 500_000_000:
            raise ValueError('Expanded package too large')
        z.extractall(destination)
    return destination
