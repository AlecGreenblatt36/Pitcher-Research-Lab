"""PITCH-01: one row per pitch from the sealed pitch-physics study of a season (research_lab.pa_model.pitch_chain reads it).

Runs in GitHub Actions, one season per run (BRL_SEASON). The study (private/statcast/season-<year>.enc, the official
play-by-play's pitch list per plate appearance) is unsealed, flattened to one row per pitch with the count before the
pitch, its type, result code and physics, and sealed again as gzip CSV at private/statcast/pitches-<year>.enc
(purpose pitch-table-season-<year>). The public receipt (pitches/<year>.json) carries counts only.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))

PITCH_COLUMNS = ['date_key', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter', 'stand', 'p_throws', 'balls', 'strikes',
                 'ptype', 'code', 'speed', 'spin', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone', 'ext', 'outcome', 'last']


def pitches_path(year: int) -> str:
    return f'private/statcast/pitches-{int(year)}.enc'


def pitches_purpose(year: int) -> str:
    return f'pitch-table-season-{int(year)}'


def receipt_path(year: int) -> str:
    return f'pitches/{int(year)}.json'


def rows_from_study(study: dict) -> list[tuple]:
    """Flatten one season's study (game -> rows with compact pitch lists) to pitch rows."""
    out = []
    for pk, game in study['games'].items():
        day = str(game.get('date') or '')[:10]
        for r in game['rows']:
            pitches = r.get('pitches') or []
            n = len(pitches)
            batter = -1 if r.get('b') is None else int(r['b'])
            for i, p in enumerate(pitches):
                ptype, code, balls, strikes, start, _end, spin, pfx_x, pfx_z, plate_x, plate_z, _rx, _rz, ext, zone = p
                out.append((day, int(pk), int(r['i']) + 1, i + 1, int(r['p']), batter, str(r.get('s') or 'R'), str(r.get('t') or ''),
                            int(balls or 0), int(strikes or 0), ptype or '', str(code or ''), start, spin, pfx_x, pfx_z, plate_x, plate_z,
                            zone if isinstance(zone, int) else None, ext, str(r.get('o') or ''), int(i == n - 1)))
    return out


def main():
    import pandas as pd
    from cloud.security import seal, unseal, key_bytes, sha
    from brl_live.bookkeeping_season import study_path, study_purpose
    sys.path.append(str(ROOT / 'tools'))          # after the packages: tools/brl_replay.py must not shadow the brl_replay package
    from brl_fit_model import read_blob, put_bytes
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key = key_bytes(os.environ['BRL_PA_PACKAGE_KEY'])
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    year = int(os.environ['BRL_SEASON'])
    t0 = time.time()
    raw = read_blob(repo, token, study_path(year), branch)
    if raw is None:
        raise ValueError(f'no sealed study for {year}')
    study = json.loads(gzip.decompress(unseal(raw, key, study_purpose(year))))
    frame = pd.DataFrame(rows_from_study(study), columns=PITCH_COLUMNS)
    frame = frame.sort_values(['date_key', 'game_pk', 'at_bat_number', 'pitch_number'], kind='mergesort').reset_index(drop=True)
    plain_csv = frame.to_csv(index=False).encode()
    cipher = seal(gzip.compress(plain_csv, mtime=0), key, pitches_purpose(year))
    put_bytes(repo, token, pitches_path(year), cipher, branch, f'BRL: pitch table {year} (PITCH-01)')
    codes = frame['code'].value_counts().to_dict()
    receipt = {'schema': 'brl.pitch-table-receipt.v1', 'experiment': 'PITCH-01', 'year': year, 'games': int(frame['game_pk'].nunique()),
               'plate_appearances': int(frame[['game_pk', 'at_bat_number']].drop_duplicates().shape[0]), 'pitches': int(len(frame)),
               'result_codes': {str(k): int(v) for k, v in sorted(codes.items(), key=lambda kv: -kv[1])},
               'with_zone_pct': round(float(frame['zone'].notna().mean()) * 100, 2), 'with_speed_pct': round(float(frame['speed'].notna().mean()) * 100, 2),
               'pitch_types': {str(k): int(v) for k, v in frame['ptype'].value_counts().items()},
               'date_min': str(frame['date_key'].min()), 'date_max': str(frame['date_key'].max()), 'columns': PITCH_COLUMNS,
               'plaintext_csv_sha256': hashlib.sha256(plain_csv).hexdigest(), 'cipher_sha256': sha(cipher), 'cipher_bytes': len(cipher),
               'path': pitches_path(year), 'format': 'AES-256-GCM; BRLAESG1 header; 12-byte nonce; AAD BRL:' + pitches_purpose(year) + '; gzip CSV inside',
               'sealed_at': datetime.now(timezone.utc).isoformat(), 'seconds': round(time.time() - t0), 'run_id': os.environ.get('GITHUB_RUN_ID')}
    put_bytes(repo, token, receipt_path(year), json.dumps(receipt, indent=1).encode(), branch, f'BRL: pitch table {year} receipt')
    print(json.dumps({k: receipt[k] for k in ('year', 'games', 'plate_appearances', 'pitches', 'with_zone_pct', 'cipher_bytes', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
