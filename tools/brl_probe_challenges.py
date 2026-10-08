"""Probe: does the official play-by-play record 2026 ball-strike challenges, and in what shape?

Reads the play-by-play of a few completed late-September 2026 regular-season games and reports only field names, value
counts and the shape of review records found on pitch events (no player rows). Output: research/challenge-probe-<run>.json.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))


def walk_keys(obj, prefix, out: Counter, depth=0):
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = prefix + '.' + k if prefix else k
            if any(w in k.lower() for w in ('review', 'challenge', 'abs')):
                out[key] += 1
            walk_keys(v, key, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:50]:
            walk_keys(v, prefix + '[]', out, depth + 1)


def main():
    from brl_bookkeeping_backfill import completed_games, get_json, put, API
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    games = [g for g in completed_games(2026, '2026-09-27') if g['game_type'] == 'R' and g['date'] >= '2026-09-20'][:12]
    keys, review_types, overturned, challengers, descriptions, shapes = Counter(), Counter(), Counter(), Counter(), Counter(), []
    pitch_events = with_review = 0
    for g in games:
        doc = get_json(f'{API}/game/{g["game_pk"]}/playByPlay') or {}
        walk_keys(doc, '', keys)
        for play in doc.get('allPlays') or []:
            for e in play.get('playEvents') or []:
                if e.get('isPitch'):
                    pitch_events += 1
                rd = e.get('reviewDetails') or (e.get('details') or {}).get('reviewDetails')
                if rd:
                    with_review += 1
                    review_types[str(rd.get('reviewType'))] += 1
                    overturned[str(rd.get('isOverturned'))] += 1
                    challengers[str(sorted(k for k in rd if 'challenge' in k.lower() or 'player' in k.lower()))] += 1
                    if len(shapes) < 6:
                        shapes.append({'event_keys': sorted(e.keys()), 'review': {k: (v if not isinstance(v, dict) else sorted(v.keys())) for k, v in rd.items()},
                                       'is_pitch': e.get('isPitch'), 'code': (e.get('details') or {}).get('code'), 'description': (e.get('details') or {}).get('description')})
                d = str((e.get('details') or {}).get('description') or '')
                if 'challenge' in d.lower() or 'abs' in d.lower():
                    descriptions[d[:80]] += 1
    out = {'schema': 'brl.challenge-probe.v1', 'run_id': run_id, 'games': [g['game_pk'] for g in games], 'pitch_events': pitch_events,
           'events_with_review': with_review, 'review_types': dict(review_types), 'overturned': dict(overturned), 'challenger_fields': dict(challengers),
           'keys_with_review_or_challenge': dict(keys.most_common(40)), 'description_examples': dict(descriptions.most_common(12)), 'shapes': shapes}
    put(repo, token, f'research/challenge-probe-{run_id}.json', json.dumps(out, indent=1).encode(), branch, 'BRL: challenge probe')
    print(json.dumps({k: out[k] for k in ('pitch_events', 'events_with_review', 'review_types', 'overturned')}))


if __name__ == '__main__':
    main()
