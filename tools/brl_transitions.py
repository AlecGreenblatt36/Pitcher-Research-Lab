"""Base-running transitions from the official play-by-play (TRANS-01), one season per job.

For every completed regular-season game it reads the official play-by-play and follows the runners play by play:
  contact    for each plate appearance with one of the seven outcomes, the bases and outs at the moment of the
             batted ball (after anything earlier in the plate appearance: steals, wild pitches, passed balls, balks,
             pickoffs) and where each runner and the batter ended up from the play itself: a base, home, or out
  running    what happened earlier in the plate appearance (event types and the change of bases and outs), by the
             bases and outs at its start
  decisions  the main advance decisions (scoring from second on a single, first to third on a single, scoring from
             first on a double, scoring from third and second to third on an out, the double play) by the runner's
             speed on the engine's scale (0.5 + (sprint speed - league mean) / 7.5, public sprint speeds)
Only aggregate counts leave the runner (research/transitions-<season>-<run>.json on the ledger branch); no plays.
"""
from __future__ import annotations
import gzip, json, os, sys, time, traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

BASES = ('1B', '2B', '3B')
DEST = {'1B': '1', '2B': '2', '3B': '3', 'score': 'H'}
SPEED_EDGES = (0.3, 0.4, 0.5, 0.6, 0.7)


def _engine_file(name: str):
    import importlib.util
    path = ROOT / 'brl_engine' / 'runtime' / 'research_lab' / 'pa_model' / (name + '.py')
    spec = importlib.util.spec_from_file_location('brl_' + name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


map_event = _engine_file('outcomes').map_event


def _pi(r):
    v = (r.get('details') or {}).get('playIndex')
    return v if isinstance(v, int) else None


def _rid(r):
    v = ((r.get('details') or {}).get('runner') or {}).get('id')
    return None if v is None else str(v)


def apply_moves(bases: dict, moves: list) -> int:
    """Apply runner movements that happen together (one event); returns outs made. Every runner leaves his base first."""
    outs = 0
    parsed = []
    for r in moves:
        mv = r.get('movement') or {}
        parsed.append((mv.get('start'), mv.get('end'), bool(mv.get('isOut')), _rid(r)))
    for start, _, _, rid in parsed:
        if start in bases and (rid is None or bases[start] == rid or bases[start] is None):
            bases[start] = None
        elif start in bases:
            bases[start] = None
    for _, end, out, rid in parsed:
        if out:
            outs += 1
        elif end in bases and rid is not None:
            bases[end] = rid
    return outs


def mask_of(bases: dict) -> int:
    return sum(1 << i for i, b in enumerate(BASES) if bases.get(b))


def game_rows(doc: dict) -> dict:
    """Contact transitions, earlier running events and decision rows of one game's play-by-play."""
    out = {'contact': [], 'running': [], 'decisions': [], 'mismatch': 0, 'plays': 0}
    key = None
    bases = {b: None for b in BASES}
    outs = 0
    for play in (doc or {}).get('allPlays') or []:
        about = play.get('about') or {}
        if not about.get('isComplete'):
            continue
        half = (about.get('inning'), bool(about.get('isTopInning')))
        if half != key:
            key, bases, outs = half, {b: None for b in BASES}, 0
        result = play.get('result') or {}
        event = str(result.get('eventType') or '')
        outcome = map_event(event)
        events = play.get('playEvents') or []
        pitches = [e.get('index', i) for i, e in enumerate(events) if e.get('isPitch') is True]
        last = max(pitches) if pitches else None
        runners = play.get('runners') or []
        groups = defaultdict(list)
        for order, r in enumerate(runners):
            groups[_pi(r) if _pi(r) is not None else 10_000 + order].append(r)
        start_bases, start_outs = dict(bases), outs
        pre_types, pre_outs = [], 0
        contact_moves = []
        for idx in sorted(groups):
            if outcome is not None and last is not None and idx < last:
                types = sorted(set(str((r.get('details') or {}).get('eventType') or '') for r in groups[idx]))
                pre_types.extend(types)
                pre_outs += apply_moves(bases, groups[idx])
            else:
                contact_moves.append(groups[idx])
        end_outs = (play.get('count') or {}).get('outs')
        if outcome is None:
            for g in contact_moves:
                apply_moves(bases, g)
            outs = min(3, int(end_outs)) if isinstance(end_outs, int) else outs
            if outs >= 3:
                bases = {b: None for b in BASES}
            continue
        out['plays'] += 1
        c_bases, c_outs = dict(bases), min(3, start_outs + pre_outs)
        out['running'].append((mask_of(start_bases), start_outs, tuple(pre_types), mask_of(c_bases), c_outs))
        batter = str(((play.get('matchup') or {}).get('batter') or {}).get('id') or '')
        dest = {}
        for g in contact_moves:
            for r in g:
                rid, mv = _rid(r), (r.get('movement') or {})
                if rid is None:
                    continue
                dest[rid] = 'X' if mv.get('isOut') else DEST.get(mv.get('end'), dest.get(rid))
        if c_outs >= 3:
            out['mismatch'] += 1       # the inning ended before the batted ball (should not happen for a completed plate appearance)
            bases = {b: None for b in BASES}; outs = 3
            continue
        pattern = []
        for b in BASES:
            rid = c_bases[b]
            pattern.append('-' if rid is None else (dest.get(rid) or b[0]))
        pb = dest.get(batter)
        if pb is None:
            pb = 'X' if outcome in ('K', 'BIP_OUT') else ('H' if outcome == 'HR' else '?')
        pattern.append(pb)
        for g in contact_moves:
            apply_moves(bases, g)
        new_outs = min(3, int(end_outs)) if isinstance(end_outs, int) else c_outs + sum(1 for p in pattern if p == 'X')
        made = new_outs - c_outs
        if made != sum(1 for p in pattern if p == 'X'):
            out['mismatch'] += 1
        out['contact'].append((outcome, event, mask_of(c_bases), c_outs, ''.join(pattern), made,
                               tuple(c_bases[b] for b in BASES), batter))
        outs = new_outs
        if outs >= 3:
            bases = {b: None for b in BASES}
    return out


def speed_table(doc: dict, season: int):
    """Engine-scale speed per runner from the season's sprint speed (the previous season when missing)."""
    seasons = doc['seasons']
    sprint, weighted = {}, [0.0, 0.0]
    for y in (season, season - 1):
        for pid, r in ((seasons.get(str(y)) or {}).get('runners') or {}).items():
            if r.get('sprint') is not None:
                sprint.setdefault(pid, float(r['sprint']))
                if y == season:
                    weighted[0] += r.get('pa', 0) * float(r['sprint']); weighted[1] += r.get('pa', 0)
    mean = weighted[0] / weighted[1] if weighted[1] else 27.35
    return {pid: min(1.0, max(0.0, 0.5 + (s - mean) / 7.5)) for pid, s in sprint.items()}, mean


def bucket(speed) -> str:
    if speed is None:
        return 'na'
    return str(sum(speed >= e for e in SPEED_EDGES))


def decisions(row, speed: dict) -> list:
    """(decision, outs, speed bucket, result) rows for the advance decisions in one contact transition."""
    outcome, event, mask, outs, pattern, made, ids, batter = row
    d1, d2, d3, db = pattern
    r1, r2, r3 = ids
    out = []
    if outcome == '1B' and r2:
        out.append(('single_from_2nd', outs, bucket(speed.get(r2)), d2))
    if outcome == '1B' and r1 and not r2:
        out.append(('single_from_1st', outs, bucket(speed.get(r1)), d1))
    if event == 'double' and r1:
        out.append(('double_from_1st', outs, bucket(speed.get(r1)), d1))
    if outcome == 'BIP_OUT' and r3 and outs < 2:
        out.append(('out_from_3rd', outs, bucket(speed.get(r3)), d3))
    if outcome == 'BIP_OUT' and r2 and not r3 and outs < 2:
        out.append(('out_from_2nd', outs, bucket(speed.get(r2)), d2))
    if outcome == 'BIP_OUT' and r1 and outs < 2:
        dp = 'dp' if made >= 2 else 'no'
        out.append(('double_play_runner', outs, bucket(speed.get(r1)), dp))
        out.append(('double_play_batter', outs, bucket(speed.get(batter)), dp))
    return out


def main():
    from brl_bookkeeping_backfill import completed_games, get_json, put, API
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    season = int(os.environ['BRL_SEASON']); run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    workers = int(os.environ.get('BRL_WORKERS') or 6)
    t0 = time.time()
    receipt = {'schema': 'brl.transitions.v1', 'season': season, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat()}
    try:
        games = [g for g in completed_games(season, f'{season}-11-30') if g['game_type'] == 'R']
        receipt['games_listed'] = len(games)
        speed, sprint_mean = speed_table(json.loads(gzip.decompress((ROOT / 'brl_live' / 'running.json.gz').read_bytes())), season)
        receipt['sprint_mean'] = round(sprint_mean, 3); receipt['runners_with_speed'] = len(speed)
        contact = defaultdict(Counter); running = defaultdict(Counter); dec = defaultdict(Counter); events = defaultdict(Counter)
        stats = Counter()

        def fetch(g):
            for attempt in range(3):
                try:
                    return g, get_json(f'{API}/game/{g["game_pk"]}/playByPlay')
                except Exception as exc:
                    if attempt == 2:
                        return g, exc
                    time.sleep(3 * (attempt + 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for g, doc in pool.map(fetch, games):
                if isinstance(doc, Exception) or not doc:
                    stats['failed_games'] += 1
                    continue
                rows = game_rows(doc)
                stats['games'] += 1; stats['plays'] += rows['plays']; stats['mismatch'] += rows['mismatch']
                for row in rows['contact']:
                    outcome, event, mask, outs, pattern, made, ids, batter = row
                    contact[f'{outcome}|{mask}|{outs}'][f'{pattern}|{made}'] += 1
                    if outcome == '2B_3B':
                        events[f'{event}|{mask}|{outs}'][f'{pattern}|{made}'] += 1
                    for d in decisions(row, speed):
                        dec['|'.join(map(str, d[:3]))][d[3]] += 1
                for m0, o0, types, m1, o1 in rows['running']:
                    running[f'{m0}|{o0}'][f'{"+".join(types) or "none"}>{m1}|{o1}'] += 1
        receipt.update(dict(stats))
        receipt['contact'] = {k: dict(v) for k, v in sorted(contact.items())}
        receipt['doubles_and_triples'] = {k: dict(v) for k, v in sorted(events.items())}
        receipt['running'] = {k: dict(v) for k, v in sorted(running.items())}
        receipt['decisions'] = {k: dict(v) for k, v in sorted(dec.items())}
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        receipt['where'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in traceback.extract_tb(exc.__traceback__)[-6:]]
    receipt['seconds'] = round(time.time() - t0, 1); receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(receipt, indent=0, sort_keys=False).replace(token, '[redacted]')
    from brl_bookkeeping_backfill import put as put_file
    put_file(repo, token, f'research/transitions-{season}-{run_id}.json', text.encode(), branch, f'BRL: transitions {season}')
    print(json.dumps({k: receipt.get(k) for k in ('status', 'error', 'games', 'plays', 'mismatch', 'failed_games', 'seconds')}))
    if receipt['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
