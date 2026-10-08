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
    """Apply runner movements that happen together (one event); returns outs made. Every runner leaves his base first.
    A runner listed more than once in the event (a steal and then an error on the throw) moves from his first start to
    his last end, so he is never left on two bases."""
    outs = 0
    per, order, parsed = {}, [], []
    for r in moves:
        mv = r.get('movement') or {}
        rid = _rid(r)
        if rid is None:
            parsed.append((mv.get('start'), mv.get('end'), bool(mv.get('isOut')), None))
        elif rid not in per:
            per[rid] = [mv.get('start'), mv.get('end'), bool(mv.get('isOut'))]
            order.append(rid)
        else:
            per[rid][1] = mv.get('end')
            per[rid][2] = per[rid][2] or bool(mv.get('isOut'))
    parsed = [(per[rid][0], per[rid][1], per[rid][2], rid) for rid in order] + parsed
    for start, _, _, rid in parsed:
        if start in bases:
            bases[start] = None
    for _, end, out, rid in parsed:
        if out:
            outs += 1
        elif end in bases and rid is not None:
            bases[end] = rid
    return outs


STEAL_TYPES = {'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home', 'caught_stealing_2b', 'caught_stealing_3b',
               'caught_stealing_home', 'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home'}


def pre_pattern(start_bases: dict, groups: list) -> str:
    """Where each runner on base at the start of the plate appearance is after the running events before the batted
    ball (or before the play ended without one): '1' '2' '3' a base, 'H' scored, 'X' out, '-' no runner; keyed by the
    base he started on, so a pinch runner or the extra-inning runner is followed by base, not by id."""
    dest, first = {}, {}
    for g in groups:
        for r in g:
            rid, mv = _rid(r), (r.get('movement') or {})
            if rid is None:
                continue
            if rid not in first:
                first[rid] = mv.get('start') if mv.get('start') in BASES else None
            dest[rid] = 'X' if mv.get('isOut') else DEST.get(mv.get('end'), dest.get(rid))
    by_base = {first[rid]: d for rid, d in dest.items() if first.get(rid) in BASES and d is not None}
    return ''.join('-' if start_bases.get(b) is None else (by_base.get(b) or b[0]) for b in BASES)


def _types(groups: list) -> list:
    return sorted(set(str((r.get('details') or {}).get('eventType') or '') for g in groups for r in g) - {''})


def resync(bases: dict, play: dict) -> None:
    """Take the official runners after the play when the feed lists them (fixes pinch runners and the extra-inning runner)."""
    m = play.get('matchup') or {}
    keys = {'1B': 'postOnFirst', '2B': 'postOnSecond', '3B': 'postOnThird'}
    if not any(k in m for k in keys.values()):
        return
    for b, k in keys.items():
        v = (m.get(k) or {}).get('id')
        bases[b] = None if v is None else str(v)


def mask_of(bases: dict) -> int:
    return sum(1 << i for i, b in enumerate(BASES) if bases.get(b))


def game_rows(doc: dict) -> dict:
    """Contact transitions, earlier running events and decision rows of one game's play-by-play."""
    out = {'contact': [], 'running': [], 'pre': [], 'phase': Counter(), 'decisions': [], 'mismatch': 0, 'plays': 0}
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
            if int(about.get('inning') or 0) >= 10:
                bases['2B'] = 'auto'           # extra innings start with a runner on second (since 2020; id unknown until he moves)
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
        contact_moves, pre_moves = [], []
        for idx in sorted(groups):
            if outcome is not None and last is not None and idx < last:
                types = sorted(set(str((r.get('details') or {}).get('eventType') or '') for r in groups[idx]))
                pre_types.extend(types)
                pre_outs += apply_moves(bases, groups[idx])
                pre_moves.append(groups[idx])
            else:
                contact_moves.append(groups[idx])
        end_outs = (play.get('count') or {}).get('outs')
        if outcome is None:
            # A plate appearance cut short by a running play (the third out on a pickoff or caught stealing, a walk-off
            # wild pitch or balk): every movement came before any batted ball.
            if contact_moves and mask_of(start_bases) and start_outs < 3:
                types = _types(contact_moves)
                scratch = dict(start_bases)
                made = sum(apply_moves(scratch, g) for g in contact_moves)
                out['pre'].append((mask_of(start_bases), start_outs, tuple(types), pre_pattern(start_bases, contact_moves), min(3 - start_outs, made), 'T'))
                for t in types:
                    out['phase'][('truncated', t)] += 1
            for g in contact_moves:
                apply_moves(bases, g)
            outs = min(3, int(end_outs)) if isinstance(end_outs, int) else outs
            if outs >= 3:
                bases = {b: None for b in BASES}
            else:
                resync(bases, play)
            continue
        out['plays'] += 1
        c_bases, c_outs = dict(bases), min(3, start_outs + pre_outs)
        out['running'].append((mask_of(start_bases), start_outs, tuple(pre_types), mask_of(c_bases), c_outs))
        if mask_of(start_bases):
            out['pre'].append((mask_of(start_bases), start_outs, tuple(sorted(set(pre_types) - {''})), pre_pattern(start_bases, pre_moves), c_outs - start_outs, 'P'))
        for t in set(pre_types) - {''}:
            out['phase'][('pre', t)] += 1
        for t in _types(contact_moves):
            out['phase'][('final_pitch', t)] += 1
        batter = str(((play.get('matchup') or {}).get('batter') or {}).get('id') or '')
        # Each runner's destination, keyed by the base he started the play from (a pinch runner or the extra-inning runner
        # may carry another id than the one tracked on that base); the batter's by his id or a start off the bases.
        dest, first_start = {}, {}
        for g in contact_moves:
            for r in g:
                rid, mv = _rid(r), (r.get('movement') or {})
                if rid is None:
                    continue
                if rid not in first_start:
                    first_start[rid] = mv.get('start') if mv.get('start') in BASES else None
                dest[rid] = 'X' if mv.get('isOut') else DEST.get(mv.get('end'), dest.get(rid))
        by_base = {first_start[rid]: d for rid, d in dest.items() if first_start.get(rid) in BASES and rid != batter}
        batter_dest = dest.get(batter)
        if batter_dest is None:
            starts_off = [d for rid, d in dest.items() if first_start.get(rid) is None and rid not in c_bases.values()]
            batter_dest = starts_off[0] if len(starts_off) == 1 else None
        if c_outs >= 3:
            out['mismatch'] += 1       # the inning ended before the batted ball (should not happen for a completed plate appearance)
            bases = {b: None for b in BASES}; outs = 3
            continue
        pattern = []
        for b in BASES:
            rid = c_bases[b]
            pattern.append('-' if rid is None else (by_base.get(b) or b[0]))
        pb = batter_dest
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
        else:
            resync(bases, play)
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
        pre = defaultdict(Counter); phase = Counter()
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
                for m0, o0, types, pattern, made, kind in rows['pre']:
                    pre[f'{m0}|{o0}'][f'{"+".join(types) or "none"}>{pattern}|{made}|{kind}'] += 1
                for (ph, t), n in rows['phase'].items():
                    phase[f'{ph}|{t}'] += n
        receipt.update(dict(stats))
        receipt['contact'] = {k: dict(v) for k, v in sorted(contact.items())}
        receipt['doubles_and_triples'] = {k: dict(v) for k, v in sorted(events.items())}
        receipt['running'] = {k: dict(v) for k, v in sorted(running.items())}
        # Running events by the bases and outs at the start of each plate appearance with runners on (P) or cut short
        # by one (T): event types, where each runner was after them ('1' '2' '3' 'H' 'X' '-', runner on first first) and
        # the outs made. phase: how often each event type happens before the last pitch, on it, or ends the play.
        receipt['pre'] = {k: dict(v) for k, v in sorted(pre.items())}
        receipt['phase'] = dict(sorted(phase.items()))
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
