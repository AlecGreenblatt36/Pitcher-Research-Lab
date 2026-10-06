"""Prior-day player-history refresh, with immutable encrypted source revisions.

No PA/starter coefficients are fitted or edited. This module consumes public
Statcast/MLB bytes; it never writes unencrypted data to the public Git store.
The seed corpus is preserved byte-for-byte when assembling updated history.
Official game PA identities/results must reconcile before a date is accepted.
A provider's empty/partial/error response never advances coverage.

Schema references: https://baseballsavant.mlb.com/csv-docs
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.safety import Blocked
from app.common import canonical, content_hash, timestamp
from cloud.security import seal, unseal
from research_lab.pa_model.outcomes import map_event, EXCLUDED_EVENTS

GAME_TYPES = frozenset(('R', 'F', 'D', 'L', 'W'))
# Explicit source-observed non-PA action; an atBatIndex alone is not a PA.
NON_PA_EVENTS = EXCLUDED_EVENTS | {'pickoff_caught_stealing_2b'}
BASE_END = '2026-09-27'
BASE_SHA256 = 'e503e223eb4d6a2222ed35abe0439a0e6eea7fe85bf9061bcac96d0065a8a197'
INDEX_PATH = 'private/history-index.enc'
INDEX_PURPOSE = 'private:history-index:v1'
SCHEMA = 'brl.history-day.v1'
REQUIRED = {'game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'events',
            'batter', 'pitcher', 'stand', 'p_throws', 'home_team', 'away_team',
            'inning', 'inning_topbot', 'outs_when_up', 'game_type'}
# Only these source fields enter the inherited history; future-appearance
# fields (e.g. days_until_next_game) are never copied.
SOURCE_FIELDS = ['game_date', 'game_year', 'game_pk', 'at_bat_number', 'pitch_number',
                 'batter', 'pitcher', 'stand', 'p_throws', 'home_team', 'away_team',
                 'inning', 'inning_topbot', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b',
                 'home_score', 'away_score', 'bat_score', 'fld_score', 'bat_score_diff',
                 'n_thruorder_pitcher', 'batter_days_since_prev_game',
                 'pitcher_days_since_prev_game', 'age_bat', 'age_pit', 'fielder_2', 'game_type']


def clock() -> datetime:
    return datetime.now(timezone.utc)


def previous_day(now: datetime) -> date:
    if now.tzinfo is None or now.utcoffset() is None:
        raise Blocked('Refresh clock must include timezone')
    return now.astimezone(ZoneInfo('America/New_York')).date() - timedelta(days=1)


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    requested_at: str
    finished_at: str
    body: bytes

    def check(self, origin: datetime) -> None:
        requested, finished = timestamp(self.requested_at), timestamp(self.finished_at)
        if not requested <= finished <= origin:
            raise Blocked('Source capture is after refresh origin or has invalid ordering')
        if not self.url.startswith(('https://statsapi.mlb.com/', 'https://baseballsavant.mlb.com/')):
            raise Blocked('Unapproved public source')

    def to_dict(self) -> dict:
        return {'name': self.name, 'url': self.url, 'requested_at': self.requested_at,
                'finished_at': self.finished_at,
                'sha256': hashlib.sha256(self.body).hexdigest(),
                'body_base64': base64.b64encode(self.body).decode()}

    @classmethod
    def from_dict(cls, value: dict) -> 'Source':
        raw = base64.b64decode(value['body_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != value['sha256']:
            raise Blocked('Stored source hash mismatch')
        return cls(value['name'], value['url'], value['requested_at'], value['finished_at'], raw)


class Fetcher:
    """Bounded public reads; errors contain no source body or credentials."""
    def get(self, name: str, url: str) -> Source:
        requested = clock().isoformat()
        try:
            request = Request(url, headers={'User-Agent': 'BaseballResearchLab-history/1.0',
                                            'Accept': 'text/csv,application/json'})
            with urlopen(request, timeout=55) as response:
                raw = response.read(25_000_001)
                if response.status != 200 or len(raw) > 25_000_000:
                    raise Blocked('Source status or response size invalid')
        except Exception as exc:
            raise Blocked('History source unavailable: ' + type(exc).__name__) from None
        return Source(name, url, requested, clock().isoformat(), raw)


def schedule_cases(source: Source, day: str, origin: datetime) -> tuple[list[dict], list[dict]]:
    source.check(origin)
    doc = json.loads(source.body)
    if not isinstance(doc.get('dates'), list) or type(doc.get('totalGames')) is not int:
        raise Blocked('Schedule missing explicit completeness fields')
    listings = [g for b in doc['dates'] for g in b.get('games', [])]
    if len(listings) != doc['totalGames'] or any(b.get('date') != day for b in doc['dates']):
        raise Blocked('Schedule date/count mismatch')
    ids = [g['gamePk'] for g in listings]
    if len(ids) != len(set(ids)):
        raise Blocked('Duplicate schedule game IDs')
    finals, exclusions = [], []
    for g in listings:
        if g['gameType'] not in GAME_TYPES:
            exclusions.append({'game_pk': g['gamePk'], 'reason': 'unsupported_game_type'})
            continue
        status = g['status']
        detail = status.get('detailedState', '')
        if detail.startswith(('Cancelled', 'Canceled', 'Postponed')):
            exclusions.append({'game_pk': g['gamePk'], 'reason': detail})
        elif status.get('codedGameState') == 'F' and detail in ('Final', 'Completed Early'):
            finals.append(g)
        else:
            raise Blocked('Prior-day game is not final; coverage remains incomplete')
    return finals, exclusions


def official_pa(feed: Source, expected: dict, day: str, origin: datetime) -> tuple[dict, dict]:
    """Complete PA identities are distinct from between-PA/pickoff actions."""
    feed.check(origin)
    doc = json.loads(feed.body)
    pk = int(expected['gamePk'])
    gd, ld = doc['gameData'], doc['liveData']
    if doc.get('gamePk') != pk or gd['game']['type'] != expected['gameType']:
        raise Blocked('Official game identity/type mismatch')
    if gd['datetime']['officialDate'] != day:
        raise Blocked('Rescheduled/resumed game requires an explicit completion-date adapter')
    if gd['status'].get('codedGameState') != 'F':
        raise Blocked('Official game not final')
    # Reconcile independently exposed final totals before admitting labels.
    scores = {}
    for side in ('away', 'home'):
        a = ld['linescore']['teams'][side]['runs']
        b = ld['boxscore']['teams'][side]['teamStats']['batting']['runs']
        if type(a) is not int or a != b or a < 0:
            raise Blocked('Final score cross-check failed')
        if gd['teams'][side]['id'] != expected['teams'][side]['team']['id']:
            raise Blocked('Official team identity mismatch')
        scores[side] = a
    if scores['away'] == scores['home']:
        raise Blocked('Final tied game cannot populate winner history')
    identities = {}
    plays = ld.get('plays', {}).get('allPlays')
    if not isinstance(plays, list) or not plays:
        raise Blocked('Final game lacks play-by-play')
    for play in plays:
        about, result = play.get('about', {}), play.get('result', {})
        event = result.get('eventType', '')
        outcome = map_event(event)
        if outcome is None:
            if event in NON_PA_EVENTS:
                continue
            raise Blocked('Unmapped official terminal event; do not silently drop PA')
        if not about.get('isComplete'):
            raise Blocked('Unfinished official PA')
        end = timestamp(about['endTime'])
        if end > timestamp(feed.finished_at):
            raise Blocked('Label resolution after source capture')
        m = play['matchup']
        key = int(about['atBatIndex']) + 1
        if key in identities:
            raise Blocked('Duplicate official PA identity')
        identities[key] = {'outcome': outcome, 'batter': int(m['batter']['id']),
                           'pitcher': int(m['pitcher']['id']), 'label_resolved_at': end.isoformat(),
                           'terminal_event': event}
    baseline = {'game_pk': pk, 'date': day, 'away_id': gd['teams']['away']['id'],
                'home_id': gd['teams']['home']['id'], 'away_runs': scores['away'],
                'home_runs': scores['home'], 'game_type': gd['game']['type']}
    return identities, baseline


def _optional_number(row: dict, name: str) -> float | None:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise Blocked('Non-numeric optional history field') from None
    if not np.isfinite(number):
        raise Blocked('Nonfinite history field')
    return number


def collapse_statcast(source: Source, day: str, expected: dict[int, dict],
                      origin: datetime) -> tuple[list[dict], dict]:
    source.check(origin)
    try:
        frame = pd.read_csv(io.BytesIO(source.body), low_memory=False)
    except Exception as exc:
        raise Blocked('Statcast response is not readable CSV: ' + type(exc).__name__) from None
    if REQUIRED - set(frame):
        raise Blocked('Statcast schema missing required fields')
    if frame.empty and expected:
        raise Blocked('Statcast empty on a date with completed games')
    if len(frame) and not frame.game_date.astype(str).eq(day).all():
        raise Blocked('Statcast includes wrong or future dates')
    if len(frame) and not frame.game_type.astype(str).isin(GAME_TYPES).all():
        raise Blocked('Unexpected Statcast game type')
    for name in ('game_pk', 'at_bat_number', 'pitch_number', 'batter', 'pitcher', 'inning', 'outs_when_up'):
        values = pd.to_numeric(frame[name], errors='coerce')
        if values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any():
            raise Blocked('Nonintegral required Statcast field')
        frame[name] = values.astype(np.int64)
    if ((frame.game_pk <= 0) | (frame.at_bat_number <= 0) | (frame.pitch_number < 0)).any():
        raise Blocked('Invalid PA or pitch identity')
    key = ['game_pk', 'at_bat_number', 'pitch_number', 'batter', 'pitcher']
    if frame.duplicated(key).any():
        raise Blocked('Duplicate pitch identities; revisions require a clean source snapshot')
    observed_games = set(int(x) for x in frame.game_pk.unique())
    if observed_games != set(expected):
        raise Blocked('Statcast game coverage differs from official final-game schedule')
    frame = frame.sort_values(['game_pk', 'at_bat_number', 'pitch_number'], kind='mergesort')
    rows, diagnostics = [], {'raw_pitch_rows': len(frame), 'mapped_PAs': 0,
                             'excluded_non_PA_events': 0, 'terminal_pitch_counts': []}
    found = {pk: set() for pk in expected}
    for (pk, ab), part in frame.groupby(['game_pk', 'at_bat_number'], sort=True):
        terminal = part.loc[part.events.notna()]
        if terminal.empty:
            # May be an interrupted PA terminated by a baserunner out. It may
            # not silently stand in for a completed official PA.
            continue
        mapped = terminal.loc[terminal.events.map(map_event).notna()]
        unknown = set(terminal.events.astype(str)) - set(NON_PA_EVENTS) - {
            str(event) for event in terminal.events if map_event(event) is not None}
        if unknown:
            raise Blocked('Unmapped Statcast event')
        if mapped.empty:
            diagnostics['excluded_non_PA_events'] += len(terminal)
            continue
        if len(mapped) != 1:
            raise Blocked('Multiple completed PA terminals for one identity')
        first, last = part.iloc[0].to_dict(), mapped.iloc[0].to_dict()
        truth = expected[int(pk)].get(int(ab))
        if truth is None:
            raise Blocked('Statcast PA missing in official feed')
        if (truth['outcome'] != map_event(last['events']) or
                truth['batter'] != int(last['batter']) or truth['pitcher'] != int(last['pitcher'])):
            raise Blocked('Statcast/official PA result or player mismatch')
        if int(first['batter']) != int(last['batter']) or int(first['pitcher']) != int(last['pitcher']):
            raise Blocked('Mid-PA player substitution requires explicit attribution; no silent merge')
        row = {name: first.get(name) for name in SOURCE_FIELDS}
        for name in list(row):
            if isinstance(row[name], (float, np.floating)) and pd.isna(row[name]):
                row[name] = None
            if isinstance(row[name], np.generic):
                row[name] = row[name].item()
        if row['stand'] not in ('R', 'L') or row['p_throws'] not in ('R', 'L'):
            raise Blocked('Unresolved handedness in observed PA')
        if str(row['inning_topbot']).lower() not in ('top', 'bot'):
            raise Blocked('Invalid half-inning')
        row.update(terminal_event=str(last['events']), outcome=truth['outcome'],
                   season=int(day[:4]), date_key=day, game_year=int(day[:4]),
                   platoon=int(row['stand'] == row['p_throws']),
                   is_home_batter=int(str(row['inning_topbot']).lower() == 'bot'),
                   park=str(row['home_team']), matchup_key=f"{int(row['batter'])}_{int(row['pitcher'])}")
        for base in ('1b', '2b', '3b'):
            row['runner_' + base] = int(row['on_' + base] is not None)
        # Match inherited data-builder transformations without copying future fields.
        lo_hi_default = {'outs_when_up': (0, 2, 0), 'inning': (1, 20, 1),
                         'n_thruorder_pitcher': (1, 8, 1),
                         'batter_days_since_prev_game': (0, 30, None),
                         'pitcher_days_since_prev_game': (0, 30, None),
                         'age_bat': (18, 50, None), 'age_pit': (18, 50, None)}
        for name, (lo, hi, default) in lo_hi_default.items():
            n = _optional_number(row, name)
            row[name] = default if n is None else min(hi, max(lo, n))
        if row['bat_score_diff'] is None:
            a, b = _optional_number(row, 'bat_score'), _optional_number(row, 'fld_score')
            if a is None or b is None:
                raise Blocked('Missing pre-PA score context')
            row['bat_score_diff'] = a - b
        row['bat_score_diff'] = min(10, max(-10, float(row['bat_score_diff'])))
        found[int(pk)].add(int(ab)); rows.append(row)
        # Kept as supplemental audit evidence; not misused as new model inputs.
        diagnostics['terminal_pitch_counts'].append({'game_pk': int(pk), 'at_bat_number': int(ab),
                                                     'pa_pitches': int(last['pitch_number'])})
    if any(found[pk] != set(expected[pk]) for pk in expected):
        raise Blocked('Missing completed PA coverage; refresh not accepted')
    diagnostics['mapped_PAs'] = len(rows)
    diagnostics['schema_fields'] = list(frame.columns)
    return rows, diagnostics


def capture_day(day: str, fetcher: Fetcher, origin_clock=clock) -> dict:
    today = origin_clock()
    if day > previous_day(today).isoformat() or day <= BASE_END:
        raise Blocked('Refresh date must be after seed corpus and no later than yesterday')
    schedule = fetcher.get('schedule_' + day,
                           'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=' + day)
    finals, excluded = schedule_cases(schedule, day, origin_clock())
    sources, expected, results = [schedule], {}, []
    for game in finals:
        pk = int(game['gamePk'])
        source = fetcher.get('final_' + str(pk), f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live')
        identities, result = official_pa(source, game, day, origin_clock())
        sources.append(source); expected[pk] = identities; results.append(result)
    rows, diagnostics = [], {'raw_pitch_rows': 0, 'mapped_PAs': 0, 'terminal_pitch_counts': []}
    if finals:
        query = urlencode({'all': 'true', 'type': 'details', 'player_type': 'pitcher',
                           'game_date_gt': day, 'game_date_lt': day,
                           'hfGT': 'R|F|D|L|W|', 'group_by': 'name', 'min_pitches': '0',
                           'min_results': '0', 'sort_col': 'pitches', 'sort_order': 'desc'})
        source = fetcher.get('statcast_' + day, 'https://baseballsavant.mlb.com/statcast_search/csv?' + query)
        rows, diagnostics = collapse_statcast(source, day, expected, origin_clock()); sources.append(source)
    completed = origin_clock().isoformat()
    return {'schema': SCHEMA, 'day': day, 'captured_at': completed,
            'status': 'complete_day' if finals else 'no_played_games',
            'source_availability': 'captured_now_not_backdated',
            'n_games': len(finals), 'n_PA': len(rows), 'rows': rows,
            'results': results, 'exclusions': excluded, 'diagnostics': diagnostics,
            'sources': [source.to_dict() for source in sources]}


def capture_seed_baseline(fetcher: Fetcher, origin_clock=clock) -> dict:
    """Fill the known one-day gap between bundled team results and PA history."""
    day = BASE_END
    if day > previous_day(origin_clock()).isoformat():
        raise Blocked('Seed baseline would use future data')
    source = fetcher.get('schedule_' + day,
                         'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=' + day)
    finals, exclusions = schedule_cases(source, day, origin_clock())
    sources, results = [source], []
    for game in finals:
        pk = int(game['gamePk'])
        feed = fetcher.get('final_' + str(pk), f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live')
        _, result = official_pa(feed, game, day, origin_clock())
        results.append(result); sources.append(feed)
    return {'schema': SCHEMA, 'day': day, 'captured_at': origin_clock().isoformat(),
            'status': 'baseline_seed_gap', 'source_availability': 'captured_now_not_backdated',
            'n_games': len(finals), 'n_PA': 0, 'rows': [], 'results': results,
            'exclusions': exclusions, 'diagnostics': {'purpose': 'team baseline only; seed PA bytes unchanged'},
            'sources': [s.to_dict() for s in sources]}


class Store(Protocol):
    def read(self, path: str): ...
    def put(self, path: str, raw: bytes, immutable: bool = False): ...


class HistoryCache:
    def __init__(self, store: Store, key: bytes, origin_clock=clock):
        self.store, self.key, self.clock = store, key, origin_clock

    def index(self) -> dict:
        saved = self.store.read(INDEX_PATH)
        if saved is None:
            return {'schema': 'brl.history-index.v1', 'base_sha256': BASE_SHA256, 'days': {}}
        doc = json.loads(unseal(saved[0], self.key, INDEX_PURPOSE))
        if doc['schema'] != 'brl.history-index.v1' or doc['base_sha256'] != BASE_SHA256:
            raise Blocked('History index identity mismatch')
        return doc

    def load_day(self, identifier: str) -> dict:
        saved = self.store.read('private/history-days/' + identifier + '.enc')
        if saved is None:
            raise Blocked('Referenced encrypted history revision missing')
        raw = gzip.decompress(unseal(saved[0], self.key, 'private:history-day:' + identifier))
        day = json.loads(raw)
        if content_hash(day) != identifier or day.get('schema') != SCHEMA:
            raise Blocked('History revision identity mismatch')
        for source in day['sources']:
            Source.from_dict(source).check(timestamp(day['captured_at']))
        return day

    def refresh(self, fetcher: Fetcher, recheck_days: int = 3) -> dict:
        origin = self.clock(); target = previous_day(origin).isoformat()
        if target < BASE_END or recheck_days < 1:
            raise Blocked('Invalid refresh window')
        index = self.index()
        # Reuse an actually complete daily run on subsequent 15-minute ticks.
        if index.get('checked_calendar_day') == origin.astimezone(ZoneInfo('America/New_York')).date().isoformat():
            self.require_complete(index, target)
            return index
        first = date.fromisoformat(BASE_END)
        recent = date.fromisoformat(target) - timedelta(days=recheck_days - 1)
        staged = json.loads(json.dumps(index))
        for offset in range((date.fromisoformat(target) - first).days + 1):
            day = (first + timedelta(days=offset)).isoformat()
            if day in staged['days'] and day < recent.isoformat():
                continue
            captured = capture_seed_baseline(fetcher, self.clock) if day == BASE_END else capture_day(day, fetcher, self.clock)
            ident = content_hash(captured)
            payload = gzip.compress(canonical(captured), mtime=0)
            cipher = seal(payload, self.key, 'private:history-day:' + ident)
            path = 'private/history-days/' + ident + '.enc'
            existing = self.store.read(path)
            if existing is None:
                self.store.put(path, cipher, immutable=True)
            elif gzip.decompress(unseal(existing[0], self.key, 'private:history-day:' + ident)) != canonical(captured):
                raise Blocked('Immutable history-day conflict')
            staged['days'][day] = {'id': ident, 'captured_at': captured['captured_at'],
                                   'n_games': captured['n_games'], 'n_PA': captured['n_PA']}
        self.require_complete(staged, target)
        staged.update(coverage_through=target,
                      checked_calendar_day=origin.astimezone(ZoneInfo('America/New_York')).date().isoformat(),
                      committed_at=self.clock().isoformat())
        # Publish only the pointer after ALL missing/recent dates have passed.
        # Previous index and prior forecast inputs survive any earlier failure.
        self.store.put(INDEX_PATH, seal(canonical(staged), self.key, INDEX_PURPOSE))
        return staged

    @staticmethod
    def require_complete(index: dict, target: str) -> None:
        first = date.fromisoformat(BASE_END)
        for offset in range((date.fromisoformat(target) - first).days + 1):
            if (first + timedelta(days=offset)).isoformat() not in index['days']:
                raise Blocked('Gap in accepted history dates')

    def assemble(self, index: dict, base: Path, output: Path, forecast_origin: datetime) -> dict:
        if hashlib.sha256(base.read_bytes()).hexdigest() != BASE_SHA256:
            raise Blocked('Original locked history bytes changed')
        self.require_complete(index, previous_day(forecast_origin).isoformat())
        if timestamp(index['committed_at']) > forecast_origin:
            raise Blocked('History revision was not available at forecast origin')
        with gzip.open(base, 'rt', newline='') as source:
            columns = next(csv.reader(source))
        additions, new_results, identities = [], [], set()
        for day, ref in sorted(index['days'].items()):
            if day >= forecast_origin.astimezone(ZoneInfo('America/New_York')).date().isoformat():
                raise Blocked('Same-day/future history partition')
            item = self.load_day(ref['id'])
            if item['day'] != day or timestamp(item['captured_at']) > forecast_origin:
                raise Blocked('Future or mismatched history revision')
            for row in item['rows']:
                identity = (row['game_pk'], row['at_bat_number'])
                if row['date_key'] != day or identity in identities:
                    raise Blocked('History PA identity/date conflict')
                if set(row) != set(columns):
                    raise Blocked('Updated PA schema differs from locked history schema')
                identities.add(identity); additions.append(row)
            new_results.extend(item['results'])
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.part')
        # Stream unchanged seed CSV followed by append-only updates.
        with gzip.open(base, 'rb') as original, temporary.open('wb') as target:
            with gzip.GzipFile(fileobj=target, mode='wb', filename='', mtime=0) as compressed:
                last = b''
                while True:
                    chunk = original.read(1024 * 1024)
                    if not chunk: break
                    compressed.write(chunk); last = chunk[-1:]
                if last != b'\n': compressed.write(b'\n')
                text = io.StringIO(newline=''); writer = csv.DictWriter(text, fieldnames=columns, lineterminator='\n')
                writer.writerows(additions); compressed.write(text.getvalue().encode())
        os.replace(temporary, output)
        return {'history_path': str(output), 'history_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'base_sha256': BASE_SHA256, 'coverage_through': index['coverage_through'],
                'captured_before': forecast_origin.isoformat(), 'added_PA': len(additions),
                'added_games': len({r['game_pk'] for r in additions}), 'index_sha256': content_hash(index),
                'new_team_results': new_results, 'source_rows_public': False}
