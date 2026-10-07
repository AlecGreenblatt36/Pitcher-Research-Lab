"""Pitch-by-pitch bookkeeping for simulated plate appearances.

The locked engine decides each plate appearance's outcome; nothing here changes that. For the
box score and play-by-play, every simulated plate appearance is given a pitch sequence drawn
from real sequences in prior official game feeds:

1. Count path. A real sequence of pitch results (ball, called strike, swinging strike, foul,
   foul tip, in play, hit by pitch) is sampled from prior plate appearances with the same
   outcome, from the same pitcher when he has enough of them, otherwise the league pool for
   the outcome and batter hand. Sequences are kept only when they are consistent with the
   outcome (a strikeout ends on strike three, a walk on ball four, a hit by pitch on the
   pitch that hit him, a ball in play on the pitch put in play).
2. Pitch types and speeds. Each pitch in the path gets a type from the pitcher's own mix for
   the batter hand and count (league mix by hand and count when his sample is thin) and a
   speed from his distribution for that type (league by type when thin).

Everything is sampled on the box-score bookkeeping stream, independent of the engine, and
never feeds back into probabilities or pitching changes. Counts, types and speeds are
estimates with real shapes, not predictions of the actual pitches.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

RESULT_CODES = {'B': 'B', 'I': 'B', 'V': 'B', 'P': 'B', '*B': 'B',       # balls
                'C': 'C', 'S': 'S', 'W': 'S', 'M': 'S', 'Q': 'S',         # strikes (called / swinging)
                'T': 'T', 'F': 'F', 'L': 'F', 'R': 'F', 'O': 'F',         # foul tip / fouls
                'X': 'X', 'D': 'X', 'E': 'X', 'J': 'X', 'Z': 'X',         # in play
                'H': 'H'}                                                 # hit by pitch
RESULT_TEXT = {'B': 'ball', 'C': 'called strike', 'S': 'swinging strike', 'F': 'foul', 'T': 'foul tip', 'X': 'in play', 'H': 'hit by pitch'}
STRIKE_ENDINGS = {'C', 'S', 'T'}
MIN_PITCHER_SEQ = 12      # sequences before a pitcher's own count paths are used
MIN_MIX = 25              # pitches before a pitcher's own type mix is used (per hand and count group)
MIN_SPEED = 8             # pitches before a pitcher's own speed for a type is used


def count_group(balls: int, strikes: int) -> str:
    if balls == 0 and strikes == 0:
        return 'first'
    if strikes == 2 and balls < 3:
        return 'two_strikes'
    if balls == 3 and strikes < 2:
        return 'three_balls'
    if balls == 3 and strikes == 2:
        return 'full'
    return 'ahead' if strikes > balls else 'behind' if balls > strikes else 'even'


def pitch_list(play: dict) -> list | None:
    """Pitches of one official play as [{b, s, r, t, v}, ...]; None when any pitch code is unknown."""
    pitches = []
    balls = strikes = 0
    for e in play.get('playEvents') or []:
        if e.get('isPitch') is not True:
            continue
        d = e.get('details') or {}
        code = RESULT_CODES.get(str(d.get('code') or ''))
        if code is None:
            return None
        ptype = str(((d.get('type') or {}).get('code')) or '').upper() or None
        speed = (e.get('pitchData') or {}).get('startSpeed')
        pitches.append({'b': balls, 's': strikes, 'r': code, 't': ptype, 'v': None if speed is None else float(speed)})
        c = e.get('count') or {}
        balls, strikes = int(c.get('balls', balls)), int(c.get('strikes', strikes))
    return pitches or None


def sequences_from_feed(feed: dict, map_event) -> list:
    """One record per complete plate appearance: pitcher, batter hand, outcome label, pitch list."""
    out = []
    for play in (feed.get('liveData', {}).get('plays', {}).get('allPlays') or []):
        outcome = map_event((play.get('result') or {}).get('eventType', ''))
        if outcome is None or not (play.get('about') or {}).get('isComplete'):
            continue
        pitches = pitch_list(play)
        if not pitches:
            continue
        m = play.get('matchup') or {}
        out.append({'pitcher': int(m['pitcher']['id']), 'stand': str((m.get('batSide') or {}).get('code') or 'R'),
                    'outcome': str(outcome).lower(), 'terminal_event': str((play.get('result') or {}).get('eventType') or ''),
                    'pitches': pitches})
    return out


def _consistent(rec: dict) -> bool:
    o, last = rec['outcome'], rec['pitches'][-1]['r']
    n = len(rec['pitches'])
    if o in ('k', 'strikeout'):
        return last in STRIKE_ENDINGS and n >= 3
    if o in ('bb_hbp', 'walk_hbp'):
        if rec['terminal_event'] == 'hit_by_pitch':
            return last == 'H'
        if rec['terminal_event'] == 'intent_walk':
            return last == 'B'
        return last == 'B' and n >= 4
    return last == 'X'


LABEL = {'k': 'strikeout', '1b': 'single', '2b_3b': 'double_triple', 'hr': 'home_run', 'walk_hbp': 'bb_hbp', 'out': 'bip_out'}


class PitchBridge:
    """Count-path pools and pitch-type/speed tables from prior sequences."""

    def __init__(self, records: list):
        self.records = [r for r in records if r.get('pitches') and _consistent(r)]
        for r in self.records:
            r['outcome'] = LABEL.get(r['outcome'], r['outcome'])
        self.paths = defaultdict(list)       # (tier, *key) -> record indices
        for i, r in enumerate(self.records):
            hbp = r['terminal_event'] == 'hit_by_pitch'
            label = 'hbp' if (r['outcome'] == 'bb_hbp' and hbp) else r['outcome']
            self.paths[('po', str(r['pitcher']), label)].append(i)
            self.paths[('oh', label, r['stand'])].append(i)
            self.paths[('o', label)].append(i)
        self.paths = {k: np.array(v) for k, v in self.paths.items() if k[0] != 'po' or len(v) >= MIN_PITCHER_SEQ}
        self.mix = defaultdict(Counter)       # (scope...) -> Counter of pitch types
        self.speed = defaultdict(list)        # (scope..., type) -> speeds
        for r in self.records:
            pid = str(r['pitcher'])
            for p in r['pitches']:
                if not p['t']:
                    continue
                g = count_group(p['b'], p['s'])
                self.mix[('p', pid, r['stand'], g)][p['t']] += 1
                self.mix[('p', pid, r['stand'])][p['t']] += 1
                self.mix[('p', pid)][p['t']] += 1
                self.mix[('l', r['stand'], g)][p['t']] += 1
                self.mix[('l', r['stand'])][p['t']] += 1
                if p['v'] is not None:
                    self.speed[('p', pid, p['t'])].append(p['v'])
                    self.speed[('l', p['t'])].append(p['v'])
        self.speed = {k: (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 1.0, len(v)) for k, v in self.speed.items()}
        self.fallbacks = Counter()

    @property
    def n_sequences(self) -> int:
        return len(self.records)

    def _path(self, pitcher_id: str, label: str, hand: str, rng) -> list | None:
        for key in [('po', pitcher_id, label), ('oh', label, hand), ('o', label)]:
            idx = self.paths.get(key)
            if idx is not None and len(idx):
                self.fallbacks[key[0]] += 1
                return self.records[int(idx[int(rng.integers(len(idx)))])]['pitches']
        return None

    def _type(self, pitcher_id: str, hand: str, balls: int, strikes: int, rng) -> str | None:
        g = count_group(balls, strikes)
        for key in [('p', pitcher_id, hand, g), ('p', pitcher_id, hand), ('p', pitcher_id), ('l', hand, g), ('l', hand)]:
            c = self.mix.get(key)
            if c and sum(c.values()) >= (MIN_MIX if key[0] == 'p' else 1):
                types, counts = zip(*c.items())
                return str(rng.choice(types, p=np.array(counts, float) / sum(counts)))
        return None

    def _speed(self, pitcher_id: str, ptype: str, rng) -> int | None:
        for key in [('p', pitcher_id, ptype), ('l', ptype)]:
            s = self.speed.get(key)
            if s and (s[2] >= MIN_SPEED or key[0] == 'l'):
                return int(round(max(55.0, min(106.0, rng.normal(s[0], max(0.6, min(s[1], 2.5)))))))
        return None

    def draw(self, pitcher_id: str, label: str, hand: str, hbp: bool, rng) -> list | None:
        """Pitch list for one simulated plate appearance, or None when no path exists."""
        path = self._path(str(pitcher_id), 'hbp' if (label == 'bb_hbp' and hbp) else label, hand, rng)
        if path is None:
            return None
        out = []
        for p in path:
            t = self._type(str(pitcher_id), hand, p['b'], p['s'], rng)
            v = self._speed(str(pitcher_id), t, rng) if t else None
            out.append([str(t or ''), int(v) if v is not None else 0, str(p['r'])])
        return out
