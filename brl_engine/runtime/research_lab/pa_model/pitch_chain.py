"""The pitch-level chain (CHAIN-01, CHAIN-02): a plate appearance built one pitch at a time.

Instead of reading a hitter's and a pitcher's past outcomes, the chain reads what each one does on a pitch and lets the
seven-outcome probabilities of a plate appearance fall out of the count. Every pitch is placed in one of 15 cells
(pitch family: fastball, breaking, offspeed; by location: top, middle or bottom of the zone, above it or below it).
For a matchup and a count the chain asks, cell by cell:

  the pitcher's mix      how often he throws each cell in this count situation (his own pitches, shrunk to the league);
  swing or take          the league's swing rate for this platoon, count and cell, moved on the log-odds scale by the
                         hitter's deviation for that pitch family in or out of the zone and the pitcher's deviation for
                         the family (both measured against what the league does on the same pitches, shrunk);
  miss or contact        the same shape for whiffs per swing;
  foul or in play        the league's foul share of contacts for the count and cell, moved by the hitter's foul tendency;
  ball or strike         on a take, the league's called-strike rate for the count and cell moved by the pitcher's command;
  the ball in play       the league's outcome mix for the cell (out, single, extra bases, home run, other reach) scaled
                         by the hitter's, the pitcher's and the park's observed-over-expected ratios per class.

Those per-count transition probabilities make a small Markov chain over the twelve counts (fouls with two strikes
stay put) whose absorbing probabilities are the plate appearance's outcome mix. The chain also gives the expected
number of pitches. Each quantity is a count table: the league's are cumulative, the players' and parks' decay with a
half-life, and every deviation is log-odds of (observed + k x expected) / (n + k) against the expected, so an unseen
player is the league and a seen one moves away from it by his own evidence.

Two uses, one code path: ``build_features`` walks the history date by date (features for a date come from pitches
before it, then the date's pitches are absorbed) and returns one row per plate appearance in the benchmark's order;
``ChainState`` is the same state built to a cutoff date for the live provider. The per-pitch log loss of the chain's own
predictions against league-only tables is accumulated as it goes (CHAIN-01's check).
"""
from __future__ import annotations

from datetime import date as _date

import numpy as np
import pandas as pd

LABELS = ['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH']
OUT_CLASSES = ['BIP_OUT', '1B', '2B_3B', 'HR', 'OTHER_REACH']
OUT_TO_LABEL = np.array([0, 3, 4, 5, 6])
OUT_INDEX = {o: i for i, o in enumerate(OUT_CLASSES)}
FAMILY = {'FF': 0, 'SI': 0, 'FA': 0, 'FT': 0, 'FC': 0, 'SL': 1, 'ST': 1, 'SV': 1, 'CU': 1, 'KC': 1, 'CS': 1, 'KN': 1, 'SC': 1,
          'CH': 2, 'FS': 2, 'FO': 2, 'EP': 2}
N_FAM, N_REG, N_CELL, N_GROUP, N_COUNT, N_BUCKET = 3, 5, 15, 6, 12, 5
CELL_FAM = np.repeat(np.arange(N_FAM), N_REG)                      # cell -> family
CELL_OUT = np.tile((np.arange(N_REG) >= 3).astype(int), N_FAM)     # cell -> out of the zone
CELL_GROUP = CELL_FAM * 2 + CELL_OUT                                # cell -> hitter group (family x in/out)
# count index = balls * 3 + strikes; bucket: first pitch, pitcher ahead, even, hitter ahead, full
BUCKET = np.array([0, 1, 1, 3, 2, 1, 3, 3, 2, 3, 3, 4])
CALLED, WHIFF, FOUL, INPLAY, HBP = {'C', 'AC'}, {'S', 'W', 'M', 'Q'}, {'F', 'T', 'L', 'R', 'O'}, {'X', 'D', 'E', 'J', 'Z'}, {'H'}
CLS_BALL, CLS_CALLED, CLS_WHIFF, CLS_FOUL, CLS_INPLAY, CLS_HBP = range(6)
FEATURES = [f'ch_{l}' for l in LABELS] + ['ch_pitches']
# Shrinkage from STAB-01's next-season constants (2021-2026): hitters' swing and chase about 200 pitches, whiffs 137 swings,
# balls in play 200 to 300; pitchers' chase and zone rates about 1,000 pitches, whiffs 366 swings, called strikes about
# 1,000 takes, balls in play about 700 (home runs allowed 1,040 plate appearances).
DEFAULT_PARAMS = {'k_b_swing': 200.0, 'k_b_whiff': 140.0, 'k_b_foul': 150.0, 'k_b_out': 220.0, 'k_p_swing': 800.0, 'k_p_whiff': 360.0,
                  'k_p_called': 1000.0, 'k_p_out': 650.0, 'k_park_out': 1500.0, 'k_mix': 40.0, 'k_league': 50.0,
                  'half_life': 400.0, 'park_half_life': 2000.0,
                  # CHAIN-02B: hitters' swing and whiff deviations split by two strikes or not; pitchers' by family in or out
                  # of the zone (instead of family alone); the matchup's own per-count rates as further features
                  'two_strike_groups': False, 'pitcher_cells': False, 'extra_features': False}
EXTRA_FEATURES = ['ch_sw00', 'ch_wh00', 'ch_fo00', 'ch_ip00', 'ch_ball00', 'ch_sw02', 'ch_wh02', 'ch_sw30']
TWO_STRIKES = (np.arange(N_COUNT) % 3 == 2).astype(int)                # count index -> two strikes
PRIOR = {'swing': 0.47, 'whiff': 0.24, 'foul': 0.50, 'called': 0.32, 'hbp': 0.004}
OUT_PRIOR = np.array([0.67, 0.21, 0.065, 0.045, 0.01])
EPS = 1e-4
_ORDER = sorted(range(N_COUNT), key=lambda c: -(c // 3 + c % 3))    # counts with more pitches thrown first


def feature_names(params: dict | None = None) -> list[str]:
    p = dict(DEFAULT_PARAMS, **(params or {}))
    return list(FEATURES) + (list(EXTRA_FEATURES) if p.get('extra_features') else [])


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rate(y, n, prior, k):
    return (y + k * prior) / (n + k)


def _dev(y, e, n, k):
    """Log-odds of the shrunk observed rate against the league-expected rate on the same pitches (0 when unseen)."""
    base = np.where(n > 0, e / np.maximum(n, 1e-9), 0.5)
    base = np.clip(base, EPS, 1 - EPS)
    est = (y + k * base) / (n + k)
    return _logit(est) - _logit(base)


def _dev_multi(y, e, n, k):
    """Per-class log ratio of the shrunk observed share against the expected share (rows sum of e equals n)."""
    base = np.where(n[:, None] > 0, e / np.maximum(n[:, None], 1e-9), OUT_PRIOR[None, :])
    base = np.clip(base, 1e-4, 1.0)
    est = (y + k * base) / (n[:, None] + k)
    return np.log(est) - np.log(base)


def day_ordinal(values) -> np.ndarray:
    return np.array([_date.fromisoformat(str(v)[:10]).toordinal() for v in values], dtype=np.int64)


def region_of(zone, plate_x, plate_z) -> np.ndarray:
    """0 top, 1 middle, 2 bottom of the zone; 3 above it, 4 below it. From MLB's zone number, else from the location."""
    z = pd.to_numeric(pd.Series(zone), errors='coerce').to_numpy(float)
    px = pd.to_numeric(pd.Series(plate_x), errors='coerce').to_numpy(float)
    pz = pd.to_numeric(pd.Series(plate_z), errors='coerce').to_numpy(float)
    reg = np.full(len(z), 1, dtype=np.int64)
    has = np.isfinite(z)
    zi = np.where(has, z, 5).astype(int)
    reg[has & (zi >= 1) & (zi <= 9)] = (zi[has & (zi >= 1) & (zi <= 9)] - 1) // 3
    reg[has & ((zi == 11) | (zi == 12))] = 3
    reg[has & ((zi == 13) | (zi == 14))] = 4
    loc = ~has & np.isfinite(px) & np.isfinite(pz)
    inz = loc & (np.abs(px) <= 0.83) & (pz >= 1.5) & (pz <= 3.5)
    reg[inz & (pz > 2.83)] = 0; reg[inz & (pz < 2.17)] = 2; reg[inz & (pz >= 2.17) & (pz <= 2.83)] = 1
    outz = loc & ~inz
    reg[outz & (pz >= 2.5)] = 3; reg[outz & (pz < 2.5)] = 4
    return reg


def result_class(codes) -> np.ndarray:
    c = pd.Series(codes).astype(str).to_numpy()
    out = np.zeros(len(c), dtype=np.int64)
    out[np.isin(c, list(CALLED))] = CLS_CALLED
    out[np.isin(c, list(WHIFF))] = CLS_WHIFF
    out[np.isin(c, list(FOUL))] = CLS_FOUL
    out[np.isin(c, list(INPLAY))] = CLS_INPLAY
    out[np.isin(c, list(HBP))] = CLS_HBP
    return out


PITCH_COLUMNS = ['date_key', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter', 'stand', 'p_throws', 'balls', 'strikes',
                 'ptype', 'code', 'speed', 'spin', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone', 'ext', 'outcome', 'last']


def pitches_from_feed(feed: dict, map_event) -> pd.DataFrame:
    """One row per pitch from one official live feed (complete plays with a mapped outcome), the sealed tables' columns."""
    rows = []
    gd = feed.get('gameData') or {}
    day = str(((gd.get('datetime') or {}).get('officialDate')) or '')[:10]
    pk = int(feed.get('gamePk') or 0)
    for play in (feed.get('liveData', {}).get('plays', {}).get('allPlays') or []):
        about = play.get('about') or {}
        outcome = map_event((play.get('result') or {}).get('eventType', ''))
        if outcome is None or not about.get('isComplete'):
            continue
        m = play.get('matchup') or {}
        batter = (m.get('batter') or {}).get('id')
        stand = str((m.get('batSide') or {}).get('code') or 'R'); throws = str((m.get('pitchHand') or {}).get('code') or '')
        events = [e for e in (play.get('playEvents') or []) if e.get('isPitch') is True]
        balls = strikes = 0
        for i, e in enumerate(events):
            d = e.get('details') or {}; pd_ = e.get('pitchData') or {}; co = pd_.get('coordinates') or {}; br = pd_.get('breaks') or {}
            rows.append((day, pk, int(about['atBatIndex']) + 1, i + 1, int(m['pitcher']['id']), -1 if batter is None else int(batter), stand, throws,
                         balls, strikes, str(((d.get('type') or {}).get('code')) or '').upper(), str(d.get('code') or ''),
                         pd_.get('startSpeed'), br.get('spinRate'), co.get('pfxX'), co.get('pfxZ'), co.get('pX'), co.get('pZ'),
                         pd_.get('zone') if isinstance(pd_.get('zone'), int) else None, pd_.get('extension'), outcome, int(i == len(events) - 1)))
            c = e.get('count') or {}
            balls, strikes = int(c.get('balls', balls)), int(c.get('strikes', strikes))
    return pd.DataFrame(rows, columns=PITCH_COLUMNS)


def prepare(pitches: pd.DataFrame, park_of_game: dict | None = None) -> pd.DataFrame:
    """The compact numeric per-pitch frame the state absorbs, sorted by day."""
    p = pitches
    fam = p['ptype'].astype(str).map(FAMILY).fillna(0).astype(int).to_numpy()
    reg = region_of(p['zone'].to_numpy(), p['plate_x'].to_numpy(), p['plate_z'].to_numpy())
    cls = result_class(p['code'].astype(str).to_numpy())
    balls = np.clip(pd.to_numeric(p['balls'], errors='coerce').fillna(0).to_numpy(int), 0, 3)
    strikes = np.clip(pd.to_numeric(p['strikes'], errors='coerce').fillna(0).to_numpy(int), 0, 2)
    stand = p['stand'].astype(str).str.upper().str[:1].to_numpy()
    throws = p['p_throws'].astype(str).str.upper().str[:1].to_numpy()
    out = pd.DataFrame({
        'day': day_ordinal(p['date_key'].to_numpy()), 'game_pk': p['game_pk'].to_numpy(np.int64),
        'batter': pd.to_numeric(p['batter'], errors='coerce').fillna(-1).to_numpy(np.int64),
        'pitcher': pd.to_numeric(p['pitcher'], errors='coerce').fillna(-1).to_numpy(np.int64),
        'pl': (stand == throws).astype(np.int64), 'c': balls * 3 + strikes, 'cell': fam * N_REG + reg, 'cls': cls,
        'out': p['outcome'].astype(str).map(OUT_INDEX).fillna(-1).astype(np.int64).to_numpy() if 'outcome' in p.columns else -1,
    })
    out['park'] = [park_of_game.get(int(g), 'UNK') for g in out['game_pk']] if park_of_game else 'UNK'
    return out.sort_values('day', kind='mergesort').reset_index(drop=True)


class _Bank:
    """Named count arrays per key (player or park) with a lazy half-life: a row is decayed to the day it is touched."""

    def __init__(self, shapes: dict, half_life: float):
        self.shapes, self.half = shapes, float(half_life)
        self.index: dict = {}
        self.cap = 0
        self.last = np.zeros(0, dtype=np.int64)
        self.a = {k: np.zeros((0,) + tuple(s)) for k, s in shapes.items()}

    def _grow(self, need: int):
        cap = max(need, 2 * self.cap, 256)
        self.last = np.concatenate([self.last, np.zeros(cap - self.cap, dtype=np.int64)])
        for k, s in self.shapes.items():
            self.a[k] = np.concatenate([self.a[k], np.zeros((cap - self.cap,) + tuple(s))])
        self.cap = cap

    def rows(self, keys, day: int) -> np.ndarray:
        """Row indices for the keys (created when new), decayed to ``day``."""
        rows = np.empty(len(keys), dtype=np.int64)
        fresh = []
        for i, k in enumerate(keys):
            r = self.index.get(k)
            if r is None:
                r = len(self.index); self.index[k] = r; fresh.append(r)
            rows[i] = r
        if len(self.index) > self.cap:
            self._grow(len(self.index))
        if fresh:
            self.last[fresh] = day
        u = np.unique(rows)
        gap = day - self.last[u]
        move = u[gap > 0]
        if len(move) and self.half > 0:
            f = 0.5 ** (gap[gap > 0] / self.half)
            for k, s in self.shapes.items():
                self.a[k][move] *= f.reshape((-1,) + (1,) * len(s))
            self.last[move] = day
        return rows


class ChainState:
    """The chain's tables as they stand at the start of a day."""

    def __init__(self, params: dict | None = None):
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        shape = (2, N_COUNT, N_CELL)
        self.L = {k: np.zeros(shape) for k in ('n', 'sw', 'wh', 'fo', 'ip', 'cs', 'hb')}
        self.Lout = np.zeros((2, N_CELL, len(OUT_CLASSES)))
        self.Lmix = np.zeros((N_BUCKET, N_CELL))
        h = float(self.p['half_life'])
        self.nb = N_GROUP * (2 if self.p.get('two_strike_groups') else 1)      # hitter deviation columns
        self.npc = N_GROUP if self.p.get('pitcher_cells') else N_FAM            # pitcher deviation columns
        self.bat = _Bank({'sw_y': (self.nb,), 'sw_e': (self.nb,), 'sw_n': (self.nb,), 'wh_y': (self.nb,), 'wh_e': (self.nb,), 'wh_n': (self.nb,),
                          'fo_y': (), 'fo_e': (), 'fo_n': (), 'out_y': (5,), 'out_e': (5,), 'out_n': ()}, h)
        self.pit = _Bank({'mix': (N_BUCKET, N_CELL), 'sw_y': (self.npc,), 'sw_e': (self.npc,), 'sw_n': (self.npc,), 'wh_y': (self.npc,), 'wh_e': (self.npc,), 'wh_n': (self.npc,),
                          'cs_y': (), 'cs_e': (), 'cs_n': (), 'out_y': (5,), 'out_e': (5,), 'out_n': ()}, h)
        self.park = _Bank({'out_y': (5,), 'out_e': (5,), 'out_n': ()}, float(self.p['park_half_life']))
        self.names = feature_names(self.p)
        self.pitches_seen = 0
        self.through_day: int | None = None
        # CHAIN-01 check: summed log loss of the chain's own per-pitch predictions against league-only tables
        self.check = {k: [0.0, 0.0, 0] for k in ('swing', 'whiff', 'outcome')}

    def _bcol(self, cell, count):
        """Hitter deviation column for a cell and count (group, or group x two strikes)."""
        g = CELL_GROUP[cell]
        return g + N_GROUP * TWO_STRIKES[count] if self.p.get('two_strike_groups') else g

    def _pcol(self, cell):
        return CELL_GROUP[cell] if self.p.get('pitcher_cells') else CELL_FAM[cell]

    # ----- league tables as rates ------------------------------------------------------------------------------------
    def _league(self):
        k = float(self.p['k_league']); L = self.L
        n_take = L['n'] - L['sw']; n_contact = L['sw'] - L['wh']
        return {'sw': _rate(L['sw'], L['n'], PRIOR['swing'], k), 'wh': _rate(L['wh'], L['sw'], PRIOR['whiff'], k),
                'fo': _rate(L['fo'], n_contact, PRIOR['foul'], k), 'cs': _rate(L['cs'], n_take, PRIOR['called'], k),
                'hb': _rate(L['hb'], n_take, PRIOR['hbp'], k),
                'out': (self.Lout + k * OUT_PRIOR) / (self.Lout.sum(axis=2, keepdims=True) + k),
                'mix': (self.Lmix + 1.0) / (self.Lmix.sum(axis=1, keepdims=True) + N_CELL)}

    # ----- features for matchups --------------------------------------------------------------------------------------
    def features_for(self, batters, pitchers, parks, stands, throws, day: int) -> np.ndarray:
        """One row per matchup: log of the seven outcome probabilities and the expected pitches."""
        n = len(batters)
        if n == 0:
            return np.zeros((0, len(self.names)))
        P = self.p
        pl = (np.asarray(stands).astype(str) == np.asarray(throws).astype(str)).astype(int)
        br = self.bat.rows([int(b) for b in batters], day); pr = self.pit.rows([int(x) for x in pitchers], day)
        kr = self.park.rows([str(x) for x in parks], day)
        B, Pi, K = self.bat.a, self.pit.a, self.park.a
        cells = np.arange(N_CELL); counts = np.arange(N_COUNT)
        bcol = self._bcol(cells[None, :], counts[:, None])                                                            # (count, cell)
        pcol = self._pcol(cells)                                                                                      # (cell,)
        db_sw = _dev(B['sw_y'][br], B['sw_e'][br], B['sw_n'][br], P['k_b_swing'])[:, bcol]                           # (n, count, cell)
        db_wh = _dev(B['wh_y'][br], B['wh_e'][br], B['wh_n'][br], P['k_b_whiff'])[:, bcol]
        db_fo = _dev(B['fo_y'][br], B['fo_e'][br], B['fo_n'][br], P['k_b_foul'])
        dp_sw = _dev(Pi['sw_y'][pr], Pi['sw_e'][pr], Pi['sw_n'][pr], P['k_p_swing'])[:, pcol]                        # (n, cell)
        dp_wh = _dev(Pi['wh_y'][pr], Pi['wh_e'][pr], Pi['wh_n'][pr], P['k_p_whiff'])[:, pcol]
        dp_cs = _dev(Pi['cs_y'][pr], Pi['cs_e'][pr], Pi['cs_n'][pr], P['k_p_called'])
        W = np.exp(_dev_multi(B['out_y'][br], B['out_e'][br], B['out_n'][br], P['k_b_out'])
                   + _dev_multi(Pi['out_y'][pr], Pi['out_e'][pr], Pi['out_n'][pr], P['k_p_out'])
                   + _dev_multi(K['out_y'][kr], K['out_e'][kr], K['out_n'][kr], P['k_park_out']))
        Lg = self._league()
        mix = (Pi['mix'][pr] + P['k_mix'] * Lg['mix'][None]) / (Pi['mix'][pr].sum(axis=2, keepdims=True) + P['k_mix'])   # (n, bucket, cell)
        M = mix[:, BUCKET, :]                                                                                           # (n, count, cell)
        p_sw = _expit(_logit(Lg['sw'][pl]) + db_sw + dp_sw[:, None, :])
        p_wh = _expit(_logit(Lg['wh'][pl]) + db_wh + dp_wh[:, None, :])
        p_fo = _expit(_logit(Lg['fo'][pl]) + db_fo[:, None, None])
        p_cs = _expit(_logit(Lg['cs'][pl]) + dp_cs[:, None, None])
        p_hb = np.minimum(Lg['hb'][pl], 1 - p_cs - EPS)
        take = 1 - p_sw
        T = {'ball': (M * take * (1 - p_cs - p_hb)).sum(2), 'cs': (M * take * p_cs).sum(2), 'hb': (M * take * p_hb).sum(2),
             'wh': (M * p_sw * p_wh).sum(2), 'fo': (M * p_sw * (1 - p_wh) * p_fo).sum(2)}
        w_ip = M * p_sw * (1 - p_wh) * (1 - p_fo)                                                                      # (n, count, cell)
        T['ip'] = w_ip.sum(2)
        O = Lg['out'][pl] * W[:, None, :]                                                                               # (n, cell, 5)
        O /= O.sum(axis=2, keepdims=True)
        D = np.einsum('ncj,njo->nco', w_ip, O) / np.maximum(T['ip'][:, :, None], 1e-12)                                 # (n, count, 5)
        V = np.zeros((n, N_COUNT, 7)); Npitch = np.zeros((n, N_COUNT))
        eK = np.zeros(7); eK[1] = 1.0; eBB = np.zeros(7); eBB[2] = 1.0
        for c in _ORDER:
            b, s = divmod(c, 3)
            D7 = np.zeros((n, 7)); D7[:, OUT_TO_LABEL] = D[:, c, :]
            nbV = V[:, (b + 1) * 3 + s] if b < 3 else eBB[None, :]; nbN = Npitch[:, (b + 1) * 3 + s] if b < 3 else 0.0
            nsV = V[:, b * 3 + s + 1] if s < 2 else eK[None, :]; nsN = Npitch[:, b * 3 + s + 1] if s < 2 else 0.0
            pb, pcs, phb, pwh, pfo, pip = (T[k][:, c] for k in ('ball', 'cs', 'hb', 'wh', 'fo', 'ip'))
            base = pb[:, None] * nbV + (pcs + pwh)[:, None] * nsV + pip[:, None] * D7 + phb[:, None] * eBB[None, :]
            if s < 2:
                V[:, c] = base + pfo[:, None] * nsV
                Npitch[:, c] = 1 + pb * nbN + (pcs + pwh + pfo) * nsN
            else:
                V[:, c] = base / (1 - pfo)[:, None]
                Npitch[:, c] = (1 + pb * nbN) / (1 - pfo)
        out = np.empty((n, len(self.names)))
        out[:, :7] = np.log(np.clip(V[:, 0, :], 1e-6, 1.0))
        out[:, 7] = Npitch[:, 0]
        if P.get('extra_features'):
            # the matchup's own rates at 0-0, 0-2 and 3-0: swing, whiff per swing, foul per contact, in play per pitch, ball per pitch
            sw = (M * p_sw).sum(2); wh = (M * p_sw * p_wh).sum(2) / np.maximum(sw, 1e-9); fo = T['fo'] / np.maximum(T['fo'] + T['ip'], 1e-9)
            c00, c02, c30 = 0, 2, 9
            out[:, 8:] = np.stack([sw[:, c00], wh[:, c00], fo[:, c00], T['ip'][:, c00], T['ball'][:, c00], sw[:, c02], wh[:, c02], sw[:, c30]], axis=1)
        return out

    def features(self, batter: int, pitcher: int, park: str, stand: str, throws: str, day: int) -> np.ndarray:
        return self.features_for([batter], [pitcher], [park], [stand], [throws], day)[0]

    # ----- learning from a day ----------------------------------------------------------------------------------------
    def absorb_date(self, day: int, q: pd.DataFrame) -> None:
        """Add one day's pitches (a ``prepare`` frame) after the day's features were read."""
        if len(q) == 0:
            self.through_day = day; return
        P = self.p; Lg = self._league()
        pl = q['pl'].to_numpy(); c = q['c'].to_numpy(); cell = q['cell'].to_numpy(); cls = q['cls'].to_numpy()
        fam = self._pcol(cell); grp = self._bcol(cell, c); bucket = BUCKET[c]
        swing = np.isin(cls, [CLS_WHIFF, CLS_FOUL, CLS_INPLAY]); take = ~swing
        whiff = cls == CLS_WHIFF; contact = np.isin(cls, [CLS_FOUL, CLS_INPLAY]); foul = cls == CLS_FOUL; inplay = cls == CLS_INPLAY
        called = cls == CLS_CALLED; hbp = cls == CLS_HBP
        e_sw = Lg['sw'][pl, c, cell]; e_wh = Lg['wh'][pl, c, cell]; e_fo = Lg['fo'][pl, c, cell]; e_cs = Lg['cs'][pl, c, cell]
        batters = q['batter'].to_numpy(); pitchers = q['pitcher'].to_numpy()
        br = self.bat.rows([int(b) for b in batters], day); pr = self.pit.rows([int(x) for x in pitchers], day)
        kr = self.park.rows([str(x) for x in q['park'].to_numpy()], day)
        B, Pi, K = self.bat.a, self.pit.a, self.park.a
        # CHAIN-01 check: the chain's own per-pitch prediction (league + the two deviations) against league-only
        db_sw = _dev(B['sw_y'][br, grp], B['sw_e'][br, grp], B['sw_n'][br, grp], P['k_b_swing'])
        dp_sw = _dev(Pi['sw_y'][pr, fam], Pi['sw_e'][pr, fam], Pi['sw_n'][pr, fam], P['k_p_swing'])
        pred_sw = _expit(_logit(e_sw) + db_sw + dp_sw)
        self._tally('swing', swing.astype(float), e_sw, pred_sw, np.ones(len(q), bool))
        db_wh = _dev(B['wh_y'][br, grp], B['wh_e'][br, grp], B['wh_n'][br, grp], P['k_b_whiff'])
        dp_wh = _dev(Pi['wh_y'][pr, fam], Pi['wh_e'][pr, fam], Pi['wh_n'][pr, fam], P['k_p_whiff'])
        pred_wh = _expit(_logit(e_wh) + db_wh + dp_wh)
        self._tally('whiff', whiff.astype(float), e_wh, pred_wh, swing)
        out_idx = q['out'].to_numpy()
        ok_out = inplay & (out_idx >= 0)
        if ok_out.any():
            e_out = Lg['out'][pl[ok_out], cell[ok_out]]                                                              # (m, 5)
            W = np.exp(_dev_multi(B['out_y'][br[ok_out]], B['out_e'][br[ok_out]], B['out_n'][br[ok_out]], P['k_b_out'])
                       + _dev_multi(Pi['out_y'][pr[ok_out]], Pi['out_e'][pr[ok_out]], Pi['out_n'][pr[ok_out]], P['k_p_out'])
                       + _dev_multi(K['out_y'][kr[ok_out]], K['out_e'][kr[ok_out]], K['out_n'][kr[ok_out]], P['k_park_out']))
            pred_out = e_out * W; pred_out /= pred_out.sum(axis=1, keepdims=True)
            rows = np.arange(ok_out.sum()); y = out_idx[ok_out]
            t = self.check['outcome']
            t[0] += float(-np.log(np.clip(e_out[rows, y], 1e-9, 1)).sum()); t[1] += float(-np.log(np.clip(pred_out[rows, y], 1e-9, 1)).sum()); t[2] += int(ok_out.sum())
        # league
        L = self.L
        np.add.at(L['n'], (pl, c, cell), 1.0); np.add.at(L['sw'], (pl, c, cell), swing.astype(float)); np.add.at(L['wh'], (pl, c, cell), whiff.astype(float))
        np.add.at(L['fo'], (pl, c, cell), foul.astype(float)); np.add.at(L['ip'], (pl, c, cell), inplay.astype(float))
        np.add.at(L['cs'], (pl, c, cell), called.astype(float)); np.add.at(L['hb'], (pl, c, cell), hbp.astype(float))
        np.add.at(self.Lmix, (bucket, cell), 1.0)
        if ok_out.any():
            np.add.at(self.Lout, (pl[ok_out], cell[ok_out], out_idx[ok_out]), 1.0)
        # hitters (unknown batter -1 rows are kept but never asked for)
        np.add.at(B['sw_y'], (br, grp), swing.astype(float)); np.add.at(B['sw_e'], (br, grp), e_sw); np.add.at(B['sw_n'], (br, grp), 1.0)
        np.add.at(B['wh_y'], (br[swing], grp[swing]), whiff[swing].astype(float)); np.add.at(B['wh_e'], (br[swing], grp[swing]), e_wh[swing]); np.add.at(B['wh_n'], (br[swing], grp[swing]), 1.0)
        np.add.at(B['fo_y'], br[contact], foul[contact].astype(float)); np.add.at(B['fo_e'], br[contact], e_fo[contact]); np.add.at(B['fo_n'], br[contact], 1.0)
        # pitchers
        np.add.at(Pi['mix'], (pr, bucket, cell), 1.0)
        np.add.at(Pi['sw_y'], (pr, fam), swing.astype(float)); np.add.at(Pi['sw_e'], (pr, fam), e_sw); np.add.at(Pi['sw_n'], (pr, fam), 1.0)
        np.add.at(Pi['wh_y'], (pr[swing], fam[swing]), whiff[swing].astype(float)); np.add.at(Pi['wh_e'], (pr[swing], fam[swing]), e_wh[swing]); np.add.at(Pi['wh_n'], (pr[swing], fam[swing]), 1.0)
        np.add.at(Pi['cs_y'], pr[take], called[take].astype(float)); np.add.at(Pi['cs_e'], pr[take], e_cs[take]); np.add.at(Pi['cs_n'], pr[take], 1.0)
        if ok_out.any():
            for bank, rows in ((B, br[ok_out]), (Pi, pr[ok_out]), (K, kr[ok_out])):
                np.add.at(bank['out_y'], (rows, out_idx[ok_out]), 1.0); np.add.at(bank['out_e'], rows, e_out); np.add.at(bank['out_n'], rows, 1.0)
        self.pitches_seen += int(len(q)); self.through_day = day

    def _tally(self, key: str, y, p_league, p_chain, mask):
        if not mask.any():
            return
        y, a, b = y[mask], np.clip(p_league[mask], 1e-9, 1 - 1e-9), np.clip(p_chain[mask], 1e-9, 1 - 1e-9)
        t = self.check[key]
        t[0] += float(-(y * np.log(a) + (1 - y) * np.log(1 - a)).sum()); t[1] += float(-(y * np.log(b) + (1 - y) * np.log(1 - b)).sum()); t[2] += int(mask.sum())

    def check_report(self) -> dict:
        """Per-pitch log loss (nats) of league-only tables and of the chain, over every pitch absorbed so far."""
        out = {}
        for k, (lg, ch, n) in self.check.items():
            out[k] = {'n': n, 'league_only': round(lg / max(n, 1), 6), 'chain': round(ch / max(n, 1), 6), 'gain_per_1000': round((lg - ch) / max(n, 1) * 1000, 3)}
        return out

    # ----- building to a cutoff -------------------------------------------------------------------------------------
    def absorb_through(self, q: pd.DataFrame, cutoff_day: int) -> None:
        """Absorb every prepared pitch with day < cutoff_day (and after what was absorbed before), day by day."""
        days = q['day'].to_numpy()
        lo = 0 if self.through_day is None else int(np.searchsorted(days, self.through_day + 1))
        hi = int(np.searchsorted(days, cutoff_day))
        if hi <= lo:
            self.through_day = max(self.through_day or -1, cutoff_day - 1); return
        part = q.iloc[lo:hi]
        for day, g in part.groupby('day', sort=True):
            self.absorb_date(int(day), g)
        self.through_day = cutoff_day - 1

    @classmethod
    def build(cls, pitches: pd.DataFrame, cutoff_date: str, params: dict | None = None, park_of_game: dict | None = None, prepared: bool = False) -> 'ChainState':
        state = cls(params)
        q = pitches if prepared else prepare(pitches, park_of_game)
        state.absorb_through(q, int(day_ordinal([cutoff_date])[0]))
        return state


def build_features(pa: pd.DataFrame, pitches: pd.DataFrame, params: dict | None = None, park_of_game: dict | None = None, prepared: bool = False) -> tuple[pd.DataFrame, dict]:
    """Chronological, date-blocked chain features for every PA row, in the engine builder's order.
    ``pitches`` is the per-pitch table, or (prepared=True) the compact frame ``prepare`` makes from it."""
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    parks = dict(park_of_game or {})
    if 'park' in ordered.columns:
        parks.update({int(g): str(k) for g, k in zip(ordered['game_pk'], ordered['park'])})
    q = pitches.sort_values('day', kind='mergesort').reset_index(drop=True) if prepared else prepare(pitches, parks)
    state = ChainState(params)
    pa_day = day_ordinal(ordered['date_key'].to_numpy())
    out = np.zeros((len(ordered), len(state.names)), dtype=np.float32)
    batters = ordered['batter'].to_numpy(np.int64); pitchers = ordered['pitcher'].to_numpy(np.int64)
    stands = ordered['stand'].astype(str).to_numpy(); throws = ordered['p_throws'].astype(str).to_numpy()
    park_col = ordered['park'].astype(str).to_numpy() if 'park' in ordered.columns else np.array(['UNK'] * len(ordered))
    pa_days = np.unique(pa_day); q_days = q['day'].to_numpy()
    all_days = np.unique(np.concatenate([pa_days, np.unique(q_days)]))
    for day in all_days:
        day = int(day)
        i0, i1 = np.searchsorted(pa_day, day), np.searchsorted(pa_day, day + 1)
        if i1 > i0:
            out[i0:i1] = state.features_for(batters[i0:i1], pitchers[i0:i1], park_col[i0:i1], stands[i0:i1], throws[i0:i1], day)
        j0, j1 = np.searchsorted(q_days, day), np.searchsorted(q_days, day + 1)
        if j1 > j0:
            state.absorb_date(day, q.iloc[j0:j1])
    audit = {'params': state.p, 'pitches': int(len(q)), 'pitch_days': int(len(np.unique(q_days))), 'pa_rows': int(len(ordered)),
             'batters': len(state.bat.index), 'pitchers': len(state.pit.index), 'parks': len(state.park.index), 'per_pitch_check': state.check_report()}
    return pd.DataFrame(out, columns=state.names), audit
