"""Pitch-quality and contact-quality features from official pitch physics (public code).

One row per plate appearance is kept from the official play-by-play: per-pitch sums
(pitches, swings, whiffs, out-of-zone pitches and chases, in-zone pitches, swings and
contacts, called strikes, fastballs with their velocity, spin, induced vertical break and
horizontal break) and the batted ball's exit velocity and launch angle. From those rows,
every feature is a shrunk rate or mean over a player's prior dates:

  batter:  exit velocity, launch angle, hard-hit rate (95+), barrel rate (98+ at 8-40 degrees),
           whiff per swing, chase per out-of-zone pitch, swing rate, in-zone contact per swing,
           balls in play seen; optionally an expected run value by (exit velocity, launch angle)
           cell and 30-day deviations from his own rates
  pitcher: fastball velocity, spin, induced vertical and horizontal break, velocity change of
           the previous outing against his norm, whiff, chase, zone and called-strike-plus-whiff
           rates, exit velocity, hard-hit and ground-ball share allowed, pitches seen; optionally
           expected run value allowed and 30-day deviations

Two implementations must agree exactly: the chronological builder used for model fitting
(``build_features``: date-blocked like the engine's own feature builder) and ``PhysicsState``
(the counters as they stand at the start of one date, used by the live provider). The test
suite checks them against each other.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

PITCH_SUMS = ['n', 'sw', 'wh', 'oz', 'ch', 'iz', 'izs', 'izc', 'cs', 'fb', 'velo', 'spin', 'ivb', 'hb']
PS = {name: i for i, name in enumerate(PITCH_SUMS)}
BIP_SUMS = ['bip', 'ev', 'la', 'hard', 'barrel', 'gb']
BS = {name: i for i, name in enumerate(BIP_SUMS)}
TABLE_COLUMNS = ['date_key', 'game_pk', 'at_bat_number', 'pitcher', 'batter'] + PITCH_SUMS + ['ev', 'la']
SWING = {'S', 'W', 'M', 'Q', 'F', 'L', 'R', 'O', 'T', 'X', 'D', 'E', 'J', 'Z'}
WHIFF = {'S', 'W', 'M', 'Q'}
CALLED = {'C'}
FASTBALL = {'FF', 'SI', 'FA', 'FT'}
NO_CONTACT = {'K', 'BB_HBP'}

BASE_FEATURES = ['b_ev', 'b_la', 'b_hard', 'b_barrel', 'b_whiff', 'b_chase', 'b_swing', 'b_zcontact', 'b_bip_n',
                 'p_velo', 'p_velo_delta', 'p_spin', 'p_ivb', 'p_hb', 'p_whiff', 'p_chase', 'p_zone', 'p_csw', 'p_ev', 'p_hard', 'p_gb', 'p_pitch_n']
XVALUE_FEATURES = ['b_xv', 'p_xv']
RECENT_FEATURES = ['p_velo_rec', 'p_whiff_rec', 'p_csw_rec', 'b_ev_rec', 'b_whiff_rec', 'b_hard_rec']
DEFAULT_PARAMS = {'k_rate': 150.0, 'k_bip': 60.0, 'k_velo': 100.0, 'xvalue': False, 'recent_days': 0,
                  'k_recent_pitch': 100.0, 'k_recent_bip': 40.0, 'k_cell': 40.0}
RUN_VALUE = {'BIP_OUT': 0.0, 'K': 0.0, 'BB_HBP': 0.69, '1B': 0.88, '2B_3B': 1.4, 'HR': 2.03, 'OTHER_REACH': 0.6}
EV_BINS, LA_BINS = 16, 18


def feature_names(params: dict | None = None) -> list[str]:
    p = dict(DEFAULT_PARAMS, **(params or {}))
    names = list(BASE_FEATURES)
    if p.get('xvalue'):
        names += XVALUE_FEATURES
    if p.get('recent_days'):
        names += RECENT_FEATURES
    return names


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def play_physics(play: dict) -> tuple[np.ndarray, float, float]:
    """Per-pitch sums and the batted ball (exit velocity, launch angle; NaN when none) of one official play."""
    a = np.zeros(len(PITCH_SUMS))
    ev = la = np.nan
    for e in play.get('playEvents') or []:
        if e.get('isPitch') is not True:
            continue
        d = e.get('details') or {}
        code = str(d.get('code') or '')
        ptype = str(((d.get('type') or {}).get('code')) or '').upper()
        pd_ = e.get('pitchData') or {}
        co = pd_.get('coordinates') or {}
        br = pd_.get('breaks') or {}
        zone = pd_.get('zone')
        swing = code in SWING
        whiff = code in WHIFF
        in_zone = isinstance(zone, int) and 1 <= zone <= 9
        out_zone = isinstance(zone, int) and zone > 9
        a[PS['n']] += 1; a[PS['sw']] += swing; a[PS['wh']] += whiff; a[PS['cs']] += code in CALLED
        a[PS['oz']] += out_zone; a[PS['ch']] += out_zone and swing
        a[PS['iz']] += in_zone; a[PS['izs']] += in_zone and swing; a[PS['izc']] += in_zone and swing and not whiff
        start = _num(pd_.get('startSpeed'))
        if ptype in FASTBALL and start is not None:
            a[PS['fb']] += 1; a[PS['velo']] += start
            a[PS['spin']] += _num(br.get('spinRate')) or 0.0
            a[PS['ivb']] += _num(co.get('pfxZ')) or 0.0
            a[PS['hb']] += abs(_num(co.get('pfxX')) or 0.0)
        hd = e.get('hitData')
        if hd:
            v, g = _num(hd.get('launchSpeed')), _num(hd.get('launchAngle'))
            if v is not None and g is not None:
                ev, la = v, g
    return a, ev, la


def study_physics(pitches: list, hit: list | None) -> tuple[np.ndarray, float, float]:
    """The same sums from a sealed study row (compact pitch lists written by the backfill)."""
    a = np.zeros(len(PITCH_SUMS))
    for p in pitches:
        ptype, code, _, _, start, _, spin_rate, pfx_x, pfx_z, _, _, _, _, _, zone = p
        swing = code in SWING; whiff = code in WHIFF
        in_zone = isinstance(zone, int) and 1 <= zone <= 9
        out_zone = isinstance(zone, int) and zone > 9
        a[PS['n']] += 1; a[PS['sw']] += swing; a[PS['wh']] += whiff; a[PS['cs']] += code in CALLED
        a[PS['oz']] += out_zone; a[PS['ch']] += out_zone and swing
        a[PS['iz']] += in_zone; a[PS['izs']] += in_zone and swing; a[PS['izc']] += in_zone and swing and not whiff
        if ptype in FASTBALL and start is not None:
            a[PS['fb']] += 1; a[PS['velo']] += start; a[PS['spin']] += spin_rate or 0.0
            a[PS['ivb']] += pfx_z or 0.0; a[PS['hb']] += abs(pfx_x or 0.0)
    ev = la = np.nan
    if hit and hit[0] is not None and hit[1] is not None:
        ev, la = float(hit[0]), float(hit[1])
    return a, ev, la


def table_from_study(doc: dict) -> pd.DataFrame:
    """Per-plate-appearance physics table from one sealed study season."""
    rows = []
    for pk, game in doc['games'].items():
        day = str(game.get('date') or '')[:10]
        for r in game['rows']:
            a, ev, la = study_physics(r['pitches'], r.get('hit'))
            if r['o'] in NO_CONTACT:
                ev = la = np.nan
            rows.append((day, int(pk), int(r['i']) + 1, int(r['p']), -1 if r.get('b') is None else int(r['b']), *a.tolist(), ev, la))
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def table_from_feed(feed: dict, map_event) -> pd.DataFrame:
    """Per-plate-appearance physics table from one official live feed (complete plays with a mapped outcome)."""
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
        a, ev, la = play_physics(play)
        if outcome in NO_CONTACT:
            ev = la = np.nan
        rows.append((day, pk, int(about['atBatIndex']) + 1, int(m['pitcher']['id']), -1 if batter is None else int(batter), *a.tolist(), ev, la))
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def shrunk(s, num: int, den: int, league, k: float) -> float:
    """(sum + k * league ratio) / (count + k): a rate or a mean pulled toward the league value."""
    return (s[num] + k * (league[num] / max(league[den], 1e-9))) / (s[den] + k)


def cell_of(ev: float, la: float) -> tuple[int, int]:
    return int(min(max((ev - 40.0) // 5, 0), EV_BINS - 1)), int(min(max((la + 90.0) // 10, 0), LA_BINS - 1))


def bip_vector(ev: float, la: float) -> np.ndarray:
    return np.array([1.0, ev, la, float(ev >= 95.0), float(ev >= 98.0 and 8.0 <= la <= 40.0), float(la < 10.0)])


class _Sums:
    """The accumulated state the features are computed from (shared by both implementations)."""

    def __init__(self, params: dict):
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        n_p, n_b = len(PITCH_SUMS), len(BIP_SUMS)
        self.zero_p, self.zero_b = np.zeros(n_p), np.zeros(n_b)
        self.bp, self.pp, self.Lp = defaultdict(lambda: np.zeros(n_p)), defaultdict(lambda: np.zeros(n_p)), np.zeros(n_p)
        self.bb, self.pb, self.Lb = defaultdict(lambda: np.zeros(n_b)), defaultdict(lambda: np.zeros(n_b)), np.zeros(n_b)
        self.last_velo: dict = {}                       # pitcher -> (fastballs, velocity sum) of his most recent prior outing
        self.cell_sum, self.cell_n = np.zeros((EV_BINS, LA_BINS)), np.zeros((EV_BINS, LA_BINS))
        self.cell_value_total = [0.0, 0.0]
        self.bcells, self.pcells = defaultdict(lambda: np.zeros((EV_BINS, LA_BINS))), defaultdict(lambda: np.zeros((EV_BINS, LA_BINS)))
        self.rec_p, self.rec_b = defaultdict(list), defaultdict(list)     # player -> [[day ordinal, pitch sums, bip sums], ...] per date
        self.names = feature_names(self.p)
        self.X = {name: i for i, name in enumerate(self.names)}

    def cell_values(self) -> np.ndarray:
        league = self.cell_value_total[0] / max(self.cell_value_total[1], 1e-9)
        k = float(self.p['k_cell'])
        return (self.cell_sum + k * league) / (self.cell_n + k)

    def _recent(self, entries: list, today: int):
        days = int(self.p['recent_days'])
        sp_, sb_ = self.zero_p.copy(), self.zero_b.copy()
        for e in entries:
            if e[0] >= today - days:
                sp_ += e[1]; sb_ += e[2]
        return sp_, sb_

    def features(self, batter: int, pitcher: int, today: int) -> np.ndarray:
        P, X = self.p, self.X
        k_rate, k_bip, k_velo = float(P['k_rate']), float(P['k_bip']), float(P['k_velo'])
        Lp, Lb = self.Lp, self.Lb
        sb = self.bp.get(batter, self.zero_p); sp = self.pp.get(pitcher, self.zero_p)
        cb = self.bb.get(batter, self.zero_b); cp = self.pb.get(pitcher, self.zero_b)
        out = np.empty(len(self.names))
        out[X['b_ev']] = shrunk(cb, BS['ev'], BS['bip'], Lb, k_bip)
        out[X['b_la']] = shrunk(cb, BS['la'], BS['bip'], Lb, k_bip)
        out[X['b_hard']] = shrunk(cb, BS['hard'], BS['bip'], Lb, k_bip)
        out[X['b_barrel']] = shrunk(cb, BS['barrel'], BS['bip'], Lb, k_bip)
        out[X['b_whiff']] = shrunk(sb, PS['wh'], PS['sw'], Lp, k_rate)
        out[X['b_chase']] = shrunk(sb, PS['ch'], PS['oz'], Lp, k_rate)
        out[X['b_swing']] = shrunk(sb, PS['sw'], PS['n'], Lp, k_rate)
        out[X['b_zcontact']] = shrunk(sb, PS['izc'], PS['izs'], Lp, k_rate)
        out[X['b_bip_n']] = cb[BS['bip']]
        own_velo = shrunk(sp, PS['velo'], PS['fb'], Lp, k_velo)
        out[X['p_velo']] = own_velo
        out[X['p_spin']] = shrunk(sp, PS['spin'], PS['fb'], Lp, k_velo)
        out[X['p_ivb']] = shrunk(sp, PS['ivb'], PS['fb'], Lp, k_velo)
        out[X['p_hb']] = shrunk(sp, PS['hb'], PS['fb'], Lp, k_velo)
        lv = self.last_velo.get(pitcher)
        out[X['p_velo_delta']] = (lv[1] / lv[0] - own_velo) if (lv is not None and lv[0] >= 5 and sp[PS['fb']] >= 100) else 0.0
        own_whiff = shrunk(sp, PS['wh'], PS['sw'], Lp, k_rate)
        own_csw = (sp[PS['cs']] + sp[PS['wh']] + k_rate * ((Lp[PS['cs']] + Lp[PS['wh']]) / max(Lp[PS['n']], 1e-9))) / (sp[PS['n']] + k_rate)
        out[X['p_whiff']] = own_whiff
        out[X['p_chase']] = shrunk(sp, PS['ch'], PS['oz'], Lp, k_rate)
        out[X['p_zone']] = shrunk(sp, PS['iz'], PS['n'], Lp, k_rate)
        out[X['p_csw']] = own_csw
        out[X['p_ev']] = shrunk(cp, BS['ev'], BS['bip'], Lb, k_bip)
        out[X['p_hard']] = shrunk(cp, BS['hard'], BS['bip'], Lb, k_bip)
        out[X['p_gb']] = shrunk(cp, BS['gb'], BS['bip'], Lb, k_bip)
        out[X['p_pitch_n']] = sp[PS['n']]
        if P.get('xvalue'):
            values = self.cell_values()
            league = self.cell_value_total[0] / max(self.cell_value_total[1], 1e-9)
            for who, table, name in ((batter, self.bcells, 'b_xv'), (pitcher, self.pcells, 'p_xv')):
                h = table.get(who)
                total = float(h.sum()) if h is not None else 0.0
                value_sum = float((h * values).sum()) if h is not None else 0.0
                out[X[name]] = (value_sum + k_bip * league) / (total + k_bip)
        if P.get('recent_days'):
            k_rp, k_rb = float(P['k_recent_pitch']), float(P['k_recent_bip'])
            rp, rpb = self._recent(self.rec_p[pitcher], today) if pitcher in self.rec_p else (self.zero_p, self.zero_b)
            rb, rbb = self._recent(self.rec_b[batter], today) if batter in self.rec_b else (self.zero_p, self.zero_b)
            dev = lambda s_, d_, own, k: (s_ + k * own) / (d_ + k) - own
            out[X['p_velo_rec']] = dev(rp[PS['velo']], rp[PS['fb']], own_velo, k_rp)
            out[X['p_whiff_rec']] = dev(rp[PS['wh']], rp[PS['sw']], own_whiff, k_rp)
            out[X['p_csw_rec']] = dev(rp[PS['cs']] + rp[PS['wh']], rp[PS['n']], own_csw, k_rp)
            out[X['b_ev_rec']] = dev(rbb[BS['ev']], rbb[BS['bip']], out[X['b_ev']], k_rb)
            out[X['b_whiff_rec']] = dev(rb[PS['wh']], rb[PS['sw']], out[X['b_whiff']], k_rp)
            out[X['b_hard_rec']] = dev(rbb[BS['hard']], rbb[BS['bip']], out[X['b_hard']], k_rb)
        return out

    def absorb_date(self, today: int, batters, pitchers, M: np.ndarray, ev: np.ndarray, la: np.ndarray, run_value: np.ndarray | None):
        """Add one whole date's rows (arrays of equal length) after that date has been scored."""
        cur_velo: dict = {}
        day_p: dict = {}; day_b: dict = {}
        want_xv, recent = bool(self.p.get('xvalue')), int(self.p.get('recent_days') or 0)
        for r in range(len(batters)):
            if np.isnan(M[r, 0]):
                continue
            b, p = int(batters[r]), int(pitchers[r])
            self.bp[b] += M[r]; self.pp[p] += M[r]; self.Lp += M[r]
            if M[r, PS['fb']] > 0:
                cv = cur_velo.get(p, (0.0, 0.0)); cur_velo[p] = (cv[0] + M[r, PS['fb']], cv[1] + M[r, PS['velo']])
            v = None
            if not np.isnan(ev[r]):
                v = bip_vector(ev[r], la[r])
                self.bb[b] += v; self.pb[p] += v; self.Lb += v
                if want_xv:
                    ci, cj = cell_of(ev[r], la[r])
                    self.bcells[b][ci, cj] += 1.0; self.pcells[p][ci, cj] += 1.0
                    rv = float(run_value[r]) if run_value is not None else 0.0
                    self.cell_sum[ci, cj] += rv; self.cell_n[ci, cj] += 1.0
                    self.cell_value_total[0] += rv; self.cell_value_total[1] += 1.0
            if recent:
                for table, key in ((day_p, p), (day_b, b)):
                    e = table.get(key)
                    if e is None:
                        e = table[key] = [today, self.zero_p.copy(), self.zero_b.copy()]
                    e[1] += M[r]
                    if v is not None:
                        e[2] += v
        if recent:
            for key, e in day_p.items():
                self.rec_p[key].append(e)
            for key, e in day_b.items():
                self.rec_b[key].append(e)
        self.last_velo.update(cur_velo)


def _aligned(pa: pd.DataFrame, table: pd.DataFrame):
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    ph = table.drop_duplicates(['game_pk', 'at_bat_number']).set_index(['game_pk', 'at_bat_number'])
    joined = ordered[['game_pk', 'at_bat_number', 'batter', 'pitcher']].join(ph.drop(columns=['date_key'], errors='ignore'), on=['game_pk', 'at_bat_number'], rsuffix='_ph')
    has = joined['n'].notna().to_numpy()
    match = has & (joined['batter'].to_numpy() == joined['batter_ph'].to_numpy()) & (joined['pitcher'].to_numpy() == joined['pitcher_ph'].to_numpy())
    M = joined[PITCH_SUMS].to_numpy(float, copy=True); M[~match] = np.nan
    ev = joined['ev'].to_numpy(float, copy=True); la = joined['la'].to_numpy(float, copy=True); ev[~match] = np.nan; la[~match] = np.nan
    audit = {'pa_rows': int(len(ordered)), 'physics_rows': int(len(ph)), 'joined': int(has.sum()), 'ids_match': int(match.sum())}
    return ordered, M, ev, la, audit


def build_features(pa: pd.DataFrame, table: pd.DataFrame, params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Chronological, date-blocked physics features for every PA row (sorted like the engine's builder)."""
    S = _Sums(params or {})
    ordered, M, ev, la, audit = _aligned(pa, table)
    audit['params'] = S.p
    batters = ordered['batter'].to_numpy(int); pitchers = ordered['pitcher'].to_numpy(int)
    dates = ordered['date_key'].astype(str).to_numpy()
    day_ord = pd.to_datetime(ordered['date_key']).map(pd.Timestamp.toordinal).to_numpy()
    rv = ordered['outcome'].map(RUN_VALUE).to_numpy(float) if 'outcome' in ordered.columns else None
    out = np.full((len(ordered), len(S.names)), np.nan, dtype=np.float32)
    i, n = 0, len(ordered)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        today = int(day_ord[i])
        for r in range(i, j):
            out[r] = S.features(int(batters[r]), int(pitchers[r]), today)
        S.absorb_date(today, batters[i:j], pitchers[i:j], M[i:j], ev[i:j], la[i:j], None if rv is None else rv[i:j])
        i = j
    audit['league_fastball_velocity'] = float(S.Lp[PS['velo']] / max(S.Lp[PS['fb']], 1)); audit['league_exit_velocity'] = float(S.Lb[BS['ev']] / max(S.Lb[BS['bip']], 1))
    audit['league_hard_hit_rate'] = float(S.Lb[BS['hard']] / max(S.Lb[BS['bip']], 1)); audit['league_whiff_per_swing'] = float(S.Lp[PS['wh']] / max(S.Lp[PS['sw']], 1))
    audit['league_ground_ball_share'] = float(S.Lb[BS['gb']] / max(S.Lb[BS['bip']], 1))
    return pd.DataFrame(out, columns=S.names), audit


class PhysicsState:
    """The physics counters as they stand at the start of ``cutoff_date`` (live provider side).

    Built from a physics table that carries date_key and outcome (run value) per row; rows dated
    on or after the cutoff are ignored. ``features(batter, pitcher)`` returns the same vector the
    chronological builder would have produced for a PA on the cutoff date.
    """

    def __init__(self, sums: _Sums, cutoff_date: str, n_rows: int):
        self.sums, self.cutoff_date, self.n_rows = sums, cutoff_date, n_rows
        self.today = pd.Timestamp(cutoff_date).toordinal()
        self.names = sums.names

    @classmethod
    def build(cls, table: pd.DataFrame, cutoff_date: str, params: dict | None = None, outcomes: pd.Series | None = None) -> 'PhysicsState':
        S = _Sums(params or {})
        frame = table.loc[table['date_key'].astype(str) < str(cutoff_date)[:10]]
        frame = frame.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
        if 'run_value' in frame.columns:
            rv = frame['run_value'].to_numpy(float)
        elif 'outcome' in frame.columns:
            rv = frame['outcome'].map(RUN_VALUE).fillna(0.0).to_numpy(float)
        else:
            rv = None
        M = frame[PITCH_SUMS].to_numpy(float, copy=True)
        ev = frame['ev'].to_numpy(float, copy=True); la = frame['la'].to_numpy(float, copy=True)
        batters = frame['batter'].to_numpy(int); pitchers = frame['pitcher'].to_numpy(int)
        dates = frame['date_key'].astype(str).to_numpy()
        day_ord = pd.to_datetime(frame['date_key']).map(pd.Timestamp.toordinal).to_numpy()
        i, n = 0, len(frame)
        while i < n:
            j = i
            while j < n and dates[j] == dates[i]:
                j += 1
            S.absorb_date(int(day_ord[i]), batters[i:j], pitchers[i:j], M[i:j], ev[i:j], la[i:j], None if rv is None else rv[i:j])
            i = j
        return cls(S, str(cutoff_date)[:10], n)

    def features(self, batter: int, pitcher: int) -> np.ndarray:
        return self.sums.features(int(batter), int(pitcher), self.today)
