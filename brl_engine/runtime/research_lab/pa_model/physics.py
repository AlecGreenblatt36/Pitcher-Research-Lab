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

PITCH_SUMS = ['n', 'sw', 'wh', 'oz', 'ch', 'iz', 'izs', 'izc', 'cs', 'fb', 'velo', 'spin', 'ivb', 'hb',
              'fb_sw', 'fb_wh', 'br_n', 'br_sw', 'br_wh', 'os_n', 'os_sw', 'os_wh']
TABLE_VERSION = 2                      # per-pitch-type sums were added in version 2; older tables must be rebuilt
BREAKING = {'SL', 'ST', 'SV', 'CU', 'KC', 'CS', 'KN', 'SC'}
OFFSPEED = {'CH', 'FS', 'FO', 'EP'}
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
TYPE_FEATURES = ['b_whiff_fb', 'b_whiff_br', 'b_whiff_os', 'p_whiff_fb', 'p_whiff_br', 'p_whiff_os', 'p_share_fb', 'p_share_br']
# Matchup interactions: the batter's whiff weakness by pitch family (beyond his overall rate and the league's family rates),
# weighted by tonight's pitcher's family mix; and his fastball weakness times the pitcher's velocity above the league.
MATCHUP_FEATURES = ['mx_whiff', 'mx_velo']
# Pitcher workload: pitched yesterday, back within three days, pitches in his last outing and over the last 14 days (hundreds).
WORKLOAD_FEATURES = ['p_b2b', 'p_rest_short', 'p_last_n', 'p_load14']
DEFENSE_FEATURES = ['f_def']            # the fielding team's out rate on fieldable balls in play above the league, prior window
ENV_FEATURES = ['env_hr', 'env_k', 'env_bb', 'env_out', 'season_day']   # league run environment of the last 30 days vs the whole history
ENV_CLASSES = {'env_hr': ('HR',), 'env_k': ('K',), 'env_bb': ('BB_HBP',), 'env_out': ('BIP_OUT',)}
DEFAULT_PARAMS = {'k_rate': 150.0, 'k_bip': 60.0, 'k_velo': 100.0, 'xvalue': False, 'recent_days': 0,
                  'k_recent_pitch': 100.0, 'k_recent_bip': 40.0, 'k_cell': 40.0, 'pitch_types': False, 'k_type': 60.0,
                  'defense': False, 'k_def': 400.0, 'defense_days': 365, 'environment': False, 'env_days': 30, 'k_env': 2000.0,
                  'matchup': False, 'workload': False}
FIELDABLE = {'BIP_OUT', '1B', '2B_3B', 'OTHER_REACH'}
ENV_LABELS = ['BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH']
ENV_INDEX = {l: i for i, l in enumerate(ENV_LABELS)}


def _season_start(day: int) -> int:
    """Ordinal of March 15 of that day's year: a fixed origin for the day-of-season feature."""
    import datetime
    year = datetime.date.fromordinal(day).year
    return datetime.date(year, 3, 15).toordinal()
RUN_VALUE = {'BIP_OUT': 0.0, 'K': 0.0, 'BB_HBP': 0.69, '1B': 0.88, '2B_3B': 1.4, 'HR': 2.03, 'OTHER_REACH': 0.6}
EV_BINS, LA_BINS = 16, 18


def feature_names(params: dict | None = None) -> list[str]:
    p = dict(DEFAULT_PARAMS, **(params or {}))
    names = list(BASE_FEATURES)
    if p.get('xvalue'):
        names += XVALUE_FEATURES
    if p.get('recent_days'):
        names += RECENT_FEATURES
    if p.get('pitch_types'):
        names += TYPE_FEATURES
    if p.get('matchup'):
        names += MATCHUP_FEATURES
    if p.get('workload'):
        names += WORKLOAD_FEATURES
    if p.get('defense'):
        names += DEFENSE_FEATURES
    if p.get('environment'):
        names += ENV_FEATURES
    return names


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _type_sums(a: np.ndarray, ptype: str, swing: bool, whiff: bool) -> None:
    if ptype in FASTBALL:
        a[PS['fb_sw']] += swing; a[PS['fb_wh']] += whiff
    elif ptype in BREAKING:
        a[PS['br_n']] += 1; a[PS['br_sw']] += swing; a[PS['br_wh']] += whiff
    elif ptype in OFFSPEED:
        a[PS['os_n']] += 1; a[PS['os_sw']] += swing; a[PS['os_wh']] += whiff


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
        _type_sums(a, ptype, swing, whiff)
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
        _type_sums(a, ptype, swing, whiff)
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
        self.env_days: list = []                                            # [day ordinal, counts by class (7)] per date, league wide
        self.outings: dict = defaultdict(list)                              # pitcher -> [[day ordinal, pitches], ...] per date pitched
        self.env_total = np.zeros(7)
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

    def features(self, batter: int, pitcher: int, today: int, team_defense=None) -> np.ndarray:
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
        if P.get('pitch_types'):
            k_t = float(P['k_type'])
            out[X['b_whiff_fb']] = shrunk(sb, PS['fb_wh'], PS['fb_sw'], Lp, k_t)
            out[X['b_whiff_br']] = shrunk(sb, PS['br_wh'], PS['br_sw'], Lp, k_t)
            out[X['b_whiff_os']] = shrunk(sb, PS['os_wh'], PS['os_sw'], Lp, k_t)
            out[X['p_whiff_fb']] = shrunk(sp, PS['fb_wh'], PS['fb_sw'], Lp, k_t)
            out[X['p_whiff_br']] = shrunk(sp, PS['br_wh'], PS['br_sw'], Lp, k_t)
            out[X['p_whiff_os']] = shrunk(sp, PS['os_wh'], PS['os_sw'], Lp, k_t)
            out[X['p_share_fb']] = shrunk(sp, PS['fb'], PS['n'], Lp, k_rate)
            out[X['p_share_br']] = shrunk(sp, PS['br_n'], PS['n'], Lp, k_rate)
        if P.get('matchup'):
            k_t = float(P['k_type'])
            league_all = Lp[PS['wh']] / max(Lp[PS['sw']], 1e-9)
            rel_all = out[X['b_whiff']] - league_all
            dev, share = {}, {}
            for g, (wh, sw, n_) in {'fb': ('fb_wh', 'fb_sw', 'fb'), 'br': ('br_wh', 'br_sw', 'br_n'), 'os': ('os_wh', 'os_sw', 'os_n')}.items():
                league_g = Lp[PS[wh]] / max(Lp[PS[sw]], 1e-9)
                dev[g] = (shrunk(sb, PS[wh], PS[sw], Lp, k_t) - league_g) - rel_all
                share[g] = shrunk(sp, PS[n_], PS['n'], Lp, k_rate)
            total = max(sum(share.values()), 1e-9)
            out[X['mx_whiff']] = sum(share[g] / total * dev[g] for g in dev)
            league_velo = Lp[PS['velo']] / max(Lp[PS['fb']], 1e-9)
            out[X['mx_velo']] = dev['fb'] * (own_velo - league_velo) / 5.0
        if P.get('workload'):
            outs_ = self.outings.get(pitcher)
            if outs_:
                rest = today - outs_[-1][0]
                out[X['p_b2b']] = float(rest == 1)
                out[X['p_rest_short']] = float(rest <= 3)
                out[X['p_last_n']] = outs_[-1][1] / 100.0
                out[X['p_load14']] = sum(n for d_, n in outs_[-14:] if d_ >= today - 14) / 100.0
            else:
                out[X['p_b2b']] = out[X['p_rest_short']] = out[X['p_last_n']] = out[X['p_load14']] = 0.0
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
        if P.get('defense'):
            out[X['f_def']] = np.nan if team_defense is None else float(team_defense)
        if P.get('environment'):
            days, k = int(P['env_days']), float(P['k_env'])
            recent = np.zeros(7)
            for e in self.env_days:
                if e[0] >= today - days:
                    recent += e[1]
            total = self.env_total
            n_recent, n_total = recent.sum(), max(total.sum(), 1e-9)
            for name, classes in ENV_CLASSES.items():
                idx = [ENV_LABELS.index(c) for c in classes]
                base = total[idx].sum() / n_total
                rate = (recent[idx].sum() + k * base) / (n_recent + k)
                out[X[name]] = float(np.log(max(rate, 1e-6) / max(base, 1e-6)))
            out[X['season_day']] = float(today - _season_start(today))
        return out

    def absorb_date(self, today: int, batters, pitchers, M: np.ndarray, ev: np.ndarray, la: np.ndarray, run_value: np.ndarray | None, outcomes=None):
        """Add one whole date's rows (arrays of equal length) after that date has been scored."""
        if self.p.get('environment') and outcomes is not None:
            counts = np.zeros(7)
            for o in outcomes:
                i = ENV_INDEX.get(o)
                if i is not None:
                    counts[i] += 1.0
            self.env_days.append([today, counts]); self.env_total += counts
        cur_velo: dict = {}
        day_p: dict = {}; day_b: dict = {}
        day_n: dict = {}
        want_xv, recent = bool(self.p.get('xvalue')), int(self.p.get('recent_days') or 0)
        for r in range(len(batters)):
            if np.isnan(M[r, 0]):
                continue
            b, p = int(batters[r]), int(pitchers[r])
            self.bp[b] += M[r]; self.pp[p] += M[r]; self.Lp += M[r]
            day_n[p] = day_n.get(p, 0.0) + float(M[r, PS['n']])
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
        for p_, n_ in day_n.items():
            self.outings[p_].append([today, n_])


def check_table(table: pd.DataFrame) -> None:
    missing = [c for c in PITCH_SUMS + ['ev', 'la', 'game_pk', 'at_bat_number', 'pitcher', 'batter'] if c not in table.columns]
    if missing:
        raise ValueError('physics table is from an older schema (missing ' + ', '.join(missing[:4]) + '); rebuild it with the backfill')


class DefenseState:
    """Each fielding team's out rate on fieldable balls in play over the prior window, above the league rate.

    Built from plate-appearance rows (date_key, home_team, away_team, inning_topbot, outcome) dated strictly
    before the cutoff; value(team) is the shrunk rate difference the model sees as f_def. The chronological
    builder computes the same number for every PA from the same sums.
    """

    def __init__(self, params: dict | None = None):
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        self.k, self.days = float(self.p['k_def']), int(self.p['defense_days'])
        self.entries: dict = defaultdict(list)      # team -> [[day ordinal, bip, outs], ...] per date
        self.league = np.zeros(2)                   # bip, outs (all time)
        self.cutoff_day = None

    def absorb_date(self, today: int, teams, fieldable: np.ndarray, outs: np.ndarray):
        day: dict = {}
        for t, f, o in zip(teams, fieldable, outs):
            if not f:
                continue
            e = day.get(t)
            if e is None:
                e = day[t] = [today, 0.0, 0.0]
            e[1] += 1.0; e[2] += float(o)
            self.league += (1.0, float(o))
        for t, e in day.items():
            self.entries[t].append(e)

    def value(self, team, today: int | None = None) -> float:
        today = self.cutoff_day if today is None else today
        bip = outs = 0.0
        for e in self.entries.get(str(team), ()):
            if e[0] >= today - self.days:
                bip += e[1]; outs += e[2]
        league = self.league[1] / max(self.league[0], 1e-9)
        return (outs + self.k * league) / (bip + self.k) - league

    @staticmethod
    def fielding_teams(frame: pd.DataFrame) -> np.ndarray:
        top = frame['inning_topbot'].astype(str).str.lower().str.startswith('top').to_numpy()
        return np.where(top, frame['home_team'].astype(str).to_numpy(), frame['away_team'].astype(str).to_numpy())

    @classmethod
    def build(cls, history: pd.DataFrame, cutoff_date: str, params: dict | None = None) -> 'DefenseState':
        D = cls(params)
        frame = history.loc[history['date_key'].astype(str) < str(cutoff_date)[:10], ['date_key', 'home_team', 'away_team', 'inning_topbot', 'outcome']]
        frame = frame.sort_values('date_key', kind='mergesort').reset_index(drop=True)
        teams = cls.fielding_teams(frame)
        fieldable = frame['outcome'].isin(FIELDABLE).to_numpy()
        outs = (frame['outcome'] == 'BIP_OUT').to_numpy()
        dates = frame['date_key'].astype(str).to_numpy()
        day_ord = pd.to_datetime(frame['date_key']).map(pd.Timestamp.toordinal).to_numpy()
        i, n = 0, len(frame)
        while i < n:
            j = i
            while j < n and dates[j] == dates[i]:
                j += 1
            D.absorb_date(int(day_ord[i]), teams[i:j], fieldable[i:j], outs[i:j])
            i = j
        D.cutoff_day = pd.Timestamp(cutoff_date).toordinal()
        return D


def _aligned(pa: pd.DataFrame, table: pd.DataFrame):
    check_table(table)
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
    want_def = bool(S.p.get('defense'))
    if want_def:
        D = DefenseState(S.p)
        teams = DefenseState.fielding_teams(ordered)
        fieldable = ordered['outcome'].isin(FIELDABLE).to_numpy(); outs = (ordered['outcome'] == 'BIP_OUT').to_numpy()
        def_col = S.X['f_def']
    i, n = 0, len(ordered)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        today = int(day_ord[i])
        for r in range(i, j):
            out[r] = S.features(int(batters[r]), int(pitchers[r]), today, team_defense=(D.value(teams[r], today) if want_def else None))
        S.absorb_date(today, batters[i:j], pitchers[i:j], M[i:j], ev[i:j], la[i:j], None if rv is None else rv[i:j],
                      outcomes=(ordered['outcome'].to_numpy()[i:j] if S.p.get('environment') else None))
        if want_def:
            D.absorb_date(today, teams[i:j], fieldable[i:j], outs[i:j])
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
        check_table(table)
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
        outcomes = frame['outcome'].astype(str).to_numpy() if ('outcome' in frame.columns and S.p.get('environment')) else None
        if S.p.get('environment') and outcomes is None:
            raise ValueError('environment features need the outcome column in the physics table')
        i, n = 0, len(frame)
        while i < n:
            j = i
            while j < n and dates[j] == dates[i]:
                j += 1
            S.absorb_date(int(day_ord[i]), batters[i:j], pitchers[i:j], M[i:j], ev[i:j], la[i:j], None if rv is None else rv[i:j],
                          outcomes=None if outcomes is None else outcomes[i:j])
            i = j
        return cls(S, str(cutoff_date)[:10], n)

    def features(self, batter: int, pitcher: int, team_defense=None) -> np.ndarray:
        return self.sums.features(int(batter), int(pitcher), self.today, team_defense=team_defense)
