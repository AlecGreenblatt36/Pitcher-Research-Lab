"""The matchup report: for every game of a date, what the credited pitch-level models say about each hitter against
the pitchers he will face, built only from pitches thrown before the report's as-of date, and, once the game is
played, graded against what happened.

Runs inside Actions (brl-report workflow) on the sealed pitch tables; only model outputs, per-player summaries and
public box-score facts leave the runner. Output: public/reports/<date>.json on the ledger branch (and an index), which
the live site reads straight from the public data branch.

Per hitter against each pitcher (the starter and the relievers used most in the last 14 days):
- chase_points: extra chases per 100 outside pitches from his own swing map at the decision moment against this
  pitcher's own pitches (MATCHUP-01F representation with the league's family part and his family maps, MATCHUP-05F),
  with the pair chase calibration (0.95);
- k_points and bb_points: the count-by-count chain's strikeout and walk change against the league (ENGINE-01 scale);
- aim: where to put outside pitches of each family (the best third of the pitcher's own spots by the hitter's own part,
  under 0.6 ft of command scatter) and the runs per 100 plate appearances that is worth (VALUE-08F);
- the hitter's own chase map by family and his whiff map (7 by 7 grids), his zone top and bottom.

Grade (finished games, every pitch in the table): for each pitcher-hitter pair, the pitches thrown outside the zone,
how many landed in the recommended cells against the pitcher's usual share, the hitter's chases against what his map
predicted on those very pitches (the league alone and with his map), and the plate appearance results. The as-of date
is stated on every report; a backfilled month uses the maps as of the first of that month, so no report ever uses a
pitch from its own game or later.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('brl_discovery', ROOT / 'tools' / 'brl_discovery.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)

SCHEMA = 'brl.report.v1'
GU = np.array([-1.25, -0.83, -0.42, 0.0, 0.42, 0.83, 1.25]); GZ = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
FAMILIES = (('fastball', (0, 1, 2, 6), 0, 94.0), ('breaking', (3, 4), 3, 85.0), ('offspeed', (5,), 5, 86.0))
B_OUT_FAMILY, N_OUT = -0.000657, 1.876          # VALUE-08F (family maps): runs per point of the own part, outside pitches per plate appearance
STRUCTURAL = True                                # VALUE-18: aims chosen and priced by the engine's components (whiff, called strike, foul, contact value, count values)
OWN_PART_CALIBRATION = {'outside': 0.95, 'inside': 0.82}   # VALUE-18 (2025 development run): how much of a fitted point of the own part shows up in actual swings
CHASE_SLOPE, K_SCALE, BB_SCALE = 0.95, 0.40, 0.53   # MATCHUP-01F pair slope; ENGINE-01 coefficients over calibrated
sig = lambda v: 1 / (1 + np.exp(-v))


BINS = ('under -2', '-2 to 0', '0 to 2', 'over 2')


def bin_of(chase_points: float) -> str:
    return BINS[0] if chase_points < -2 else BINS[1] if chase_points < 0 else BINS[2] if chase_points < 2 else BINS[3]


def record_from(days: dict) -> dict:
    """The forward record over every graded game: chases against the league's and the maps' expectations, by what the
    map predicted for the pair, and staffs' aiming against their usual rates."""
    rec = {'schema': 'brl.report-record.v1', 'built_at': datetime.now(timezone.utc).isoformat(), 'games': 0, 'dates': 0, 'first_date': None, 'last_date': None,
           'outside_pitches': 0, 'in_recommended': 0, 'usual_expected': 0.0, 'chases': 0, 'chases_expected_league': 0.0, 'chases_expected_map': 0.0,
           'bins': {k: {'pairs': 0, 'outside': 0, 'chases': 0, 'league': 0.0, 'map': 0.0} for k in BINS}, 'by_month': {}}
    for day in sorted(days):
        doc = days[day]; used = False
        for g in (doc.get('games') or {}).values():
            gr = g.get('grade')
            if not gr or not gr.get('pairs'):
                continue
            used = True; rec['games'] += 1
            for k in ('outside_pitches', 'in_recommended', 'usual_expected', 'chases', 'chases_expected_league', 'chases_expected_map'):
                rec[k] += gr.get(k, 0)
            mo = rec['by_month'].setdefault(day[:7], {'games': 0, 'outside_pitches': 0, 'in_recommended': 0, 'usual_expected': 0.0, 'chases': 0, 'chases_expected_league': 0.0, 'chases_expected_map': 0.0})
            mo['games'] += 1
            for k in ('outside_pitches', 'in_recommended', 'usual_expected', 'chases', 'chases_expected_league', 'chases_expected_map'):
                mo[k] += gr.get(k, 0)
            for k, bn in (gr.get('bins') or {}).items():
                if k in rec['bins']:
                    for kk in ('pairs', 'outside', 'chases', 'league', 'map'):
                        rec['bins'][k][kk] += bn.get(kk, 0)
        if used:
            rec['dates'] += 1; rec['first_date'] = rec['first_date'] or day; rec['last_date'] = day
    for d_ in [rec] + list(rec['by_month'].values()):
        for k in ('usual_expected', 'chases_expected_league', 'chases_expected_map'):
            d_[k] = round(d_[k], 1)
    for bn in rec['bins'].values():
        bn['league'] = round(bn['league'], 1); bn['map'] = round(bn['map'], 1)
    return rec


def put(repo, token, path, text, branch, message, tries=14):
    """put_text with patience: several backfills commit to the same branch at once, so a 409 is ordinary."""
    import random
    from urllib.error import HTTPError
    for attempt in range(tries):
        try:
            return D.put_text(repo, token, path, text, branch, message)
        except HTTPError as exc:
            if exc.code not in (409, 422) or attempt == tries - 1:
                raise
            time.sleep(random.uniform(2, 6) * (1 + attempt / 3))


def rebuild_index(repo, token, branch) -> tuple[dict, dict]:
    """The reports index from what is on the branch (never a read-modify-write, which parallel backfills would race):
    every public/reports/<date>/index.json, as {date: summary} and the index document."""
    listing = D.api(f'https://api.github.com/repos/{repo}/contents/public/reports?ref={branch}', token)
    days = {}
    for item in listing if isinstance(listing, list) else []:
        if item.get('type') == 'dir' and len(item.get('name', '')) == 10:
            raw = D.read_blob(repo, token, f"public/reports/{item['name']}/index.json", branch)
            if raw is not None:
                try:
                    days[item['name']] = json.loads(raw.decode())
                except ValueError:
                    pass
    idx = {'schema': 'brl.reports.v1', 'dates': {}}
    for day, doc in sorted(days.items()):
        games = doc.get('games') or {}
        idx['dates'][day] = {'asof': doc.get('asof'), 'games': len(games), 'graded': sum(1 for g in games.values() if (g.get('grade') or {}).get('pairs'))}
    return idx, days


def famof(g: int) -> str:
    return 'fastball' if g in (0, 1, 2, 6) else ('breaking' if g in (3, 4) else 'offspeed')


class Fitted:
    """Everything fitted on the pitches before the as-of date: league models, hitter maps, pools of pitcher spots."""

    def __init__(self, T: dict, asof_day: int, stage, min_pitches=400, min_swings=200):
        F = D.rebuild(T)
        keep = F['ok'] & (T['group'] >= 0) & (T['call'] <= 2) & (T['balls'] >= 0) & (T['balls'] <= 3) & (T['strikes'] >= 0) & (T['strikes'] <= 2) & ~((T['bunt_pa'] == 1) & (T['last_in_pa'] == 1))
        self.T = T = D.take(T, keep); F = {k: v[keep] for k, v in F.items()}
        self.train = tr = T['day'] < asof_day
        self.n_train = int(tr.sum())
        swing = ((T['call'] == 1) | (T['call'] == 2)).astype(np.float64); whiff = (T['call'] == 2).astype(np.float64)
        self.swing, self.whiff = swing, whiff
        self.xp, self.zp = D.projected(T, F, None, 'straight', 0.26)
        self.xt, self.zt = T['px'].astype(np.float64), T['pz'].astype(np.float64)
        u_t = np.where(T['stand_r'] == 1, self.xt, -self.xt)
        self.outside = (np.abs(u_t) > D.ZONE_HALF) | (self.zt > D.ZONE_TOP) | (self.zt < D.ZONE_BOT)
        rng = self.rng = np.random.default_rng(11)
        # propensities use earlier dates only (as the maps will); the report's games sit after every training pitch
        Tp = dict(T); Tp['day'] = np.where(tr, T['day'], asof_day)
        prop_s = D.swing_propensity(Tp); prop_p = D.pitcher_propensity(Tp)
        self.Ls = D.location_block(self.xp, self.zp, T['stand_r'], T['strikes'])
        self.Bh0 = D.hitter_basis(self.xp, self.zp, T['stand_r'], T['strikes']); nh = self.nh = self.Bh0.shape[1] - 1
        lf = [(self.Bh0[:, :-1] * np.isin(T['group'], (3, 4))[:, None]).astype(np.float32), (self.Bh0[:, :-1] * (T['group'] == 5)[:, None]).astype(np.float32)]
        Xs = np.hstack([self.Ls] + lf + [D.control_block(T, prop_s), prop_p[:, None].astype(np.float32)]); del lf
        self.i_ps = self.Ls.shape[1] + 2 * nh + 32
        idx = np.flatnonzero(tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
        self.m_s = D.fit_logistic(Xs[idx], swing[idx]); self.off_s = self.m_s.decision_function(Xs); del Xs
        cf_ = self.m_s.coef_[0].astype(np.float64); nL = self.Ls.shape[1]
        self.wL, self.wB, self.wO = cf_[:nL], cf_[nL:nL + nh], cf_[nL + nh:nL + 2 * nh]
        self.Bs = D.family_basis(self.Bh0, T['group'])
        stage('league swing model')
        gb_tr = self.gb_tr = D._groups(T['batter'], tr)
        self.maps_s = D._hitter_maps(self.Bs, swing, self.off_s, gb_tr, D.FAMILY_SHRINKAGE_LEAGUE_FAMILY['location_by_family'], min_pitches)
        self.side = {h: int(np.round(T['stand_r'][r].mean())) for h, r in gb_tr.items()}
        Ssum = {0: 0.0, 1: 0.0}; Nn = {0: 0, 1: 0}
        for h, m in self.maps_s.items():
            Ssum[self.side[h]] = Ssum[self.side[h]] + m; Nn[self.side[h]] += 1
        self.mbar = {h: (Ssum[self.side[h]] - m) / max(Nn[self.side[h]] - 1, 1) for h, m in self.maps_s.items()}
        self.mbar_side = {sd: Ssum[sd] / max(Nn[sd], 1) for sd in (0, 1)}
        stage(f'hitter swing maps {len(self.maps_s)}')
        # whiff model on the true crossing (MATCHUP-03's representation)
        tw = dict(Tp); tw['call'] = np.where(whiff == 1, 1, np.where(swing == 1, 0, 3)); prop_w = D.swing_propensity(tw, 200.0)
        grp = np.zeros((len(swing), 7), np.float32); grp[np.arange(len(swing)), np.clip(T['group'], 0, 6)] = 1
        Lw = D.location_block(self.xt, self.zt, T['stand_r'], T['strikes'])
        Xw = np.hstack([Lw, grp, D.hats(T['v0'].astype(np.float64), D.V_KNOTS), (T['strikes'] == 2)[:, None].astype(np.float32), prop_w[:, None].astype(np.float32),
                        (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)])
        sw_tr = tr & (swing == 1)
        idx = np.flatnonzero(sw_tr); idx = rng.choice(idx, min(len(idx), 600000), replace=False)
        self.m_w = D.fit_logistic(Xw[idx], whiff[idx]); self.off_w = self.m_w.decision_function(Xw)
        fam = np.column_stack([np.isin(T['group'], (0, 1, 2)), np.isin(T['group'], (3, 4)), np.isin(T['group'], (5,))]).astype(np.float64)
        self.Bw = np.hstack([D.hitter_basis(self.xt, self.zt, T['stand_r'], T['strikes']), fam, D.hats(self.zt, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
        self.maps_w = D._hitter_maps(self.Bw, whiff, self.off_w, D._groups(T['batter'], sw_tr), 30.0, min_swings)
        Wsum = {0: 0.0, 1: 0.0}; Wn = {0: 0, 1: 0}
        for h, m in self.maps_w.items():
            sd_ = self.side.get(h, 1); Wsum[sd_] = Wsum[sd_] + m; Wn[sd_] += 1
        self.mbar_w = {h: (Wsum[self.side.get(h, 1)] - m) / max(Wn[self.side.get(h, 1)] - 1, 1) for h, m in self.maps_w.items()}
        # called strikes on takes and fouls on contact (league), for the count chain
        Xc = np.hstack([Lw, (T['stand_r'] == T['throw_r'])[:, None].astype(np.float32)])
        tk = tr & (T['call'] == 0); idx = np.flatnonzero(tk); idx = rng.choice(idx, min(len(idx), 400000), replace=False)
        self.p_cs = D.fit_logistic(Xc[idx], (T['cs'] == 1).astype(np.float64)[idx]).predict_proba(Xc)[:, 1]
        ct = tr & (T['call'] == 1); foul = ((T['call'] == 1) & (T['last_in_pa'] == 0)).astype(np.float64)
        idx = np.flatnonzero(ct); idx = rng.choice(idx, min(len(idx), 400000), replace=False)
        self.p_fo = D.fit_logistic(Xw[idx], foul[idx]).predict_proba(Xw)[:, 1]
        del Xw, Xc
        stage(f'whiff maps {len(self.maps_w)}')
        # batter zones from the training rows' official zone numbers (MATCHUP-08)
        top_b, bot_b, _ = D.batter_zones(T, tr)
        self.zone = {}
        for h, r in gb_tr.items():
            self.zone[h] = (float(top_b[r[0]]), float(bot_b[r[0]]))
        # league reference rates on the training rows, for the cards
        u_p = np.where(T['stand_r'] == 1, self.xp, -self.xp)
        looks_in = ~((np.abs(u_p) > D.ZONE_HALF) | (self.zp > D.ZONE_TOP) | (self.zp < D.ZONE_BOT))
        self.looks_in_ends_out = looks_in & self.outside
        self.league = {'chase_rate': round(float(swing[tr & self.outside].mean()), 3), 'whiff_rate': round(float(whiff[tr & (swing == 1)].sum() / max((tr & (swing == 1)).sum(), 1)), 3),
                       'looks_in_ends_out': round(float(self.looks_in_ends_out[tr].mean()), 3)}
        # the pitcher's training pitches
        self.gp = D._groups(T['pitcher'], tr)
        self.cgrp = np.where(T['strikes'] == 2, 2, np.where(T['balls'] > T['strikes'], 1, 0))
        self.cidx = T['balls'].astype(np.int64) * 3 + T['strikes'].astype(np.int64)
        K = 16; jit = np.random.default_rng(3).standard_normal((K, 2)); self.jit = (jit - jit.mean(0)) / jit.std(0); self.K = K
        self.pools = {}; self.chains = {}
        # VALUE-18: the engine's components for pricing aims (the planner's fitted pieces) and the training count values
        self.PM = None
        if STRUCTURAL:
            self.PM = D.PAModels(T, tr, rng, {'league_n': 500000, 'min_pitches': min_pitches, 'min_swings': min_swings}, stage)
            ci_all = np.clip(T['balls'], 0, 3) * 3 + np.clip(T['strikes'], 0, 2)
            fin_all = np.where(T['out7'] >= 0, D.LW7[np.clip(T['out7'], 0, 6)], np.nan)
            self.cv = np.array([float(np.nanmean(fin_all[tr & (ci_all == c_)])) if (tr & (ci_all == c_) & np.isfinite(fin_all)).any() else 0.0 for c_ in range(12)])
            self.xoff, self.zoff = self.xp - self.xt, self.zp - self.zt
        # grids
        UU, ZZ = np.meshgrid(GU, GZ); self.uu, self.zz = UU.ravel(), ZZ.ravel()
        self.Bg = D.hitter_basis(self.uu, self.zz, np.ones(len(self.uu), np.int64), np.zeros(len(self.uu), np.int64))
        self.Bgw = np.hstack([self.Bg, np.tile(np.array([1.0, 0.0, 0.0]), (len(self.uu), 1)), D.hats(self.zz, (1.0, 2.0, 3.0, 4.0)).astype(np.float64)])
        self.out_cells = [k for k in range(len(self.uu)) if (abs(self.uu[k]) > D.ZONE_HALF or self.zz[k] > D.ZONE_TOP or self.zz[k] < D.ZONE_BOT)]
        stage('fitted')

    def league_loc(self, LBm, Bh_m, g):
        return LBm.astype(np.float64) @ self.wL + np.isin(g, (3, 4)) * (Bh_m[:, :-1] @ self.wB) + (g == 5) * (Bh_m[:, :-1] @ self.wO)

    def hitter_card(self, h: int) -> dict | None:
        if h not in self.maps_s:
            return None
        T = self.T; r = self.gb_tr.get(h)
        sd = self.side[h]
        own = {}
        for fname, _, gcode, _ in FAMILIES:
            dgf = D.family_basis(self.Bg, np.full(len(self.uu), gcode, np.int64)) @ (self.maps_s[h] - self.mbar[h])
            own[fname] = {'grid': [round(float(v), 2) for v in dgf],
                          'top': [[float(self.uu[k]), float(self.zz[k]), round(float(dgf[k]), 3)] for k in sorted(self.out_cells, key=lambda k: -dgf[k])[:3] if dgf[k] > 0]}
        card = {'side': 'R' if sd == 1 else 'L', 'pitches': int(len(r)) if r is not None else 0, 'own_chase': own,
                'chase_rate': round(float(self.swing[r][self.outside[r]].mean()), 3) if r is not None and self.outside[r].any() else None,
                'whiff_rate': round(float(self.whiff[r].sum() / max(self.swing[r].sum(), 1)), 3) if r is not None else None}
        if h in self.maps_w:
            dw = self.Bgw @ self.maps_w[h]
            card['whiff_grid'] = [round(float(v), 2) for v in dw]
        if h in self.zone:
            card['zone_top_ft'], card['zone_bottom_ft'] = round(self.zone[h][0], 2), round(self.zone[h][1], 2)
        return card

    def pitcher_card(self, p: int) -> dict | None:
        r = self.gp.get(p)
        if r is None or len(r) < 150:
            return None
        T = self.T; mix = {}
        for fname, groups, _, _ in FAMILIES:
            mm = np.isin(T['group'][r], groups)
            if mm.any():
                mix[fname] = {'share': round(float(mm.mean()), 3), 'speed': round(float(np.nanmean(T['v0'][r][mm])), 1)}
        return {'throws': 'R' if T['throw_r'][r][0] == 1 else 'L', 'pitches': int(len(r)), 'mix': mix,
                'chase_rate_against': round(float(self.swing[r][self.outside[r]].mean()), 3) if self.outside[r].any() else None,
                'looks_in_ends_out': round(float(self.looks_in_ends_out[r].mean()), 3)}

    def _chain(self, ps_, pw_, pcs_, pfo_, ci_):
        use_c = np.bincount(ci_, minlength=12) >= 15; ki = ci_ % 3
        def agg(w):
            return np.where(use_c, np.bincount(ci_, weights=w, minlength=12), np.bincount(ki, weights=w, minlength=3)[np.arange(12) % 3])
        n_ = agg(np.ones(len(ps_))); s_ = agg(ps_); sw_ = agg(ps_ * pw_); t_ = agg(1 - ps_); tc_ = agg((1 - ps_) * pcs_)
        c_ = agg(ps_ * (1 - pw_)); cf_ = agg(ps_ * (1 - pw_) * pfo_)
        d = lambda v: {(b, k): float(v[b * 3 + k]) for b in range(4) for k in range(3)}
        return D._count_chain(d(s_ / np.maximum(n_, 1e-9)), d(sw_ / np.maximum(s_, 1e-9)), d(tc_ / np.maximum(t_, 1e-9)), d(cf_ / np.maximum(c_, 1e-9)))

    def _arsenal(self, p, sd):
        key = (p, sd)
        if key in self.chains:
            return self.chains[key]
        T = self.T; r = self.gp[p]; side = T['stand_r'][r] == sd
        r = r[side] if side.sum() >= 100 else r
        A = {'r': r, 'ci': self.cidx[r], 'pcs': self.p_cs[r], 'pfo': self.p_fo[r], 'os': self.off_s[r], 'ow': self.off_w[r], 'Bs': self.Bs[r], 'Bw': self.Bw[r], 'out': self.outside[r]}
        A['league'] = self._chain(sig(A['os']), sig(A['ow']), A['pcs'], A['pfo'], A['ci'])
        self.chains[key] = A
        return A

    def _pool(self, p, sd, c3, tg):
        key = (p, sd, c3, tg)
        if key in self.pools:
            return self.pools[key]
        T = self.T; r = self.gp[p]
        r = r[(T['stand_r'][r] == sd) & (self.cgrp[r] == c3) & self.outside[r] & (np.clip(T['group'][r], 0, 6) == tg)]
        if len(r) < 15:
            self.pools[key] = None; return None
        n_all = len(r)
        if len(r) > 250:
            r = self.rng.choice(r, 250, replace=False)
        pl_ = sig(self.off_s[r]); third = np.searchsorted(np.percentile(pl_, [33.3, 66.7]), pl_)
        lb0 = self.league_loc(self.Ls[r], self.Bh0[r], T['group'][r]); K = self.K; sgm = 0.6
        xj = (self.xp[r][:, None] + sgm * self.jit[None, :, 0]).ravel(); zj = (self.zp[r][:, None] + sgm * self.jit[None, :, 1]).ravel()
        sj = np.repeat(np.full(len(r), sd), K); kj = np.repeat(T['strikes'][r], K); gj = np.repeat(T['group'][r], K)
        Bj0 = D.hitter_basis(xj, zj, sj, kj)
        offj = np.repeat(self.off_s[r] - lb0, K) + self.league_loc(D.location_block(xj, zj, sj, kj), Bj0, gj)
        uu_ = np.where(sd == 1, self.xt[r], -self.xt[r])
        cell = np.argmin(np.abs(uu_[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[r][:, None] - GZ[None, :]), 1)
        struct = None
        if self.PM is not None:
            # the scattered pitch's true crossing (the same scatter), its side of the zone and the engine's blocks there
            xtj = xj - np.repeat(self.xoff[r], K); ztj = zj - np.repeat(self.zoff[r], K)
            uj = np.where(sd == 1, xtj, -xtj); e_j = np.maximum(np.maximum(np.abs(uj) - D.ZONE_HALF, ztj - D.ZONE_TOP), D.ZONE_BOT - ztj)
            lam_j = np.where(e_j > 0, OWN_PART_CALIBRATION['outside'], OWN_PART_CALIBRATION['inside'])
            bj_ = np.repeat(T['balls'][r], K)
            blk = self.PM.blocks_at(xj, zj, xtj, ztj, sj, np.repeat(T['throw_r'][r], K), gj, np.repeat(T['v0'][r].astype(np.float64), K), bj_, kj)
            struct = (blk, bj_, kj, lam_j, e_j > 0)
        self.pools[key] = (offj, D.family_basis(Bj0, gj), third, cell, len(r), n_all, struct)
        return self.pools[key]

    def pair(self, h: int, p: int) -> dict | None:
        """One hitter against one pitcher: chase, strikeout and walk changes, and the aim plan with its value."""
        if h not in self.maps_s or p not in self.gp or len(self.gp[p]) < 150:
            return None
        sd = self.side[h]; A = self._arsenal(p, sd)
        ph = sig(A['os'] + A['Bs'] @ self.maps_s[h]); pl = sig(A['os'])
        wmap = self.maps_w.get(h)
        hit = self._chain(ph, sig(A['ow'] + (A['Bw'] @ wmap if wmap is not None else 0.0)), A['pcs'], A['pfo'], A['ci'])
        chase = float((ph - pl)[A['out']].mean()) * 100 * CHASE_SLOPE if A['out'].any() else 0.0
        out = {'chase_points': round(chase, 2), 'k_points': round((hit[0] - A['league'][0]) * 100 * K_SCALE, 2), 'bb_points': round((hit[1] - A['league'][1]) * 100 * BB_SCALE, 2),
               'arsenal_pitches': int(len(A['r']))}
        T = self.T; rp = self.gp[p][(T['stand_r'][self.gp[p]] == sd) & self.outside[self.gp[p]]]
        # how much the numbers rest on: the hitter's training pitches behind his map, the pitcher's pitches to this side (the audit's insufficient-evidence state)
        n_h = int(len(self.gb_tr.get(h, ()))); n_ps = int((T['stand_r'][self.gp[p]] == sd).sum())
        out['evidence'] = {'hitter_pitches': n_h, 'pitcher_pitches_to_side': n_ps, 'whiff_map': bool(h in self.maps_w),
                           'level': 'thin' if (n_h < 700 or n_ps < 150) else 'ok'}
        if len(rp) < 40:
            return out
        tot = 0.0; wsum = 0.0; tot_runs = 0.0; cells = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
        # where each chosen spot's value comes from: the scattered pitches that stay outside (his extra chases) or land inside (strikes he takes)
        v_out = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}; v_in = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
        by_count = {}

        def labeled(c_, vo_, vi_):
            return [[float(GU[k % len(GU)]), float(GZ[k // len(GU)]), 'chase' if vo_[k] <= vi_[k] else 'take'] for k in np.argsort(c_)[::-1][:3] if c_[k] > 0]
        p_scalar = self.PM.p_scalar.get(p, (0.0, self.PM.lg_bip)) if self.PM is not None else None
        for c3, cname in ((0, 'even_or_ahead'), (1, 'behind'), (2, 'two_strikes')):
            ct = 0.0; cw = 0.0; ct_runs = 0.0; ccells = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
            cv_out = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}; cv_in = {f_: np.zeros(len(GU) * len(GZ)) for f_, *_ in FAMILIES}
            for tg in range(7):
                P_ = self._pool(p, sd, c3, tg)
                if P_ is None:
                    continue
                offj, Bj, third, cell, npool, n_all, struct = P_
                dev_jk = (sig(offj + Bj @ self.maps_s[h]) - sig(offj + Bj @ self.mbar[h])) * 100
                dj = dev_jk.reshape(npool, self.K).mean(1)
                if struct is not None:
                    # VALUE-18: the structural value of each aim, runs per pitch (negative is good for the pitcher): the calibrated own part
                    # times the value if the hitter swings minus if he takes, averaged over the scatter
                    blk, bj_, kj, lam_j, out_j = struct
                    tau_j = 0.01 * self.PM.swing_minus_take(blk, h, p_scalar, bj_, kj, self.cv)
                    vjk = dev_jk * lam_j * tau_j
                    vj = vjk.reshape(npool, self.K).mean(1)
                    vo_j = (vjk * out_j).reshape(npool, self.K).mean(1); vi_j = (vjk * ~out_j).reshape(npool, self.K).mean(1)
                gains = []; gains_runs = []
                for t3 in range(3):
                    sel = np.flatnonzero(third == t3)
                    if len(sel) >= 3:
                        k3 = max(1, len(sel) // 3)
                        best = sel[np.argsort(vj[sel])[:k3]] if struct is not None else sel[np.argsort(dj[sel])[::-1][:k3]]
                        gains.append(float(dj[best].mean() - dj[sel].mean()))
                        if struct is not None:
                            gains_runs.append(float(vj[best].mean() - vj[sel].mean()))
                            np.add.at(v_out[famof(tg)], cell[best], n_all * vo_j[best]); np.add.at(v_in[famof(tg)], cell[best], n_all * vi_j[best])
                            np.add.at(cv_out[famof(tg)], cell[best], n_all * vo_j[best]); np.add.at(cv_in[famof(tg)], cell[best], n_all * vi_j[best])
                        np.add.at(cells[famof(tg)], cell[best], n_all / len(rp)); np.add.at(ccells[famof(tg)], cell[best], n_all)
                if gains:
                    tot += n_all * float(np.mean(gains)); wsum += n_all; ct += n_all * float(np.mean(gains)); cw += n_all
                    if gains_runs:
                        tot_runs += n_all * float(np.mean(gains_runs)); ct_runs += n_all * float(np.mean(gains_runs))
            if cw > 0:
                by_count[cname] = {'gain_points': round(ct / cw, 2), 'pitches': int(cw),
                                   'cells': {f_: labeled(c_, cv_out[f_], cv_in[f_]) for f_, c_ in ccells.items()}}
                if self.PM is not None:
                    by_count[cname]['runs_per_100_pa'] = round(float(ct_runs / cw * N_OUT) * 100, 2)
        if wsum > 0:
            g_ = tot / wsum
            out['aim'] = {'runs_per_100_pa': round(float(tot_runs / wsum * N_OUT) * 100, 2) if self.PM is not None else round(float(B_OUT_FAMILY * g_ * N_OUT) * 100, 2),
                          'gain_points': round(g_, 2),
                          'cells': {f_: labeled(c_, v_out[f_], v_in[f_]) for f_, c_ in cells.items()},
                          'by_count': by_count, 'pricing': 'structural' if self.PM is not None else 'regression'}
            if self.PM is not None:
                out['aim']['runs_per_100_pa_regression'] = round(float(B_OUT_FAMILY * g_ * N_OUT) * 100, 2)
        # miss spots (unpriced): among the pitcher's two-strike pitches to this side, where this hitter's own whiff map
        # (his map minus the same-side mean) says he misses most, by family
        if wmap is not None and h in self.mbar_w:
            r2 = self.gp[p][(T['stand_r'][self.gp[p]] == sd) & (T['strikes'][self.gp[p]] == 2)]
            if len(r2) >= 40:
                dev = self.Bw[r2] @ (wmap - self.mbar_w[h])
                uu_ = np.where(sd == 1, self.xt[r2], -self.xt[r2])
                cell2 = np.argmin(np.abs(uu_[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[r2][:, None] - GZ[None, :]), 1)
                miss = {}
                for fname, groups, _, _ in FAMILIES:
                    mm = np.flatnonzero(np.isin(T['group'][r2], groups))
                    if len(mm) < 12:
                        continue
                    k3 = max(1, len(mm) // 3); best = mm[np.argsort(dev[mm])[::-1][:k3]]
                    acc = np.bincount(cell2[best], minlength=len(GU) * len(GZ))
                    miss[fname] = {'cells': [[float(GU[k % len(GU)]), float(GZ[k // len(GU)])] for k in np.argsort(acc)[::-1][:3] if acc[k] > 0],
                                   'own_whiff_points': round(float((sig(self.off_w[r2[best]] + self.Bw[r2[best]] @ wmap) - sig(self.off_w[r2[best]] + self.Bw[r2[best]] @ self.mbar_w[h])).mean()) * 100, 1)}
                if miss:
                    out['miss_spots'] = miss
        return out

    def grade(self, game: int, h: int, p: int, aim_cells: dict | None) -> dict | None:
        """The pitches p threw to h in this game against the report: outside pitches in the recommended cells against
        the pitcher's usual share, and the hitter's chases against the league's and his map's predictions."""
        T = self.T; r = np.flatnonzero((T['game'] == game) & (T['batter'] == h) & (T['pitcher'] == p))
        if len(r) == 0:
            return None
        out = {'pitches': int(len(r)), 'outside': int(self.outside[r].sum())}
        o = r[self.outside[r]]
        if len(o) and h in self.maps_s:
            pl = sig(self.off_s[o]); ph = sig(self.off_s[o] + self.Bs[o] @ self.maps_s[h])
            out['chases'] = int(self.swing[o].sum()); out['chases_expected_league'] = round(float(pl.sum()), 2); out['chases_expected_map'] = round(float(ph.sum()), 2)
        if len(o) and aim_cells:
            sd = self.side.get(h, 1); uu_ = np.where(sd == 1, self.xt[o], -self.xt[o])
            cell = np.argmin(np.abs(uu_[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[o][:, None] - GZ[None, :]), 1)
            rec = np.zeros(len(o), bool)
            for j, g in enumerate(T['group'][o]):
                cells = aim_cells.get(famof(int(g))) or []
                want = {int(np.argmin(np.abs(GU - c[0]))) + len(GU) * int(np.argmin(np.abs(GZ - c[1]))) for c in cells}
                rec[j] = cell[j] in want
            out['in_recommended_cells'] = int(rec.sum())
            # the pitcher's usual share: his training outside pitches to this side in those cells
            rp = self.gp.get(p)
            if rp is not None:
                rp = rp[(T['stand_r'][rp] == sd) & self.outside[rp]]
                if len(rp) >= 40:
                    uu2 = np.where(sd == 1, self.xt[rp], -self.xt[rp])
                    cell2 = np.argmin(np.abs(uu2[:, None] - GU[None, :]), 1) + len(GU) * np.argmin(np.abs(self.zt[rp][:, None] - GZ[None, :]), 1)
                    hits = 0
                    for fname, groups, _, _ in FAMILIES:
                        want = {int(np.argmin(np.abs(GU - c[0]))) + len(GU) * int(np.argmin(np.abs(GZ - c[1]))) for c in (aim_cells.get(fname) or [])}
                        mm = np.isin(T['group'][rp], groups)
                        hits += int(np.isin(cell2[mm], list(want)).sum())
                    out['usual_share_in_cells'] = round(hits / len(rp), 3)
        last = r[T['last_in_pa'][r] == 1]
        if len(last):
            o7 = T['out7'][last]; names = D.OUT7
            out['results'] = [names[int(v)] for v in o7 if 0 <= v < 7]
        return out


def mlb(url: str):
    import urllib.request
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'BRL-report/1.0', 'Accept': 'application/json'}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 + 3 * attempt)


def schedule(day: str) -> list:
    doc = mlb(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day}&gameType=R,F,D,L,W&hydrate=probablePitcher,lineups,team')
    out = []
    for d in doc.get('dates', []):
        for g in d.get('games', []):
            st = g.get('status') or {}
            teams = g.get('teams') or {}
            row = {'game_pk': int(g['gamePk']), 'game_type': g.get('gameType'), 'status': st.get('detailedState'), 'final': st.get('abstractGameState') == 'Final',
                   'start': g.get('gameDate'), 'teams': {}, 'lineups': {}}
            for side in ('away', 'home'):
                t = teams.get(side, {}); team = t.get('team') or {}
                row['teams'][side] = {'id': int(team.get('id') or 0), 'name': team.get('name'), 'abbr': team.get('abbreviation'),
                                      'probable': int(((t.get('probablePitcher') or {}).get('id')) or 0) or None}
            lu = g.get('lineups') or {}
            for side, key in (('away', 'awayPlayers'), ('home', 'homePlayers')):
                row['lineups'][side] = [int(x['id']) for x in (lu.get(key) or []) if x.get('id')]
            out.append(row)
    return out


def boxscore_players(game_pk: int) -> dict:
    """Finished games: who batted and pitched for each side, in batting order (public box score)."""
    doc = mlb(f'https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore')
    out = {}
    for side in ('away', 'home'):
        t = (doc.get('teams') or {}).get(side) or {}
        out[side] = {'batters': [int(v) for v in t.get('batters') or []], 'pitchers': [int(v) for v in t.get('pitchers') or []], 'battingOrder': [int(v) for v in t.get('battingOrder') or []]}
    return out


def recent_players(T: dict, game_pks_by_team: dict, team_games: dict) -> tuple[dict, dict]:
    """From the pitch table: each team's hitters (by plate appearances) and relievers (by appearances) in its recent games."""
    hitters, relievers = {}, {}
    starters_of = {}
    g = T['game']; h = T['half']; b = T['batter']; p = T['pitcher']; ab = T['ab']; pn = T['pitch_no']
    first = pn == 0
    for tid, pks in game_pks_by_team.items():
        hc = {}; pc = {}
        for pk in pks:
            side = team_games[(pk, tid)]            # 'away' or 'home'
            bat_half = 0 if side == 'away' else 1; fld_half = 1 - bat_half
            m = (g == pk) & (h == bat_half) & first
            for x in b[m]:
                hc[int(x)] = hc.get(int(x), 0) + 1
            m2 = (g == pk) & (h == fld_half)
            if m2.any():
                order = np.argsort(ab[m2] * 100 + pn[m2], kind='stable'); ps = p[m2][order]
                starters_of[(pk, tid)] = int(ps[0])
                for x in (np.unique(ps[1:]) if len(ps) > 1 else []):
                    if int(x) != int(ps[0]):
                        pc[int(x)] = pc.get(int(x), 0) + 1
        hitters[tid] = sorted(hc, key=lambda k: -hc[k])[:13]
        relievers[tid] = sorted(pc, key=lambda k: -pc[k])[:6]
    return hitters, relievers


def names_for(ids: set) -> dict:
    out = {}
    ids = sorted(int(i) for i in ids if i)
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        try:
            doc = mlb('https://statsapi.mlb.com/api/v1/people?personIds=' + ','.join(str(v) for v in chunk))
            for pers in doc.get('people', []):
                out[int(pers['id'])] = pers.get('fullName') or str(pers['id'])
        except Exception:
            pass
    return out


def build_report(fit: Fitted, T_all: dict, day: str, asof: str, stage, max_relievers=4) -> dict:
    """The date's report: a summary document and one document per game (each with the players it needs)."""
    games = schedule(day)
    rep = {'schema': SCHEMA, 'date': day, 'asof': asof, 'built_at': datetime.now(timezone.utc).isoformat(), 'training_pitches': fit.n_train,
           'grid': {'side_ft': GU.tolist(), 'height_ft': GZ.tolist()}, 'games': {}, 'players': {}}
    if not games:
        return rep
    # recent games per team (last 14 days before the report date) from the public schedule
    d0 = date.fromisoformat(day)
    sched = mlb(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={(d0 - timedelta(days=16)).isoformat()}&endDate={(d0 - timedelta(days=1)).isoformat()}&gameType=R,F,D,L,W')
    by_team = {}; team_games = {}
    for d in sched.get('dates', []):
        for g in d.get('games', []):
            if (g.get('status') or {}).get('abstractGameState') != 'Final':
                continue
            pk = int(g['gamePk'])
            for side in ('away', 'home'):
                tid = int((((g.get('teams') or {}).get(side) or {}).get('team') or {}).get('id') or 0)
                if tid:
                    by_team.setdefault(tid, []).append(pk); team_games[(pk, tid)] = side
    rec_h, rec_p = recent_players(fit.T, by_team, team_games)
    stage('recent players')
    need_names = set()
    for g in games:
        pk = g['game_pk']
        entry = {'teams': g['teams'], 'status': g['status'], 'final': g['final'], 'start': g['start'], 'game_type': g['game_type'], 'sides': {}}
        box = None
        if g['final']:
            try:
                box = boxscore_players(pk)
            except Exception:
                box = None
        for bat_side in ('away', 'home'):
            fld_side = 'home' if bat_side == 'away' else 'away'
            tid = g['teams'][bat_side]['id']; fid = g['teams'][fld_side]['id']
            if box:
                order = box[bat_side]['battingOrder'] or box[bat_side]['batters']
                hitters = [x for x in order if x in fit.maps_s][:12]
                pitchers = [x for x in box[fld_side]['pitchers']]
                starter = pitchers[0] if pitchers else None
                pens = pitchers[1:]
            else:
                lineup = g['lineups'].get(bat_side) or []
                hitters = [x for x in (lineup or rec_h.get(tid, [])) if x in fit.maps_s][:12]
                starter = g['teams'][fld_side]['probable']
                pens = rec_p.get(fid, [])[:max_relievers]
            staff = ([starter] if starter else []) + [x for x in pens if x != starter]
            staff = [x for x in staff if x in fit.gp and len(fit.gp[x]) >= 150]
            side_entry = {'lineup_source': 'box score' if box else ('posted lineup' if g['lineups'].get(bat_side) else 'recent games'), 'hitters': hitters,
                          'pitchers': [{'id': x, 'role': 'starter' if x == starter else 'reliever'} for x in staff], 'pairs': {}}
            need_names.update(hitters); need_names.update(staff)
            for h in hitters:
                for p in staff:
                    pr = fit.pair(h, p)
                    if pr is None:
                        continue
                    if g['final']:
                        gr = fit.grade(pk, h, p, (pr.get('aim') or {}).get('cells'))
                        if gr:
                            pr['grade'] = gr
                    side_entry['pairs'][f'{h}:{p}'] = pr
            entry['sides'][bat_side] = side_entry
        # game grade summary, with the chases binned by what the map predicted for the pair (the forward calibration record)
        if g['final']:
            tot = {'pairs': 0, 'outside_pitches': 0, 'in_recommended': 0, 'usual_expected': 0.0, 'chases': 0, 'chases_expected_league': 0.0, 'chases_expected_map': 0.0,
                   'bins': {k: {'pairs': 0, 'outside': 0, 'chases': 0, 'league': 0.0, 'map': 0.0} for k in BINS}}
            for se in entry['sides'].values():
                for pr in se['pairs'].values():
                    gr = pr.get('grade')
                    if not gr or 'chases' not in gr:
                        continue
                    tot['pairs'] += 1; tot['outside_pitches'] += gr['outside']; tot['chases'] += gr['chases']
                    tot['chases_expected_league'] += gr['chases_expected_league']; tot['chases_expected_map'] += gr['chases_expected_map']
                    if 'in_recommended_cells' in gr and 'usual_share_in_cells' in gr:
                        tot['in_recommended'] += gr['in_recommended_cells']; tot['usual_expected'] += gr['usual_share_in_cells'] * gr['outside']
                    bn = tot['bins'][bin_of(pr['chase_points'])]
                    bn['pairs'] += 1; bn['outside'] += gr['outside']; bn['chases'] += gr['chases']; bn['league'] += gr['chases_expected_league']; bn['map'] += gr['chases_expected_map']
            for k in ('usual_expected', 'chases_expected_league', 'chases_expected_map'):
                tot[k] = round(tot[k], 2)
            for bn in tot['bins'].values():
                bn['league'] = round(bn['league'], 2); bn['map'] = round(bn['map'], 2)
            entry['grade'] = tot
        entry['players'] = {}
        ids = set()
        for se in entry['sides'].values():
            ids.update(se['hitters']); ids.update(x['id'] for x in se['pitchers'])
        for pid in sorted(ids):
            card = fit.hitter_card(pid) if pid in fit.maps_s else None
            pc = fit.pitcher_card(pid)
            if card or pc:
                entry['players'][str(pid)] = {'hitter': card, 'pitcher': pc}
        rep['games'][str(pk)] = entry
        stage(f'game {pk}')
    names = names_for(need_names)
    for entry in rep['games'].values():
        for pid in list(entry['players']):
            entry['players'][pid]['name'] = names.get(int(pid), pid)
        for side in ('away', 'home'):
            entry['teams'][side]['probable_name'] = names.get(entry['teams'][side].get('probable') or -1)
    return rep


def fit_n(fit):
    return fit.n_train


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']
    from cloud.security import unseal, key_bytes
    from brl_live.bookkeeping_season import STUDY_SCHEMA, study_path, study_purpose
    key = key_bytes(os.environ['BRL_PA_PACKAGE_KEY'])
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    params = json.loads((ROOT / 'tools' / 'report_params.json').read_text()) if (ROOT / 'tools' / 'report_params.json').exists() else {}
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York'))
    if os.environ.get('BRL_REPORT_DAILY'):        # the scheduled runs: yesterday (now graded) and today, each as of its own date
        params = {'publish': True, 'dates': [(now_et.date() - timedelta(days=1)).isoformat(), now_et.date().isoformat()]}
    dates = params.get('dates') or [params.get('date') or now_et.date().isoformat()]
    asof = params.get('asof')                      # one as-of date for every report in the run (a backfilled month); default: each report's own date
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    receipt = {'schema': 'brl.report-receipt.v1', 'run_id': run_id, 'params': params, 'started_at': datetime.now(timezone.utc).isoformat(), 'stages': [], 'reports': {}}
    t0 = time.time()

    def stage(name):
        receipt['stages'].append({'stage': name, 'at_seconds': round(time.time() - t0, 1)}); print(name, round(time.time() - t0), 's', flush=True)
    try:
        years = {int(d[:4]) for d in dates}
        seasons = sorted(years | {y - 1 for y in years} | set(int(s) for s in params.get('seasons', ())))
        tables = []
        for year in seasons:
            raw = D.read_blob(repo, token, study_path(int(year)), branch)
            if raw is None:
                continue
            doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(int(year)))))
            if doc.get('schema') != STUDY_SCHEMA:
                raise ValueError('study schema mismatch')
            tables.append(D.pitch_table(doc, int(year), ('R', 'F', 'D', 'L', 'W'))); del doc, raw
            stage(f'load {year}')
        T = D.concat(tables); del tables
        fits = {}
        for day in sorted(dates):
            a = asof or day
            if a not in fits:
                fits.clear()
                fits[a] = Fitted(T, date.fromisoformat(a).toordinal(), stage, int(params.get('min_pitches', 300)), int(params.get('min_swings', 200)))
            rep = build_report(fits[a], T, day, a, stage, int(params.get('max_relievers', 4)))
            total = 0
            summary = {k: v for k, v in rep.items() if k != 'games'}
            summary['games'] = {}
            for pk, entry in rep['games'].items():
                doc = dict(entry); doc.update({'schema': SCHEMA, 'date': day, 'asof': a, 'game_pk': int(pk), 'grid': rep['grid'], 'training_pitches': fit_n(fits[a]), 'league': fits[a].league})
                text = json.dumps(doc, separators=(',', ':')); total += len(text)
                if params.get('publish', True):
                    put(repo, token, f'public/reports/{day}/{pk}.json', text, branch, f'BRL report {day} game {pk} (as of {a})')
                summary['games'][pk] = {'teams': {sd: {'id': entry['teams'][sd]['id'], 'name': entry['teams'][sd]['name'], 'abbr': entry['teams'][sd]['abbr']} for sd in ('away', 'home')},
                                        'status': entry['status'], 'final': entry['final'], 'start': entry['start'], 'grade': entry.get('grade'),
                                        'pairs': sum(len(se['pairs']) for se in entry['sides'].values())}
            if params.get('publish', True):
                put(repo, token, f'public/reports/{day}/index.json', json.dumps(summary, separators=(',', ':')), branch, f'BRL report {day} summary (as of {a})')
            receipt['reports'][day] = {'games': len(rep['games']), 'bytes': total, 'asof': a, 'graded': sum(1 for g in rep['games'].values() if g.get('grade'))}
            stage(f'report {day}')
        if params.get('publish', True):
            stage('index')
            idx, days = rebuild_index(repo, token, branch)
            put(repo, token, 'public/reports/index.json', json.dumps(idx, separators=(',', ':'), sort_keys=True), branch, 'BRL reports index')
            rec = record_from(days)
            put(repo, token, 'public/reports/record.json', json.dumps(rec, separators=(',', ':')), branch, 'BRL reports record')
            receipt['record'] = {k: rec[k] for k in ('games', 'dates', 'first_date', 'last_date', 'chases', 'chases_expected_league', 'chases_expected_map', 'in_recommended', 'usual_expected', 'outside_pitches')}
        receipt['status'] = 'completed'
    except Exception as exc:
        import traceback
        receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        receipt['trace'] = traceback.format_exc()[-2500:]
        print(receipt['trace'], flush=True)
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat(); receipt['seconds'] = round(time.time() - t0, 1)
    put(repo, token, f'research/report-{run_id}.json', json.dumps(receipt, indent=1), branch, f'BRL report receipt {run_id}')
    if receipt['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
