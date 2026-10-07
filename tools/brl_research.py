"""Plate-appearance model research inside Actions, where the sealed data and the key live.

Experiment: does pitch quality and contact quality (from the sealed pitch-physics seasons)
improve the locked plate-appearance model? The locked features are rebuilt exactly as the
engine builds them; the candidate adds time-valid player features computed from prior dates
only (a batter's exit velocity, hard-hit and barrel rates, whiff, chase and swing rates; a
pitcher's fastball velocity, its change since his previous outing, spin, movement, whiff,
chase, zone and called-strike-plus-whiff rates, contact quality and ground-ball share
allowed). Both models are fitted on 2023-2024, tuned and calibrated on 2025 exactly like the
locked run, and scored on the 2025 blend dates and on 2026, paired with a game-clustered
bootstrap. Only metrics and coefficients are written: research/<experiment>-<run>.json on
the ledger branch. No player rows leave the runner.
"""
from __future__ import annotations
import base64, gzip, importlib.util, json, os, sys, time, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))
from cloud.security import unseal, key_bytes  # noqa: E402
from brl_live.bookkeeping_season import STUDY_SCHEMA, study_path, study_purpose  # noqa: E402

LABELS = ('BIP_OUT', 'K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_REACH')
SWING = {'S', 'W', 'M', 'Q', 'F', 'L', 'R', 'O', 'T', 'X', 'D', 'E', 'J', 'Z'}
WHIFF = {'S', 'W', 'M', 'Q'}
CALLED = {'C'}
FASTBALL = {'FF', 'SI', 'FA', 'FT'}
K_RATE, K_MEAN, K_BIP = 150.0, 150.0, 60.0           # pseudo-counts: pitches / pitches / balls in play
EXTRA = ['b_ev', 'b_la', 'b_hard', 'b_barrel', 'b_whiff', 'b_chase', 'b_swing', 'b_zcontact', 'b_bip_n',
         'p_velo', 'p_velo_delta', 'p_spin', 'p_ivb', 'p_hb', 'p_whiff', 'p_chase', 'p_zone', 'p_csw', 'p_ev', 'p_hard', 'p_gb', 'p_pitch_n']


def api(url, token, method='GET', payload=None):
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-research/1.0'}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode(); headers['Content-Type'] = 'application/json'
    with urlopen(Request(url, headers=headers, data=body, method=method), timeout=60) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def read_blob(repo, token, path, branch):
    try:
        value = api(f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}', token)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if value.get('encoding') == 'base64' and value.get('content'):
        return base64.b64decode(''.join(value['content'].split()))
    blob = api(f'https://api.github.com/repos/{repo}/git/blobs/{value["sha"]}', token)
    return base64.b64decode(''.join(blob['content'].split()))


def put_text(repo, token, path, text, branch, message):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    payload = {'message': message, 'content': base64.b64encode(text.encode()).decode(), 'branch': branch}
    try:
        payload['sha'] = api(url + '?ref=' + branch, token)['sha']
    except HTTPError as exc:
        if exc.code != 404:
            raise
    api(url, token, 'PUT', payload)


def entrypoint_module():
    spec = importlib.util.spec_from_file_location('brl_entrypoint', ROOT / 'brl_engine' / 'entrypoint.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# Per-pitch aggregates kept for every plate appearance (sums over its pitches).
PITCH_SUMS = ['n', 'sw', 'wh', 'oz', 'ch', 'iz', 'izs', 'izc', 'cs', 'fb', 'velo', 'spin', 'ivb', 'hb']
PS = {name: i for i, name in enumerate(PITCH_SUMS)}
BIP_SUMS = ['bip', 'ev', 'la', 'hard', 'barrel', 'gb']
BS = {name: i for i, name in enumerate(BIP_SUMS)}


def per_pa_physics(doc: dict) -> pd.DataFrame:
    """One row per plate appearance with the pitch and contact aggregates the features need."""
    rows = []
    for pk, game in doc['games'].items():
        for r in game['rows']:
            a = np.zeros(len(PITCH_SUMS))
            for p in r['pitches']:
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
            hit = r.get('hit')
            ev = la = np.nan
            if hit and hit[0] is not None and hit[1] is not None and r['o'] not in ('K', 'BB_HBP'):
                ev, la = float(hit[0]), float(hit[1])
            rows.append((int(pk), int(r['i']) + 1, int(r['p']), -1 if r.get('b') is None else int(r['b']), *a.tolist(), ev, la))
    return pd.DataFrame(rows, columns=['game_pk', 'at_bat_number', 'pitcher', 'batter'] + PITCH_SUMS + ['ev', 'la'])


def shrunk(s, num, den, league, k):
    """(sum + k * league ratio) / (count + k): a rate or a mean pulled toward the league value."""
    return (s[num] + k * (league[num] / max(league[den], 1e-9))) / (s[den] + k)


RUN_VALUE = {'BIP_OUT': 0.0, 'K': 0.0, 'BB_HBP': 0.69, '1B': 0.88, '2B_3B': 1.4, 'HR': 2.03, 'OTHER_REACH': 0.6}
XV_K = 40.0                  # pseudo balls in play per (exit velocity, launch angle) cell
DEFAULT_PARAMS = {'k_rate': 150.0, 'k_bip': 60.0, 'k_velo': 100.0, 'xvalue': False, 'recent_days': 0, 'k_recent_pitch': 100.0, 'k_recent_bip': 40.0}
XVALUE_COLS = ['b_xv', 'p_xv']
RECENT_COLS = ['p_velo_rec', 'p_whiff_rec', 'p_csw_rec', 'b_ev_rec', 'b_whiff_rec', 'b_hard_rec']


def columns_for(params: dict) -> list[str]:
    cols = list(EXTRA)
    if params.get('xvalue'):
        cols += XVALUE_COLS
    if params.get('recent_days'):
        cols += RECENT_COLS
    return cols


def _cell(ev_value: float, la_value: float) -> tuple[int, int]:
    return int(min(max((ev_value - 40.0) // 5, 0), 15)), int(min(max((la_value + 90.0) // 10, 0), 17))


def build_extras(pa: pd.DataFrame, physics: pd.DataFrame, params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Time-valid batter and pitcher physics features aligned to the PA frame (sorted like the engine's builder)."""
    P = dict(DEFAULT_PARAMS, **(params or {}))
    k_rate, k_bip, k_velo = float(P['k_rate']), float(P['k_bip']), float(P['k_velo'])
    want_xv, recent_days = bool(P['xvalue']), int(P['recent_days'] or 0)
    k_rp, k_rb = float(P['k_recent_pitch']), float(P['k_recent_bip'])
    cols = columns_for(P)
    ordered = pa.sort_values(['date_key', 'game_pk', 'at_bat_number'], kind='mergesort').reset_index(drop=True)
    ph = physics.drop_duplicates(['game_pk', 'at_bat_number']).set_index(['game_pk', 'at_bat_number'])
    joined = ordered[['game_pk', 'at_bat_number', 'batter', 'pitcher']].join(ph, on=['game_pk', 'at_bat_number'], rsuffix='_ph')
    has = joined['n'].notna().to_numpy()
    id_match = has & (joined['batter'].to_numpy() == joined['batter_ph'].to_numpy()) & (joined['pitcher'].to_numpy() == joined['pitcher_ph'].to_numpy())
    audit = {'pa_rows': int(len(ordered)), 'physics_rows': int(len(ph)), 'joined': int(has.sum()), 'ids_match': int(id_match.sum()), 'params': P}
    M = joined[PITCH_SUMS].to_numpy(float, copy=True); M[~id_match] = np.nan
    ev = joined['ev'].to_numpy(float, copy=True); la = joined['la'].to_numpy(float, copy=True); ev[~id_match] = np.nan; la[~id_match] = np.nan
    batters = ordered['batter'].to_numpy(int); pitchers = ordered['pitcher'].to_numpy(int); dates = ordered['date_key'].astype(str).to_numpy()
    rv = ordered['outcome'].map(RUN_VALUE).to_numpy(float)
    day_ord = pd.to_datetime(ordered['date_key']).map(pd.Timestamp.toordinal).to_numpy()
    out = np.full((len(ordered), len(cols)), np.nan, dtype=np.float32)
    X = {name: i for i, name in enumerate(cols)}
    zero_p, zero_b = np.zeros(len(PITCH_SUMS)), np.zeros(len(BIP_SUMS))
    bp, pp, Lp = defaultdict(lambda: np.zeros(len(PITCH_SUMS))), defaultdict(lambda: np.zeros(len(PITCH_SUMS))), np.zeros(len(PITCH_SUMS))
    bb, pb, Lb = defaultdict(lambda: np.zeros(len(BIP_SUMS))), defaultdict(lambda: np.zeros(len(BIP_SUMS))), np.zeros(len(BIP_SUMS))
    last_velo, cur_velo = {}, {}      # pitcher -> (fastballs, velocity sum) of his most recent prior outing / of the date being scored
    # expected run value by (exit velocity, launch angle) cell, learned online; per-player sums of the cell values of their batted balls
    xv_sum, xv_n = np.zeros((16, 18)), np.zeros((16, 18)); xv_league = [0.0, 0.0]
    bxv, pxv = defaultdict(lambda: np.zeros(2)), defaultdict(lambda: np.zeros(2))
    # recent windows: per player, list of (day ordinal, pitch sums, bip sums) per date
    rec_p, rec_b = defaultdict(list), defaultdict(list)
    def recent(entries, today):
        sp_, sb_ = zero_p.copy(), zero_b.copy()
        keep = []
        for e in entries:
            if e[0] >= today - recent_days:
                keep.append(e); sp_ += e[1]; sb_ += e[2]
        entries[:] = keep
        return sp_, sb_
    def dev(recent_sum, recent_den, own_rate, k):
        return (recent_sum + k * own_rate) / (recent_den + k) - own_rate
    i, n = 0, len(ordered)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        today = day_ord[i]
        for r in range(i, j):
            b, p = batters[r], pitchers[r]
            sb = bp.get(b, zero_p); sp = pp.get(p, zero_p); cb = bb.get(b, zero_b); cp = pb.get(p, zero_b)
            out[r, X['b_ev']] = shrunk(cb, BS['ev'], BS['bip'], Lb, k_bip)
            out[r, X['b_la']] = shrunk(cb, BS['la'], BS['bip'], Lb, k_bip)
            out[r, X['b_hard']] = shrunk(cb, BS['hard'], BS['bip'], Lb, k_bip)
            out[r, X['b_barrel']] = shrunk(cb, BS['barrel'], BS['bip'], Lb, k_bip)
            out[r, X['b_whiff']] = shrunk(sb, PS['wh'], PS['sw'], Lp, k_rate)
            out[r, X['b_chase']] = shrunk(sb, PS['ch'], PS['oz'], Lp, k_rate)
            out[r, X['b_swing']] = shrunk(sb, PS['sw'], PS['n'], Lp, k_rate)
            out[r, X['b_zcontact']] = shrunk(sb, PS['izc'], PS['izs'], Lp, k_rate)
            out[r, X['b_bip_n']] = cb[BS['bip']]
            own_velo = shrunk(sp, PS['velo'], PS['fb'], Lp, k_velo)
            out[r, X['p_velo']] = own_velo
            out[r, X['p_spin']] = shrunk(sp, PS['spin'], PS['fb'], Lp, k_velo)
            out[r, X['p_ivb']] = shrunk(sp, PS['ivb'], PS['fb'], Lp, k_velo)
            out[r, X['p_hb']] = shrunk(sp, PS['hb'], PS['fb'], Lp, k_velo)
            lv = last_velo.get(p)
            out[r, X['p_velo_delta']] = (lv[1] / lv[0] - own_velo) if (lv is not None and lv[0] >= 5 and sp[PS['fb']] >= 100) else 0.0
            own_whiff = shrunk(sp, PS['wh'], PS['sw'], Lp, k_rate)
            own_csw = (sp[PS['cs']] + sp[PS['wh']] + k_rate * ((Lp[PS['cs']] + Lp[PS['wh']]) / max(Lp[PS['n']], 1e-9))) / (sp[PS['n']] + k_rate)
            out[r, X['p_whiff']] = own_whiff
            out[r, X['p_chase']] = shrunk(sp, PS['ch'], PS['oz'], Lp, k_rate)
            out[r, X['p_zone']] = shrunk(sp, PS['iz'], PS['n'], Lp, k_rate)
            out[r, X['p_csw']] = own_csw
            out[r, X['p_ev']] = shrunk(cp, BS['ev'], BS['bip'], Lb, k_bip)
            out[r, X['p_hard']] = shrunk(cp, BS['hard'], BS['bip'], Lb, k_bip)
            out[r, X['p_gb']] = shrunk(cp, BS['gb'], BS['bip'], Lb, k_bip)
            out[r, X['p_pitch_n']] = sp[PS['n']]
            if want_xv:
                league_xv = xv_league[0] / max(xv_league[1], 1e-9)
                bx = bxv.get(b); px = pxv.get(p)
                out[r, X['b_xv']] = ((bx[0] if bx is not None else 0.0) + k_bip * league_xv) / ((bx[1] if bx is not None else 0.0) + k_bip)
                out[r, X['p_xv']] = ((px[0] if px is not None else 0.0) + k_bip * league_xv) / ((px[1] if px is not None else 0.0) + k_bip)
            if recent_days:
                rp, rpb = recent(rec_p[p], today) if p in rec_p else (zero_p, zero_b)
                rb, rbb = recent(rec_b[b], today) if b in rec_b else (zero_p, zero_b)
                out[r, X['p_velo_rec']] = dev(rp[PS['velo']], rp[PS['fb']], own_velo, k_rp)
                out[r, X['p_whiff_rec']] = dev(rp[PS['wh']], rp[PS['sw']], own_whiff, k_rp)
                out[r, X['p_csw_rec']] = dev(rp[PS['cs']] + rp[PS['wh']], rp[PS['n']], own_csw, k_rp)
                out[r, X['b_ev_rec']] = dev(rbb[BS['ev']], rbb[BS['bip']], out[r, X['b_ev']], k_rb)
                out[r, X['b_whiff_rec']] = dev(rb[PS['wh']], rb[PS['sw']], out[r, X['b_whiff']], k_rp)
                out[r, X['b_hard_rec']] = dev(rbb[BS['hard']], rbb[BS['bip']], out[r, X['b_hard']], k_rb)
        day_p, day_b = {}, {}
        for r in range(i, j):
            if np.isnan(M[r, 0]):
                continue
            b, p = batters[r], pitchers[r]
            bp[b] += M[r]; pp[p] += M[r]; Lp += M[r]
            if M[r, PS['fb']] > 0:
                cv = cur_velo.get(p, (0.0, 0.0)); cur_velo[p] = (cv[0] + M[r, PS['fb']], cv[1] + M[r, PS['velo']])
            v = None
            if not np.isnan(ev[r]):
                v = np.array([1.0, ev[r], la[r], float(ev[r] >= 95.0), float(ev[r] >= 98.0 and 8.0 <= la[r] <= 40.0), float(la[r] < 10.0)])
                bb[b] += v; pb[p] += v; Lb += v
                if want_xv:
                    ci, cj = _cell(ev[r], la[r])
                    league_xv = xv_league[0] / max(xv_league[1], 1e-9)
                    value = (xv_sum[ci, cj] + XV_K * league_xv) / (xv_n[ci, cj] + XV_K)
                    bxv[b] += (value, 1.0); pxv[p] += (value, 1.0)
                    xv_sum[ci, cj] += rv[r]; xv_n[ci, cj] += 1.0; xv_league[0] += rv[r]; xv_league[1] += 1.0
            if recent_days:
                for table, key in ((day_p, p), (day_b, b)):
                    e = table.get(key)
                    if e is None:
                        e = table[key] = [today, zero_p.copy(), zero_b.copy()]
                    e[1] += M[r]
                    if v is not None:
                        e[2] += v
        if recent_days:
            for key, e in day_p.items():
                rec_p[key].append(e)
            for key, e in day_b.items():
                rec_b[key].append(e)
        last_velo.update(cur_velo); cur_velo = {}
        i = j
    extras = pd.DataFrame(out, columns=cols)
    audit['league_fastball_velocity'] = float(Lp[PS['velo']] / max(Lp[PS['fb']], 1)); audit['league_exit_velocity'] = float(Lb[BS['ev']] / max(Lb[BS['bip']], 1))
    audit['league_hard_hit_rate'] = float(Lb[BS['hard']] / max(Lb[BS['bip']], 1)); audit['league_whiff_per_swing'] = float(Lp[PS['wh']] / max(Lp[PS['sw']], 1))
    audit['league_ground_ball_share'] = float(Lb[BS['gb']] / max(Lb[BS['bip']], 1))
    return extras, audit


# Experiment sets: each variant is a parameter dict for build_extras; the locked model is the shared reference.
SETS = {
    'physics': {'physics': {}},
    'physics2': {'base': {}, 'k_low': {'k_rate': 60.0, 'k_bip': 30.0}, 'k_high': {'k_rate': 300.0, 'k_bip': 120.0},
                 'xvalue': {'xvalue': True}, 'recent30': {'recent_days': 30}, 'xvalue_recent30': {'xvalue': True, 'recent_days': 30}},
}


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key_hex = os.environ['BRL_PA_PACKAGE_KEY']; key = key_bytes(key_hex)
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data')
    experiment = (os.environ.get('BRL_EXPERIMENT') or 'physics').strip()
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    work = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'brl-research'
    receipt = {'schema': 'brl.research-receipt.v1', 'experiment': experiment, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat(), 'stages': []}
    t0 = time.time()
    def stage(name):
        receipt['stages'].append({'stage': name, 'at_seconds': round(time.time() - t0, 1)}); print(name, round(time.time() - t0), 's', flush=True)
    try:
        stage('restore data package')
        ep = entrypoint_module()
        manifest = json.loads((ROOT / 'brl_engine' / 'data_package.json').read_text())
        data_root = work / 'data'
        if data_root.exists():
            import shutil; shutil.rmtree(data_root)
        data_root.mkdir(parents=True)
        ep.restore_package(repo, token, key_hex, manifest, data_root)
        pa_path = next(data_root.rglob('plate_appearances.csv.gz'))
        from research_lab.pa_model.config import PAConfig
        from research_lab.pa_model.features import build_time_valid_features
        from research_lab.pa_model.model import fit_frozen_model, validation_partitions
        from research_lab.pa_model.evaluation import probability_metrics, clustered_log_loss_difference_ci
        config = PAConfig(train_years=(2023, 2024), validation_years=(2025,), test_years=(2026,), evaluation_mode='locked_final')
        stage('load plate appearances')
        pa = pd.read_csv(pa_path, low_memory=False)
        pa['date_key'] = pa['date_key'].astype(str)
        receipt['pa_rows'] = int(len(pa)); receipt['seasons'] = sorted(int(s) for s in pa['season'].unique())
        stage('load pitch physics seasons')
        physics = []; seasons_loaded = []
        for year in receipt['seasons']:
            raw = read_blob(repo, token, study_path(year), branch)
            if raw is None:
                continue
            doc = json.loads(gzip.decompress(unseal(raw, key, study_purpose(year))))
            if doc.get('schema') != STUDY_SCHEMA:
                raise ValueError('study schema mismatch for ' + str(year))
            physics.append(per_pa_physics(doc)); seasons_loaded.append(year); del doc
        receipt['physics_seasons'] = seasons_loaded
        if not physics:
            raise ValueError('No pitch-physics seasons are sealed yet')
        physics = pd.concat(physics, ignore_index=True)
        stage('build locked features')
        features, locked_columns = build_time_valid_features(pa, config)
        label_to_index = {label: i for i, label in enumerate(LABELS)}
        y = features['outcome'].map(label_to_index).to_numpy(int)
        partitions, _ = validation_partitions(features, config)
        test_mask = features['season'].isin(config.test_years).to_numpy(); blend_mask = partitions['blend']
        parts = (('validation_blend_2025', blend_mask), ('test_2026', test_mask))
        results = {}
        probs = {}

        def fit_and_score(name, cols, extra_cols):
            fitted, tuning = fit_frozen_model(features, cols, config)
            out = {'feature_count': len(cols), 'best_regularization_c': tuning['best_regularization_c'], 'best_tune_log_loss': tuning['best_tune_log_loss'],
                   'calibration': {k: tuning['calibration'][k] for k in ('temperature', 'pre_calibration_log_loss', 'post_calibration_log_loss')}}
            probs[name] = {}
            for part, mask in parts:
                p = fitted.predict_proba(features.loc[mask]); probs[name][part] = p
                out[part] = probability_metrics(y[mask], p).to_dict()
                months = pd.to_datetime(features.loc[mask, 'date_key']).dt.month.to_numpy()
                out[part + '_by_month'] = {int(m): float(probability_metrics(y[mask][months == m], p[months == m]).log_loss) for m in np.unique(months)}
            if extra_cols:
                coef = fitted.estimator.named_steps['model'].coef_
                idx = [cols.index(c) for c in extra_cols]
                out['extra_coefficients_by_class'] = {LABELS[k]: {c: round(float(coef[k, i]), 4) for c, i in zip(extra_cols, idx)} for k in range(coef.shape[0])}
            return out

        stage('fit locked')
        results['locked'] = fit_and_score('locked', list(locked_columns), [])
        variants = SETS.get(experiment) or {experiment: {}}
        for vname, params in variants.items():
            stage('build physics features ' + vname)
            extras, audit = build_extras(pa, physics, params)
            results.setdefault('physics_join', {})[vname] = audit
            cols = list(extras.columns)
            for c in cols:
                features[c] = extras[c].to_numpy()
            stage('fit ' + vname)
            results[vname] = fit_and_score(vname, list(locked_columns) + cols, cols)
            for part, mask in parts:
                games = features.loc[mask, 'game_pk'].to_numpy()
                results[vname][part + '_minus_locked'] = clustered_log_loss_difference_ci(y[mask], probs[vname][part], probs['locked'][part], games, replicates=600)
            features.drop(columns=cols, inplace=True)
            del extras
        receipt['results'] = results
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        frames = traceback.extract_tb(exc.__traceback__)
        receipt['where'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in frames[-6:]]
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat(); receipt['seconds'] = round(time.time() - t0, 1)
    text = json.dumps(receipt, indent=1, default=float)
    for secret in (key_hex, token):
        text = text.replace(secret, '[redacted]')
    put_text(repo, token, f'research/{experiment}-{run_id}.json', text, branch, 'BRL: research ' + experiment)
    print(json.dumps({k: receipt.get(k) for k in ('status', 'error', 'seconds')}))
    if receipt['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
