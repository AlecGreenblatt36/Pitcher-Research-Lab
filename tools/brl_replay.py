"""Frozen season replay inside Actions: simulate every game of a season with a chosen PA model,
score the simulator, the team model and the production blend, and compare with a reference replay.

Settings come from tools/replay_params.json on the trigger branch:
  model        'locked-pa-2026-v1' (data package) or a fitted model name with a manifest in brl_engine/models/
  season       2026 (default)
  n_sims       worlds per game (default 200; 0 with real_pa_check runs only the check on real plate appearances)
  date_from    first date to replay (default the season start)
  step         keep every step-th game (default 1)
  reference    path on the ledger branch of an earlier replay's per-game file to pair against (optional)
  offsets      apply the production context offsets (default true), or a path in the repository of another table in that format
  environment  path in the repository of a run-environment table (brl_live/environment.py) to apply per game
  conditions   path on the ledger branch of the game conditions file (tools/brl_game_conditions.py), needed with environment
  team_offsets {'rows': ledger-branch path of team residual rows (research/team-resid-*.json.gz), 'k', 'half_life', 'sides', 'center'}:
               team offsets as they stood at the start of each date (brl_live/team_offsets.py)
  age_layer    {'receipt': ledger-branch path of a stage2 research receipt, 'fit': 'fit_2025' | 'fit_2026' | 'fit_all', 'variant': 'v2'}:
               the aging and recency layer (brl_live/age_layer.py)
  steals       {'per_pa': attempt scale}: runner speeds and stolen bases (brl_live/running.py), statistics through the
               season before each replayed date
  hitter_lines record each lineup hitter's simulated chances and his actual line (brl_replay.harness.replay_dates)
  transitions  path in the repository of a base-running kernel (tools/brl_transition_kernel.py) used after each outcome
  running_events path in the repository of a running-plays table (tools/brl_running_events.py): wild pitches, passed
               balls, balks, pickoffs and the rest between plate appearances
  reliever_choice path in the repository of a fitted reliever choice (tools/brl_reliever_choice.py): which reliever
               enters at each pitching change
  relief_exit  path in the repository of fitted reliever exits (tools/brl_relief_exit.py): when a reliever comes out
  base_state   path in the repository of base-state offsets (tools/brl_base_state.py): the stack shaped by bases and outs
  leash        path in the repository of a starter-leash table (tools/brl_leash.py): each starter's expected batters
               faced moved for short rest, a relief outing before, a return from a layoff and the month
  physics_years 'recent' (default: the previous and the replayed season, as the live runtime loads), 'all' (every sealed
               season from 2023, as the model was fitted) or a list of years
  physics_join  keep only physics rows whose game, plate appearance, batter and pitcher are in the plate-appearance
               history (as the fit's chronological builder does; drops postseason rows)
Outputs on the ledger branch: research/replay-<tag>-<run>.jsonl.gz (one record per game: win counts,
run histograms, starter outs, final score) and research/replay-<tag>-<run>.json (scores, paired
comparison, timing). Per-game model outputs and final scores are not private data; the plate
appearances never leave the runner.
"""
from __future__ import annotations
import base64, gzip, hashlib, importlib.util, io, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'brl_engine' / 'runtime'))


def api(url, token, method='GET', payload=None):
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-replay/1.0'}
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


def put_bytes(repo, token, path, raw, branch, message):
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    for attempt in range(6):
        payload = {'message': message, 'content': base64.b64encode(raw).decode(), 'branch': branch}
        try:
            payload['sha'] = api(url + '?ref=' + branch, token)['sha']
        except HTTPError as exc:
            if exc.code != 404:
                raise
        try:
            return api(url, token, 'PUT', payload)
        except HTTPError as exc:
            if exc.code != 409 or attempt == 5:
                raise
            time.sleep(2 + 3 * attempt)


def entrypoint_module():
    spec = importlib.util.spec_from_file_location('brl_entrypoint', ROOT / 'brl_engine' / 'entrypoint.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


_SHARED: dict = {}


def _worker(dates: list) -> list:
    from brl_replay.harness import replay_dates
    s = _SHARED
    return replay_dates(s['h'], s['app'], s['games'], dates, model_path=s['model_path'], model_sha256=s['model_sha256'], history_path=s['history_path'],
                        hazard_path=s['hazard_path'], n_sims=s['n_sims'], physics_table=s['physics_table'], offsets=s['offsets'], rest=s.get('rest', False),
                        environment=s.get('environment'), team_offsets=s.get('team_offsets'), age_layer=s.get('age_layer'), steals=s.get('steals'),
                        win_states=s.get('win_states') or False, starter_lines=bool(s.get('starter_lines')), role_offsets=s.get('role_offsets'), real_pa_check=bool(s.get('real_pa_check')),
                        transitions=s.get('transitions'), hitter_lines=bool(s.get('hitter_lines')), running_events=s.get('running_events'),
                        reliever_choice=s.get('reliever_choice'), leash=s.get('leash'), base_state=s.get('base_state'), relief_exit=s.get('relief_exit'),
                        log=lambda m: print(m, flush=True))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score(frame: pd.DataFrame) -> dict:
    y = frame['y'].to_numpy(float)
    out = {}
    for name in ('p_sim', 'p_team', 'p_blend', 'p_coin'):
        p = frame[name].to_numpy(float)
        out[name] = {'brier': float(np.mean((p - y) ** 2)), 'log_loss': float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))),
                     'better_than_coin_pct': float(100 * (0.25 - np.mean((p - y) ** 2)) / 0.25)}
    out['n_games'] = int(len(frame))
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, column: str, replicates: int = 4000, seed: int = 1) -> dict:
    """Brier(a) - Brier(b) on the common games with a paired bootstrap over dates."""
    m = a.merge(b[['game_pk', column]], on='game_pk', suffixes=('', '_ref'))
    da = (m[column] - m['y']) ** 2; db = (m[column + '_ref'] - m['y']) ** 2
    diff = (da - db).to_numpy(); dates = m['date'].to_numpy()
    uniq = np.unique(dates); idx = {u: np.where(dates == u)[0] for u in uniq}
    rng = np.random.default_rng(seed); draws = np.empty(replicates)
    for i in range(replicates):
        pick = rng.choice(uniq, len(uniq), replace=True); ii = np.concatenate([idx[u] for u in pick]); draws[i] = diff[ii].mean()
    return {'n_common_games': int(len(m)), 'brier_difference': float(diff.mean()), 'ci_95_low': float(np.percentile(draws, 2.5)), 'ci_95_high': float(np.percentile(draws, 97.5)),
            'brier_candidate': float(da.mean()), 'brier_reference': float(db.mean())}


def main():
    repo = os.environ['GITHUB_REPOSITORY']; token = os.environ['GH_TOKEN']; key_hex = os.environ['BRL_PA_PACKAGE_KEY']
    branch = os.environ.get('BRL_LEDGER_BRANCH', 'brl-live-data'); run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    params = {}
    settings = ROOT / 'tools' / 'replay_params.json'
    if settings.exists():
        params = json.loads(settings.read_text())
    model = str(params.get('model') or 'locked-pa-2026-v1'); season = int(params.get('season') or 2026)
    n_sims = int(200 if params.get('n_sims') is None else params['n_sims']); date_from = str(params.get('date_from') or f'{season}-01-01'); step = int(params.get('step') or 1)
    reference = params.get('reference'); use_offsets = bool(params.get('offsets', True)); use_rest = bool(params.get('rest', False))
    tag = str(params.get('tag') or model)
    n_shards = max(1, int(params.get('shards') or 1)); shard = int(os.environ.get('BRL_REPLAY_SHARD') or 0)
    suffix = f'-s{shard}of{n_shards}' if n_shards > 1 else ''
    workers = int(params.get('workers') or max(1, (os.cpu_count() or 2)))
    work = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'brl-replay'
    receipt = {'schema': 'brl.replay-receipt.v1', 'tag': tag, 'model': model, 'season': season, 'n_sims': n_sims, 'date_from': date_from, 'step': step,
               'offsets': use_offsets, 'rest': use_rest, 'workers': workers, 'run_id': run_id, 'shard': [shard, n_shards],
               'started_at': datetime.now(timezone.utc).isoformat(), 'stages': []}
    t0 = time.time()
    def stage(label):
        receipt['stages'].append({'stage': label, 'at_seconds': round(time.time() - t0, 1)}); print(label, round(time.time() - t0), 's', flush=True)
    try:
        stage('restore data package')
        ep = entrypoint_module()
        manifest = json.loads((ROOT / 'brl_engine' / 'data_package.json').read_text())
        data_root = work / 'data'
        if data_root.exists():
            import shutil; shutil.rmtree(data_root)
        data_root.mkdir(parents=True)
        ep.restore_package(repo, token, key_hex, manifest, data_root)
        history_path = next(data_root.rglob('plate_appearances.csv.gz'))
        hazard_path = next(data_root.rglob('starter_hazard.joblib'))
        if model == 'locked-pa-2026-v1':
            model_path = next(data_root.rglob('pa_model.joblib'))
            from research_lab.game_sim.locked_pa_provider import EXPECTED_MODEL_SHA256 as model_sha256
            physics_table = None
        else:
            os.environ['BRL_DATA_ROOT'] = str(data_root)
            (ROOT / 'brl_engine' / 'model.json').write_text(json.dumps({'model': model, 'manifest': f'brl_engine/models/{model}.json'}))
            selected = ep.select_model(repo, token, key_hex, data_root)
            model_path = Path(os.environ['BRL_MODEL_PATH']); model_sha256 = os.environ['BRL_MODEL_SHA256']
            receipt['model_selected'] = selected
            stage('load physics tables')
            from brl_live.bookkeeping_season import physics_path, physics_purpose
            from cloud.security import unseal, key_bytes
            tables = []
            py = params.get('physics_years') or 'recent'
            years = list(range(2023, season + 1)) if py == 'all' else ([int(y) for y in py] if isinstance(py, list) else [season - 1, season])
            receipt['physics_years'] = years
            for year in years:
                raw = read_blob(repo, token, physics_path(year), branch)
                if raw is None:
                    raise ValueError(f'physics table for {year} is not sealed')
                tables.append(pd.read_csv(io.BytesIO(gzip.decompress(unseal(raw, key_bytes(key_hex), physics_purpose(year))))))
            physics_table = pd.concat(tables, ignore_index=True)
            physics_table['date_key'] = physics_table['date_key'].astype(str).str[:10]
            receipt['physics_rows'] = int(len(physics_table))
        stage('load history and reconstruct games')
        from brl_replay.games import load_history, reconstruct
        from brl_replay.harness import appearances
        h = load_history(history_path)
        if physics_table is not None and params.get('physics_join'):
            keys = h[['game_pk', 'at_bat_number', 'batter', 'pitcher']].drop_duplicates(['game_pk', 'at_bat_number']).astype('int64')
            before = int(len(physics_table))
            physics_table = physics_table.drop_duplicates(['game_pk', 'at_bat_number'])
            physics_table = physics_table.astype({'game_pk': 'int64', 'at_bat_number': 'int64', 'batter': 'int64', 'pitcher': 'int64'})
            physics_table = physics_table.merge(keys, on=['game_pk', 'at_bat_number', 'batter', 'pitcher'], how='inner').reset_index(drop=True)
            receipt['physics_join'] = {'before': before, 'after': int(len(physics_table))}
        app = appearances(h)
        games = reconstruct(h)
        games['date'] = games['date'].astype(str)
        games = games[(games['season'] == season) & (~games['ambiguous']) & games['valid_lineups'] & (games['date'] >= date_from)].sort_values(['date', 'game_pk'])
        games = games.iloc[::step].reset_index(drop=True)
        if n_shards > 1:
            # Parallel jobs split the season by date (every n-th date), so each date's state is built once.
            keep = sorted(games['date'].unique())[shard::n_shards]
            games = games[games['date'].isin(keep)].reset_index(drop=True)
        receipt['games'] = int(len(games))
        offsets = None
        if use_offsets:
            from brl_live.provider_adjust import load_offsets
            offsets = load_offsets(ROOT / str(params['offsets'])) if isinstance(params.get('offsets'), str) else load_offsets()
            receipt['offsets_table'] = {'path': params['offsets'] if isinstance(params.get('offsets'), str) else 'brl_live/context_offsets.json',
                                        'estimated_through': offsets.get('estimated_through')}
        environment = None
        if params.get('environment'):
            from brl_live.environment import load_table, conditions, log_multipliers
            table = load_table(ROOT / str(params['environment']))
            raw = read_blob(repo, token, str(params['conditions']), branch)
            if raw is None:
                raise ValueError('conditions file missing on the ledger branch')
            rows = [json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l.strip()]
            by_pk = {int(r['game_pk']): r for r in rows}
            environment = {}
            for g in games.itertuples():
                r = by_pk.get(int(g.game_pk))
                if r is None:
                    continue
                c = conditions(venue=r.get('venue'), temp=r.get('temp_f'), condition=r.get('condition'), day_night=r.get('day_night'),
                               date=r.get('date'), wind_mph=r.get('wind_mph'), wind_dir=r.get('wind_dir'))
                environment[int(g.game_pk)] = log_multipliers(c, table)
            receipt['environment'] = {'table': table.get('name'), 'conditions': params['conditions'], 'games_with_conditions': len(environment)}
        team_offsets = None
        if params.get('team_offsets'):
            from brl_live.team_offsets import by_date
            spec = params['team_offsets']
            raw = read_blob(repo, token, str(spec['rows']), branch)
            if raw is None:
                raise ValueError('team residual rows missing on the ledger branch')
            doc = json.loads(gzip.decompress(raw))
            team_offsets = by_date(doc['rows'], sorted(games['date'].astype(str).unique()), k=float(spec.get('k', 4000.0)),
                                   half_life=spec.get('half_life'), sides=tuple(spec.get('sides', ('bat', 'fld'))), center=bool(spec.get('center')))
            sizes = [max((abs(v) for t in day.values() for s in t.values() for v in s), default=0.0) for day in team_offsets.values()]
            receipt['team_offsets'] = {**spec, 'dates': len(team_offsets), 'largest_offset': round(float(max(sizes, default=0.0)), 4)}
        age_layer = None
        if params.get('age_layer'):
            from brl_live.age_layer import layer_from_receipt
            spec = params['age_layer']
            raw = read_blob(repo, token, str(spec['receipt']), branch)
            if raw is None:
                raise ValueError('stage2 receipt missing on the ledger branch')
            age_layer = layer_from_receipt(json.loads(raw), fit=str(spec.get('fit', 'fit_all')), variant=str(spec.get('variant', 'v2')))
            receipt['age_layer'] = {**spec, 'columns': len(age_layer['columns'])}
        role_offsets = None
        if params.get('role_offsets'):
            from brl_live.role_offsets import by_date as role_by_date
            rspec = params['role_offsets']
            local = ROOT / str(rspec['rows'])
            rraw = local.read_bytes() if str(rspec['rows']).startswith('brl_replay/') and local.exists() else read_blob(repo, token, rspec['rows'], branch)
            if rraw is None:
                raise ValueError('role residual rows missing on the ledger branch')
            rdoc = json.loads(gzip.decompress(rraw) if rraw[:2] == b'\x1f\x8b' else rraw)
            role_offsets = role_by_date(rdoc['rows'], sorted(games['date'].astype(str).unique()), k=float(rspec.get('k', 2000.0)), half_life=rspec.get('half_life'))
            receipt['role_offsets'] = {**rspec, 'dates': len(role_offsets)}
        transitions = None
        if params.get('transitions'):
            from research_lab.game_sim.transitions import EmpiricalKernel
            kdoc = json.loads((ROOT / str(params['transitions'])).read_text())
            transitions = EmpiricalKernel(kdoc)
            receipt['transitions'] = {'path': params['transitions'], 'name': transitions.name, 'cells': len(transitions.cells)}
        running_events = None
        if params.get('running_events'):
            from research_lab.game_sim.running_events import RunningEvents
            running_events = RunningEvents(json.loads((ROOT / str(params['running_events'])).read_text()))
            receipt['running_events'] = {'path': params['running_events'], 'name': running_events.name, 'cells': len(running_events.cells)}
        reliever_choice = None
        if params.get('reliever_choice'):
            from research_lab.game_sim.reliever_choice import RelieverChoice
            reliever_choice = RelieverChoice(json.loads((ROOT / str(params['reliever_choice'])).read_text()))
            receipt['reliever_choice'] = {'path': params['reliever_choice'], 'name': reliever_choice.name}
        relief_exit = None
        if params.get('relief_exit'):
            from research_lab.game_sim.relief_exit import ReliefExit
            relief_exit = ReliefExit(json.loads((ROOT / str(params['relief_exit'])).read_text()))
            receipt['relief_exit'] = {'path': params['relief_exit'], 'name': relief_exit.name}
        base_state = None
        if params.get('base_state'):
            from brl_live.provider_adjust import load_base_state
            base_state = load_base_state(ROOT / str(params['base_state']))
            receipt['base_state'] = {'path': params['base_state'], 'name': base_state.get('name')}
        leash = None
        if params.get('leash'):
            from research_lab.game_sim.starter_leash import AppearanceIndex, Leash
            regular = h[h['game_type'] == 'R'] if 'game_type' in h.columns else h
            leash = (Leash(json.loads((ROOT / str(params['leash'])).read_text())), AppearanceIndex.from_frame(appearances(regular)))
            receipt['leash'] = {'path': params['leash'], 'name': leash[0].name, 'terms': leash[0].terms}
        steals = None
        if params.get('steals'):
            steals = {k: float(v) for k, v in dict(params['steals']).items() if k in ('per_pa', 'third')}
            receipt['steals'] = steals
        _SHARED.update(h=h, app=app, games=games, model_path=model_path, model_sha256=model_sha256, history_path=history_path, hazard_path=hazard_path,
                       n_sims=n_sims, physics_table=physics_table, offsets=offsets, rest=use_rest, environment=environment, team_offsets=team_offsets,
                       age_layer=age_layer, steals=steals, win_states=(params.get('win_states') if params.get('win_states') == 'split' else bool(params.get('win_states'))), starter_lines=bool(params.get('starter_lines')), role_offsets=role_offsets, real_pa_check=bool(params.get('real_pa_check')), transitions=transitions,
                       hitter_lines=bool(params.get('hitter_lines')), running_events=running_events, reliever_choice=reliever_choice, leash=leash, base_state=base_state, relief_exit=relief_exit)
        if params.get('win_states'):
            receipt['win_states'] = True
        stage(f'replay {len(games)} games with {workers} workers')
        dates = sorted(games['date'].unique())
        shards = [dates[w::workers] for w in range(workers)]
        records = []
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for part in pool.map(_worker, shards):
                    records.extend(part)
        else:
            records = _worker(dates)
        stage('score')
        sim = pd.DataFrame(records)
        sim['p_sim'] = (sim['home_wins'] + 0.5 * sim['ties'] + 0.5) / (sim['n'] + 1.0)
        sim['y'] = (sim['home_runs'] > sim['away_runs']).astype(float)
        from brl_live.team_model import TeamModel
        corpus_games = reconstruct(h)       # the team model uses every unambiguous prior game of the corpus
        corpus_games['date'] = corpus_games['date'].astype(str)
        corpus_games = corpus_games[~corpus_games['ambiguous']]
        tm = TeamModel([{'date': r.date, 'home_id': r.home, 'away_id': r.away, 'home_runs': r.home_runs, 'away_runs': r.away_runs} for r in corpus_games.itertuples()])
        p_team = []
        for r in sim.itertuples():
            try:
                p_team.append(tm.probability(r.date, r.home, r.away)['p_home'])
            except Exception:
                p_team.append(0.5)
        sim['p_team'] = p_team
        sim['p_blend'] = sigmoid(0.5 * (logit(sim['p_sim']) + logit(sim['p_team'])))
        sim['p_coin'] = 0.5
        receipt['scores'] = score(sim)
        sim['month'] = sim['date'].str[5:7]
        receipt['scores_by_month'] = {m: score(g) for m, g in sim.groupby('month')}
        if reference:
            raw = read_blob(repo, token, reference, branch)
            if raw is None:
                receipt['reference_missing'] = reference
            else:
                ref = pd.DataFrame([json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l.strip()])
                ref['p_sim'] = (ref['home_wins'] + 0.5 * ref['ties'] + 0.5) / (ref['n'] + 1.0)
                ref['y'] = (ref['home_runs'] > ref['away_runs']).astype(float)
                ref = ref.merge(sim[['game_pk', 'p_team']], on='game_pk')
                ref['p_blend'] = sigmoid(0.5 * (logit(ref['p_sim']) + logit(ref['p_team'])))
                receipt['paired_vs_reference'] = {'reference': reference, 'p_sim': paired(sim, ref, 'p_sim'), 'p_blend': paired(sim, ref, 'p_blend')}
        payload = gzip.compress('\n'.join(json.dumps(r) for r in records).encode() + b'\n', mtime=0)
        out_path = f'research/replay-{tag}-{run_id}{suffix}.jsonl.gz'
        put_bytes(repo, token, out_path, payload, branch, f'BRL: replay {tag}')
        receipt['per_game_file'] = out_path; receipt['per_game_sha256'] = hashlib.sha256(payload).hexdigest()
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt['status'] = 'failed'; receipt['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        frames = traceback.extract_tb(exc.__traceback__)
        receipt['where'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno} for f in frames[-6:]]
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat(); receipt['seconds'] = round(time.time() - t0, 1)
    text = json.dumps(receipt, indent=1, default=float)
    for secret in (key_hex, token):
        text = text.replace(secret, '[redacted]')
    put_bytes(repo, token, f'research/replay-{tag}-{run_id}{suffix}.json', text.encode(), branch, 'BRL: replay receipt ' + tag)
    print(json.dumps({k: receipt.get(k) for k in ('status', 'error', 'seconds', 'games')}))
    if receipt['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
