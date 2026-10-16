"""Pair two full-season replays (2025 and 2026) game by game on the simulator's win Brier and on the simulator-plus-team
blend the replay lane scores (logit average of the two), with 95% intervals from resampling dates. The team model's
chances come from a team_probs research receipt (tools/brl_research.py), the same for both replays; as a check, the
reference blend rebuilt here is compared with the blend Brier its own shards recorded.
Usage: python tools/replay_blend_compare.py <reference tag prefix> <candidate tag prefix> <team_probs receipt path> [out.json]
Data-branch aggregates only (per-game replay rows and game-level chances)."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = str(Path(__file__).resolve().parents[1])


def _git(*args, binary=False):
    out = subprocess.run(['git', '-C', REPO, *args], capture_output=True)
    return out.stdout if binary else out.stdout.decode()


def files(prefix, ext):
    return sorted(f for f in _git('ls-tree', '-r', '--name-only', 'origin/brl-live-data', 'research').split() if f.startswith(prefix) and f.endswith(ext))


def replay(tag):
    rows = []
    for f in files(f'research/replay-{tag}-', '.jsonl.gz'):
        rows += [json.loads(l) for l in gzip.decompress(_git('show', 'origin/brl-live-data:' + f, binary=True)).decode().splitlines() if l.strip()]
    d = pd.DataFrame(rows).drop_duplicates('game_pk')
    d['p_sim'] = (d['home_wins'] + 0.5 * d['ties'] + 0.5) / (d['n'] + 1.0)
    d['y'] = (d['home_runs'] > d['away_runs']).astype(float)
    return d[['game_pk', 'date', 'p_sim', 'y']]


def recorded_blend(tag):
    """The blend Brier the replay's own shards recorded, weighted by games."""
    tot, n = 0.0, 0
    for f in files(f'research/replay-{tag}-', '.json'):
        r = json.loads(_git('show', 'origin/brl-live-data:' + f))
        s = (r.get('scores') or {})
        if 'p_blend' in s:
            tot += s['p_blend']['brier'] * s['n_games']; n += s['n_games']
    return tot / n if n else None


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend(p_sim, p_team):
    return 1 / (1 + np.exp(-0.5 * (logit(p_sim) + logit(p_team))))


def paired(dates, diff, reps=4000, seed=7):
    u = np.unique(dates); idx = {d_: np.flatnonzero(dates == d_) for d_ in u}; rng = np.random.default_rng(seed)
    vals = [float(np.mean(np.concatenate([diff[idx[d_]] for d_ in rng.choice(u, len(u))]))) for _ in range(reps)]
    return {'difference': round(float(diff.mean()), 6), 'ci_95': [round(float(v), 6) for v in np.percentile(vals, [2.5, 97.5])]}


def compare(ref_prefix, new_prefix, p_team: dict) -> dict:
    out = {'reference': ref_prefix, 'candidate': new_prefix}
    pool = {'sim': [], 'blend': [], 'dates': []}
    for season in (2025, 2026):
        a, b = replay(f'{ref_prefix}-{season}'), replay(f'{new_prefix}-{season}')
        m = a.merge(b, on='game_pk', suffixes=('_ref', '_new'))
        m['p_team'] = [p_team.get(str(int(g)), np.nan) for g in m['game_pk']]
        missing = int(m['p_team'].isna().sum()); m = m[m['p_team'].notna()]
        y = m['y_ref'].to_numpy(float); dates = m['date_ref'].astype(str).to_numpy()
        sim_ref, sim_new = (m['p_sim_ref'] - y) ** 2, (m['p_sim_new'] - y) ** 2
        bl_ref, bl_new = pd.Series((blend(m['p_sim_ref'], m['p_team']) - y) ** 2), pd.Series((blend(m['p_sim_new'], m['p_team']) - y) ** 2)
        out[str(season)] = {'games': int(len(m)), 'games_without_team_chance': missing,
                            'sim_brier': {'reference': round(float(sim_ref.mean()), 6), 'candidate': round(float(sim_new.mean()), 6), **paired(dates, (sim_new - sim_ref).to_numpy(float))},
                            'blend_brier': {'reference': round(float(bl_ref.mean()), 6), 'candidate': round(float(bl_new.mean()), 6), **paired(dates, (bl_new - bl_ref).to_numpy(float)),
                                            'reference_recorded_by_its_shards': recorded_blend(f'{ref_prefix}-{season}')}}
        pool['sim'].append((sim_new - sim_ref).to_numpy(float)); pool['blend'].append((bl_new - bl_ref).to_numpy(float)); pool['dates'].append(np.char.add(f'{season}:', dates.astype(str)))
    dates = np.concatenate(pool['dates'])
    out['pooled'] = {'sim_brier': paired(dates, np.concatenate(pool['sim'])), 'blend_brier': paired(dates, np.concatenate(pool['blend']))}
    return out


if __name__ == '__main__':
    ref, new, team_path = sys.argv[1], sys.argv[2], sys.argv[3]
    p_team = json.loads(_git('show', 'origin/brl-live-data:' + team_path))['p_team']
    result = compare(ref, new, p_team)
    print(json.dumps(result, indent=1))
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(json.dumps(result, indent=1))
